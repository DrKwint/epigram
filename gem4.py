import jax
import jax.numpy as jnp
from flax import nnx
import heapq
import functools
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

# --- IMPORTS FROM YOUR PROJECT ---
# Replace these with your actual imports if they are in a different file
# from gem import AbstractENN, Zonotope
# For this standalone file to work, I will assume AbstractENN and Zonotope are available.
# If they are in 'gem.py', uncomment the line below:
from gem import AbstractENN, Zonotope
from src.net import ENN

# --- HELPER FUNCTIONS ---

def box_to_zonotope(min_vals: jax.Array, max_vals: jax.Array) -> Zonotope:
    """Creates a diagonal (axis-aligned) Zonotope from a hyper-rectangle."""
    center = (max_vals + min_vals) / 2.0
    radii = (max_vals - min_vals) / 2.0
    
    # Generators: Diagonal matrix [Dim, Dim] -> Expanded to [1, Dim, Dim]
    # We expand dims to match [Batch, N_Gens, Dim]
    generators = jnp.expand_dims(jnp.diag(radii), axis=0)
    center = jnp.expand_dims(center, axis=0)
    
    return Zonotope(center, generators)

def calculate_z_probability(zono: Zonotope, start_dim: int) -> float:
    """
    Approximates probability mass of the Z-region (starting at feature index `start_dim`).
    Uses simple Gaussian integration over the bounding box of the Z-slice.
    """
    import jax.scipy.stats.norm as norm
    
    c_z = zono.center[:, start_dim:]
    g_z = zono.generators[:, :, start_dim:]
    
    r_z = jnp.sum(jnp.abs(g_z), axis=1)
    lb_z = c_z - r_z
    ub_z = c_z + r_z
    
    # P(lb < z < ub) for standard normal
    probs = norm.cdf(ub_z) - norm.cdf(lb_z)
    total_prob = jnp.prod(probs)
    
    return float(total_prob)

def get_action_range(zono: Zonotope, start_dim: int, a_dim: int):
    """Extracts bounds of action dimensions."""
    end = start_dim + a_dim
    c_a = zono.center[:, start_dim:end]
    g_a = zono.generators[:, :, start_dim:end]
    
    r_a = jnp.sum(jnp.abs(g_a), axis=1)
    lb_a = c_a - r_a
    ub_a = c_a + r_a
    
    return lb_a, ub_a, c_a

# --- TRACING SYSTEM ---

@dataclass
class TrajectoryNode:
    step: int
    zono_id: int
    parent_id: int              # ID of the input zonotope at step t-1
    action_range: Tuple[jnp.ndarray, jnp.ndarray] # (lb, ub) of action used
    prob_mass: float
    zonotope: Zonotope

class VerificationTrace:
    def __init__(self):
        self.nodes: Dict[int, TrajectoryNode] = {}
        
    def add_node(self, node: TrajectoryNode):
        self.nodes[node.zono_id] = node
        
    def get_action_sequence(self, final_zono_id: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Backtracks from final node to root to reconstruct action sequence."""
        actions = []
        curr_id = final_zono_id
        
        while curr_id in self.nodes:
            node = self.nodes[curr_id]
            
            # Skip root node (step -1) which has no action
            if node.step >= 0:
                actions.insert(0, node.action_range)
            
            if node.parent_id == -1: # Root reached
                break
            curr_id = node.parent_id
            
        return actions

# --- CORE ENGINE ---

class ReachabilityEngine:
    def __init__(self, model: AbstractENN, safe_min: jax.Array, safe_max: jax.Array):
        self.model = model
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.input_dim = model.x_dim + model.a_dim 
        
        # JIT the gradient function for speed
        self._grad_loss_fn = jax.jit(jax.grad(self._loss_fn, argnums=0))

    def _loss_fn(self, generators, center_x, center_z, safe_min, safe_max):
        """Internal loss function for gradient calculation."""
        # Reconstruct zonotopes inside JIT boundary
        # Note: We must know the slice index. 
        # Since this is JITed, 'self.input_dim' is static constant.
        g_x = generators[:, :, :self.input_dim]
        g_z = generators[:, :, self.input_dim:]
        
        x_in = Zonotope(center_x, g_x)
        z_in = Zonotope(center_z, g_z)
        
        out = self.model(x_in, z_in)
        l, u = out.concrete_bounds()
        
        # Minimizing violation
        return jnp.sum(jnp.maximum(0.0, u - safe_max)) + \
               jnp.sum(jnp.maximum(0.0, safe_min - l))

    def verify_step(
        self, 
        zonotopes: List[Zonotope], 
        action_zonotope: Optional[Zonotope],
        step_idx: int, 
        max_splits=50
    ) -> List[Tuple[Zonotope, Zonotope]]:
        """
        Returns list of (RefinedInput, NextOutput) pairs.
        RefinedInput contains the safe split of Action/State/Z for step T.
        NextOutput contains the state for step T+1.
        """
        
        print(f"--- Verifying Step {step_idx} (Input Regions: {len(zonotopes)}) ---")
        
        successful_pairs = []
        pq = []
        entry_id = 0
        
        # 1. Initialize Queue
        for z_in in zonotopes:
            # Slicing
            c_x = z_in.center[:, :self.input_dim]
            c_z = z_in.center[:, self.input_dim:]
            g_x = z_in.generators[:, :, :self.input_dim]
            g_z = z_in.generators[:, :, self.input_dim:]
            
            # Initial Propagation
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            margin = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            # Push to queue. 
            # IMPORTANT: Ensure z_in has an ID. If not, this is a bug in the caller.
            if not hasattr(z_in, 'id'):
                z_in.id = -999 # Fallback
                
            heapq.heappush(pq, (-float(margin), entry_id, z_in))
            entry_id += 1

        splits_done = 0
        
        # 2. Refinement Loop
        while len(pq) > 0:
            neg_margin, _, curr_z_in = heapq.heappop(pq)
            margin = -neg_margin
            
            # Re-slice and Propagate
            c_x = curr_z_in.center[:, :self.input_dim]
            c_z = curr_z_in.center[:, self.input_dim:]
            g_x = curr_z_in.generators[:, :, :self.input_dim]
            g_z = curr_z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            worst_viol = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            # --- SAFE CASE ---
            if worst_viol < 0:
                # 1. Align Epistemic Errors (Axis 1)
                n_out = z_out.generators.shape[1]
                n_in_z = g_z.shape[1]
                
                if n_out > n_in_z:
                    pad = jnp.zeros((g_z.shape[0], n_out - n_in_z, g_z.shape[2]))
                    g_z_padded = jnp.concatenate([g_z, pad], axis=1)
                    g_x_next = z_out.generators
                elif n_out < n_in_z:
                    pad = jnp.zeros((z_out.generators.shape[0], n_in_z - n_out, z_out.generators.shape[2]))
                    g_x_next = jnp.concatenate([z_out.generators, pad], axis=1)
                    g_z_padded = g_z
                else:
                    g_x_next = z_out.generators
                    g_z_padded = g_z

                # 2. Inject Future Action (Axis 2)
                # Calculate gap between Output (State) and Input (State+Action)
                # We assume self.input_dim = x_dim + a_dim
                gap_size = self.input_dim - z_out.center.shape[1]
                
                if gap_size > 0:
                    if action_zonotope is None:
                        # Default zero action
                        act_c = jnp.zeros((1, gap_size))
                        act_g = jnp.zeros((1, g_x_next.shape[1], gap_size))
                    else:
                        # Slice/Pad action zonotope
                        act_c = action_zonotope.center[:, :gap_size]
                        act_g_raw = action_zonotope.generators[:, :, :gap_size]
                        
                        # Pad action gens to match depth
                        if act_g_raw.shape[1] < g_x_next.shape[1]:
                            diff = g_x_next.shape[1] - act_g_raw.shape[1]
                            pad = jnp.zeros((1, diff, act_g_raw.shape[2]))
                            act_g = jnp.concatenate([act_g_raw, pad], axis=1)
                        elif act_g_raw.shape[1] > g_x_next.shape[1]:
                             # Pad others to match action
                             diff = act_g_raw.shape[1] - g_x_next.shape[1]
                             pad_x = jnp.zeros((g_x_next.shape[0], diff, g_x_next.shape[2]))
                             g_x_next = jnp.concatenate([g_x_next, pad_x], axis=1)
                             pad_z = jnp.zeros((g_z_padded.shape[0], diff, g_z_padded.shape[2]))
                             g_z_padded = jnp.concatenate([g_z_padded, pad_z], axis=1)
                             act_g = act_g_raw
                        else:
                            act_g = act_g_raw
                    
                    new_center = jnp.concatenate([z_out.center, act_c, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_next, act_g, g_z_padded], axis=2)
                else:
                    # No gap
                    new_center = jnp.concatenate([z_out.center, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_next, g_z_padded], axis=2) 
                
                next_step_zono = Zonotope(new_center, new_gens)
                
                # Save pair
                successful_pairs.append((curr_z_in, next_step_zono))
                continue
                
            # --- SPLIT CASE ---
            if splits_done >= max_splits:
                continue

            # Calculate Gradient
            grads = self._grad_loss_fn(
                curr_z_in.generators, c_x, c_z, self.safe_min, self.safe_max
            )
            impact = jnp.sum(jnp.abs(grads * curr_z_in.generators), axis=(0, 2))
            split_idx = jnp.argmax(impact)
            
            splits_done += 1
            gen_vec = curr_z_in.generators[:, split_idx, :]
            
            # Create Children
            c_l = curr_z_in.center - 0.5 * gen_vec
            g_l = curr_z_in.generators.at[:, split_idx, :].set(0.5 * gen_vec)
            
            c_r = curr_z_in.center + 0.5 * gen_vec
            
            z_l = Zonotope(c_l, g_l)
            z_r = Zonotope(c_r, g_l) # Same gens
            
            # INHERIT ID so we can trace back to parent
            z_l.id = curr_z_in.id
            z_r.id = curr_z_in.id
            
            heapq.heappush(pq, (-float(margin), entry_id, z_l))
            entry_id += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_r))
            entry_id += 1
            
        return successful_pairs

# --- MAIN DRIVER ---

def run_trajectory_rollout_with_trace(
    model, 
    x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=100
):
    print(f"--- Starting Robust Rollout (H={horizon}) ---")
    
    # Setup
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    
    # Dimensions
    x_dim = model.x_dim
    a_dim = model.a_dim
    
    # Root Zonotope
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    root_zono = box_to_zonotope(joint_min, joint_max)
    
    # ID tracking
    root_zono.id = 0
    next_id = 1
    
    # Add Root to Trace
    trace.add_node(TrajectoryNode(
        step=-1, zono_id=0, parent_id=-1, 
        action_range=(None, None), prob_mass=1.0, zonotope=root_zono
    ))
    
    current_regions = [root_zono]
    
    # Reusable Action Constraints
    act_min = x_min[x_dim:]
    act_max = x_max[x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)

    for t in range(horizon):
        if not current_regions:
            print(f"Horizon {t}: Trajectory Dead (No safe regions).")
            break
        
        # Verify Step
        # Pass t > 0 check: if t=0, action is embedded in root. If t>0, we inject fresh action.
        # Actually, standard pattern: Inject fresh action for NEXT step always?
        # No, 'action_zonotope' in verify_step is used to BUILD the Output for T+1.
        # So we always need it.
        
        step_results = engine.verify_step(
            current_regions, 
            action_zonotope=action_zono, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )
        
        next_regions = []
        
        for (refined_input, next_output) in step_results:
            # 1. Extract Safe Action Range from Input
            # Action is at [x_dim : x_dim+a_dim]
            lb_a, ub_a, _ = get_action_range(refined_input, x_dim, a_dim)
            prob = calculate_z_probability(refined_input, x_dim + a_dim)
            
            # 2. Get Parent ID
            parent_id = getattr(refined_input, 'id', -1)
            
            # 3. Assign New ID to Output
            next_output.id = next_id
            
            # 4. Record Trace
            trace.add_node(TrajectoryNode(
                step=t,
                zono_id=next_id,
                parent_id=parent_id,
                action_range=(lb_a, ub_a),
                prob_mass=prob,
                zonotope=next_output
            ))
            
            next_regions.append(next_output)
            next_id += 1
            
        current_regions = next_regions
        print(f"  Step {t+1}: {len(current_regions)} safe regions found.")
        yield current_regions, trace

def print_best_trajectory(safe_regions, trace: VerificationTrace, model):
    """
    Identifies the most robust final state and prints the full action sequence 
    that led to it.
    """
    if not safe_regions:
        print("❌ No safe trajectory found.")
        return

    print(f"\n--- Analysis of {len(safe_regions)} Safe Trajectories ---")

    # 1. Find the "Best" final region
    # We judge "Best" by the one that covers the largest probability mass of Z
    # (i.e., the most robust to physics uncertainty)
    best_node = None
    max_prob = -1.0

    for zono in safe_regions:
        # Look up the trace node for this zonotope ID
        if hasattr(zono, 'id') and zono.id in trace.nodes:
            node = trace.nodes[zono.id]
            if node.prob_mass > max_prob:
                max_prob = node.prob_mass
                best_node = node
    
    if best_node is None:
        print("Error: Could not link final regions to trace.")
        return

    print(f"🏆 Best Final Region (ID {best_node.zono_id})")
    print(f"   Robustness Probability: {best_node.prob_mass:.4f}")
    
    # 2. Extract History
    # This returns a list of (lb, ub) tuples for t=0 to t=H
    actions = trace.get_action_sequence(best_node.zono_id)
    
    # 3. Print Sequence
    print("\n--- Optimal Robust Action Sequence ---")
    for t, (lb, ub) in enumerate(actions):
        # Calculate width to see how 'tight' the constraint is
        width = ub - lb
        center = (ub + lb) / 2.0
        
        # Determine strictness string
        strictness = "Flexible" if jnp.mean(width) > 0.1 else "Strict"
        
        print(f"Step {t}:")
        print(f"  Action Range: [{lb[0,0]:.4f}, {ub[0,0]:.4f}]")
        print(f"  Nominal Act : {center[0,0]:.4f}")
        print(f"  Uncertainty : +/- {width[0,0]/2:.4f} ({strictness})")
        print("-" * 30)

def print_search_debug(trace, step, parent_id, current_action_range, current_prob):
    """
    Prints the current search depth, the specific action being attempted, 
    and the robustness probability of the path so far.
    """
    # 1. Reconstruct History (Actions 0 to t-1)
    # We look up the parent node to find the path that led here
    history = []
    if parent_id != -1 and parent_id in trace.nodes:
        # Reuses the existing backtrack logic
        history = trace.get_action_sequence(parent_id)
        
    # 2. Format Current Action
    lb, ub = current_action_range
    width = ub - lb
    
    # 3. Print
    indent = "  " * step
    print(f"\n{indent}🔍 DFS Depth {step}: Exploring Candidate...")
    print(f"{indent}   Robustness Mass : {current_prob:.4f}")
    print(f"{indent}   Current Action  : [{lb[0,0]:.3f}, {ub[0,0]:.3f}] (Width {width[0,0]:.3f})")
    
    if history:
        print(f"{indent}   Path Context    : {len(history)} previous steps")
        # Optional: Print last few steps of context
        # last_step = history[-1]
        # print(f"{indent}   Prev Action     : [{last_step[0][0,0]:.2f}, {last_step[1][0,0]:.2f}]")

class GeneratorTracker:
    def __init__(self, z_dim):
        # We start with Z columns
        self.map = [] # List of (start_col, end_col, type, step_idx)
        self.map.append((0, z_dim, 'z', -1))
        self.current_cols = z_dim
        
    def register_action(self, step_idx, n_cols):
        # When we inject a new action zonotope, we record its column location
        start = self.current_cols
        end = start + n_cols
        self.map.append((start, end, 'action', step_idx))
        self.current_cols = end
        
    def identify_culprit(self, impact_scores):
        """
        Given a vector of impacts (one per column), find the source.
        Returns: (step_to_blame, type)
        """
        # Find index of max impact
        max_idx = jnp.argmax(impact_scores)
        
        # Look up in map
        for (start, end, type_, step) in self.map:
            if start <= max_idx < end:
                return step, type_
        
        return -1, 'unknown' # Likely a ReLU noise term (not actionable)

def run_trajectory_search_dfs(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=50
):
    print(f"--- Starting Depth-First Robust Search (H={horizon}) ---")
    
    # 1. Setup Engine & Trace
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    
    # Setup Root
    x_dim = model.x_dim
    a_dim = model.a_dim
    z_min = jnp.ones(z_dim) * -3.0; z_max = jnp.ones(z_dim) * 3.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    root_zono = box_to_zonotope(joint_min, joint_max)
    
    # Assign IDs
    root_zono.id = 0
    trace.global_id_counter = 1
    
    trace.add_node(TrajectoryNode(
        step=-1, zono_id=0, parent_id=-1, 
        action_range=(None, None), prob_mass=1.0, zonotope=root_zono
    ))
    
    # Reusable Action Constraints
    act_min = x_min[x_dim:]; act_max = x_max[x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)

    # 2. Define Recursive DFS Function
    def dfs_step(current_zono, t):
        # Base Case: Goal Reached
        if t == horizon:
            return [current_zono] # Success! Return the successful leaf
        
        print(f"  DFS Depth {t} -> {t+1} (Parent ID: {current_zono.id})")
        
        # Run Verification Step for ONE node
        # We increase splits here because we are focused on ONE path, so we can afford it.
        results = engine.verify_step(
            [current_zono], 
            action_zonotope=action_zono, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )

        # Heuristic Sort: Try the "Most Robust" children first
        # (Regions with higher probability mass are more likely to be stable)
        results.sort(
            key=lambda p: calculate_z_probability(p[0], x_dim + a_dim), 
            reverse=True
        )
        
        for (refined_input, next_output) in results:
            # Register in Trace
            node_id = trace.global_id_counter
            trace.global_id_counter += 1
            
            # Extract Info
            lb_a, ub_a, _ = get_action_range(refined_input, x_dim, a_dim)
            prob = calculate_z_probability(refined_input, x_dim + a_dim)
            parent_id = getattr(refined_input, 'id', -1)
            
            next_output.id = node_id # Tag for next step
            
            trace.add_node(TrajectoryNode(
                step=t, zono_id=node_id, parent_id=parent_id,
                action_range=(lb_a, ub_a), prob_mass=prob, zonotope=next_output
            ))
            
            # --- RECURSION (The Magic) ---
            # Immediately try to extend this child to the end.
            print_search_debug(trace, t, parent_id, (lb_a, ub_a), prob)
            path_result = dfs_step(next_output, t + 1)
            
            if path_result:
                # If the child returned success, bubble it up!
                return path_result

        # If loop finishes, ALL children failed. 
        # Backtrack (return None implicitly)
        # print(f"  Backtracking from Depth {t}")
        return None

    # 3. Start Search
    final_leafs = dfs_step(root_zono, 0)
    
    if final_leafs:
        print("✅ Trajectory Verified via DFS!")
        return final_leafs, trace
    else:
        print("❌ DFS Search Failed to find any safe trajectory.")
        return [], trace
    
def run_directed_dfs(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=5, max_splits_per_step=50
):
    print(f"--- Starting Trace-Based Sensitivity Search (H={horizon}) ---")
    
    # 1. Setup
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    trace.global_id_counter = 1
    
    # Dimensions
    x_dim = model.x_dim
    a_dim = model.a_dim
    
    # Initialize Tracker
    tracker = GeneratorTracker(z_dim)
    
    # Root Zono
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    root_zono = box_to_zonotope(joint_min, joint_max)
    
    root_zono.id = 0
    trace.add_node(TrajectoryNode(-1, 0, -1, (None,None), 1.0, root_zono))
    
    # Action Consts
    act_min = x_min[x_dim:]; act_max = x_max[x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)

    # ---------------------------------------------------------
    # DFS Function
    # ---------------------------------------------------------
    def dfs_step(current_zono, t):
        # Base Case
        if t == horizon:
            return [current_zono]

        # 1. Update Tracker
        # We need to register that we are adding 'Action' dimensions at this step.
        # Note: In the 'verify_step' logic, we inject Action noise (gap_size).
        # We assume the gap is always 'a_dim'.
        # IMPORTANT: We must only register ONCE per depth if strictly tracking,
        # but since 'tracker' is global/mutable, this is tricky in recursion.
        # Hack: We calculate the 'expected' cols based on depth.
        # Cols = z_dim + (t+1) * a_dim + (t * hidden_relu_noise?)
        # For simplicity in this demo, we assume the tracker just appends.
        tracker.register_step(t, a_dim)

        # 2. Verify / Split
        # print(f"  > Depth {t} verifying...") # Debug print
        results = engine.verify_step(
            [current_zono], 
            action_zonotope=action_zono, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )
        
        # 3. Dead End Analysis
        if not results:
            print(f"  🛑 Dead End at Depth {t+1}")
            
            # --- BLAME ANALYSIS ---
            # To blame, we need the FAILED output. verify_step only returns SAFE ones.
            # We quickly re-run the propagation on the 'worst case' point to fail it.
            # (Simplified: we just blame the current inputs)
            
            # Heuristic Blame: We calculate blame on the CURRENT zonotope w.r.t
            # the future safety constraint (Lookahead Blame).
            # But simpler: just return a generic request to split parent.
            return RefinementRequest(target_step=t-1, reason="dead_end")

        # 4. Sort (Greedy Heuristic)
        results.sort(
            key=lambda p: calculate_z_probability(p[0], x_dim+a_dim), 
            reverse=True
        )

        # 5. Recurse
        for i, (refined_input, next_output) in enumerate(results):
            # Trace Logging
            node_id = trace.global_id_counter
            trace.global_id_counter += 1
            parent_id = getattr(refined_input, 'id', -1)
            
            lb_a, ub_a, _ = get_action_range(refined_input, x_dim, a_dim)
            prob = calculate_z_probability(refined_input, x_dim+a_dim)
            
            # Tag ID
            next_output.id = node_id
            
            trace.add_node(TrajectoryNode(
                t, node_id, parent_id, (lb_a, ub_a), prob, next_output
            ))
            
            # DEBUG PRINT
            if i == 0: # Print only the primary candidate to avoid spam
                width = ub_a - lb_a
                print(f"  Step {t}: Trying Action [{lb_a[0,0]:.2f}, {ub_a[0,0]:.2f}] (P={prob:.2f})")

            # RECURSE
            res = dfs_step(next_output, t + 1)
            
            # Handle Return
            if isinstance(res, list):
                return res # Success!
            
            elif isinstance(res, RefinementRequest):
                # Backtracking Signal
                if res.target_step == t:
                    print(f"  ⚡ Backtrack caught at Step {t}. Switching branch...")
                    continue # Try next sibling in 'results' loop
                else:
                    return res # Bubble up

        # All siblings failed
        return None

    # ---------------------------------------------------------
    # Start Search
    # ---------------------------------------------------------
    final = dfs_step(root_zono, 0)
    
    if isinstance(final, list) and len(final) > 0:
        print("\n✅ Success! Safe Trajectory Found.")
        return final, trace
    else:
        print("\n❌ Failure: Search exhausted or root failed.")
        return [], trace

if __name__ == "__main__":
    import dill as pickle
    from pathlib import Path

    rngs = nnx.Rngs(0)
    enn: ENN = pickle.loads(Path("model.pkl").read_bytes())

    model = AbstractENN.from_concrete(enn, rngs)

    import gymnasium as gym
    env = gym.make('InvertedPendulum-v5')
    obs, _ = env.reset()
    input_min = jnp.concatenate([obs, jnp.array([-3])])
    input_max = jnp.concatenate([obs, jnp.array([3])])

    safe_min = jnp.array([-10., -0.2, -10., -10.])
    safe_max = jnp.array([10., 0.2, 10., 10.])

    run_directed_dfs(model, input_min, input_max, safe_min, safe_max, horizon=10)

    # run_trajectory_search_dfs(model, input_min, input_max, safe_min, safe_max, horizon=10)

    # for horizon, (regions, trace) in enumerate(run_trajectory_rollout_with_trace(model, input_min, input_max, safe_min, safe_max, horizon=10)):
        # print("Horizon:", horizon)
        # print_best_trajectory(regions, trace, model)