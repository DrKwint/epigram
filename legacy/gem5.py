import jax
import jax.numpy as jnp
from flax import nnx
import heapq
import functools
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass

from gem import AbstractENN, Zonotope

# --- DATA STRUCTURES ---

@dataclass
class RefinementRequest:
    """Signal sent up the recursion stack to request a split at a specific past step."""
    target_step: int
    reason: str  # 'action' or 'z'

@dataclass
class TrajectoryNode:
    """A node in the search tree for backtracking and visualization."""
    step: int
    zono_id: int
    parent_id: int
    action_range: Tuple[jnp.ndarray, jnp.ndarray]
    prob_mass: float
    zonotope: 'Zonotope'

class VerificationTrace:
    def __init__(self):
        self.nodes: Dict[int, TrajectoryNode] = {}
        self.global_id_counter = 1
        
    def add_node(self, node: TrajectoryNode):
        self.nodes[node.zono_id] = node
        
    def get_action_sequence(self, final_zono_id: int) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
        actions = []
        curr_id = final_zono_id
        while curr_id in self.nodes:
            node = self.nodes[curr_id]
            if node.step >= 0:
                actions.insert(0, node.action_range)
            if node.parent_id == -1: break
            curr_id = node.parent_id
        return actions

class GeneratorTracker:
    """Tracks which columns of the zonotope generators correspond to which step/type."""
    def __init__(self, z_dim):
        # (start_col, end_col, type, step_idx)
        # Initialize with Z parameters (Step -1)
        self.map = [(0, z_dim, 'z', -1)]
        self.current_cols = z_dim
        
    def register_step(self, step_idx, added_cols):
        if added_cols > 0:
            start = self.current_cols
            end = start + added_cols
            # Check if already registered (simple check for recursion)
            if not any(m[3] == step_idx and m[2] == 'action' for m in self.map):
                self.map.append((start, end, 'action', step_idx))
                self.current_cols = end

    def identify_culprit(self, impact_scores):
        max_idx = jnp.argmax(impact_scores)
        for (start, end, type_, step) in self.map:
            if start <= max_idx < end:
                return step, type_
        return -1, 'noise'

# --- HELPER FUNCTIONS ---

def box_to_zonotope(min_vals: jax.Array, max_vals: jax.Array) -> Zonotope:
    center = (max_vals + min_vals) / 2.0
    radii = (max_vals - min_vals) / 2.0
    generators = jnp.expand_dims(jnp.diag(radii), axis=0)
    center = jnp.expand_dims(center, axis=0)
    return Zonotope(center, generators)

def calculate_z_probability(zono: Zonotope, start_dim: int) -> float:
    import jax.scipy.stats.norm as norm
    c_z = zono.center[:, start_dim:]
    g_z = zono.generators[:, :, start_dim:]
    r_z = jnp.sum(jnp.abs(g_z), axis=1)
    lb, ub = c_z - r_z, c_z + r_z
    return float(jnp.prod(norm.cdf(ub) - norm.cdf(lb)))

def get_action_range(zono: Zonotope, start_dim: int, a_dim: int):
    end = start_dim + a_dim
    c = zono.center[:, start_dim:end]
    g = zono.generators[:, :, start_dim:end]
    r = jnp.sum(jnp.abs(g), axis=1)
    return c - r, c + r, c

def calculate_blame(zonotope, safe_min, safe_max, tracker):
    """Identifies which generator component causes the most safety violation."""
    lb, ub = zonotope.concrete_bounds()
    
    # lb, ub have shape [Batch, Dim]
    viol_max = jnp.maximum(0.0, ub - safe_max)
    viol_min = jnp.maximum(0.0, safe_min - lb)
    
    if jnp.sum(viol_max) + jnp.sum(viol_min) == 0:
        return -1, 'none'
    
    # Gradient Direction w has shape [Batch, Dim]
    w = (viol_max > 0).astype(jnp.float32) - (viol_min > 0).astype(jnp.float32)
    
    # Project Generators onto Violation Direction
    # We use einsum to handle the dimensions explicitly:
    # 'bnd' = Generators [Batch, N_Cols, Dim]
    # 'bd'  = w [Batch, Dim]
    # -> 'bn' = Projection [Batch, N_Cols]
    projection = jnp.einsum('bnd,bd->bn', zonotope.generators, w)
    
    impacts = jnp.sum(jnp.abs(projection), axis=0)
    
    return tracker.identify_culprit(impacts)

# --- CORE ENGINE ---

class ReachabilityEngine:
    def __init__(self, model, safe_min, safe_max):
        self.model = model
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.input_dim = model.x_dim + model.a_dim
        self._grad_loss_fn = jax.jit(jax.grad(self._loss_fn, argnums=0))

    def _loss_fn(self, generators, center_x, center_z, safe_min, safe_max):
        g_x = generators[:, :, :self.input_dim]
        g_z = generators[:, :, self.input_dim:]
        out = self.model(Zonotope(center_x, g_x), Zonotope(center_z, g_z))
        l, u = out.concrete_bounds()
        return jnp.sum(jnp.maximum(0.0, u - safe_max)) + \
               jnp.sum(jnp.maximum(0.0, safe_min - l))

    def verify_step(self, zonotopes, action_zonotope, step_idx, max_splits=50):
        successful_pairs = []
        pq = []
        entry_id = 0
        
        # 1. Init Queue
        for z_in in zonotopes:
            c_x = z_in.center[:, :self.input_dim]
            c_z = z_in.center[:, self.input_dim:]
            g_x = z_in.generators[:, :, :self.input_dim]
            g_z = z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            margin = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            if not hasattr(z_in, 'id'): z_in.id = -999
            heapq.heappush(pq, (-float(margin), entry_id, z_in))
            entry_id += 1

        splits_done = 0
        
        # 2. Refine
        while len(pq) > 0:
            neg_margin, _, curr_z_in = heapq.heappop(pq)
            
            c_x = curr_z_in.center[:, :self.input_dim]
            c_z = curr_z_in.center[:, self.input_dim:]
            g_x = curr_z_in.generators[:, :, :self.input_dim]
            g_z = curr_z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            worst_viol = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            # --- SAFE ---
            if worst_viol < 0:
                # Reconstruct Next State
                n_out = z_out.generators.shape[1]
                n_in_z = g_z.shape[1]
                
                # Align Epistemic Axis (1)
                if n_out > n_in_z:
                    pad = jnp.zeros((g_z.shape[0], n_out - n_in_z, g_z.shape[2]))
                    g_z_padded = jnp.concatenate([g_z, pad], axis=1)
                    g_x_next = z_out.generators
                elif n_out < n_in_z:
                    pad = jnp.zeros((z_out.generators.shape[0], n_in_z - n_out, z_out.generators.shape[2]))
                    g_x_next = jnp.concatenate([z_out.generators, pad], axis=1)
                    g_z_padded = g_z
                else:
                    g_x_next, g_z_padded = z_out.generators, g_z
                
                # Inject Action (Axis 2)
                gap_size = self.input_dim - z_out.center.shape[1]
                if gap_size > 0:
                    if action_zonotope is None:
                        act_c = jnp.zeros((1, gap_size))
                        act_g = jnp.zeros((1, g_x_next.shape[1], gap_size))
                    else:
                        act_c = action_zonotope.center[:, :gap_size]
                        act_g = action_zonotope.generators[:, :, :gap_size]
                        # Pad Action Generators to match depth
                        if act_g.shape[1] < g_x_next.shape[1]:
                            diff = g_x_next.shape[1] - act_g.shape[1]
                            pad = jnp.zeros((1, diff, act_g.shape[2]))
                            act_g = jnp.concatenate([act_g, pad], axis=1)
                    
                    new_center = jnp.concatenate([z_out.center, act_c, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_next, act_g, g_z_padded], axis=2)
                else:
                    new_center = jnp.concatenate([z_out.center, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_next, g_z_padded], axis=2)
                
                next_zono = Zonotope(new_center, new_gens)
                successful_pairs.append((curr_z_in, next_zono))
                continue
            
            # --- SPLIT ---
            if splits_done >= max_splits: continue

            grads = self._grad_loss_fn(curr_z_in.generators, c_x, c_z, self.safe_min, self.safe_max)
            impact = jnp.sum(jnp.abs(grads * curr_z_in.generators), axis=(0, 2))
            
            # Boost Action Impact
            a_start = self.model.x_dim
            a_end = a_start + self.model.a_dim
            impact = impact.at[a_start:a_end].multiply(2.0)
            
            split_idx = jnp.argmax(impact)
            gen_vec = curr_z_in.generators[:, split_idx, :]
            
            c_l = curr_z_in.center - 0.5 * gen_vec
            g_l = curr_z_in.generators.at[:, split_idx, :].set(0.5 * gen_vec)
            
            z_l = Zonotope(c_l, g_l); z_l.id = curr_z_in.id
            z_r = Zonotope(curr_z_in.center + 0.5 * gen_vec, g_l); z_r.id = curr_z_in.id
            
            splits_done += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_l)); entry_id += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_r)); entry_id += 1
            
        return successful_pairs

    def force_split(self, zono: Zonotope, reason: str) -> List[Zonotope]:
        """
        Ignores gradients and blindly splits the widest/most impactful 
        generator of the requested type (z or action).
        """
        # 1. Identify relevant column indices
        if reason == 'z':
            # Z parameters are usually the first few columns (0 to z_dim)
            # We assume self.input_dim is where Z starts in the CENTER vector
            # But in GENERATORS, Z noise is usually the first N columns 
            # if we initialized it that way. 
            # Let's rely on the fact that Z generators are usually 
            # the ones affecting the Z-dimensions of the center.
            
            # Heuristic: Look at generators affecting the Z-output dimensions
            start_feat = self.input_dim # Z dimensions start here
            relevant_gens = zono.generators[:, :, start_feat:]
            # Sum abs values to find which column has most "power" over Z
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
            
        elif reason == 'action':
            # Look at generators affecting Action dimensions
            start_feat = self.model.x_dim
            end_feat = start_feat + self.model.a_dim
            relevant_gens = zono.generators[:, :, start_feat:end_feat]
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
            
        else:
            return []

        # 2. Pick best split index
        split_idx = jnp.argmax(power)
        
        # 3. Execute Split
        gen_vec = zono.generators[:, split_idx, :]
        
        # Left Child
        c_l = zono.center - 0.5 * gen_vec
        g_l = zono.generators.at[:, split_idx, :].set(0.5 * gen_vec)
        z_l = Zonotope(c_l, g_l); z_l.id = zono.id
        
        # Right Child
        c_r = zono.center + 0.5 * gen_vec
        z_r = Zonotope(c_r, g_l); z_r.id = zono.id
        
        return [z_l, z_r]

# --- MAIN DIRECTED DFS ---

def run_directed_dfs(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=50
):
    print(f"--- Starting Trace-Based Sensitivity Search (H={horizon}) ---")
    
    x_dim, a_dim = model.x_dim, model.a_dim
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    tracker = GeneratorTracker(z_dim)
    
    # Root Setup
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    
    root_zono = box_to_zonotope(joint_min, joint_max)
    root_zono.id = 0
    trace.add_node(TrajectoryNode(-1, 0, -1, (None,None), 1.0, root_zono))
    
    act_min, act_max = x_min[x_dim:], x_max[x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)

    def dfs_step(current_zono, t):
        if t == horizon: return [current_zono]

        # 1. Update Tracker (Register new action cols for this depth)
        tracker.register_step(t, a_dim)

        # 2. Verify
        results = engine.verify_step(
            [current_zono], 
            action_zonotope=action_zono, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )
        
        # 3. Dead End Analysis
        if not results:
            # Re-propagate to get failed state for blame analysis
            # Construct input manually (simplified reconstruction)
            c_x = current_zono.center[:, :x_dim+a_dim]
            c_z = current_zono.center[:, x_dim+a_dim:]
            g_x = current_zono.generators[:, :, :x_dim+a_dim]
            g_z = current_zono.generators[:, :, x_dim+a_dim:]
            
            # Simple Forward
            failed_zono = engine.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            
            step_blame, type_blame = calculate_blame(failed_zono, safe_min, safe_max, tracker)
            print(f"  🛑 Dead End at Depth {t+1}. Blaming '{type_blame}' at Step {step_blame}")
            
            return RefinementRequest(target_step=step_blame, reason=type_blame)

        # 4. Sort & Recurse
        # Sort & Recurse Loop
        # We wrap the results in a mutable list so we can append forced splits
        candidates = results 
        candidates.sort(key=lambda p: calculate_z_probability(p[0], x_dim+a_dim), reverse=True)
        
        candidate_idx = 0
        while candidate_idx < len(candidates):
            refined_input, next_output = candidates[candidate_idx]
            candidate_idx += 1
            
            # ... (Trace logging setup same as before) ...
            
            # Recurse
            res = dfs_step(next_output, t + 1)
            
            if isinstance(res, list): return res
            
            if isinstance(res, RefinementRequest):
                # --- NEW LOGIC ---
                if res.target_step == t:
                    print(f"  ⚡ Backtrack caught at Step {t}. Reason: '{res.reason}'")
                    
                    # Instead of just 'continue' (trying next stale sibling),
                    # we FORCE a split on the requested reason.
                    
                    print(f"     -> Forcing new split on {res.reason}...")
                    
                    # 1. Force split the CURRENT input zonotope (current_zono)
                    # Note: We need to split the PARENT of the failed node.
                    # 'current_zono' IS the parent of the failed node (next_output).
                    # Actually, 'refined_input' was the specific slice we just tried.
                    # We should try to split 'refined_input' further!
                    
                    new_inputs = engine.force_split(refined_input, res.reason)
                    
                    if not new_inputs:
                        print("     -> Could not split further. Continuing...")
                        continue
                        
                    # 2. Verify the new children immediately
                    # We treat them as new priority candidates
                    for inp in new_inputs:
                        # Verify to get the corresponding output
                        # We pass list size 1
                        new_pairs = engine.verify_step([inp], action_zonotope=action_zono, step_idx=t)
                        if new_pairs:
                            # Add to the FRONT of the candidates list to try immediately
                            # This implements Depth-First Priority for the refinement
                            candidates.insert(candidate_idx, new_pairs[0])
                    
                    # Continue the while loop; it will pick up the new inserted candidates next.
                    continue
                    
                else:
                    return res # Bubble up
                    
        return None

    final = dfs_step(root_zono, 0)
    
    if isinstance(final, list) and len(final) > 0:
        print("\n✅ Success! Safe Trajectory Found.")
        return final, trace
    else:
        print("\n❌ Failure: Search exhausted.")
        return [], trace

def print_best_trajectory(safe_regions, trace: VerificationTrace, model):
    if not safe_regions: return
    print(f"\n--- Optimal Robust Action Sequence ---")
    
    # Trace back from first leaf
    actions = trace.get_action_sequence(safe_regions[0].id)
    
    for t, (lb, ub) in enumerate(actions):
        width = ub - lb
        center = (ub + lb) / 2.0
        print(f"Step {t}: Act [{lb[0,0]:.3f}, {ub[0,0]:.3f}] (Nom: {center[0,0]:.3f})")

# --- EXECUTION ---
if __name__ == "__main__":
    import dill as pickle
    from pathlib import Path
    from src.net import ENN

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
