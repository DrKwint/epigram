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
    target_step: int
    reason: str

@dataclass
class TrajectoryNode:
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

def calculate_blame(impacts, x_dim, a_dim, z_dim, n_action_gens, action_bias=10.0):
    """
    Analyzes gradient impacts to assign blame to 'action', 'z', or the past ('history').
    
    The 'impacts' array contains the summed absolute values of (gradient * generator)
    for each generator of the FAILED input zonotope. We need to map this back to what
    each generator represents.
    
    Based on `verify_step` reconstruction:
    - The final N_ACTION_GENS columns are from the current action_zono.
    - The columns before that are from the previous step's model output and Z-noise.
    """
    if jnp.sum(impacts) < 1e-7:
        # If total impact is negligible, the violation is likely due to the
        # concrete center point. This is a "nominal crash". We can't use
        # gradient attribution, so we heuristically blame the most recent action.
        return 0, 'action' # Blame current step 't' for action

    # Identify the generators corresponding to the current action
    n_total_gens = impacts.shape[0]
    n_history_gens = n_total_gens - n_action_gens
    
    action_impact = jnp.sum(impacts[n_history_gens:])
    history_impact = jnp.sum(impacts[:n_history_gens])

    # To be more fine-grained, we could try to separate Z from X impacts,
    # but this requires a much more complex generator tracking system.
    # For now, we distinguish "past" (history) from "present" (action).
    # A simple heuristic is to check if the action is the dominant cause.
    
    # We can approximate Z's impact. In the initial state, the last z_dim
    # generators are for Z. This is a rough heuristic.
    z_impact = 0.0
    if n_history_gens > z_dim:
        # This is a major simplification. Assumes Z generators are at the end of the history block.
        z_impact = jnp.sum(impacts[n_history_gens - z_dim : n_history_gens])

    # Apply bias to encourage blaming the action, as it's controllable.
    # Compare action impact vs. all non-action impact.
    if (action_impact * action_bias) > history_impact:
        # Blame the current action at step 't'
        return 0, 'action' 
    elif z_impact > (history_impact - z_impact):
        # If Z is the biggest part of the history blame, blame Z.
        return 0, 'z'
    else:
        # Otherwise, the blame lies in the accumulated error from previous steps.
        # Send a signal to refine a previous step.
        return -1, 'history' # Signal to bubble up

# --- ENGINE ---

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
    
    def calculate_blame_gradients(self, zono: Zonotope) -> jnp.ndarray:
        """
        Calculates the gradient of the violation loss w.r.t. the input generators
        of a failing zonotope. Returns an array of 'impacts' for each generator.
        """
        c_x = zono.center[:, :self.input_dim]
        c_z = zono.center[:, self.input_dim:]
        
        # Get gradients of the loss w.r.t the generators of the input zonotope
        grads = self._grad_loss_fn(zono.generators, c_x, c_z, self.safe_min, self.safe_max)
        
        # Impact = |gradient * generator_magnitude|. Sum over feature dimensions.
        impacts = jnp.sum(jnp.abs(grads * zono.generators), axis=(0, 2))
        return impacts

    def force_split(self, zono: Zonotope, reason: str) -> List[Zonotope]:
        # If we blame history, we need a splitting strategy. For now, we'll
        # just split on the action space as a fallback. A more advanced
        # implementation would need to identify the most sensitive historical
        # generator.
        if reason == 'history':
            # If history is blamed, split the state part of the zonotope.
            relevant_gens = zono.generators[:, :, :self.model.x_dim]
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
        elif reason == 'z':
            start_feat = self.input_dim 
            relevant_gens = zono.generators[:, :, start_feat:]
            if relevant_gens.shape[2] == 0: return []
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
        elif reason == 'action':
            start_feat = self.model.x_dim
            end_feat = start_feat + self.model.a_dim
            relevant_gens = zono.generators[:, :, start_feat:end_feat]
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
        else: return []

        if power.shape[0] == 0: return []
        split_idx = jnp.argmax(power)
        
        gen_vec = zono.generators[:, split_idx, :]
        c_l = zono.center - 0.5 * gen_vec
        g_l = zono.generators.at[:, split_idx, :].set(0.5 * gen_vec)
        
        z_l = Zonotope(c_l, g_l); z_l.id = getattr(zono, 'id', -99)
        z_r = Zonotope(zono.center + 0.5 * gen_vec, g_l); z_r.id = getattr(zono, 'id', -99)
        
        return [z_l, z_r]

    def verify_step(self, zonotopes, action_zonotope, step_idx, max_splits=50):
        successful_pairs = []
        pq = []
        entry_id = 0
        
        for z_in in zonotopes:
            c_x = z_in.center[:, :self.input_dim]
            c_z = z_in.center[:, self.input_dim:]
            g_x = z_in.generators[:, :, :self.input_dim]
            g_z = z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            margin = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            if not hasattr(z_in, 'id'): z_in.id = -999
            heapq.heappush(pq, (-float(margin), entry_id, z_in)); entry_id += 1

        splits_done = 0
        
        while len(pq) > 0:
            neg_margin, _, curr_z_in = heapq.heappop(pq)
            
            c_x = curr_z_in.center[:, :self.input_dim]
            c_z = curr_z_in.center[:, self.input_dim:]
            g_x = curr_z_in.generators[:, :, :self.input_dim]
            g_z = curr_z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            worst_viol = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            if worst_viol < 0:
                # --- RECONSTRUCTION WITH COLUMN EXPANSION ---
                
                n_out_gens = z_out.generators.shape[1]
                n_in_z_gens = g_z.shape[1]

                # Align State (from model) and Z generator counts
                if n_out_gens > n_in_z_gens:
                    pad = jnp.zeros((g_z.shape[0], n_out_gens - n_in_z_gens, g_z.shape[2]))
                    g_z_padded = jnp.concatenate([g_z, pad], axis=1)
                    g_x_next = z_out.generators
                else: # n_out_gens <= n_in_z_gens
                    pad = jnp.zeros((z_out.generators.shape[0], n_in_z_gens - n_out_gens, z_out.generators.shape[2]))
                    g_x_next = jnp.concatenate([z_out.generators, pad], axis=1)
                    g_z_padded = g_z
                
                # 2. Inject Action into NEW COLUMNS
                gap_size = self.input_dim - z_out.center.shape[1]
                
                if gap_size > 0 and action_zonotope is not None:
                    act_c = action_zonotope.center[:, :gap_size]
                    act_g_raw = action_zonotope.generators[:, :, :gap_size]
                    
                    n_history_gens = g_x_next.shape[1]
                    n_action_gens = act_g_raw.shape[1]
                    
                    # Pad history generators (state, z) to make space for new action generators
                    g_x_expanded = jnp.pad(g_x_next, ((0,0), (0, n_action_gens), (0,0)))
                    g_z_expanded = jnp.pad(g_z_padded, ((0,0), (0, n_action_gens), (0,0)))
                    
                    # Pad action generators to align with history
                    act_g_expanded = jnp.pad(act_g_raw, ((0,0), (n_history_gens, 0), (0,0)))
                    
                    # 3. Concatenate Features (Axis 2)
                    new_center = jnp.concatenate([z_out.center, act_c, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_expanded, act_g_expanded, g_z_expanded], axis=2)
                        
                else:
                    # No Action to inject
                    new_center = jnp.concatenate([z_out.center, c_z], axis=1)
                    new_gens = jnp.concatenate([g_x_next, g_z_padded], axis=2)
                
                next_zono = Zonotope(new_center, new_gens)
                successful_pairs.append((curr_z_in, next_zono))
                continue
            
            # --- SPLIT LOGIC (Unchanged) ---
            if splits_done >= max_splits: continue

            # Gradients are calculated on the FAILED input zonotope `curr_z_in`
            grads = self._grad_loss_fn(curr_z_in.generators, c_x, c_z, self.safe_min, self.safe_max)
            impact = jnp.sum(jnp.abs(grads * curr_z_in.generators), axis=(0, 2))
            
            # Bias towards splitting action-related generators
            a_start_feat = self.model.x_dim
            a_end_feat = a_start_feat + self.model.a_dim
            
            # This part is tricky because generator columns don't map cleanly to features anymore.
            # A simple heuristic: find which generators most affect the action part of the input.
            action_related_power = jnp.sum(jnp.abs(curr_z_in.generators[:, :, a_start_feat:a_end_feat]), axis=(0, 2))
            
            # Boost impact by how much a generator affects the action space.
            # This is a heuristic; a better way might involve tracking generator lineage.
            biased_impact = impact * (1.0 + action_related_power)

            split_idx = jnp.argmax(biased_impact)
            gen_vec = curr_z_in.generators[:, split_idx, :]
            
            c_l = curr_z_in.center - 0.5 * gen_vec
            g_l = curr_z_in.generators.at[:, split_idx, :].set(0.5 * gen_vec)
            z_l = Zonotope(c_l, g_l); z_l.id = curr_z_in.id
            z_r = Zonotope(curr_z_in.center + 0.5 * gen_vec, g_l); z_r.id = curr_z_in.id
            
            splits_done += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_l)); entry_id += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_r)); entry_id += 1
            
        return successful_pairs

# --- DIRECTED DFS ---

def run_directed_dfs(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=50
):
    print(f"--- Starting Trace-Based Sensitivity Search (H={horizon}) ---")
    
    x_dim, a_dim = model.x_dim, model.a_dim
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    # x_min here is assumed to contain state and action bounds
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    
    root_zono = box_to_zonotope(joint_min, joint_max)
    root_zono.id = 0
    trace.add_node(TrajectoryNode(-1, 0, -1, (None,None), 1.0, root_zono))
    
    act_min, act_max = x_min[x_dim:], x_max[x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)
    n_action_gens = action_zono.generators.shape[1]

    def dfs_step(current_zono, t):
        if t == horizon: return [current_zono]

        results = engine.verify_step(
            [current_zono], action_zonotope=action_zono, step_idx=t, max_splits=max_splits_per_step
        )
        
        # Dead End Handling
        if not results:
            # `current_zono` is the zonotope that failed to be verified.
            # We calculate blame based on its sensitivity.
            impacts = engine.calculate_blame_gradients(current_zono)
            
            step_offset, blame_type = calculate_blame(impacts, x_dim, a_dim, z_dim, n_action_gens)
            
            # step_offset is -1 for history, 0 for current step.
            # We want the absolute step index.
            blame_step = t + step_offset

            print(f"  🛑 Dead End at Depth {t+1}. Blaming '{blame_type}' at Step {blame_step}")
            
            req = RefinementRequest(target_step=blame_step, reason=blame_type)
            return req

        candidates = results
        candidates.sort(key=lambda p: calculate_z_probability(p[0], x_dim+a_dim), reverse=True)
        
        cand_idx = 0
        while cand_idx < len(candidates):
            refined_input, next_output = candidates[cand_idx]
            cand_idx += 1
            
            # Trace Logging
            node_id = trace.global_id_counter; trace.global_id_counter += 1
            parent_id = getattr(refined_input, 'id', -1)
            lb_a, ub_a, _ = get_action_range(refined_input, x_dim, a_dim)
            prob = calculate_z_probability(refined_input, x_dim+a_dim)
            
            next_output.id = node_id
            trace.add_node(TrajectoryNode(t, node_id, parent_id, (lb_a, ub_a), prob, next_output))
            
            if cand_idx == 1: 
                print(f"  Step {t}: Action [{lb_a[0,0]:.2f}, {ub_a[0,0]:.2f}] (P={prob:.2f})")
            
            res = dfs_step(next_output, t + 1)
            
            # --- RECURSION HANDLING ---
            if isinstance(res, list): return res
            
            if isinstance(res, RefinementRequest):
                # Bubble up if the request is for a previous step
                if res.target_step < t:
                    return res 

                # Handle if the request is for THIS step
                if res.target_step == t:
                    print(f"  ⚡ Backtrack caught at Step {t}. Reason: {res.reason}. Forcing split...")
                    
                    # We split the input that LED to the failure.
                    new_inputs = engine.force_split(refined_input, res.reason)
                    if not new_inputs:
                         print("     -> Could not split further.")
                         continue
                         
                    # Verify new candidates immediately
                    for inp in new_inputs:
                        new_pairs = engine.verify_step([inp], action_zonotope=action_zono, step_idx=t)
                        if new_pairs:
                            # Insert at front to try immediately
                            candidates.insert(cand_idx, new_pairs[0])
                    
                    continue # Loop back to try new candidates
                else: # target_step > t
                    # This means the child at t+1 failed and blamed itself.
                    # From this step's (t) perspective, that path is simply
                    # exhausted. We should continue to the next sibling candidate.
                    print(f"  - Child at step {t+1} failed and is handling it internally. Trying next sibling at step {t}.")
                    continue

        # End of Loop - All siblings failed, this path is a dead end
        # We need to assign blame for THIS failure.
        impacts = engine.calculate_blame_gradients(current_zono)
        step_offset, blame_type = calculate_blame(impacts, x_dim, a_dim, z_dim, n_action_gens)
        blame_step = t + step_offset
        print(f"  🛑 Exhausted Siblings at Depth {t+1}. Blaming '{blame_type}' at Step {blame_step}")
        return RefinementRequest(target_step=blame_step, reason=blame_type)


    final = dfs_step(root_zono, 0)
    
    if isinstance(final, list) and len(final) > 0:
        print("\n✅ Success! Safe Trajectory Found.")
        return final, trace
    else:
        print("\n❌ Failure: Search exhausted.")
        return [], trace

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
