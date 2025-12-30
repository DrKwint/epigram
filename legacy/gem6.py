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
    reason: str  # 'action' or 'z'

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

class GeneratorTracker:
    def __init__(self, z_dim):
        # (start, end, type, step)
        self.map = [(0, z_dim, 'z', -1)]
        self.current_cols = z_dim
        
    def register_step(self, step_idx, added_cols):
        if added_cols > 0:
            start = self.current_cols
            end = start + added_cols
            if not any(m[3] == step_idx and m[2] == 'action' for m in self.map):
                self.map.append((start, end, 'action', step_idx))
                self.current_cols = end

    def get_aggregated_blame(self, impact_scores, action_bias=10.0):
        """
        Sums impact scores by category and applies bias to Actions.
        Returns: (best_step, best_type)
        """
        best_score = -1.0
        best_step = -1
        best_type = 'none'
        
        # 1. Calculate Z score (Base)
        # We assume Z is always the first entry in map or we search for it
        # (Simplified: iterate all segments)
        
        for (start, end, type_, step) in self.map:
            # Sum the impacts for this block of columns
            # Handle out-of-bounds safety if impact_scores is shorter (rare)
            segment = impact_scores[start:min(end, len(impact_scores))]
            if len(segment) == 0: continue
            
            score = jnp.sum(segment)
            
            # Apply Bias
            if type_ == 'action':
                score *= action_bias
                
            # Log max
            if score > best_score:
                best_score = score
                best_step = step
                best_type = type_
                
        return best_step, best_type

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
    lb, ub = zonotope.concrete_bounds()
    viol_max = jnp.maximum(0.0, ub - safe_max)
    viol_min = jnp.maximum(0.0, safe_min - lb)
    
    if jnp.sum(viol_max) + jnp.sum(viol_min) == 0:
        return -1, 'none'
    
    w = (viol_max > 0).astype(jnp.float32) - (viol_min > 0).astype(jnp.float32)
    
    # Projection
    projection = jnp.einsum('bnd,bd->bn', zonotope.generators, w)
    impacts = jnp.sum(jnp.abs(projection), axis=0)
    
    # USE AGGREGATION WITH BIAS
    # Increase action_bias if it's still blaming Z too much
    return tracker.get_aggregated_blame(impacts, action_bias=100.0)

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

    def force_split(self, zono: Zonotope, reason: str) -> List[Zonotope]:
        """Blindly splits the most impactful generator of the requested type."""
        if reason == 'z':
            start_feat = self.input_dim 
            relevant_gens = zono.generators[:, :, start_feat:]
            # Only consider columns that actually exist
            if relevant_gens.shape[2] == 0: return []
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
        elif reason == 'action':
            start_feat = self.model.x_dim
            end_feat = start_feat + self.model.a_dim
            relevant_gens = zono.generators[:, :, start_feat:end_feat]
            power = jnp.sum(jnp.abs(relevant_gens), axis=(0, 2))
        else:
            return []

        if power.shape[0] == 0: return []
        split_idx = jnp.argmax(power)
        
        # Execute Split
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
                # Reconstruct
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
                    g_x_next, g_z_padded = z_out.generators, g_z
                
                # Gap Fill for Action
                gap_size = self.input_dim - z_out.center.shape[1]
                if gap_size > 0:
                    if action_zonotope is None:
                        act_c = jnp.zeros((1, gap_size))
                        act_g = jnp.zeros((1, g_x_next.shape[1], gap_size))
                    else:
                        act_c = action_zonotope.center[:, :gap_size]
                        act_g = action_zonotope.generators[:, :, :gap_size]
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
            
            if splits_done >= max_splits: continue

            grads = self._grad_loss_fn(curr_z_in.generators, c_x, c_z, self.safe_min, self.safe_max)
            impact = jnp.sum(jnp.abs(grads * curr_z_in.generators), axis=(0, 2))
            
            # Action Bias
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

# --- MAIN DFS LOOP ---

def run_directed_dfs(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=50
):
    print(f"--- Starting Trace-Based Sensitivity Search (H={horizon}) ---")
    
    x_dim, a_dim = model.x_dim, model.a_dim
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    tracker = GeneratorTracker(z_dim)
    
    # Root
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

        tracker.register_step(t, a_dim)

        results = engine.verify_step(
            [current_zono], action_zonotope=action_zono, step_idx=t, max_splits=max_splits_per_step
        )
        
        # Dead End Handling
        if not results:
            c_x = current_zono.center[:, :x_dim+a_dim]
            c_z = current_zono.center[:, x_dim+a_dim:]
            g_x = current_zono.generators[:, :, :x_dim+a_dim]
            g_z = current_zono.generators[:, :, x_dim+a_dim:]
            
            failed_zono = engine.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            step_blame, type_blame = calculate_blame(failed_zono, safe_min, safe_max, tracker)
            print(f"  🛑 Dead End at Depth {t+1}. Blaming '{type_blame}' at Step {step_blame}")
            
            return RefinementRequest(target_step=step_blame, reason=type_blame)

        # Candidates
        candidates = results
        candidates.sort(key=lambda p: calculate_z_probability(p[0], x_dim+a_dim), reverse=True)
        
        cand_idx = 0
        while cand_idx < len(candidates):
            refined_input, next_output = candidates[cand_idx]
            cand_idx += 1
            
            # Trace
            node_id = trace.global_id_counter; trace.global_id_counter += 1
            parent_id = getattr(refined_input, 'id', -1)
            lb_a, ub_a, _ = get_action_range(refined_input, x_dim, a_dim)
            prob = calculate_z_probability(refined_input, x_dim+a_dim)
            
            next_output.id = node_id
            trace.add_node(TrajectoryNode(t, node_id, parent_id, (lb_a, ub_a), prob, next_output))
            
            if cand_idx == 1: # Print first try
                print(f"  Step {t}: Action [{lb_a[0,0]:.2f}, {ub_a[0,0]:.2f}] (P={prob:.2f})")
            
            res = dfs_step(next_output, t + 1)
            
            if isinstance(res, list): return res
            if isinstance(res, RefinementRequest):
                is_target = (res.target_step == t)
                is_root_catch = (t == 0 and res.target_step == -1)
                
                if is_target or is_root_catch:
                    # Logic to choose what to split
                    # If the request is explicitly Z, we split Z.
                    # If the request is Action, we split Action.
                    
                    split_type = res.reason
                    
                    # FALLBACK: If we are at Root (t=0) and it asks to split Z,
                    # we should Double Check if we can split Action instead?
                    # No, let's trust the biased blame. If blame says Z (despite 100x bias),
                    # it really IS Z.
                    
                    print(f"  ⚡ Backtrack caught at Step {t}. Reason: {split_type}. Forcing split...")
                    
                    new_inputs = engine.force_split(refined_input, split_type)
                    
                    if not new_inputs:
                        print("     -> Could not split further. Continuing...")
                        continue
                        
                    # Verify
                    valid_children_found = False
                    for inp in new_inputs:
                        new_pairs = engine.verify_step([inp], action_zonotope=action_zono, step_idx=t)
                        if new_pairs:
                            candidates.insert(cand_idx, new_pairs[0])
                            valid_children_found = True
                    
                    # CRITICAL: If Action split yielded NO valid children, 
                    # we must try splitting Z as a last resort.
                    if not valid_children_found and split_type == 'action':
                         print("     -> Action split failed to produce safe children. Falling back to Z split.")
                         fallback_inputs = engine.force_split(refined_input, 'z')
                         for inp in fallback_inputs:
                             new_pairs = engine.verify_step([inp], action_zonotope=action_zono, step_idx=t)
                             if new_pairs:
                                 candidates.insert(cand_idx, new_pairs[0])

                    continue
            
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
    actions = trace.get_action_sequence(safe_regions[0].id)
    for t, (lb, ub) in enumerate(actions):
        center = (ub + lb) / 2.0
        print(f"Step {t}: Act [{lb[0,0]:.3f}, {ub[0,0]:.3f}] (Nom: {center[0,0]:.3f})")

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
