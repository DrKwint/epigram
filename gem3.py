import jax
import jax.numpy as jnp
from flax import nnx
import heapq
from typing import List, Optional

from gem import AbstractENN, Zonotope, box_to_zonotope
from src.net import ENN

# (Assuming AbstractENN, Zonotope classes exist in gem.py or above)
# If copying this into a new file, ensure imports work.

class ReachabilityEngine:
    def __init__(self, model: AbstractENN, safe_min: jax.Array, safe_max: jax.Array):
        self.model = model
        self.safe_min = safe_min
        self.safe_max = safe_max
        # Model input is State + Action. 
        self.input_dim = model.x_dim + model.a_dim 

    def verify_step(
        self, 
        zonotopes: List[Zonotope], 
        action_zonotope: Optional[Zonotope], # <--- NEW ARGUMENT
        step_idx: int, 
        max_splits=50
    ) -> List[tuple[Zonotope, Zonotope]]:
        
        print(f"--- Verifying Step {step_idx} (Input Regions: {len(zonotopes)}) ---")
        
        next_step_zonotopes = []
        pq = []
        entry_id = 0
        
        for z_in in zonotopes:
            # Slicing: z_in is [State, Action, Z]
            c_x = z_in.center[:, :self.input_dim]
            c_z = z_in.center[:, self.input_dim:]
            g_x = z_in.generators[:, :, :self.input_dim]
            g_z = z_in.generators[:, :, self.input_dim:]
            
            # Propagate
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            # Check Margin
            # (Assuming z_out is just State, dim = x_dim)
            margin = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            heapq.heappush(pq, (-float(margin), entry_id, z_in))
            entry_id += 1

        splits_done = 0
        # Change output list to store pairs
        successful_pairs = []
        
        while len(pq) > 0:
            neg_margin, _, curr_z_in = heapq.heappop(pq)
            margin = -neg_margin
            
            # 1. Re-Propagate
            c_x = curr_z_in.center[:, :self.input_dim]
            c_z = curr_z_in.center[:, self.input_dim:]
            g_x = curr_z_in.generators[:, :, :self.input_dim]
            g_z = curr_z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            # 2. Safety Check
            worst_viol = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            if worst_viol < 0:
                # SAFE! 
                # -------------------------------------------------------------
                # CAPTURE THE REFINED INPUT (Contains the safe Action for Step T)
                # -------------------------------------------------------------
                refined_input = curr_z_in
                
                # --- A. Align Epistemic Error Terms (Axis 1) ---
                n_out = z_out.generators.shape[1]
                n_in_z = g_z.shape[1]
                
                # Align Z
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

                # --- B. Handle Action (Axis 1 & 2) ---
                # We need to append the action for the NEXT step.
                if action_zonotope is None:
                    # Default: Zero action with 0 uncertainty
                    act_c = jnp.zeros((1, self.model.a_dim))
                    # Generators match the Error Depth (Axis 1) of the others
                    act_g = jnp.zeros((1, g_x_next.shape[1], self.model.a_dim))
                else:
                    # Use provided action zono
                    act_c = action_zonotope.center
                    act_g = action_zonotope.generators
                    # Pad Action generators to match depth if needed
                    if act_g.shape[1] < g_x_next.shape[1]:
                        diff = g_x_next.shape[1] - act_g.shape[1]
                        pad = jnp.zeros((1, diff, act_g.shape[2]))
                        act_g = jnp.concatenate([act_g, pad], axis=1)
                
                # --- C. Concatenate All Features (Axis 2) ---
                # Order: [Next_State, Next_Action, Epistemic_Z]
                
                new_center = jnp.concatenate([z_out.center, act_c, c_z], axis=1)
                new_gens = jnp.concatenate([g_x_next, act_g, g_z_padded], axis=2) 
                
                next_step_zono = Zonotope(new_center, new_gens)
                next_step_zonotopes.append(next_step_zono)
                successful_pairs.append((refined_input, next_step_zono))
                continue
                
            # 3. Check Split Budget
            if splits_done >= max_splits:
                continue

            # 4. Split (Weighted Gradient on INPUT)
            def loss_fn(gens):
                g_x_t = gens[:, :, :self.input_dim]
                g_z_t = gens[:, :, self.input_dim:]
                x_t = Zonotope(c_x, g_x_t)
                z_t = Zonotope(c_z, g_z_t)
                res = self.model(x_t, z_t)
                l, u = res.concrete_bounds()
                return jnp.sum(jnp.maximum(0.0, u - self.safe_max)) + \
                       jnp.sum(jnp.maximum(0.0, self.safe_min - l))
            
            grads = jax.grad(loss_fn)(curr_z_in.generators)
            impact = jnp.sum(jnp.abs(grads * curr_z_in.generators), axis=(0, 2))
            split_idx = jnp.argmax(impact)
            
            splits_done += 1
            gen_vec = curr_z_in.generators[:, split_idx, :]
            
            # Left/Right Children
            c_l = curr_z_in.center - 0.5 * gen_vec
            g_l = curr_z_in.generators.at[:, split_idx, :].set(0.5 * gen_vec)
            
            c_r = curr_z_in.center + 0.5 * gen_vec
            
            heapq.heappush(pq, (-float(margin), entry_id, Zonotope(c_l, g_l)))
            entry_id += 1
            heapq.heappush(pq, (-float(margin), entry_id, Zonotope(c_r, g_l)))
            entry_id += 1
            
        return successful_pairs

# --- Updated Rollout Function ---

def run_trajectory_rollout(
    model, 
    x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=100
):
    # Setup
    a_dim = 1
    # Check if x_min includes action (user passed 'input_min' in previous snippet)
    # We assume x_min is [State, Action]
    
    # Initial Joint Zonotope: [x0, u0, z]
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    
    root_zono = box_to_zonotope(joint_min, joint_max)
    current_regions = [root_zono]
    
    engine = ReachabilityEngine(model, safe_min, safe_max)
    
    # Define Action Constraints for future steps
    # Extract action bounds from initial set for consistency
    # Assuming action is the last dim of x_min
    act_min = x_min[-a_dim:]
    act_max = x_max[-a_dim:]
    
    # Create a reusable Action Zonotope for steps 1..H
    # (Assuming we verify "Any valid action in this range")
    action_zono = box_to_zonotope(act_min, act_max)
    
    # We maintain just the 'next states' for the loop input
    current_regions = [root_zono]
    
    for t in range(horizon):
        if len(current_regions) == 0:
            break
            
        # Get pairs: (Input_at_T, Output_for_T+1)
        step_results = engine.verify_step(
            current_regions, 
            action_zonotope=action_zono, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )
        
        # Unzip results
        # refined_inputs: The zonotopes with the SAFE ACTIONS for step T
        # next_states: The zonotopes with the FRESH ACTIONS for step T+1
        refined_inputs = [p[0] for p in step_results]
        next_states    = [p[1] for p in step_results]
        
        # Yield the inputs so you can inspect the Action Splits at step T
        yield refined_inputs
        
        # Prepare for next loop
        current_regions = next_states

        print(f"Step {t+1} Complete. Safe Sub-regions: {len(current_regions)}")
        yield current_regions

import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class TrajectoryNode:
    step: int
    zono_id: int
    parent_id: int  # The ID of the input zono at step t-1
    action_range: Tuple[jnp.ndarray, jnp.ndarray] # (lb, ub) of action used
    prob_mass: float
    zonotope: 'Zonotope'

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
            # Prepend action (since we are walking backwards)
            actions.insert(0, node.action_range)
            
            if node.parent_id == -1: # Root
                break
            curr_id = node.parent_id
            
        return actions

# --- Updated Run Function ---

def run_trajectory_rollout_with_trace(
    model, x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=5, max_splits_per_step=100
):
    # Setup Engine
    engine = ReachabilityEngine(model, safe_min, safe_max)
    trace = VerificationTrace()
    
    # 1. Setup Root
    # ... (Standard init code) ...
    z_min = jnp.ones(z_dim) * -2.0; z_max = jnp.ones(z_dim) * 2.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    root_zono = box_to_zonotope(joint_min, joint_max)
    
    # Assign ID to root
    root_zono.id = 0
    next_id = 1
    
    # Add Root to trace (Step -1, no action)
    trace.add_node(TrajectoryNode(
        step=-1, zono_id=0, parent_id=-1, 
        action_range=(None, None), prob_mass=1.0, zonotope=root_zono
    ))
    
    current_regions = [root_zono]
    
    # Action constraints (reusable)
    act_min = x_min[model.x_dim:]; act_max = x_max[model.x_dim:]
    action_zono = box_to_zonotope(act_min, act_max)

    for t in range(horizon):
        if not current_regions: break
        
        # Run Verification Step
        # verify_step returns list of (RefinedInput, NextOutput)
        step_results = engine.verify_step(
            current_regions, action_zonotope=action_zono, step_idx=t, max_splits=max_splits_per_step
        )
        
        next_regions = []
        
        for (refined_input, next_output) in step_results:
            # 1. Extract Info from Refined Input (which contains the SAFE action split)
            lb_a, ub_a, _ = get_action_range(refined_input, model.x_dim, model.a_dim)
            prob = calculate_z_probability(refined_input, model.x_dim + model.a_dim)
            
            # 2. Identify Parent
            # The 'refined_input' is a split version of one of 'current_regions'.
            # We need to know WHICH 'current_region' ID it came from.
            # *Implementation Detail*: You need to pass the ID through the split logic.
            # Simple hack: Attach .id to zonotope objects in verify_step
            parent_id = getattr(refined_input, 'id', -1) 
            
            # 3. Register Child (Next Output)
            next_output.id = next_id
            
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
        
    return current_regions, trace

import jax.scipy.stats.norm as norm

def calculate_z_probability(zono: Zonotope, input_dim: int) -> float:
    """
    Approximates the probability mass of the Z-region covered by this zonotope.
    Uses the bounding box of the Z-slice under a Standard Normal N(0,I).
    """
    # 1. Extract Z components
    # The zonotope features are [State, Action, Z]
    # We assume 'input_dim' covers [State, Action]
    c_z = zono.center[:, input_dim:]
    g_z = zono.generators[:, :, input_dim:]
    
    # 2. Compute Concrete Bounds of Z
    # radius = sum(|generators|)
    r_z = jnp.sum(jnp.abs(g_z), axis=1)
    lb_z = c_z - r_z
    ub_z = c_z + r_z
    
    # 3. Integrate Gaussian over Box
    # P(lb < z < ub) = CDF(ub) - CDF(lb)
    # Since z is diagonal Gaussian, we multiply probabilities per dimension
    probs = norm.cdf(ub_z) - norm.cdf(lb_z)
    total_prob = jnp.prod(probs)
    
    return float(total_prob)

def get_action_range(zono: Zonotope, x_dim: int, a_dim: int):
    """
    Extracts the bounds of the Action dimensions from the zonotope.
    Assumes order [State(x_dim), Action(a_dim), Z]
    """
    # Action starts after State
    start = x_dim
    end = x_dim + a_dim
    
    c_a = zono.center[:, start:end]
    g_a = zono.generators[:, :, start:end]
    
    r_a = jnp.sum(jnp.abs(g_a), axis=1)
    lb_a = c_a - r_a
    ub_a = c_a + r_a
    
    return lb_a, ub_a, c_a # Returns center as the 'nominal' action

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

    for step_t, safe_input_zonos in enumerate(run_trajectory_rollout(model, input_min, input_max, safe_min, safe_max, horizon=8)):
        print(f"\n--- Horizon {step_t} Safe Actions ---")
        for zono in safe_input_zonos:
            # Extract Action Range from the Input Zonotope
            lb, ub, c = get_action_range(zono, model.x_dim, model.a_dim)
            prob = calculate_z_probability(zono, model.x_dim + model.a_dim)
            
            # Look for splits (e.g., width < 6.0)
            width = ub - lb
            print(f"  Action: [{lb[0,0]:.2f}, {ub[0,0]:.2f}] (Center {c[0,0]:.2f}) | P(z): {prob:.2f}")
