import jax
import jax.numpy as jnp
from flax import nnx
import heapq
from typing import List

from gem import AbstractENN, Zonotope, box_to_zonotope
from src.net import ENN

# (Assuming AbstractENN, Zonotope classes exist)

class ReachabilityEngine:
    def __init__(self, model: AbstractENN, safe_min: jax.Array, safe_max: jax.Array):
        self.model = model
        self.safe_min = safe_min
        self.safe_max = safe_max
        self.input_dim = model.x_dim + model.a_dim

    def verify_step(self, zonotopes: List[Zonotope], step_idx: int, max_splits=50) -> List[Zonotope]:
        """
        Takes a list of valid zonotopes at time (t), propagates them to (t+1), 
        and refines them until they are safe w.r.t current step constraints.
        Returns the list of safe output zonotopes at (t+1).
        """
        print(f"--- Verifying Step {step_idx} (Input Regions: {len(zonotopes)}) ---")
        
        # We process regions one by one, but if they fail, we split them and add children to queue.
        # Queue stores: (margin, unique_id, parent_zonotope_at_t, current_output_zonotope)
        # We need to track the PARENT (input to this step) because that is what we propagate 
        # to the next step if we split.
        
        next_step_zonotopes = []
        
        # Initialize Queue with the inputs
        pq = []
        entry_id = 0
        
        for z_in in zonotopes:
            # Slicing logic (State vs Epistemic)
            c_x = z_in.center[:, :self.input_dim]
            c_z = z_in.center[:, self.input_dim:]
            g_x = z_in.generators[:, :, :self.input_dim]
            g_z = z_in.generators[:, :, self.input_dim:]
            
            # Initial Propagation
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            # Check Margin
            margin = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            # Push to queue (Priority: Worst margin first, to attack problems early)
            heapq.heappush(pq, (-float(margin), entry_id, z_in)) # Note: Using negative for Max-Heap behavior if desired, or positive for Min
            # Actually, standard logic: We want to process AMBIGUOUS (margin > 0) regions.
            # Safe regions (margin < 0) are done.
            entry_id += 1

        splits_done = 0
        
        while len(pq) > 0:
            # Pop
            neg_margin, _, curr_z_in = heapq.heappop(pq)
            margin = -neg_margin
            
            # 1. Re-Propagate (Or cache if you optimize)
            # We need the output to check safety again
            c_x = curr_z_in.center[:, :self.input_dim]
            c_z = curr_z_in.center[:, self.input_dim:]
            g_x = curr_z_in.generators[:, :, :self.input_dim]
            g_z = curr_z_in.generators[:, :, self.input_dim:]
            
            z_out = self.model(Zonotope(c_x, g_x), Zonotope(c_z, g_z))
            lb, ub = z_out.concrete_bounds()
            
            # 2. Safety Check
            worst_viol = jnp.maximum(jnp.max(ub - self.safe_max), jnp.max(self.safe_min - lb))
            
            if worst_viol < 0:
                # SAFE for this step.
                # We need to construct the joint zonotope [x_{t+1}, z] to pass to the next step.
                
                # 1. Align Error Terms (Axis 1) --------------------------------
                # z_out has K error terms (original + new ReLUs).
                # z_in (epistemic part) has M error terms (original).
                n_out = z_out.generators.shape[1]
                n_in = g_z.shape[1]
                
                if n_out > n_in:
                    # Pad Z generators (the epistemic part) to match new depth K
                    # Shape: [Batch, Diff, Z_Dim]
                    pad = jnp.zeros((g_z.shape[0], n_out - n_in, g_z.shape[2]))
                    g_z_padded = jnp.concatenate([g_z, pad], axis=1)
                    
                    # Output generators are already size K
                    g_x_next = z_out.generators
                    
                elif n_out < n_in:
                    # Rare: Output shrank (e.g. projection). Pad Output to match M.
                    pad = jnp.zeros((z_out.generators.shape[0], n_in - n_out, z_out.generators.shape[2]))
                    g_x_next = jnp.concatenate([z_out.generators, pad], axis=1)
                    
                    # Z generators are already size M
                    g_z_padded = g_z
                else:
                    g_x_next = z_out.generators
                    g_z_padded = g_z
                
                # 2. Concatenate Features (Axis 2) -----------------------------
                # We want [Batch, Max_Errors, X_Dim + Z_Dim]
                # Center: [Batch, X_Dim + Z_Dim]
                new_center = jnp.concatenate([z_out.center, c_z], axis=1)
                
                # Generators: [Batch, Max_Errors, X_Dim + Z_Dim]
                # We stack them side-by-side along the FEATURE axis (2)
                new_gens = jnp.concatenate([g_x_next, g_z_padded], axis=2) 
                
                next_step_zono = Zonotope(new_center, new_gens)
                next_step_zonotopes.append(next_step_zono)
                continue
                
            # 3. Check Split Budget
            if splits_done >= max_splits:
                # Budget exhausted, we have to drop this or flag it unsafe
                print(f"  Warning: Step {step_idx} budget exhausted. Dropping ambiguous region.")
                continue

            # 4. Split (Weighted Gradient)
            # We differentiate the loss of THIS step w.r.t the INPUT generators
            def loss_fn(gens):
                # Reconstruction logic
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
            
            # Execute Split on the INPUT Zonotope
            splits_done += 1
            gen_vec = curr_z_in.generators[:, split_idx, :]
            
            # Left
            c_l = curr_z_in.center - 0.5 * gen_vec
            g_l = curr_z_in.generators.at[:, split_idx, :].set(0.5 * gen_vec)
            z_l = Zonotope(c_l, g_l)
            
            # Right
            c_r = curr_z_in.center + 0.5 * gen_vec
            z_r = Zonotope(c_r, g_l) # generators same size
            
            # Push back to queue to verify validity
            heapq.heappush(pq, (-float(margin), entry_id, z_l))
            entry_id += 1
            heapq.heappush(pq, (-float(margin), entry_id, z_r))
            entry_id += 1
            
        return next_step_zonotopes

# --- Usage Script ---

def run_trajectory_rollout(
    model: AbstractENN,
    x_min, x_max, safe_min, safe_max, 
    horizon=10, z_dim=4, max_splits_per_step=100
):
    # 2. Initial Joint Zonotope
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    joint_min = jnp.concatenate([x_min, z_min])
    joint_max = jnp.concatenate([x_max, z_max])
    
    root_zono = box_to_zonotope(joint_min, joint_max)

    # Current list of valid regions
    # At t=0, we just have the root
    current_regions = [root_zono]
    
    engine = ReachabilityEngine(model, safe_min, safe_max)
    
    # 3. Rollout Loop
    for t in range(horizon):
        if len(current_regions) == 0:
            print(f"Step {t}: No valid regions remaining. Trajectory Unsafe.")
            break
            
        print(f"--- TIMESTEP {t} -> {t+1} ---")
        
        # Verify and Propagate
        # This function splits ambiguous regions and returns ONLY the safe children
        # projected to the next time step.
        next_regions = engine.verify_step(
            current_regions, 
            step_idx=t, 
            max_splits=max_splits_per_step
        )
        
        # Update list for next step
        current_regions = next_regions
        
        print(f"  Step {t+1} Complete. Safe Sub-regions: {len(current_regions)}")

    if len(current_regions) > 0:
        print("✅ Trajectory Verified Safe for a subset of parameters.")

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

    run_trajectory_rollout(model, input_min, input_max, safe_min, safe_max, horizon=2)