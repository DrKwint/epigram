import dill as pickle
from pathlib import Path
import gymnasium as gym
import jax.numpy as jnp
from src.zono import AbstractENN, box_to_zonotope

from src.search import ConstraintState
from src.search import ReachabilitySolver, SearchNode
from src.zono import GeneratorGroup

# env_name = "InvertedPendulum-v5"
# env = gym.make(env_name)
# obs, _ = env.reset()

# Dummy obs for testing without gym
obs = jnp.zeros(4)

concrete_model = pickle.loads(Path("model.pkl").read_bytes())
x_val = jnp.concatenate([obs, jnp.zeros(1)])
model = AbstractENN.from_concrete(concrete_model)

safe_min = jnp.array([-100.0, -0.2, -100.0, -100.0])
safe_max = jnp.array([100.0, 0.2, 100.0, 100.0])

input_min = jnp.concatenate([obs, jnp.array([-3])])
input_max = jnp.concatenate([obs, jnp.array([3])])
x_zono = box_to_zonotope(input_min, input_max)
x_zono.history = (GeneratorGroup("input", 0, x_zono.generators.shape[1]),)

z_dim = 4
z_min = jnp.ones(z_dim) * -3.0
z_max = jnp.ones(z_dim) * 3.0
z_box = box_to_zonotope(z_min, z_max)
z_box.history = (GeneratorGroup("z", 0, z_box.generators.shape[1]),)

x_aligned, z_aligned = x_zono.stack_independent(z_box)

# 1. Setup Environment
# Only convert the State (obs) to a Zonotope. Do NOT concat actions yet.
x_init_state = box_to_zonotope(input_min[:4], input_max[:4])  # Dimensions 0-3
z_init = box_to_zonotope(z_min, z_max)

# 2. Merge Independence
# Ensure State and Z are independent noise sources
x_aligned, z_aligned = x_init_state.stack_independent(z_init)

# 4. Setup Solver
# Unsafe Condition: Pole Angle (index 1) > 0.2
unsafe_vec = jnp.array([0.0, 1.0, 0.0, 0.0])
unsafe_limit = 0.2

solver = ReachabilitySolver(model, unsafe_vec, unsafe_limit)

# IMPORTANT: Initialize Root with Z-Zonotope correctly threaded
root = SearchNode(
    priority=0.0, 
    timestep=0, 
    zonotope=x_aligned, 
    z_zonotope=z_aligned, # New field!
    constraints=ConstraintState(z_bounds=(tuple(z_min), tuple(z_max))) # Initial bounds
)
solver.push(root)

print("Starting Verification Loop...")
step_count = 0

try:
    while solver.step():
        step_count += 1
        if step_count % 10 == 0:
            print(
                f"Step {step_count}: Queue Size {len(solver.queue)}, Safe Leaves {len(solver.safe_leaves)}"
            )

        if step_count > 1000:
            print("Timeout!")
            break

    print("\n--- DONE ---")
    print(f"Total Nodes Explored: {solver.nodes_explored}")
    print(f"Safe Trajectories Found: {len(solver.safe_leaves)}")

    if len(solver.safe_leaves) > 0:
        print("System is verifiably SAFE under the found constraints.")
    else:
        print("System could not be verified safe (or no paths reached horizon).")

except KeyboardInterrupt:
    print("\nPaused.")

finally:
    print("Saving solver state to 'solver_results.pkl'...")
    with open('solver_results.pkl', 'wb') as f:
        pickle.dump(solver, f)
    print("Saved.")
