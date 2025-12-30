import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from src.zono import box_to_zonotope

def debug_env():
    env = gym.make('InvertedPendulum-v5')
    obs, _ = env.reset()
    print(f"Initial Obs: {obs}")
    print(f"Angle (idx 1): {obs[1]}")
    
    # Check Unsafe Condition
    unsafe_vec = jnp.array([0.0, 1.0, 0.0, 0.0])
    unsafe_thresh = 0.2
    
    proj = float(np.dot(obs, unsafe_vec))
    print(f"Projection (h^T x): {proj}")
    print(f"Unsafe Thresh: {unsafe_thresh}")
    print(f"Is Safe? {proj <= unsafe_thresh}")

    # Check Zonotope Scale
    x_min = jnp.array(obs)
    x_max = jnp.array(obs)
    x_zono = box_to_zonotope(x_min, x_max, source_type='input')
    
    _, ub = x_zono.project_bounds(unsafe_vec)
    print(f"Zonotope UB: {ub}")
    
if __name__ == "__main__":
    debug_env()
