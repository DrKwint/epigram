import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import dill as pickle
from pathlib import Path
from flax import nnx
import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import dill as pickle
from pathlib import Path
from flax import nnx
import jax
import polars as pl
import time

from src.data import TrajectoryDataset, collect_data
from src.net import ENN
from src.zono import AbstractENN, box_to_zonotope, GeneratorGroup
from src.search import ReachabilitySolver, SearchNode, ConstraintState
from main import train_enn
from util import TensorboardLogger

# Configuration
ENV_NAME = 'InvertedPendulum-v5'
MAX_EPOCHS = 50
BATCH_SIZE = 32
MPC_HORIZON = 10
UNSAFE_THRESH = 0.2
TARGET_SAFE_PROB = 0.8 # Early stopping for MPC
INITIAL_STEPS = 500
MPC_EPISODES = 5
LOOPS = 5

def get_mpc_action(model: AbstractENN, obs: np.ndarray, unsafe_vec: jax.Array, unsafe_thresh: float):
    """
    Uses ReachabilitySolver to find a safe action.
    Returns (action, info_dict)
    """
    x_min = jnp.array(obs)
    x_max = jnp.array(obs)
    x_zono = box_to_zonotope(x_min, x_max, source_type='input')
    
    z_dim = model.z_dim
    z_min = jnp.ones(z_dim) * -3.0
    z_max = jnp.ones(z_dim) * 3.0
    z_zono = box_to_zonotope(z_min, z_max, source_type='z')
    
    x_aligned, z_aligned = x_zono.stack_independent(z_zono)
    
    solver = ReachabilitySolver(
        model, 
        unsafe_vec, 
        unsafe_thresh, 
        target_safe_prob=TARGET_SAFE_PROB,
        verbosity=1
    )
    
    root = SearchNode(
        priority=0.0,
        timestep=0,
        zonotope=x_aligned,
        z_zonotope=z_aligned,
        constraints=ConstraintState(z_bounds=(tuple(z_min), tuple(z_max)))
    )
    solver.push(root)
    
    start_time = time.time()
    step_limit = 500
    steps = 0
    
    while solver.step():
        steps += 1
        if steps >= step_limit:
            break
            
    solve_duration = time.time() - start_time
    
    # Check Result
    action, safe_mass, is_safe = solver.get_best_action(default_action=np.array([np.random.uniform(-3.0, 3.0)]))
    
    # Print Trajectory
    try:
        traj = solver.get_best_trajectory()
        print(f"  [MPC] Safe={is_safe}, Mass={safe_mass:.4f}, Steps={len(traj)}")
        # Only print first few or if short
        for step in traj[:MPC_HORIZON]: 
             # Flatten action if needed
             a_val = step['action']
             if hasattr(a_val, 'flatten'): a_val = a_val.flatten()
             print(f"    t={step['timestep']}: Action={a_val}")
    except Exception as e:
        print(f"  [MPC] Failed to print trajectory: {e}")
    
    info = {
        "solver_success": is_safe,
        "solver_steps": steps,
        "solver_time": solve_duration,
        "safe_mass": safe_mass,
        "nodes_explored": solver.nodes_explored
    }
    return action, info, solver

def run_active_learning():
    # 0. Setup Logging
    logger = TensorboardLogger('./runs/', 'active_learning')
    mpc_logs = []
    
    # 1. Initialize Environment and Data
    print("Collecting Initial Data...")
    env = gym.make(ENV_NAME)
    (states, actions, next_states, dones), (env, last_obs) = collect_data(env, steps=INITIAL_STEPS)
    dataset = TrajectoryDataset(states, actions, next_states, dones)
    
    logger.writer.add_scalar("data/dataset_size", dataset.lengths.sum(), 0)
    
    # 2. Setup Model
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    z_dim = 4
    
    rngs = nnx.Rngs(params=0, epistemic=1)
    enn = ENN(obs_dim, act_dim, z_dim, hidden_dim=16, rngs=rngs)
    
    unsafe_vec = jnp.array([0.0, 1.0, 0.0, 0.0])
    
    global_step = 0 # For training
    total_mpc_steps = 0 # For MPC log
    total_resets = 0 
    
    # 3. Active Learning Loop
    for loop_i in range(LOOPS):
        print(f"\n=== Loop {loop_i+1}/{LOOPS} ===")
        logger.writer.add_scalar("loop/iteration", loop_i, global_step)
        
        # Log resets vs steps
        logger.writer.add_scalar("env/resets", total_resets, total_mpc_steps)
        
        # A. Train
        print(f"Training on {dataset.E} episodes ({dataset.lengths.sum()} transitions)...")
        enn, global_step = train_enn(
            enn, dataset, max_epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, rngs=rngs,
            logger=logger, global_step=global_step
        )
        
        # B. Convert to Abstract Model
        abstract_model = AbstractENN.from_concrete(enn)
        
        # C. Collect Data with MPC
        print("Collecting Data with MPC Policy...")
        new_transitions = 0
        total_reward = 0
        
        for ep_i in range(MPC_EPISODES):
            obs, _ = env.reset()
            done = False
            ep_steps = 0
            ep_reward = 0
            
            while not done:
                # Plan Action
                # Only use MPC every few steps? Or every step.
                # To be fast, maybe smaller horizon or simplified.
                # For now, every step.
                action, info, solver = get_mpc_action(abstract_model, obs, unsafe_vec, UNSAFE_THRESH)

                # Save Solver Snapshot (First step of each episode for analysis)
                if ep_steps == 0:
                    tree_dir = Path("runs/trees")
                    tree_dir.mkdir(parents=True, exist_ok=True)
                    tree_path = tree_dir / f"solver_loop_{loop_i}_ep_{ep_i}_step_{ep_steps}.pkl"
                    with open(tree_path, 'wb') as f:
                        pickle.dump(solver, f)
                
                # Execute
                next_obs, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                
                if done:
                    total_resets += 1
                
                # Store Map
                dataset.append_transition(obs, action, next_obs, done)
                
                # Log MPC Step
                log_entry = {
                    "loop": loop_i,
                    "episode_local": ep_i,
                    "global_step_mpc": total_mpc_steps,
                    "step_in_ep": ep_steps,
                    "success": info["solver_success"],
                    "solver_steps": info["solver_steps"],
                    "solver_time": info["solver_time"],
                    "safe_mass": info["safe_mass"],
                    "nodes": info["nodes_explored"],
                    "reward": float(reward),
                    "action_0": float(action[0]),
                    "state_1_angle": float(obs[1])
                }
                mpc_logs.append(log_entry)
                
                obs = next_obs
                new_transitions += 1
                ep_steps += 1
                total_mpc_steps += 1
                ep_reward += reward
                
            print(f"  Episode {ep_i+1}: {ep_steps} steps, Reward: {ep_reward:.1f}")
            total_reward += ep_reward
            
        avg_reward = total_reward / MPC_EPISODES
        logger.writer.add_scalar("mpc/avg_reward", avg_reward, loop_i)
        logger.writer.add_scalar("mpc/new_data_count", new_transitions, loop_i)
        logger.writer.add_scalar("data/total_dataset_size", dataset.lengths.sum(), loop_i)

        print(f"Collected {new_transitions} new transitions. Avg Reward: {avg_reward}")
        
        # Save CSV Per Loop
        if mpc_logs:
            df = pl.DataFrame(mpc_logs)
            df.write_csv("mpc_logs.csv")

    print("\nActive Learning Complete.")
    
    # Save final dataset
    Path('active_learning_data.pkl').write_bytes(pickle.dumps(dataset))
    logger.close()
    print("Saved dataset and logs.")

if __name__ == "__main__":
    run_active_learning()
