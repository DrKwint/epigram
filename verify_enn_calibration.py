import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import dill as pickle
from flax import nnx
from typing import Tuple, List
import polars as pl
from tqdm import tqdm

from src.data import collect_data, TrajectoryDataset
from src.net import ENN

# Metric Helpers
def compute_coverage_and_rmse(
    model: ENN, 
    dataset: TrajectoryDataset, 
    key: jax.Array, 
    num_samples: int = 100
) -> Tuple[float, float, float]:
    """
    Computes RMSE and 95% Coverage Probabilities.
    Coverage is defined as the fraction of labels falling within mu +/- 1.96 * sigma.
    """
    
    states = dataset.states
    actions = dataset.actions
    next_states = dataset.next_states
    
    # Batch processing
    batch_size = 128
    N = len(states)
    
    squared_errors = []
    coverage_counts = 0
    total_points = 0
    cis_lengths = []
    
    print(f"Evaluating Calibration on {N} samples...")
    
    for i in tqdm(range(0, N, batch_size)):
        batch_s = jnp.array(states[i:i+batch_size])
        batch_a = jnp.array(actions[i:i+batch_size])
        batch_next_s = next_states[i:i+batch_size]
        
        # Prepare inputs
        # ENN expects (x, z). x is [state, action]
        batch_x = jnp.concatenate([batch_s, batch_a], axis=-1)
        B = batch_x.shape[0]
        
        # Sample Z: [B, num_samples, z_dim]
        key, subkey = jax.random.split(key)
        z_samples = jax.random.normal(subkey, (B, num_samples, model.z_dim))
        
        # Vectorize model call over samples
        # model((B, D), (B, Z)) -> (B, Out)
        # We need (B, S, Out)
        
        def predict_ensemble(x_b, z_b_samples):
            # x_b: (D,)
            # z_b_samples: (S, Z_dim)
            # broadcast x_b
            x_b_expanded = jnp.broadcast_to(x_b, (num_samples, x_b.shape[0]))
            return model(x_b_expanded, z_b_samples)
            
        # Vmap over batch
        batch_preds = jax.vmap(predict_ensemble)(batch_x, z_samples) # (B, S, Out)
        
        # Compute Stats
        means = jnp.mean(batch_preds, axis=1) # (B, Out)
        stds = jnp.std(batch_preds, axis=1)   # (B, Out)
        
        # RMSE Contribution
        current_sq_err = np.sum((means - batch_next_s)**2)
        squared_errors.append(current_sq_err)
        total_points += B * batch_next_s.shape[1] # dimensions count? RMSE is usually per-dimension or average
        
        # Coverage (95% CI = mean +/- 1.96 * std)
        # We check per dimension
        lower = means - 1.96 * stds
        upper = means + 1.96 * stds
        
        in_bounds = (batch_next_s >= lower) & (batch_next_s <= upper)
        coverage_counts += np.sum(in_bounds)
        
        # CI Length
        cis_lengths.append(np.sum(upper - lower))

    total_samples = N * states.shape[1] # Total scalar predictions
    rmse = np.sqrt(sum(squared_errors) / total_samples)
    coverage = coverage_counts / total_samples
    avg_ci_width = sum(cis_lengths) / total_samples
    
    return rmse, coverage, avg_ci_width

def main():
    ENV_NAME = 'InvertedPendulum-v5'
    MODEL_PATH = 'model.pkl'
    
    # 1. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found! Please run active_learning.py first to train a model.")
        return

    # 2. Collect Validation Data
    print("Collecting validation data...")
    val_env = gym.make(ENV_NAME)
    (states, actions, next_states, dones), _ = collect_data(val_env, steps=2000)
    val_dataset = TrajectoryDataset(states, actions, next_states, dones)
    
    # 3. Evaluate
    key = jax.random.key(0)
    rmse, coverage, ci_width = compute_coverage_and_rmse(model, val_dataset, key)
    
    print("\n" + "="*40)
    print(f"Validation Results ({ENV_NAME})")
    print("="*40)
    print(f"RMSE:             {rmse:.4f}")
    print(f"95% Coverage:     {coverage:.4%}")
    print(f"Avg CI Width:     {ci_width:.4f}")
    print("="*40)
    
    if coverage < 0.80:
        print("\n[WARNING] Model is significantly UNDER-confident or biased (Coverage < 80%).")
        print("Search Optimization guarantees likely invalid.")
    elif coverage > 0.99:
        print("\n[WARNING] Model is OVER-confident (Coverage > 99%). CIs are too wide.")
    else:
        print("\n[OK] Model calibration looks reasonable.")

if __name__ == "__main__":
    main()
