import dill as pickle
import jax
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
from src.eval import compute_val_diagnostics
from src.data import TrajectoryDataset
from pathlib import Path

def verify_calibration(model_path='model.pkl', data_path='active_learning_data.pkl'):
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        dataset: TrajectoryDataset = pickle.load(f)
        
    # Split data to get validation set
    _, val_ds = dataset.split(0.15)
    print(f"Validation Set: {val_ds.lengths.sum()} transitions")
    
    rngs = nnx.Rngs(params=0, epistemic=42)
    
    print("Computing diagnostics...")
    metrics = compute_val_diagnostics(
        model, 
        val_ds, 
        rngs, 
        id_batch=1024,
        n_samples=50 # Robust sampling
    )
    
    print("\n=== Calibration & Uncertainty Metrics ===")
    print(f"Mean ID Sigma:       {metrics['val/mean_sigma_id']:.4f}")
    print(f"Mean OOD Sigma:      {metrics['val/mean_sigma_ood']:.4f}")
    print(f"OOD/ID Ratio:        {metrics['val/sigma_ood_over_id']:.2f} (Should be >> 1)")
    print(f"AUROC (ID vs OOD):   {metrics['val/auroc_ood']:.4f} (Should be > 0.5)")
    print(f"ECE (Sigma vs Err):  {metrics['val/calib_ece_sigma_vs_error']:.4f} (Should be low, < 0.1)")
    print(f"L2 Error (ID):       {metrics['val/mean_id_error_l2']:.4f}")
    
    # Save a simple plot of Error vs Sigma
    # We need to re-run a small batch to plot, as compute_val_diagnostics returns aggregates.
    # Or just trust the printed metrics.
    
    # Basic Check:
    if metrics['val/calib_ece_sigma_vs_error'] > 0.2:
        print("\n[WARNING] Model appears poorly calibrated (High ECE). Search reliability is low.")
    elif metrics['val/auroc_ood'] < 0.6:
        print("\n[WARNING] Model cannot distinguish OOD data. Epistemic uncertainty is weak.")
    else:
        print("\n[PASS] Model calibration looks reasonable.")

if __name__ == "__main__":
    import os
    if not os.path.exists('model.pkl'):
        # Just in case model wasn't saved in previous runs (we fixed it but maybe user wants to run this now)
        print("model.pkl not found. Please run active_learning.py first.")
    else:
        verify_calibration()
