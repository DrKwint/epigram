# Lab Notebook: Active Learning & Reachability Analysis

## Verification: Calibration Check
**Date:** 2025-12-30
**Goal:** Verify ENN is calibrated before optimizing search.

### Results
*   **ECE (Sigma vs Error):** 0.0357 (Target < 0.1). **PASS**.
    *   The model's predicted uncertainty closely matches its actual error rate.
*   **AUROC (ID vs OOD):** 0.7948 (Target > 0.6). **PASS**.
    *   The model reliably outputs higher uncertainty for OOD data.
*   **OOD/ID Sigma Ratio:** 1.53.
    *   OOD data is ~50% more uncertain than training data.

**Conclusion:** The model provides a valid signal for "safe" vs "unsafe/unknown" regions. We can proceed with maximizing safe probability.

## Experiment 1: Baseline Active Learning
**Date:** 2025-12-29
**Goal:** Run the existing active learning loop, capture search trees, and establish a baseline for learning speed (episode length) and search efficiency.

### Configuration
*   **Env:** InvertedPendulum-v5
*   **Loops:** 5
*   **Episodes per Loop:** 5
*   **MPC Horizon:** 6
*   **Solver:** ReachabilitySolver (ENN-based)

### Modifications
*   Modified `active_learning.py` to save `ReachabilitySolver` objects from the first step of each episode to `runs/trees/`.

### Results
*   **Learning Progress:** Average reward improved from 2.4 (Loop 0) to 5.2 (Loop 3) but plateaued around 5.0.
*   **Episode Length:** Capped by the short MPC horizon (6) and "safety" failures. The agent survives ~5 steps, which is close to the lookahead horizon.
*   **Search Statistics:**
    *   **Depth:** Increased from max 2 (Loop 0) to 5 (Loop 4).
    *   **ReLU Splits:** Decreased significantly from ~11.8 to ~6.0 per branch. This suggests the learned model became "simpler" or more linear in the relevant state space as training progressed.
    *   **Z Splits:** Decreased from ~2.2 to ~0.7 per branch, indicating reduced epistemic uncertainty (or uncertainty regions that don't cross decision boundaries).
    *   **Action Splits:** Remained steady at ~0.8 per branch.

### Analysis & Recommendations
1.  **Short Horizon Limiting Performance:** The agent consistently survives to ~5 steps, matching the MPC horizon of 6. To achieve longer episodes (e.g., 100+ steps), we MUST increase the MPC horizon.
2.  **Search Optimization:**
    *   **ReLU Splits** are the dominant cost. The reduction over time is promising.
    *   **Low Z-Splits** in later loops suggests we could reduce the number of Z-dimensions or splits allocated to Z, focusing more on Action/ReLU to reach deeper depths.
3.  **Optimization Proposal:**
    *   Increase `MPC_HORIZON` to 10 or 15.
    *   To offset variable computational cost, we might limit `max_splits` or use a "Safe Mass" heuristic earlier.

### Figures
![Learning Curve](analysis_learning_curve.png)
![Node Exploration](analysis_nodes_per_loop.png)
![Split Types](analysis_split_types.png)
