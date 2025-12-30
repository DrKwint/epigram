# Implementation Plan - Maximizing Safe Probability

This plan addresses the goal of optimizing the reachability search to find valid safe trajectories at a longer horizon (10) by switching to Prioritized DFS and maximizing the lower bound of safe probability. It also includes a research integrity check for epistemic uncertainty.

## User Review Required

> [!IMPORTANT]
> **Proposed Search Change**: We are moving from a "Best-First Search" (that stops at the first safe leaf) to an **Exhaustive Prioritized DFS** that accumulates safe mass from *all* disjoint safe leaves found within the node budget.

> [!TIP]
> **Verification Step**: We will run a new script `verify_enn_calibration.py` first. If the model is uncalibrated (e.g., ground truth falls outside prediction bounds too often), the search improvements may be optimizing noise.

## Proposed Changes

### 2. Search Optimization (Rigorous Lower Bound)

#### [MODIFY] [src/search.py](file:///mnt/c/Users/elean/Documents/GitHub/epigram/src/search.py)
- **Class `ReachabilitySolver`**:
    - **Tree Aggregation Logic (The "MiniMax" fix)**:
        - We cannot simply sum all safe leaves because Action splits represent *choices*, not random events.
        - **New Method `compute_safe_probability(node)`**:
            - If Leaf: Return `prob` if Safe else 0.
            - If Split Type == 'Action': Return `max(compute(child) for child in children)`.
            - If Split Type == 'Z' or 'ReLU' (State Partition): Return `sum(compute(child) for child in children)`.
        - This provides a valid lower bound on the maximum achievable safety probability.
    - **DFS Strategy**:
        - Prioritize deep exploration to find *any* safe path first.
        - Then expand siblings of Action splits to find *better* paths.
        - Expand siblings of Z splits to accumulate *more mass*.
    - **`get_mpc_action` update**:
        - Use the new recursive `compute_safe_probability` on the Root to find the global Max Safe Mass.
        - Return the action associated with the best branch.

### 1. Verification & Calibration (Research Integrity)

#### [NEW] [verify_enn_calibration.py](file:///mnt/c/Users/elean/Documents/GitHub/epigram/verify_enn_calibration.py)
- **Coverage Metric**:
    - Add `coverage_95`: Fraction of true next states falling within the model's 95% confidence zonotope/polytope.
    - If Coverage << 0.95, the safety guarantees are invalid.
- Logic:
    - Load model/data.
    - Run diagnostics.
    - **Crucial**: Check if `sigma_id` is correlated with error magnitude (ECE).

### 3. Active Learning Experiment

#### [MODIFY] [active_learning.py](file:///mnt/c/Users/elean/Documents/GitHub/epigram/active_learning.py)
- **Config**:
    - `MPC_HORIZON = 10`
    - `step_limit = 500` (in `get_mpc_action`)
- **Logging**:
    - Log `total_safe_mass` (sum of all safe leaves) instead of just single path mass.

## Verification Plan

### Automated Tests
- **Calibration Check**: Run `python3 verify_enn_calibration.py`. Expect ECE < 0.1 (heuristically) and AUROC > 0.5.
- **Search Logic**:
    - Run `active_learning.py` for 1 loop (fast).
    - Check log output: "Steps=..." should reach ~10 often.
    - Check "Safe Mass": Should be potentially > 0 (unlike before where it was often 0.125 or 0).

### Manual Verification
- **Visual**: `analyze_trees.py` plots should show depth distribution peaking at 10.
