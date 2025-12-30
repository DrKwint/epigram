# Research Log: Continuous Experimentation & details

**Role**: AI/Human continuity log.
**Purpose**: Store full verbose details, configuration parameters, raw error logs explanation, and deep analysis.

## 2025-12-30: Experiment 2 (Horizon 10) & Visualization

### Status
- **Experiment 2**: Started. Command `710b9053...` (Horizon 10).
- **Visualization**: Implemented `visualize_reachability.py` and patched `src/zono.py` with `project/vertices`.

### Detailed Configuration (Experiment 2)
- **Env**: InvertedPendulum-v5
- **Horizon**: 10 (Critical change from Exp 1).
- **Max Nodes**: 500 (Budget for search).
- **Search Strategy**: Prioritized DFS + Mass Accumulation.
- **Dependencies**: Added `polytope`, `gymnasium[mujoco]`, `tensorboardX`.

### Calibration Check (Pre-Experiment)
- **Script**: `verify_enn_calibration.py` (Patched to generate fresh data).
- **Results**:
    - **ECE**: 0.0357 (Very good).
    - **AUROC**: 0.7948 (Good separation).
    - **Sigma Ratio**: 1.53 (OOD > ID).
- **Raw Output**: `calibration_results.txt`.

### Visualization Implementation
- **Goal**: Produce `reachability.png` for paper.
- **Method**: 
    - `Zonotope.project(dims)`: Projects generators.
    - `Zonotope.vertices()`: Uses Convex Hull of sampled directions (Robust 2D).
- **Verification**: `mock_tree.py` used to generate a dummy tree, verified script works.

---

## 2025-12-29: Experiment 1 (Baseline)

### Configuration
- **MPC Horizon**: 6
- **Loops**: 5

### Detailed Results
- **Learning Curve**: Reward 2.4 -> 5.2. Plateaued.
- **Split Stats**:
    - ReLU: 11.8 -> 6.0 (Model simplification).
    - Z: 2.2 -> 0.7 (Epistemic reduction).
    - Action: ~0.8 (Constant).
- **Analysis**: Horizon 6 is too short for sustained stability. The agent "survives" exactly to the horizon limit.

### Figures
- `analysis_learning_curve.png`
- `analysis_nodes_per_loop.png`
- `analysis_split_types.png`
