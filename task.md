# Task: Maximizing Safe Probability & Research Best Practices

- [x] **1. Verification & Calibration (Research Integrity)**
    - [x] Create `verify_enn_calibration.py` to run `compute_val_diagnostics` on the saved model.
    - [x] Analyze calibration (ECE) and epistemic uncertainty quality (AUROC ID vs OOD).
    - [x] Validate core assumptions: Is the model calibrated? If not, search is effectively guessing.

- [x] **2. Search Optimization (Prioritized DFS)**
    - [x] Modify `src/search.py`: Switch to a Prioritized DFS approach (Deepest+Highest Mass).
    - [x] Implement "Mass Accumulation": Don't stop at first safe leaf; find disjoint safe volumes.
    - [x] Implement pruning for negligible mass branches.

- [/] **3. Active Learning Experiment (Horizon 10)**
    - [x] Update `active_learning.py` config: `MPC_HORIZON=10`, `MAX_NODES=500`.
    - [/] Run experiment (5 loops).
    - [ ] Save trees for analysis.

- [ ] **4. Analysis**
    - [ ] Update `analyze_trees.py` to ensure legacy safety mass calculation is robust.
    - [ ] Run `analyze_trees.py` after Loop 1 finishes.
    - [ ] Generate plots:
        - [ ] Search Depth over Loops.
        - [ ] Safe Mass Accumulation over Loops.
        - [ ] Learning Curve (Reward vs Episodes).
    - [ ] Synthesis: Update Lab Notebook with findings on Horizon 10 feasibility.

- [x] **5. Visualization**
    - [x] Modify `src/zono.py`: Add `project(dims)` and `vertices()` to `Zonotope`.
    - [x] Create `visualize_reachability.py`: Plot 2D projections of the reachable tube.
    - [x] Generate `reachability.png` from the first available tree (Mock Data for now).

- [x] **6. Code Refactoring (QA & Best Practices)**
    - [x] `src/affine.py` & `src/star.py`: Add type hints, docstrings, and ensure immutability.
    - [x] `src/polytope.py`: Clean up API, ensure JAX compatibility.
    - [x] `src/zono.py`: Strong typing for Generators, method cleanup.
    - [x] `src/net.py` & `src/data.py`: Standardize neural net and dataset classes.
    - [x] `src/search.py`: Refactor `ReachabilitySolver` for readability and type safety.

- [ ] **5. Search Strategies (Refactor)**
    - [ ] Update `src/search.py`: Introduce `SearchStrategy` abstract base class.
    - [ ] Implement `CompletePrioritizedDFS`: Port existing logic.
    - [ ] Implement `RRTStrategy`: New random sampling strategy with inverse Z-polytope constraints.
    - [ ] Update `ConstraintState` and `Zonotope`: Support Polytopic Z constraints.
