# Task: Maximizing Safe Probability & Research Best Practices

- [ ] **1. Verification & Calibration (Research Integrity)**
    - [ ] Create `verify_enn_calibration.py` to run `compute_val_diagnostics` on the saved model.
    - [ ] Analyze calibration (ECE) and epistemic uncertainty quality (AUROC ID vs OOD).
    - [ ] Validate core assumptions: Is the model calibrated? If not, search is effectively guessing.

- [ ] **2. Search Optimization (Prioritized DFS)**
    - [ ] Modify `src/search.py`: Switch to a Prioritized DFS approach (Deepest+Highest Mass).
    - [ ] Implement "Mass Accumulation": Don't stop at first safe leaf; find disjoint safe volumes.
    - [ ] Implement pruning for negligible mass branches.

- [ ] **3. Active Learning Experiment (Horizon 10)**
    - [ ] Update `active_learning.py` config: `MPC_HORIZON=10`, `MAX_NODES=500`.
    - [ ] Run experiment (5 loops).
    - [ ] Save trees for analysis.

- [ ] **4. Analysis**
    - [ ] Update `analyze_trees.py` to report "Total Safe Mass".
    - [ ] Update Lab Notebook with results.
