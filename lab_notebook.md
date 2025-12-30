# Lab Notebook: Executive Summary

**Project**: Robust MPC with ENNs & Zonotopes
**Target Venue**: CAV / NeuS / L4DC

## Current Status: Experiment 2 (Horizon 10)
**Date**: 2025-12-30

### Key Findings
1.  **Calibration Verified**: The ENN is well-calibrated (ECE < 0.04), providing valid safety signals.
2.  **Search Depth**: Experiment 1 (Horizon 6) showed the agent consistently surviving *only* to the horizon. We are now testing **Horizon 10**.
3.  **Visualization**: We have implemented a pipeline to plot 2D "Reachable Tubes" (`visualize_reachability.py`) to visually demonstrate safety guarantees.

### Next Steps
1.  **Analyze Horizon 10**: Once Experiment 2 completes, run `analyze_trees.py`.
2.  **Generate Figures**: Run `visualize_reachability.py` on the new trees.
3.  **Synthesize**: Determine if "Prioritized DFS" successfully enables deep search without exponential blowup.

---

## Past Experiments

### Experiment 1: Baseline (Horizon 6)
- **Result**: Agent learned but performance was capped by short horizon.
- **Insight**: Search complexity (ReLU splits) decreased over time, suggesting the model learns to be "easier to verify" in relevant regions.

*For full details, logs, and configuration, see [Research Log](research_log.md).*

---

## Scientific Strategy: Advanced Search & Hybrid Geometry

**Date**: 2025-12-30

### 1. Hybrid Representation for Robust RRT
To enable rigorous randomized search strategies (like RRT), we are adopting a **Hybrid Geometric Representation**:
- **Forward Reachability (State $X$)**: Maintained as **Zonotopes**.
    - *Rationale*: Neural network propagation is only computationally feasible with zonotopes (or similar centrally symmetric sets) due to the explosion of complexity with polytopes.
- **Inverse Constraints (Parameters $Z$)**: Maintained as **Polytopes** ($Hz \le d$).
    - *Rationale*: The "validity domain" of epistemic parameters for a specific trajectory is defined by the intersection of linear safety constraints mapped back from the output space. These are arbitrary half-spaces, not axis-aligned boxes.
    - *Integration*: When propagating forward, we will over-approximate the $Z$-Polytope as a Zonotope to feed into the `AbstractENN`.

### 2. Search Node Abstraction
We are refactoring `SearchNode` to be a **Strategy-Agnostic Container** representing a "State of Knowledge" at depth $t$:
> "Under the constraints $C$ (defining valid $Z$ and action history $u_{0:t}$), the system state is guaranteed to be within Zonotope $X_t$."

### 3. Modular Search Strategies
We will implement two distinct strategies sharing this core abstraction:
1.  **`CompletePrioritizedDFS`** (Current): Exhaustive branching on Action/ReLU/Z to find *guaranteed* safe volumes.
2.  **`RRTStrategy`** (New): Randomized exploration of the action space, using inverse constraints to identify the *subset of model parameters* for which the sampled trajectory is safe. This effectively turns the search from "finding a safe action for *all* models" to "finding a safe action for *some* models and quantifying their probability."
