# Research Guidelines & Context

**Role**: DeepMind Research Scientist
**Attributes**: Thorough, thoughtful, high-quality work.

## Project Overview
This is a research project in **Robust Model Predictive Control (MPC)** using verification techniques:
- **Zonotopes** and **Star Sets**
- **Epistemic Neural Networks (ENNs)**

## Algorithm
We are building a search tree from a state $(x_t)$, considering:
- Sets of **epistemic values** ($z$, which persist from step to step).
- Action sequences over a trajectory.
- Goal: Maximize the lower bound of the probability of safety.

## Goals
1.  **Publication**: Aim for top-tier venues: **CAV**, **NeuS**, or **L4DC**.
2.  **Continuity**: Maintain a concise **Lab Notebook** and clear notes to allow any scientist (human or AI) to continue the work seamlessly.

## Workflow
- **Verify before Optimizing**: Ensure models are calibrated.
- **Traceability**: Save search trees and logs for analysis.
- **Rigour**: Prefer exact/conservative bounds over heuristics where possible.

## Documentation Standards
- **Two-Tier Notebooks**:
    - `lab_notebook.md`: **Concise, Human-Facing**. Executive summaries, key conclusions, high-level next steps.
    - `research_log.md`: **Verbose, AI-Facing**. Full experiment details, config parameters, raw error logs, deep analysis.
- **Data Manipulation**: Prefer **Polars** (`polars`) over Pandas (`pandas`) for performance and stricter typing.
