import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
from pathlib import Path
import polars as pl
import re

# Import necessary classes for unpickling
from src.search import ReachabilitySolver, SearchNode
# Ensure src is in path or run from root

def analyze_trees(tree_dir='runs/trees'):
    files = sorted(Path(tree_dir).glob("*.pkl"))
    if not files:
        print(f"No tree files found in {tree_dir}")
        return

    data = []
    
    print(f"Analyzing {len(files)} trees...")
    
    for fpath in files:
        # Parse filename: solver_loop_{loop}_ep_{ep}_step_{step}.pkl
        match = re.search(r"solver_loop_(\d+)_ep_(\d+)_step_(\d+)", fpath.name)
        if not match:
            continue
        loop_i, ep_i, step_i = map(int, match.groups())
        
        try:
            with open(fpath, 'rb') as f:
                solver: ReachabilitySolver = pickle.load(f)
        except Exception as e:
            print(f"Failed to load {fpath}: {e}")
            continue

        # Basic Metrics
        is_safe = len(solver.safe_leaves) > 0
        nodes = solver.nodes_explored
        queue_size = len(solver.queue)
        
        # Depth
        all_nodes = solver.queue + solver.safe_leaves
        max_depth = max([n.timestep for n in all_nodes]) if all_nodes else 0
        
        # Split Counts
        splits = collections.defaultdict(int)
        # Verify how many splits of each type happened. 
        # Since we don't have the full tree history easily (unless we traverse), 
        # we can infer from the constraints of the leaves/frontier.
        # But that only counts *active* splits on that branch.
        # A better proxy: check `solver.stats` if it exists (it doesn't in the provided code).
        # We'll use the "leaves approach" to see what constraints form the final partitions.
        
        relu_splits = 0
        action_splits = 0
        z_splits = 0
        
        # Sample a few nodes to estimate split types
        # (Counting ALL nodes might be slow if many files)
        sample_nodes = all_nodes # Use all for accuracy on these small trees
        
        for node in sample_nodes:
            # ReLU
            for layer, layer_splits in node.constraints.relu_splits.items():
                relu_splits += len(layer_splits)
            
            # Action
            if node.constraints.action_bounds:
                mins, maxs = node.constraints.action_bounds
                # Default [-1, 1]? Check deviation
                for l, u in zip(mins, maxs):
                    if l > -0.99 or u < 0.99: action_splits += 1
            
            # Z
            if node.constraints.z_bounds:
                mins, maxs = node.constraints.z_bounds
                # Default [-3, 3]
                for l, u in zip(mins, maxs):
                    if l > -2.99 or u < 2.99: z_splits += 1
        
        # 3. Analyze Tree
        total_nodes = solver.nodes_explored
        # depth = max([n.timestep for n in solver.queue] + [0]) # Queue might be empty
        # Better max depth finding:
        visited = set()
        queue = [solver.root]
        max_depth = 0
        if solver.root:
            visited.add(solver.root)
            while queue:
                n = queue.pop(0)
                max_depth = max(max_depth, n.timestep)
                children = solver.children_map.get(n, [])
                for c in children:
                     if c not in visited:
                         visited.add(c)
                         queue.append(c)

        # Rigorous Mass Calculation (MiniMax)
        try:
            # Check for new method signature or property
            if hasattr(solver, 'compute_safe_probability'):
                 safe_mass_lower_bound = solver.compute_safe_probability(solver.root)
            else:
                 # Fallback if method missing (shouldn't happen with updated search.py)
                 safe_mass_lower_bound = 0.0
        except Exception as e:
            print(f"Error computing mass for {fpath.name}: {e}")
            safe_mass_lower_bound = 0.0

        # Normalize splits by number of nodes (average splits per branch)
        n_branches = len(sample_nodes) if len(sample_nodes) > 0 else 1
        
        data.append({
            'loop': loop_i,
            'episode': ep_i,
            'step': step_i,
            'is_safe': is_safe,
            'nodes': nodes,
            'max_depth': max_depth, # Updated max_depth calculation
            'avg_relu_splits': relu_splits / n_branches,
            'avg_action_splits': action_splits / n_branches,
            'avg_z_splits': z_splits / n_branches,
            'safe_mass_lower_bound': safe_mass_lower_bound,
            'mass_efficiency': safe_mass_lower_bound / (nodes + 1e-9)
        })

    df = pl.DataFrame(data)
    print(df)
    
    # Save metrics
    df.write_csv("tree_analysis.csv")
    
    import seaborn as sns
    
    # Plotting
    # 1. Nodes per Loop
    plt.figure()
    sns.boxplot(data=df.to_pandas(), x="loop", y="nodes")
    plt.title("Nodes Explored Distribution per Loop")
    plt.savefig("analysis_nodes_per_loop.png")
    plt.close()

    # 2. Max Depth per Loop
    plt.figure()
    sns.boxplot(data=df.to_pandas(), x="loop", y="max_depth")
    plt.title("Search Depth Distribution per Loop")
    plt.savefig("analysis_depth_per_loop.png")
    plt.close()

    # 3. Safe Mass per Loop (The Key Metric)
    plt.figure()
    sns.boxplot(data=df.to_pandas(), x="loop", y="safe_mass_lower_bound")
    plt.title("Lower Bound Safe Probability per Loop")
    plt.savefig("analysis_safe_mass_per_loop.png")
    plt.close()
    
    # 4. Split Distribution (Bar Chart)
    
    # 3. Learning Curve (from mpc_logs.csv)
    try:
        mpc_df = pl.read_csv("mpc_logs.csv")
        # Agg by Loop/Episode
        learning_curve = mpc_df.group_by(["loop", "episode_local"]).agg(
            pl.col("reward").sum().alias("ep_reward"),
            pl.col("step_in_ep").max().alias("ep_length")
        ).sort(["loop", "episode_local"])
        
        # Plot Reward per Episode (flattened)
        plt.figure()
        plt.plot(learning_curve["ep_reward"])
        plt.xlabel("Episode (Cumulative)")
        plt.ylabel("Reward")
        plt.title("Active Learning Progress")
        plt.savefig("analysis_learning_curve.png")
        
        # Avg Reward per Loop
        avg_reward_loop = learning_curve.group_by("loop").agg(pl.col("ep_reward").mean()).sort("loop")
        print("\nAvg Reward per Loop:")
        print(avg_reward_loop)

        # Correlation between safe_mass and reward?
        # Join tree data with mpc_logs?
        # Tree data has loop, episode, step=0. 
        # MPC logs usually have many steps. We can check step=0 logs.
        
    except Exception as e:
        print(f"Could not analyze mpc_logs.csv: {e}")
    
    print("Analysis complete. Saved plots and csv.")

if __name__ == "__main__":
    analyze_trees()
