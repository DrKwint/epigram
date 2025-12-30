import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
import collections
from src.search import ReachabilitySolver, SearchNode
import jax.numpy as jnp
from scipy.stats import norm
import matplotlib.patches as patches

def analyze_results(filename='solver_results.pkl'):
    print(f"Loading {filename}...")
    with open(filename, 'rb') as f:
        solver: ReachabilitySolver = pickle.load(f)
    
    print(f"Solver loaded.")
    print(f"Nodes Explored: {solver.nodes_explored}")
    print(f"Queue Size (Frontier): {len(solver.queue)}")
    print(f"Safe Leaves: {len(solver.safe_leaves)}")
    
    # --- 1. Tree Structure Analysis ---
    # We reconstruct the tree by walking up from existing nodes
    # Only "live" nodes (queue + safe) are guaranteed to exist.
    
    all_live_nodes = solver.queue + solver.safe_leaves
    
    depths = [n.timestep for n in all_live_nodes]
    max_depth = max(depths) if depths else 0
    
    plt.figure(figsize=(10, 5))
    plt.hist(depths, bins=range(0, max_depth + 2), alpha=0.7, label='All Live Nodes')
    plt.xlabel('Depth (Timestep)')
    plt.ylabel('Count')
    plt.title('Distribution of Search Depth')
    plt.legend()
    plt.savefig('analysis_depth_hist.png')
    print("Saved depth histogram to analysis_depth_hist.png")
    
    # --- 2. Split Analysis ---
    # What are we splitting on?
    # We walk up from every live node and count the constraints applied to reach it.
    
    split_counts = collections.defaultdict(int)
    
    # To avoid double counting shared paths, we might want to traverse Top-Down if we had the root.
    # But we don't have a reliable root pointer in the solver object unless we retained it.
    # Actually dev_main kept 'root', but solver probably dropped it.
    # However, we can find unique paths.
    
    # To count splits properly, we should walk from root to leaves and see what changed.
    # Since we can't easily find unique edges without a full graph reconstruction,
    # we'll approximate by inspecting the constraints of the sampled nodes.
    # Note: A deep node accumulates all splits of its ancestors.
    # So summing them up would double count.
    # Better metric: For *Unique* nodes in the sample, what is the *Latest* split type?
    # Actually, the user likely just wants to know "How many Action splits vs Z splits total in stable tree?".
    # We can iterate over all nodes in the solver.
    
    total_relu_splits = 0
    total_z_splits = 0
    total_action_splits = 0
    
    relu_layer_counts = collections.defaultdict(int)
    
    # Analyze all unique nodes in queue + safe_leaves + their parents (if we could reach them)
    # Without full tree, let's just analyze the *Constraint State* of the leaves/frontier.
    # This represents the "End State" of the search branches.
    # It shows "How constrained did we have to get?"
    
    print("\n--- Split Analysis (Approximated from Frontier) ---")
    print(f"Analyzing {len(all_live_nodes)} active branches...")
    
    for node in all_live_nodes:
        # ReLU Splits
        for layer_name, splits in node.constraints.relu_splits.items():
            relu_layer_counts[layer_name] += len(splits)
            total_relu_splits += len(splits)
            
        # Z Splits - Bounds differ from default?
        # We assume default is [-3, 3] from dev_main.
        # We count how many dimensions are tightened.
        if node.constraints.z_bounds:
            mins, maxs = node.constraints.z_bounds
            # Initial was [-3, 3]. If different, it's a split.
            # Assuming 4 dimensions
            for l, u in zip(mins, maxs):
                if l > -2.99 or u < 2.99: # Approx check
                    total_z_splits += 1

        # Action Splits - Bounds differ from default?
        # Default action is usually [-1, 1] (implied). 
        # But let's check node.constraints.action_bounds
        if node.constraints.action_bounds:
            a_mins, a_maxs = node.constraints.action_bounds
            for l, u in zip(a_mins, a_maxs):
                if l > -0.99 or u < 0.99: # Approx check assuming [-1, 1] range
                    total_action_splits += 1

    print(f"Total Cumulative Split Depth (Sum of all branches):")
    print(f"  ReLU Splits: {total_relu_splits}")
    print(f"  Action Splits: {total_action_splits}")
    print(f"  Z Splits: {total_z_splits}")
        
    print("ReLU Splits by Layer:")
    for layer, count in sorted(relu_layer_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {layer}: {count}")

    # --- 3. Safety Probability Analysis (Z-Set) ---
    print("\n--- Safe Trajectories & Z-Probability ---")
    
    if not solver.safe_leaves:
        print("No safe leaves found. Cannot analyze probabilities.")
    else:
        # Assume Z ~ N(0, I)
        # We calculate P(z in Box) = Product(P(l_i < z_i < u_i))
        
        safe_data = []
        
        for i, node in enumerate(solver.safe_leaves):
            # 1. Get Z Bounds
            if node.constraints.z_bounds:
                z_mins, z_maxs = node.constraints.z_bounds
            else:
                # Default to full range if not constrained? 
                # Or panic? logic says if it's safe, it's safe for *some* subset.
                # If z_bounds is empty, it means the WHOLE initial Z is safe? 
                # (Assuming initial Z was constrained somewhere else?)
                # Actually dev_main initializes constraints with bounds.
                print(f"Warning: Node {i} has no Z bounds.")
                continue
                
            # 2. Calculate Probability
            prob = 1.0
            for l, u in zip(z_mins, z_maxs):
                # CDF(u) - CDF(l)
                p_dim = norm.cdf(u) - norm.cdf(l)
                prob *= p_dim
            
            # 3. reconstruct Trajectory
            # Walk up to root
            traj = []
            action_seq = []
            curr = node
            while curr:
                traj.append(curr.zonotope)
                # Store action bounds
                if curr.constraints.action_bounds:
                    action_seq.append(curr.constraints.action_bounds)
                else:
                    action_seq.append(None) # Root or unconstrained?
                curr = curr.parent
            traj = traj[::-1] # Root to Leaf
            action_seq = action_seq[::-1]
            
            safe_data.append({
                'id': i,
                'prob': prob,
                'z_bounds': (z_mins, z_maxs),
                'final_depth': node.timestep,
                'trajectory': traj,
                'actions': action_seq
            })
            
        # Sort by Probability
        safe_data.sort(key=lambda x: x['prob'], reverse=True)
        
        print(f"Found {len(safe_data)} safe paths.")
        print("Top 5 Safest Paths:")
        for rank, data in enumerate(safe_data[:5]):
            print(f"  Rank {rank+1}: Prob = {data['prob']:.6e} | Depth = {data['final_depth']}")
            print(f"    Z_Bounds (Dim 0): [{data['z_bounds'][0][0]:.3f}, {data['z_bounds'][1][0]:.3f}]")
            print(f"    Action Sequence (First 3 & Last):")
            
            # Print condensed action info
            acts = data['actions']
            indices = [0, 1, 2, len(acts)-1] if len(acts) > 4 else range(len(acts))
            seen = set()
            for k in indices:
                if k in seen: continue
                seen.add(k)
                if acts[k]:
                    # Format just the first dim for brevity
                    amin = acts[k][0][0]
                    amax = acts[k][1][0]
                    print(f"      t={k}: [{amin:.3f}, {amax:.3f}] ...")
                else:
                    print(f"      t={k}: Unconstrained")

        # --- 4. Z-Space Visualization ---
        # Plot the accepted Z-regions (Dim 0 vs Dim 1)
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        total_prob_mass = 0.0
        
        for data in safe_data:
            mins, maxs = data['z_bounds']
            # Rect
            w = maxs[0] - mins[0]
            h = maxs[1] - mins[1]
            rect = patches.Rectangle((mins[0], mins[1]), w, h, 
                                     linewidth=0, facecolor='green', alpha=0.5)
            ax.add_patch(rect)
            total_prob_mass += data['prob'] # Note: Disjoint? 
            # If partitions are disjoint (which they are from splitting), we can sum.
            
        plt.xlim(-3.5, 3.5)
        plt.ylim(-3.5, 3.5)
        plt.xlabel("Z Dim 0")
        plt.ylabel("Z Dim 1")
        plt.title(f"Safe Z-Regions (Total Prob Mass: {total_prob_mass:.4f})")
        plt.grid(True, alpha=0.3)
        plt.savefig('analysis_z_plot.png')
        print(f"Saved Z-space plot to analysis_z_plot.png")

    # --- 5. Phase Plot (Angle vs Velocity) ---
    # Assuming Dimensions: [Pos, Angle, Vel, AngVel] ??
    # User env was InvertedPendulum? usually [x, theta, v, omega] or [cos, sin, v, omega]?
    # InvertedPendulum-v4/v5 obs: [x, y, vx, vy] of tip? 
    # Or standard Gym [x, theta, x_dot, theta_dot]?
    # From dev_main: "Unsafe: Pole Angle (index 1) > 0.2"
    # So Index 1 is Theta. Index 3 is likely Theta Dot.
    
    dim_x = 1 # Angle
    dim_y = 3 # AngVel
    
    plt.figure(figsize=(8, 8))
    
    def plot_zono(zono, color, style='-'):
        # Project to 2D Bounds
        # Simple Box approximation for speed
        cen = zono.center[0] # [Batch=1, Dim]
        # generators: [1, N, Dim]
        
        # Project center
        cx, cy = float(cen[dim_x]), float(cen[dim_y])
        
        # Project generators
        # We need the "Projected Zonotope" vertices for a nice plot, 
        # but a Box is easier: lb, ub.
        lb, ub = zono.concrete_bounds()
        min_x, max_x = float(lb[0, dim_x]), float(ub[0, dim_x])
        min_y, max_y = float(lb[0, dim_y]), float(ub[0, dim_y])
        
        rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                             linewidth=1, edgecolor=color, facecolor='none', linestyle=style)
        plt.gca().add_patch(rect)

    print("\nPlotting state space coverage...")
    # Plot Frontier
    for i, node in enumerate(solver.queue):
        if i > 500: break # Limit
        plot_zono(node.zonotope, 'blue')
        
    # Plot Safe
    for i, node in enumerate(solver.safe_leaves):
        plot_zono(node.zonotope, 'green')
        
    plt.xlabel(f"Dim {dim_x} (Angle)")
    plt.ylabel(f"Dim {dim_y} (AngVel)")
    plt.title("State Space Coverage (Blue=Frontier, Green=Safe)")
    plt.autoscale()
    plt.savefig('analysis_phase_plot.png')
    print("Saved phase plot to analysis_phase_plot.png")

if __name__ == "__main__":
    analyze_results()
