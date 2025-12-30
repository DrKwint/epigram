import dill as pickle
import matplotlib.pyplot as plt
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon
import argparse

from src.search import ReachabilitySolver

def load_latest_tree(tree_dir='runs/trees'):
    files = sorted(Path(tree_dir).glob("*.pkl"))
    if not files:
        print("No trees found.")
        return None
    print(f"Loading {files[-1]}...")
    with open(files[-1], 'rb') as f:
        return pickle.load(f)

def collect_zonotopes(solver):
    """
    Traverses the tree and returns a list of (step, zonotope, is_safe) tuples.
    """
    zonos = []
    # Queue + Safe Leaves
    # We want to visualize the *reachable sets*.
    # The solver has `queue` (frontier) and `safe_leaves` (terminal).
    # We might want to see the whole tree? 
    # The `SearchNode` stores the `star_set` (which is a Star). 
    # We need to convert Star -> Zonotope?
    # Or does `SearchNode` have a zonotope?
    # Let's check SearchNode structure.
    
    # Assuming nodes have 'abstract_state' which is a Star or Zonotope.
    # If Star, we might need to overapproximate to Zono for easy plotting, 
    # or plotting Star lines is harder (projection of high-dim polytope).
    
    # Let's check what solver.root has.
    pass 
    return zonos

def plot_reachability(solver, filename="reachability.png"):
    # 1. Collect nodes
    nodes = solver.safe_leaves + solver.queue
    print(f"Plotting {len(nodes)} nodes...")
    
    # 2. Setup Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    dims_list = [(0, 1), (2, 3), (0, 2)]
    labels = ["x vs vx", "theta vs vtheta", "x vs theta"]
    
    from src.zono import Zonotope
    # If nodes store Stars, we might need a helper.
    # Let's assume for now we can get a zonotope.
    
    # Helper to extract Zono from Node
    def get_zono(node):
        # SearchNode has 'zonotope' field
        if hasattr(node, 'zonotope'):
            return node.zonotope
        # Fallback for other potential node types
        return getattr(node, 'abstract_state', None)

    for ax, dims, label in zip(axs, dims_list, labels):
        ax.set_title(label)
        patches = []
        for node in nodes:
            state = get_zono(node)
            # Check type
            if not hasattr(state, 'project'):
                continue
                
            proj = state.project(dims)
            # Get vertices (convex hull)
            try:
                verts = proj.vertices() # Returns list of arrays (batch) or single?
                # The zono might be batched? Solver nodes are usually single?
                # 'abstract_state' in search is usually singular or batched?
                # Likely singular `Zonotope`.
                
                # vertices() returns list of (N, 2) arrays if batched, or just (N,2)?
                # My implementation returns `[ _get_verts(...) for i ... ]`.
                # So it returns a list.
                
                if isinstance(verts, list):
                    verts = verts[0] # Take first if batch=1
                    
                polygon = MplPolygon(verts, True)
                patches.append(polygon)
            except Exception as e:
                print(f"Error plotting node: {e}")
        
        p = PatchCollection(patches, alpha=0.4, color='green')
        ax.add_collection(p)
        ax.autoscale()
        
    plt.savefig(filename)
    print(f"Saved {filename}")

if __name__ == "__main__":
    solver = load_latest_tree()
    if solver:
        plot_reachability(solver)
