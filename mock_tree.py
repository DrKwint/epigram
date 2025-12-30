import dill as pickle
from pathlib import Path
import jax.numpy as jnp
from src.search import ReachabilitySolver, SearchNode
from src.zono import Zonotope, AbstractENN
from flax import nnx

def create_mock_tree():
    # 1. Create Dummy Solver
    # Mock AbstractENN?
    class MockModel:
        pass
    model = MockModel()
    solver = ReachabilitySolver(model, jnp.array([1., 0.]), 0.5)
    
    # 2. Create Root Zonotope (4D: x, vx, theta, vtheta)
    # Center: [0, 0, 0, 0]
    # Gens: Identity * 0.1
    dim = 4
    center = jnp.zeros((1, dim))
    generators = jnp.expand_dims(jnp.eye(dim) * 0.1, 0) # [1, 4, 4]
    
    zono = Zonotope(center, generators)
    
    # 3. Create Node
    node = SearchNode(
        priority=(0, 0.0),
        timestep=0,
        zonotope=zono
    )
    
    solver.root = node
    solver.queue = [node]
    solver.safe_leaves = []
    
    # 4. Save
    Path("runs/trees").mkdir(parents=True, exist_ok=True)
    with open("runs/trees/mock_tree.pkl", "wb") as f:
        pickle.dump(solver, f)
    print("Created runs/trees/mock_tree.pkl")

if __name__ == "__main__":
    create_mock_tree()
