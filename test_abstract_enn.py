import jax
import jax.numpy as jnp
from flax import nnx
from src.zono import Zonotope, AbstractENN, GeneratorGroup

def test_abstract_enn():
    # 1. Setup
    rngs = nnx.Rngs(0)
    # Simple model: x (1) + a (1) + z (1) -> hidden (2) -> out (1)
    # in features for base: x+a = 2
    model = AbstractENN(x_dim=1, a_dim=1, z_dim=1, hidden_dim=2, rngs=rngs)
    
    # 2. Inputs
    # Center 0, width 2 ([-1, 1])
    x_init = Zonotope(
        jnp.zeros((1, 1)), 
        jnp.eye(1)[None, :, :]
    )
    # z: center 0, width 2
    z_sample = Zonotope(
        jnp.zeros((1, 1)), 
        jnp.eye(1)[None, :, :]
    )
    
    # 3. Trajectory Rollout (No splits)
    print("Testing Rollout (No Splits)...")
    traj = model(x_init, z_sample)
    assert isinstance(traj, list)
    assert len(traj) >= 1
    print(f"Rollout produced {len(traj)} branches.")
    
    # 4. Check Shapes
    out_node = traj[0]
    assert out_node.center.shape == (1, 1)
    
    # 5. Test Branching Logic (Mocking Split)
    print("Testing Branching Logic...")
    # Force a split on the first neuron of base_fc1 layer
    # We need to assume the neuron is unstable. With random weights, it likely is.
    split_map = {
        "base_fc1": [0]
    }
    
    branches = model(x_init, z_sample, split_idxs=split_map)
    # Should produce 2 branches if unstable, or 1 if stable.
    print(f"Split produced {len(branches)} branches.")
    
    # Check if history reflects the split (for ReLU type)
    found_relu = False
    for group in branches[0].history:
        if group.source_type == "relu":
            found_relu = True
            break
            
    # Note: If unstable, we expect relu error terms. 
    # If stable active/inactive, error is 0 but we might implicitly have it or logic handles it.
    if len(branches) > 1:
        assert found_relu, "Should have ReLU error terms for split branches (or at least history)"
    
    print("Optimization/Scaling checks...")
    # Just ensure it runs without error
    scales = {"base": jnp.ones((2,))}
    _ = model(x_init, z_sample, error_scales=scales)

    print("All AbstractENN tests passed.")

if __name__ == "__main__":
    test_abstract_enn()
