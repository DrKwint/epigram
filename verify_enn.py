import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from src.net import ENN
from src.polytope import Polytope
from src.affine import Affine
from src.star import Star

def verify_propagate_star_set():
    print("Initializing model...")
    rngs = nnx.Rngs(42)
    x_dim = 2
    a_dim = 1
    z_dim = 2
    hidden_dim = 8
    
    model = ENN(x_dim, a_dim, z_dim, hidden_dim, rngs=rngs)
    
    # Define input dimensions
    # The input to propagate_star_set is a Star over [x, a, z]
    # We'll define a latent source 'w' and map it to [x, a, z]
    w_dim = 2
    
    # Input set P: [-1, 1]^2
    # Using Polytope.box
    P = Polytope.box(w_dim, (jnp.full((w_dim,), -1.0), jnp.full((w_dim,), 1.0)))
    
    # Affine transform T: w -> [x, a, z]
    # Output dim = x_dim + a_dim + z_dim = 5
    out_dim = x_dim + a_dim + z_dim
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A_mat = jax.random.normal(k1, (out_dim, w_dim))
    b_vec = jax.random.normal(k2, (out_dim,))
    
    T = Affine(A_mat, b_vec)
    input_star = Star(P, T)
    
    print("Propagating star set...")
    output_stars = model.propagate_star_set(input_star)
    print(f"Propagation complete. Generated {len(output_stars)} output stars.")
    
    # Verification
    n_samples = 100
    print(f"Verifying with {n_samples} random samples...")
    
    # Sample w uniformly from [-1, 1]^2
    w_samples = np.random.uniform(-1, 1, (n_samples, w_dim))
    
    passed = 0
    for i in range(n_samples):
        w = w_samples[i]
        
        # 1. Compute concrete output
        # Map w -> [x, a, z]
        inputs = np.array(T(w)) # (5,)
        
        # Split inputs
        # x_in for model is [x, a] (first x_dim + a_dim elements)
        xa_dim = x_dim + a_dim
        x_in = inputs[:xa_dim]
        z_in = inputs[xa_dim:]
        
        # Run model
        # Add batch dim
        y_concrete = model(x_in[None, :], z_in[None, :])[0]
        
        # 2. Check symbolic output
        # Find which star contains w
        found_star = False
        for star in output_stars:
            # Check if w is in the input polytope of this star
            if star.input_set.contains(w):
                # Compute symbolic output for this star
                y_symbolic = star.transform(w)
                
                # Compare
                if np.allclose(y_concrete, y_symbolic, atol=1e-5):
                    found_star = True
                    break
        
        if found_star:
            passed += 1
        else:
            print(f"Sample {i}: w not found in any star input set or mismatch.")
            
    print(f"Verification Result: {passed}/{n_samples} passed.")

if __name__ == "__main__":
    verify_propagate_star_set()