from src.polytope import Polytope
import jax.numpy as jnp
import numpy as np

def test_polytope():
    # 1. Box creation
    print("Testing Box...")
    p = Polytope.box(2, (jnp.zeros(2), jnp.ones(2)))
    assert p.A.shape == (4, 2)
    assert p.b.shape == (4,)
    
    # 2. Check contains
    print("Testing Contains...")
    assert p.contains(jnp.array([0.5, 0.5]))
    assert not p.contains(jnp.array([1.5, 0.5]))
    
    # 3. Empitness
    print("Testing Empty...")
    assert not p.is_empty()
    
    # 4. Infeasible Intersection
    p_far = Polytope.box(2, (jnp.array([2.0, 2.0]), jnp.array([3.0, 3.0])))
    p_int = p.intersect(p_far)
    assert p_int.is_empty()
    
    print("All Polytope tests passed.")

if __name__ == "__main__":
    test_polytope()
