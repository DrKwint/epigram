from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Optional, Callable, List, Tuple, Any, Dict
from jaxtyping import Array

import tree  # type: ignore

from src.affine import Affine
from src.polytope import Polytope

Metrics = Dict[str, float]

class Star:
    """
    Represents a Star set (an affine transformation of an input Polytope).
    S = {x | x = Ay + b, y ∈ P}

    Attributes:
        input_set (Polytope): The input constraints P.
        transform (Affine): The mapping y ↦ Ay + b.
        activation_pattern (Optional[str]): Tracked activation string (e.g., '010...').
    """
    __slots__ = ("input_set", "transform", "activation_pattern")

    def __init__(
        self,
        input_set: Polytope,
        transform: Affine,
        activation_pattern: Optional[str] = None,
    ) -> None:
        self.input_set = input_set
        self.transform = transform
        self.activation_pattern = activation_pattern

    def map_affine(self, aff: Affine) -> Star:
        """Return a new Star with composed affine transformation."""
        return Star(
            self.input_set,
            aff.map(self.transform),
            activation_pattern=self.activation_pattern,
        )

    def map_steprelu(self, dim: int) -> List[Star]:
        """
        Split the star set across the ReLU boundary at output dimension `dim`.
        Returns a list of non-empty Star sets (0, 1, or 2).
        """
        # Constraint: row_dim(y) < 0  => neg case
        # y = Ax + b. A[dim]x + b[dim] <= 0
        
        # Negative Case
        neg_set = self.input_set.add_eqn(
            self.transform.A[dim],
            -1 * self.transform.b[dim],
        )
        # In Negative case, output becomes 0. Transform row becomes 0.
        neg_transform = Affine(
            self.transform.A.at[dim].set(0.0),
            self.transform.b.at[dim].set(0.0),
        )
        neg_pat = (
            (self.activation_pattern or "") + "0"
        )
        
        # Positive Case
        # A[dim]x + b[dim] >= 0  => -A[dim]x <= b[dim] (Polytope uses <=)
        pos_set = self.input_set.add_eqn(
            -1 * self.transform.A[dim],
            self.transform.b[dim],
        )
        pos_pat = (
            (self.activation_pattern or "") + "1"
        )

        # Check emptiness for each
        result = []
        
        # Use simple creation then emptiness check
        # This can be optimized by only creating if check passed? 
        # But Star creation is cheap. Emptiness check is the cost.
        candidate_neg = Star(neg_set, neg_transform, neg_pat)
        if not candidate_neg.is_empty():
            result.append(candidate_neg)
            
        candidate_pos = Star(pos_set, self.transform, pos_pat)
        if not candidate_pos.is_empty():
            result.append(candidate_pos)
            
        return result

    def map_relu(self) -> Tuple[List[Star], Metrics]:
        """
        Apply ReLU to all dimensions sequentially.
        Note: This can cause exponential blowup.
        """
        stars = [self]
        for dim in range(self.transform.A.shape[0]):
            # Flatten one level of list-of-lists
            next_stars = []
            for s in stars:
                next_stars.extend(s.map_steprelu(dim))
            stars = next_stars
            
        metrics = {"verification/split_factor": len(stars)}
        return stars, metrics

    def map_relu_checked(
        self,
        check_fn: Callable[[Array, Array], bool],
    ) -> List[Star]:
        """
        Apply ReLU, filtering intermediate stars with a custom check function.
        Useful for pruning branches known to be safe or irrelevant.
        """
        stars = [self]
        for dim in range(self.transform.A.shape[0]):
            next_stars = []
            for s in stars:
                splits = s.map_steprelu(dim)
                # Filter splits
                valid = [sp for sp in splits if check_fn(sp.input_set.A, sp.input_set.b)]
                next_stars.extend(valid)
            stars = next_stars
        return stars

    def is_empty(self) -> bool:
        return self.input_set.is_empty()

    @property
    def output_set(self) -> Polytope:
        """The minimal H-rep polytope containing the output (only valid if map is identity)."""
        # Actually this transformation logic in Polytope is usually Vertex-based or approximation.
        # But `map_affine` in Polytope may do V-rep conversion.
        return self.input_set.map_affine(self.transform)

    def output_set_chebyshev_radius(self) -> float:
        """Radius of largest inscribed ball in the output set (proxy for volume)."""
        radius, _ = self.output_set.cheby_ball()
        return float(radius) if radius is not None else 0.0

    def intersect_polytope(self, other: Polytope) -> Star:
        """
        Constrain the Output of the star to lie within `other`.
        Adds constraints C(Ax+b) <= d  => (CA)x <= d - Cb to the input set.
        """
        # other: Cy <= d
        # y = Ax + b
        # C(Ax+b) <= d ==> CA x <= d - Cb
        C = other.A @ self.transform.A
        d = other.b - other.A @ self.transform.b
        
        inp = Polytope(
            jnp.concatenate([self.input_set.A, C], axis=0),
            jnp.concatenate([self.input_set.b, d], axis=0),
        )
        return Star(inp, self.transform, self.activation_pattern)

    def join(self, other: Star, input_dim: int) -> Star:
        """
        Join this star set with another (e.g., cross-product inputs).
        Assumes independent input sets up to dimension `input_dim`?
        (This logic was specific to previous use-cases, keeping for compatibility).
        """
        other_proj_inp = other.input_set.project(list(range(1, input_dim + 1)))
        assert other_proj_inp is not None
        
        # Intersect inputs
        join_inp = self.intersect_polytope(other_proj_inp).input_set
        
        # Compose transforms? This seems to imply specific structure (gem.py legacy?).
        # Legacy Note: This looks like composition logic from GEM 
        # where `other` might be a state-transition and `self` is state? 
        # Keeping as-is but with type hints.
        join_trans = other.transform.map(self.transform)
        
        return Star(join_inp, join_trans)

# ---------------------------------------------------------------------------
# PyTree Support
# ---------------------------------------------------------------------------

def star_flatten(obj: Star) -> Tuple[Tuple[Polytope, Affine], Tuple[Optional[str]]]:
    return (obj.input_set, obj.transform), (obj.activation_pattern,)

def star_unflatten(aux_data: Tuple[Optional[str]], children: Tuple[Polytope, Affine]) -> Star:
    return Star(children[0], children[1], aux_data[0])

jax.tree_util.register_pytree_node(Star, star_flatten, star_unflatten)

# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Smoke tests
    A = jnp.array([[1.0], [-1.0]])
    b = jnp.array([1.0, 0.0])
    poly = Polytope(A, b) # 0 <= x <= 1
    
    aff = Affine(jnp.array([[2.0]]), jnp.array([1.0])) # y = 2x + 1
    star = Star(poly, aff)

    # Test mapping
    star2 = star.map_affine(Affine.identity(1))
    assert isinstance(star2, Star)

    # Test split
    splits = star.map_steprelu(0)
    # y = 2x+1. At x=0, y=1. At x=1, y=3. Always positive.
    # Should result in 1 split (pos).
    # wait, map_steprelu checks if y < 0 or y > 0.
    # range is [1, 3]. So always > 0.
    # So "neg" set should be empty. "pos" set entire.
    assert len(splits) == 1
    assert splits[0].activation_pattern == "1"

    print("Star unit tests passed.")
