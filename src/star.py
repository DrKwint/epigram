from __future__ import annotations


import jax.numpy as jnp
import tree

from src.affine import Affine
from src.polytope import Polytope
from typing import Optional, Callable

Metrics = dict[str, float]


class Star:
    """
    Represents a Star set, defined by an input set, an affine transformation, and an optional activation pattern.

    Attributes:
        input_set (Polytope): The set of possible inputs.
        transform (Affine): The affine transformation applied to the input set.
        activation_pattern (Optional[str]): The activation pattern for the star set (e.g., for ReLU networks).
    """

    def __init__(
        self,
        input_set: Polytope,
        transform: Affine,
        activation_pattern: Optional[str] = None,
    ) -> None:
        """
        Initialize a Star set.

        Args:
            input_set (Polytope): The set of possible inputs.
            transform (Affine): The affine transformation applied to the input set.
            activation_pattern (Optional[str]): The activation pattern for the star set.
        """
        self.input_set: Polytope = input_set
        self.transform: Affine = transform
        self.activation_pattern = activation_pattern

    def map_affine(self, aff: Affine) -> Star:
        """
        Apply an affine transformation to the star set.
        """

        return Star(
            self.input_set,
            aff.map(self.transform),
            activation_pattern=self.activation_pattern,
        )

    def map_steprelu(self, dim: int) -> list[Star]:
        """
        Split the star set along a single ReLU activation at the given dimension.
        """

        neg_set = self.input_set.add_eqn(
            self.transform.A[dim],
            -1 * self.transform.b[dim],
        )
        neg_transform = Affine(
            self.transform.A.at[dim].set(0.0),
            self.transform.b.at[dim].set(0.0),
        )
        neg_pat = (
            self.activation_pattern + "0"
            if self.activation_pattern is not None
            else "0"
        )
        pos_set = self.input_set.add_eqn(
            -1 * self.transform.A[dim],
            self.transform.b[dim],
        )
        pos_pat = (
            self.activation_pattern + "1"
            if self.activation_pattern is not None
            else "1"
        )
        result = [
            s
            for s in [
                Star(neg_set, neg_transform, neg_pat),
                Star(pos_set, self.transform, pos_pat),
            ]
            if not s.is_empty()
        ]
        return result

    def map_relu(self) -> list[Star]:
        """
        Apply ReLU activation to all dimensions, splitting the star set as needed.
        Returns the list of resulting stars and metrics about the split.
        """
        stars = [self]
        for dim in range(self.transform.A.shape[0]):
            stars: list[Star] = tree.flatten([s.map_steprelu(dim) for s in stars])
        return stars

    def map_relu_checked(
        self,
        check_fn: Callable[[jnp.ndarray, jnp.ndarray], bool],
    ) -> list[Star]:
        """
        Apply ReLU activation to all dimensions, filtering with a custom check function.
        """
        stars = [self]
        for dim in range(self.transform.A.shape[0]):
            stars = tree.flatten([s.map_steprelu(dim) for s in stars])
            stars = [s for s in stars if check_fn(s.input_set.A, s.input_set.b)]
        return stars

    def is_empty(self) -> bool:
        """
        Check if the star set is empty.

        Returns:
            bool: True if the input set is empty, False otherwise.
        """
        return self.input_set.is_empty()

    @property
    def output_set(self) -> Polytope:
        """
        Get the output set by applying the affine transformation to the input set.

        Returns:
            Polytope: The output set.
        """
        return self.input_set.map_affine(self.transform)

    def output_set_chebyshev_radius(self) -> float:
        """
        Computes the radius of the largest inscribed ball in the output set.
        This serves as a proxy for the 'volume' or 'tightness' of the star.
        A smaller radius means a more precise, less conservative reachable set.
        """
        # This can be computationally expensive.
        radius, _ = self.output_set.cheby_ball()
        return float(radius) if radius is not None else 0.0

    def intersect_polytope(self, other: Polytope) -> "Star":
        """
        Intersect the star's output polytope with another polytope.
        """
        C = other.A @ self.transform.A
        d = other.b - other.A @ self.transform.b
        inp = Polytope(
            jnp.concatenate([self.input_set.A, C], axis=0),
            jnp.concatenate([self.input_set.b, d], axis=0),
        )
        result = Star(inp, self.transform)
        return result

    def join(self, other: "Star", input_dim: int) -> "Star":
        """
        Join this star set with another, assuming their input/output sets are compatible.
        """
        other_proj_inp = other.input_set.project(list(range(1, input_dim + 1)))
        assert other_proj_inp is not None
        join_inp = self.intersect_polytope(other_proj_inp).input_set
        join_trans = other.transform.map(self.transform)
        result = Star(join_inp, join_trans)
        return result


def flatten_func(obj):
    children = (obj.input_set, obj.transform)  # children must contain arrays & pytrees
    aux_data = (obj.activation_pattern,)  # aux_data must contain static, hashable data.
    return (children, aux_data)


def unflatten_func(aux_data, children):
    # Here we avoid `__init__` because it has extra logic we don't require:
    input_set, transform = children
    activation_pattern = aux_data[0]
    return Star(input_set, transform, activation_pattern)


import jax

jax.tree_util.register_pytree_node(Star, flatten_func, unflatten_func)

# =====================
# Unit Tests
# =====================
if __name__ == "__main__":
    from src.affine import Affine
    from src.polytope import Polytope

    # Simple Polytope: x >= 0, x <= 1
    A = jnp.array([[1.0], [-1.0]])
    b = jnp.array([1.0, 0.0])
    poly = Polytope(A, b)
    # Affine: y = 2x + 1
    aff = Affine(jnp.array([[2.0]]), jnp.array([1.0]))
    star = Star(poly, aff)

    # Test affine map
    star2 = star.map_affine(Affine(jnp.array([[1.0]]), jnp.array([0.0])))
    assert isinstance(star2, Star)

    # Test ReLU split
    relu_stars = star.map_steprelu(0)
    assert all(isinstance(s, Star) for s in relu_stars)

    # Test ReLU map
    relu_all, metrics = star.map_relu()
    assert (
        all(isinstance(s, Star) for s in relu_all)
        and "verification/split_factor" in metrics
    )

    # Test intersection
    poly2 = Polytope(jnp.array([[1.0]]), jnp.array([0.5]))
    star3 = star.intersect_polytope(poly2)
    assert isinstance(star3, Star)

    # Test join
    star4 = star.join(star, 1)
    assert isinstance(star4, Star)

    print("All Star unit tests passed.")
