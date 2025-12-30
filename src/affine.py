from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Tuple, Union, Optional
from jaxtyping import Array, Float
from numpy.typing import NDArray

ArrayLike = Union[NDArray, Array]

class Affine:
    """
    Represents an affine map x ↦ A x + b, where A ∈ R^{m×n} and b ∈ R^m.

    Attributes:
        A (jax.Array): Linear transformation matrix [m, n].
        b (jax.Array): Translation vector [m].
    """
    __slots__ = ("A", "b")  # Optimization for immutable-like class

    def __init__(self, A: ArrayLike, b: ArrayLike) -> None:
        """
        Initialize an Affine map. Auto-converts inputs to JAX arrays.
        """
        self.A: Float[Array, "m n"] = jnp.asarray(A)
        self.b: Float[Array, "m"] = jnp.asarray(b)

        if self.b.ndim != 1:
            raise ValueError(f"b must be a 1D vector, got shape {self.b.shape}")
        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: A has {self.A.shape[0]} rows but b has {self.b.shape[0]}."
            )

    @classmethod
    def identity(cls, dim: int) -> Affine:
        """Returns the identity affine map on R^dim."""
        return cls(jnp.eye(dim), jnp.zeros(dim))

    def map(self, other: Affine) -> Affine:
        """
        Compose two affine maps: self ∘ other.

        (self)(x) = A_self (A_other x + b_other) + b_self
                  = (A_self A_other) x + (A_self b_other + b_self)
        """
        A = self.A @ other.A
        b = self.b + self.A @ other.b
        return Affine(A, b)

    def __matmul__(self, other: Affine) -> Affine:
        """Syntactic sugar for composition: F = F2 @ F1"""
        return self.map(other)

    def __call__(self, x: ArrayLike) -> Float[Array, "m"]:
        """Apply the affine transformation to a vector."""
        x = jnp.asarray(x)
        return self.A @ x + self.b

    def __repr__(self) -> str:
        return f"Affine(output_dim={self.b.shape[0]}, input_dim={self.A.shape[1]})"

    def try_inverse(self) -> Optional[Affine]:
        """
        Attempt to invert the affine map x ↦ A x + b.

        Returns:
            Affine: The inverse map y ↦ A^{-1}(y - b) if A is invertible.
            None:   If A is not square or is numerically singular (cond > 1e12).
        """
        if self.A.shape[0] != self.A.shape[1]:
            return None

        # Check conditioning
        cond = jnp.linalg.cond(self.A)
        if jnp.isinf(cond) or cond > 1e12:
            return None

        A_inv = jnp.linalg.inv(self.A)
        b_inv = -A_inv @ self.b
        return Affine(A_inv, b_inv)

# ---------------------------------------------------------------------------
# PyTree Support
# ---------------------------------------------------------------------------

def affine_flatten(obj: Affine) -> Tuple[Tuple[Array, Array], None]:
    return (obj.A, obj.b), None

def affine_unflatten(aux_data: Any, children: Tuple[Array, Array]) -> Affine:
    return Affine(*children)

jax.tree_util.register_pytree_node(Affine, affine_flatten, affine_unflatten)
