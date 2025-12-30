from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from typing import Union, Tuple, Any
from jaxtyping import Array
from numpy.typing import NDArray


ArrayLike = Union[NDArray, Array]


class Affine:
    """
    Represents an affine map x ↦ A x + b
    where A ∈ R^{m×n} and b ∈ R^m.

    Attributes:
        A (jnp.ndarray): Linear transformation matrix.
        b (jnp.ndarray): Translation vector.
    """

    def __init__(self, A: ArrayLike, b: ArrayLike) -> None:
        # Convert to JAX arrays immediately
        A = jnp.asarray(A)
        b = jnp.asarray(b)

        if b.ndim != 1:
            raise ValueError(f"b must be a 1D vector, got shape {b.shape}")
        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                f"Incompatible dimensions: A has {A.shape[0]} rows but b has {b.shape[0]}."
            )

        self.A = A
        self.b = b

    @classmethod
    def identity(cls, dim: int) -> Affine:
        """Identity affine map on R^dim."""
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

    # syntactic sugar
    __matmul__ = map  # so you can write: F = F2 @ F1

    def __call__(self, x: ArrayLike) -> jnp.ndarray:
        """Apply the affine transformation to a vector or batch."""
        x = jnp.asarray(x)
        return self.A @ x + self.b

    def __repr__(self) -> str:
        return f"Affine(A={np.array(self.A)}, b={np.array(self.b)})"

    def try_inverse(self) -> Affine | None:
        """
        Attempt to invert the affine map x ↦ A x + b.

        Returns:
            Affine: The inverse map y ↦ A^{-1}(y - b) if A is invertible.
            None:   If A is not square or is numerically singular.
        """
        A = self.A
        b = self.b

        # Must be square
        if A.shape[0] != A.shape[1]:
            return None

        # Check conditioning: high cond ⇒ nearly singular.
        # Threshold 1e12 is conservative and common in numerical code.
        cond = jnp.linalg.cond(A)
        if jnp.isinf(cond) or cond > 1e12:
            return None

        # Compute A^{-1}
        A_inv = jnp.linalg.inv(A)
        b_inv = -A_inv @ b

        return Affine(A_inv, b_inv)


# ---------------------------------------------------------------------------
# PyTree support
# ---------------------------------------------------------------------------


def affine_flatten(obj: Affine):
    """Return ((children), aux). Children must be JAX arrays or pytrees."""
    children = (obj.A, obj.b)
    aux_data = None  # no static fields
    return children, aux_data


def affine_unflatten(aux_data: Any, children: Tuple[Any, Any]) -> Affine:
    A, b = children
    return Affine(A, b)


import jax

jax.tree_util.register_pytree_node(Affine, affine_flatten, affine_unflatten)
