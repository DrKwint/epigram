from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linprog


from jaxtyping import Array
from typing import Any, Optional, Tuple, Union
import time
import polytope as pc

from src.affine import Affine

ArrayLike = Union[np.ndarray, Array]


class Polytope:
    """
    Convex polytope in H-representation:   A x ≤ b
    """

    def __init__(self, A: ArrayLike, b: ArrayLike):
        A = jnp.asarray(A)
        b = jnp.asarray(b)

        if A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {A.shape}")
        if b.ndim != 1:
            raise ValueError(f"b must be 1D, got shape {b.shape}")
        if A.shape[0] != b.shape[0]:
            raise ValueError(
                "Number of inequalities must match: A.shape[0] != b.shape[0]"
            )

        self.A = A
        self.b = b

    @classmethod
    def box(cls, dim: int, bounds: Optional[Tuple[Array, Array]] = None) -> Polytope:
        if bounds is not None:
            lo, hi = bounds
            lo = jnp.asarray(lo)
            hi = jnp.asarray(hi)
            assert lo.shape == (dim,)
            assert hi.shape == (dim,)

            A = jnp.concatenate([-jnp.eye(dim), jnp.eye(dim)], axis=0)
            b = jnp.concatenate([-lo, hi], axis=0)
        else:
            A = jnp.zeros((0, dim))
            b = jnp.zeros((0,))

        return Polytope(A, b)

    @classmethod
    def random(cls, dim: int, key, n=10):
        A = jax.random.normal(key, (n, dim))
        b = jnp.ones((n,))
        return Polytope(A, b)

    ### H-rep algebraic functions
    # Cheap and easy

    def add_eqn(self, coeffs: Array, rhs: Array):
        return Polytope(
            jnp.concatenate([self.A, coeffs[None, :]], axis=0),
            jnp.concatenate([self.b, jnp.array([rhs])], axis=0),
        )

    def contains(self, x):
        return jnp.all(self.A @ x <= self.b)

    def intersect(self, other: Polytope) -> Polytope:
        return Polytope(
            jnp.concatenate([self.A, other.A], axis=0),
            jnp.concatenate([self.b, other.b], axis=0),
        )

    def translate(self, t: Array) -> Polytope:
        t = jnp.asarray(t)
        return Polytope(self.A, self.b + self.A @ t)

    __add__ = translate

    def map_affine(self, aff: Affine) -> Polytope:
        verts = self.extreme()
        out_pts = verts @ aff.A.T + aff.b
        return Polytope.from_v(out_pts)

    def is_empty(self) -> bool:
        # This is a potential bottleneck, so it's good to be aware of its performance.
        # Note: linprog can be slow. For many checks, a faster SAT/SMT solver
        # might be better if the problem can be framed that way.
        start_time = time.perf_counter()
        c = np.zeros(self.A.shape[1])
        out = linprog(
            c=c, A_ub=np.asarray(self.A), b_ub=np.asarray(self.b), bounds=(None, None)
        )
        return not out.success  # `success` is True if a feasible point is found.

    def project(self, dims: Any) -> Optional[Polytope]:
        """
        Project the polytope onto a subset of dimensions.

        Args:
            dims: The dimensions to project onto.

        Returns:
            Optional[Polytope]: The projected polytope, or None if empty or projection failed.
        """
        # proj = pc.projection(pc.Polytope(np.array(self.A), np.array(self.b)), dims)  # type: ignore[attr-defined]
        poly = pc.Polytope(np.array(self.A), np.array(self.b))
        proj = pc.projection(poly, dims)
        if (
            proj is not None
            and hasattr(proj, "A")
            and hasattr(proj, "b")
            and len(proj.b) > 0
        ):
            return Polytope(proj.A, proj.b)
        else:
            return None

    def extreme(self) -> Array:
        """
        Convert H-rep → V-rep using polytope package (no cdd).
        """
        # Create a polytope object from the package
        P = pc.Polytope(np.asarray(self.A), np.asarray(self.b))
        pts = pc.extreme(P)
        return jnp.asarray(pts)

    @classmethod
    def from_v(cls, V: ArrayLike):
        V = np.asarray(V)
        if V.shape[0] == 0:
             # Handle empty case
             return Polytope(jnp.zeros((0, V.shape[1])), jnp.zeros((0,)))
             
        # Use qhull from polytope package
        P = pc.qhull(V)
        return Polytope(jnp.asarray(P.A), jnp.asarray(P.b))

    def is_subset(self, bigger: Polytope) -> bool:
        pts = np.asarray(self.extreme())
        A = np.asarray(bigger.A)
        b = np.asarray(bigger.b)
        return bool(np.all(pts @ A.T <= b + 1e-12))  # tolerance

    def reduce(self):
        reduced = pc.reduce(pc.Polytope(np.array(self.A), np.array(self.b)))
        if hasattr(reduced, "A") and hasattr(reduced, "b") and len(reduced.b) > 0:
            return Polytope(reduced.A, reduced.b)
        return None

    def cheby_ball(self):
        # (radius, center)
        return pc.cheby_ball(pc.Polytope(np.array(self.A), np.array(self.b)))


# ----------------------------------------------------------------------
# PyTree registration
# ----------------------------------------------------------------------


def flatten_func(obj: Polytope):
    return (obj.A, obj.b), None


def unflatten_func(aux, children):
    A, b = children
    return Polytope(A, b)


jax.tree_util.register_pytree_node(Polytope, flatten_func, unflatten_func)
