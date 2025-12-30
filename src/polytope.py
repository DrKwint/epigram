from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import time
import polytope as pc  # type: ignore

from scipy.optimize import linprog
from jaxtyping import Array, Float
from typing import Any, Optional, Tuple, Union

from src.affine import Affine

ArrayLike = Union[np.ndarray, Array]

class Polytope:
    """
    Convex polytope in H-representation: A x ≤ b.
    
    Attributes:
        A (jax.Array): Matrix of shape [n_constr, n_dim].
        b (jax.Array): Vector of shape [n_constr].
    """
    __slots__ = ("A", "b")

    def __init__(self, A: ArrayLike, b: ArrayLike) -> None:
        self.A: Float[Array, "n_constr dim"] = jnp.asarray(A)
        self.b: Float[Array, "n_constr"] = jnp.asarray(b)

        if self.A.ndim != 2:
            raise ValueError(f"A must be 2D, got shape {self.A.shape}")
        if self.b.ndim != 1:
            raise ValueError(f"b must be 1D, got shape {self.b.shape}")
        if self.A.shape[0] != self.b.shape[0]:
            raise ValueError(
                f"Number of inequalities must match: A.shape[0] {self.A.shape[0]} != b.shape[0] {self.b.shape[0]}"
            )

    @classmethod
    def box(cls, dim: int, bounds: Optional[Tuple[Array, Array]] = None) -> Polytope:
        """Create a hyperrectangle (box)."""
        if bounds is not None:
            lo, hi = bounds
            lo = jnp.asarray(lo)
            hi = jnp.asarray(hi)
            assert lo.shape == (dim,)
            assert hi.shape == (dim,)

            # [-I; I] * x <= [-lo; hi] <=> -x <= -lo (x>=lo) AND x <= hi
            A = jnp.concatenate([-jnp.eye(dim), jnp.eye(dim)], axis=0)
            b = jnp.concatenate([-lo, hi], axis=0)
        else:
            # Empty constraints (whole R^n) or empty set?
            # Standard "empty" representation usually implies 0 constraints but that is R^n.
            # Let's assume consistent initialization.
            A = jnp.zeros((0, dim))
            b = jnp.zeros((0,))

        return Polytope(A, b)

    @classmethod
    def random(cls, dim: int, key: jax.Array, n: int = 10) -> Polytope:
        A = jax.random.normal(key, (n, dim))
        b = jnp.ones((n,))
        return Polytope(A, b)

    ### H-rep algebraic functions

    def add_eqn(self, coeffs: Array, rhs: Array) -> Polytope:
        """Add a single linear inequality: coeffs^T x <= rhs."""
        return Polytope(
            jnp.concatenate([self.A, coeffs[None, :]], axis=0),
            jnp.concatenate([self.b, jnp.array([rhs])], axis=0),
        )

    def contains(self, x: ArrayLike) -> bool:
        """Check if x satisfies A x <= b."""
        return bool(jnp.all(self.A @ jnp.asarray(x) <= self.b))

    def intersect(self, other: Polytope) -> Polytope:
        """Intersect two polytopes (concatenate constraints)."""
        return Polytope(
            jnp.concatenate([self.A, other.A], axis=0),
            jnp.concatenate([self.b, other.b], axis=0),
        )

    def translate(self, t: ArrayLike) -> Polytope:
        """Shift the polytope by vector t."""
        t = jnp.asarray(t)
        # A(x - t) <= b  => Ax <= b + At
        return Polytope(self.A, self.b + self.A @ t)

    def __add__(self, t: ArrayLike) -> Polytope:
        return self.translate(t)

    def map_affine(self, aff: Affine) -> Polytope:
        """
        Map this H-rep polytope through affine map y = Ax+b.
        WARNING: Requires vertex enumeration (expensive).
        """
        verts = self.extreme()
        # y = A v + b
        out_pts = verts @ aff.A.T + aff.b
        return Polytope.from_v(out_pts)

    def is_empty(self) -> bool:
        """
        Check emptiness via LP.
        Performance warning: Uses scipy.optimize.linprog (slow compared to compiled solvers).
        """
        # Optimize 0 subject to Ax <= b
        c = np.zeros(self.A.shape[1])
        # Convert to numpy for scipy
        A_np = np.asarray(self.A)
        b_np = np.asarray(self.b)
        
        # Check for empty constraints (R^n is not empty)
        if A_np.shape[0] == 0:
            return False

        out = linprog(
            c=c, A_ub=A_np, b_ub=b_np, bounds=(None, None)
        )
        return not out.success

    def project(self, dims: Any) -> Optional[Polytope]:
        """
        Project onto subset of dimensions using `polytope` library.
        """
        poly_np = pc.Polytope(np.array(self.A), np.array(self.b))
        try:
            proj = pc.projection(poly_np, dims)
        except Exception:
            return None
            
        if proj is None: # Empty projection?
            return None
            
        # Check standard properties availability
        if hasattr(proj, "A") and hasattr(proj, "b") and len(proj.b) > 0:
            return Polytope(jnp.array(proj.A), jnp.array(proj.b))
        return None

    def extreme(self) -> Array:
        """
        Compute vertices (V-rep) using external `polytope` library.
        Returns JAX array [N_vertices, Dim].
        """
        P = pc.Polytope(np.asarray(self.A), np.asarray(self.b))
        pts = pc.extreme(P)
        if pts is None: # Should return None if empty? Or empty array?
             return jnp.zeros((0, self.A.shape[1]))
        return jnp.asarray(pts)

    @classmethod
    def from_v(cls, V: ArrayLike) -> Polytope:
        """Create from vertices (Convex Hull)."""
        V = np.asarray(V)
        if V.shape[0] == 0:
             return Polytope(jnp.zeros((0, V.shape[1])), jnp.zeros((0,)))
             
        P = pc.qhull(V) # Uses Qhull via `polytope` wrapper
        return Polytope(jnp.asarray(P.A), jnp.asarray(P.b))

    def is_subset(self, bigger: Polytope) -> bool:
        """Check if self is subset of bigger."""
        pts = np.asarray(self.extreme())
        if pts.shape[0] == 0: return True # Empty is subset of anything
        
        A = np.asarray(bigger.A)
        b = np.asarray(bigger.b)
        # Check if all vertices of self satisfy bigger's constraints
        # Ax <= b
        return bool(np.all(pts @ A.T <= b + 1e-7))

    def reduce(self) -> Optional[Polytope]:
        """Remove redundant constraints."""
        P_np = pc.Polytope(np.array(self.A), np.array(self.b))
        reduced = pc.reduce(P_np)
        
        if hasattr(reduced, "A") and hasattr(reduced, "b") and len(reduced.b) > 0:
            return Polytope(reduced.A, reduced.b)
        return None

    def cheby_ball(self) -> Tuple[float, Optional[np.ndarray]]:
        """Return (radius, center) of largest inscribed ball."""
        P_np = pc.Polytope(np.array(self.A), np.array(self.b))
        res = pc.cheby_ball(P_np)
        # res is (radius, center)
        return res

# ----------------------------------------------------------------------
# PyTree registration
# ----------------------------------------------------------------------

def flatten_func(obj: Polytope) -> Tuple[Tuple[Array, Array], None]:
    return (obj.A, obj.b), None

def unflatten_func(aux: None, children: Tuple[Array, Array]) -> Polytope:
    return Polytope(*children)

jax.tree_util.register_pytree_node(Polytope, flatten_func, unflatten_func)
