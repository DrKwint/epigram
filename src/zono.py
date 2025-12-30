from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import jax
import jax.numpy as jnp
import polars as pl
from flax import nnx
from jaxtyping import Array, Float

from src.net import ENN


@dataclass(frozen=True)
class SensitivityResult:
    """
    Represents a potential intervention found by sensitivity analysis.
    """
    gain: float
    source_type: Literal['action', 'z', 'relu', 'input']
    
    # Range of generator columns in the zonotope matrix
    indices: Tuple[int, int]
    
    # Metadata from the ledger (e.g. layer name, neuron index, timestep)
    meta: Dict[str, Any]
    
    @property
    def desc(self) -> str:
        """Human-readable description for logging."""
        if self.source_type == 'action':
            return f"Bisect Action (Gain: {self.gain:.4f})"
        elif self.source_type == 'z':
            return f"Bisect Epistemic Z (Gain: {self.gain:.4f})"
        elif self.source_type == 'relu':
            # 'layer' is the standard key used in AbstractReLU
            layer = self.meta.get('layer', self.meta.get('layer_name', 'unknown'))
            idx = self.meta.get('index', '?')
            return f"Stabilize ReLU {layer}[{idx}] (Gain: {self.gain:.4f})"
        return f"Fix {self.source_type} (Gain: {self.gain:.4f})"


@dataclass(frozen=True)
class GeneratorGroup:
    source_type: str  # 'input', 'action', 'z', 'relu'
    start_idx: int  # Column index in generator matrix
    count: int  # Number of columns
    metadata: Dict[str, Any] = field(default_factory=dict)  # e.g. layer name

    @property
    def end_idx(self) -> int:
        return self.start_idx + self.count


class Zonotope(nnx.Module):
    """
    Represent a set of zonotopes (batched).
    Z = { c + \sum e_i g_i | e_i \in [-1, 1] }
    
    Attributes:
        center (Array): [Batch, Features]
        generators (Array): [Batch, N_Gens, Features]
        history (Tuple[GeneratorGroup]): Ledger of generator origins.
    """
    def __init__(
        self,
        center: Float[Array, "b f"],
        generators: Float[Array, "b n f"],
        history: Tuple[GeneratorGroup, ...] = (),
    ):
        # JAX arrays (Traced)
        self.center = center
        self.generators = generators

        # Static Metadata (Not Traced)
        self.history = history

    def concrete_bounds(self) -> Tuple[Float[Array, "b f"], Float[Array, "b f"]]:
        """Compute independent bounds for each feature."""
        radius = jnp.sum(jnp.abs(self.generators), axis=1)
        return self.center - radius, self.center + radius

    def append_error_terms(
        self, 
        new_generators: Float[Array, "b k f"], 
        source_type: str, 
        meta: Optional[Dict[str, Any]] = None
    ) -> "Zonotope":
        """
        Appends new error terms (columns) to the generator matrix.
        Used by AbstractReLU to add relaxation errors.
        """
        # 1. Concatenate Arrays
        combined_gens = jnp.concatenate([self.generators, new_generators], axis=1)

        # 2. Update History
        current_count = self.generators.shape[1]
        new_count = new_generators.shape[1]

        new_group = GeneratorGroup(
            source_type=source_type,
            start_idx=current_count,
            count=new_count,
            metadata=meta or {},
        )

        return Zonotope(self.center, combined_gens, self.history + (new_group,))

    def stack_independent(self, other: "Zonotope") -> Tuple["Zonotope", "Zonotope"]:
        """
        Merges this Zonotope with another INDEPENDENT source (like x and z).
        Returns two new Zonotopes that share a unified, block-diagonal generator basis.
        """
        g1 = self.generators
        g2 = other.generators

        n1 = g1.shape[1]
        n2 = g2.shape[1]

        # 1. Create Block Diagonal Generators
        # New G1: [G1, 0]
        pad1 = jnp.zeros((g1.shape[0], n2, g1.shape[2]))
        new_g1 = jnp.concatenate([g1, pad1], axis=1)

        # New G2: [0, G2]
        pad2 = jnp.zeros((g2.shape[0], n1, g2.shape[2]))
        new_g2 = jnp.concatenate([pad2, g2], axis=1)

        # 2. Merge Histories
        # The history of the second zonotope needs to be offset by n1
        shifted_history_2 = []
        for group in other.history:
            shifted_history_2.append(
                GeneratorGroup(
                    source_type=group.source_type,
                    start_idx=group.start_idx + n1,  # Offset!
                    count=group.count,
                    metadata=group.metadata,
                )
            )

        combined_history = self.history + tuple(shifted_history_2)

        z1_aligned = Zonotope(self.center, new_g1, combined_history)
        z2_aligned = Zonotope(other.center, new_g2, combined_history)

        return z1_aligned, z2_aligned

    def concatenate(self, others: List["Zonotope"], axis: int = -1) -> "Zonotope":
        """
        Concatenate features with other Zonotopes.
        Assumes all zonotopes share the same generator basis (aligned errors).
        Pads generator count to max if mismatched (assuming prefix match).
        """
        zonotopes = [self] + others
        max_errors = max(z.generators.shape[1] for z in zonotopes)

        aligned_gens = []
        for z in zonotopes:
            pad = max_errors - z.generators.shape[1]
            if pad > 0:
                padding = jnp.zeros((z.generators.shape[0], pad, z.generators.shape[2]))
                aligned_gens.append(jnp.concatenate([z.generators, padding], axis=1))
            else:
                aligned_gens.append(z.generators)

        new_center = jnp.concatenate([z.center for z in zonotopes], axis=axis)
        new_generators = jnp.concatenate(aligned_gens, axis=axis)

        # Inherit history from 'self' (assuming they share the lineage)
        return Zonotope(new_center, new_generators, self.history)

    def add(self, other: "Zonotope") -> "Zonotope":
        """
        Minkowski sum: Z_new = Z_1 + Z_2.
        c = c1 + c2
        G = G1 + G2 (assuming same error basis).
        """
        n_err1 = self.generators.shape[1]
        n_err2 = other.generators.shape[1]

        g1 = self.generators
        g2 = other.generators
        history = self.history

        if n_err1 < n_err2:
            pad = jnp.zeros((g1.shape[0], n_err2 - n_err1, g1.shape[2]))
            g1 = jnp.concatenate([g1, pad], axis=1)
            history = other.history
        elif n_err2 < n_err1:
            pad = jnp.zeros((g2.shape[0], n_err1 - n_err2, g2.shape[2]))
            g2 = jnp.concatenate([g2, pad], axis=1)

        return Zonotope(self.center + other.center, g1 + g2, history)

    def project(self, dims: Tuple[int, ...]) -> "Zonotope":
        """
        Projects the zonotope onto the specified dimensions.
        """
        dims_t = tuple(dims)
        new_center = self.center[:, dims_t]
        new_generators = self.generators[:, :, dims_t]
        return Zonotope(new_center, new_generators, self.history)

    def vertices(self) -> List[Array]:
        """
        Computes the vertices of the zonotope in 2D.
        Returns a List[Array] where each item is [NumVertices, 2].
        Uses a sampling-based convex hull approach for robustness.
        """
        from scipy.spatial import ConvexHull  # type: ignore
        import numpy as np

        def _get_verts(c: Array, g: Array) -> Array:
            dim = c.shape[-1]
            if dim != 2:
                return jnp.zeros((0, dim))

            # Robust fallback: Sample 100 random directions
            # Vertices of a zonotope are subset of hypercube vertices map.
            # Maximize x*d over the zonotope to find boundary points.
            gs_np = np.array(g)
            c_np = np.array(c)
            
            theta = np.linspace(0, 2 * np.pi, 100)
            dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
            
            verts = []
            for d in dirs:
                # Support point: c + G * sign(G^T d)
                projs = gs_np @ d
                u = np.sign(projs)
                pt = c_np + u @ gs_np
                verts.append(pt)
            
            hull = ConvexHull(verts)
            return jnp.array(np.array(verts)[hull.vertices])

        # Map over batch (returns list because vertex count differs)
        return [_get_verts(self.center[i], self.generators[i]) for i in range(self.center.shape[0])]

    def calculate_potential_gains(self, unsafe_direction: Float[Array, "f"]) -> pl.DataFrame:
        """
        Calculates how much the output bound would shrink along 'unsafe_direction'
        if we performed various splits. Returns a sorted Polars DataFrame.
        """
        # 1. Project all generators onto the unsafe vector
        # Shape: [Batch, N_generators] -> [N_generators] (Max over batch)
        # We take max over batch to capture worst-case sensitivity
        projected_mags = jnp.max(jnp.abs(self.generators @ unsafe_direction), axis=0)

        results = []

        # 2. Iterate through history
        for group in self.history:
            # Extract magnitude block
            group_mags = projected_mags[group.start_idx : group.end_idx]

            if group_mags.shape[0] == 0:
                continue

            # Find the "Max Impact" generator in this group
            best_local_idx = jnp.argmax(group_mags)
            max_val = group_mags[best_local_idx]

            # 3. Calculate Gain based on type
            if group.source_type == "relu":
                # Gain = 100% of the error (error vanishes)
                gain = float(max_val)
                global_idx = group.start_idx + int(best_local_idx)

                results.append(
                    {
                        "gain": gain,
                        "type": "relu",
                        "location": group.metadata.get("layer", "unknown"),
                        "local_index": int(best_local_idx),
                        "global_index": global_idx,
                    }
                )

            elif group.source_type in ["action", "z", "input"]:
                # Gain = 50% of the width (bisecting the interval)
                gain = 0.5 * float(max_val)
                global_idx = group.start_idx + int(best_local_idx)

                results.append(
                    {
                        "gain": gain,
                        "type": group.source_type,
                        "location": "input",
                        "local_index": int(best_local_idx),
                        "global_index": global_idx,
                    }
                )

        # 4. Create Polars DataFrame
        if not results:
            return pl.DataFrame(
                schema={
                    "gain": pl.Float64,
                    "type": pl.Utf8,
                    "location": pl.Utf8,
                    "local_index": pl.Int64,
                    "global_index": pl.Int64,
                }
            )

        df = pl.DataFrame(results)
        return df.sort("gain", descending=True)

    def project_bounds(self, direction: Float[Array, "f"]) -> Tuple[Float[Array, "b"], Float[Array, "b"]]:
        """
        Projects the zonotope onto a scalar direction vector 'h'.
        Used for checking safety against linear constraints (h^T x <= d).
        
        Args:
            direction: [Features] Normal vector of the half-space.
            
        Returns:
            lb: [Batch] Lower bound of the projection
            ub: [Batch] Upper bound of the projection
        """
        # 1. Project the Center
        c_proj = jnp.dot(self.center, direction)
        
        # 2. Project the Generators
        g_proj = jnp.dot(self.generators, direction)
        
        # 3. Calculate Radius (Sum of absolute projected generators)
        radius = jnp.sum(jnp.abs(g_proj), axis=1)
        
        # 4. Form Bounds
        lb = c_proj - radius
        ub = c_proj + radius
        return lb, ub
        
    def restrict_generators(self, restrictions: List[Tuple[int, float, float]]) -> "Zonotope":
        """
        Restricts specific error terms (generators) to a sub-interval [min, max].
        Original error terms are assumed to be [-1, 1].
        
        Args:
            restrictions: List of (global_gen_idx, new_min, new_max)
                          where -1.0 <= new_min <= new_max <= 1.0
                          
        Returns:
            New Zonotope with updated center and scaled generators.
        """
        if not restrictions:
            return self

        new_center = self.center
        new_gens = self.generators
        
        for (idx, r_min, r_max) in restrictions:
            # Linear map: e' = c + (w/2) * e_new
            mid = (r_min + r_max) / 2.0
            scale = (r_max - r_min) / 2.0
            
            # Update Center: Center += G[:, idx] * mid
            shift_vec = new_gens[:, idx, :] * mid 
            new_center = new_center + shift_vec
            
            # Scale Generator: G'[:, idx] = G[:, idx] * scale
            new_gens = new_gens.at[:, idx, :].mul(scale)
            
        return Zonotope(new_center, new_gens, self.history)

    def get_sensitivity_ranking(self, unsafe_direction: Float[Array, "f"]) -> List[SensitivityResult]:
        """
        Projects generators onto unsafe_direction and ranks sources by 
        how much splitting them would reduce the output bound.
        """
        # 1. Project generators onto the unsafe vector
        # self.generators: [Batch, N_Gens, Features]
        # projection: [Batch, N_Gens]
        projection = jnp.abs(self.generators @ unsafe_direction)
        
        # 2. Get the Worst-Case Magnitude for each generator (Max over batch)
        max_scores = jnp.max(projection, axis=0) # [N_Gens]

        ranking = []
        
        # 3. Iterate through the history ledger
        for group in self.history:
            scores = max_scores[group.start_idx : group.end_idx]
            
            if scores.shape[0] == 0:
                continue

            # Find local max in this group
            max_val = float(jnp.max(scores))
            local_best_idx = int(jnp.argmax(scores))
            
            # Calculate Gain based on source type
            if group.source_type in ['input', 'z', 'action']:
                # Bisecting reduces error by ~50%
                gain = 0.5 * max_val
            elif group.source_type == 'relu':
                # Fixing ReLU removes 100% of the error
                gain = 1.0 * max_val
            else:
                gain = max_val

            # Enrich metadata with the specific local index
            meta_enriched = group.metadata.copy()
            meta_enriched['local_index'] = local_best_idx
            meta_enriched['index'] = local_best_idx

            result = SensitivityResult(
                gain=gain,
                source_type=group.source_type, # type: ignore
                indices=(group.start_idx, group.end_idx),
                meta=meta_enriched
            )
            ranking.append(result)
            
        # 4. Sort by highest potential gain
        return sorted(ranking, key=lambda x: x.gain, reverse=True)

class AbstractLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x: Zonotope) -> Zonotope:
        # Wx + b
        new_center = self.linear(x.center)
        # W * G^T (Using einsum for batch/error dim handling)
        # Kernel: [in, out], Gens: [batch, n_err, in] -> [batch, n_err, out]
        new_gens = jnp.einsum("oi,bni->bno", self.linear.kernel.value.T, x.generators)
        return Zonotope(new_center, new_gens, x.history)

    @property
    def out_features(self) -> int:
        return self.linear.out_features

class AbstractReLU(nnx.Module):
    def __call__(
        self, 
        x: Zonotope, 
        split_idxs: Optional[List[Union[int, Tuple[int, str]]]] = None, 
        error_scale: Optional[Float[Array, "features"]] = None, 
        layer_name: str = ''
    ) -> List[Zonotope]:
        
        lb, ub = x.concrete_bounds()
        
        # 1. Initial Stability (from bounds)
        is_active = (lb >= 0)
        is_inactive = (ub <= 0)
        
        # 2. Process Overrides (Constraints)
        branch_candidates: List[int] = []
        
        final_active = is_active
        final_inactive = is_inactive
        
        # Mask to explicitly zero out error for constrained neurons
        constraint_mask = jnp.ones_like(lb)

        if split_idxs:
            for item in split_idxs:
                if isinstance(item, tuple):
                    # CONSTRAINT: Force the mask immediately
                    idx, status = item
                    
                    # Explicitly mask error for this neuron
                    constraint_mask = constraint_mask.at[:, idx].set(0.0)

                    if status == 'active':
                        final_active = final_active.at[:, idx].set(True)
                        final_inactive = final_inactive.at[:, idx].set(False)
                    elif status == 'inactive':
                        final_inactive = final_inactive.at[:, idx].set(True)
                        final_active = final_active.at[:, idx].set(False)
                else:
                    branch_candidates.append(item)

        # 3. Recalculate Unstable (CRITICAL STEP)
        # We must compute this AFTER applying constraints
        is_unstable = ~(final_active | final_inactive)

        # 4. DeepPoly Parameters (Lambda/Mu slope selection)
        # Ensure denom is not zero
        denom = jnp.maximum(ub - lb, 1e-9)
        slope_u = ub / denom
        offset_u = -lb * slope_u / 2.0
        base_error_mag = -lb * ub / (2.0 * denom)
        
        # Apply constraint mask to guarantee 0 error for forced neurons
        base_error_mag = base_error_mag * constraint_mask
        
        if error_scale is not None:
            base_error_mag = base_error_mag * error_scale

        # 5. Helper for Branching
        def apply_branch_config(extra_active: List[int], extra_inactive: List[int]) -> Zonotope:
            # Apply any *additional* splits on top of the constraints
            branch_active = final_active
            branch_inactive = final_inactive
            
            if extra_active:
                arr_act = jnp.array(extra_active)
                branch_active = branch_active.at[:, arr_act].set(True)
                branch_inactive = branch_inactive.at[:, arr_act].set(False)
            if extra_inactive:
                arr_inact = jnp.array(extra_inactive)
                branch_inactive = branch_inactive.at[:, arr_inact].set(True)
                branch_active = branch_active.at[:, arr_inact].set(False)
            
            # Recalc unstable for this specific branch
            branch_unstable = ~(branch_active | branch_inactive)
            
            # --- PARAMETER SELECTION ---
            # If Active: Slope=1, Offset=0, Error=0
            # If Inactive: Slope=0, Offset=0, Error=0
            # If Unstable: Slope=slope_u, Offset=offset_u, Error=base_error
            
            slope = jnp.where(branch_active, 1.0, 0.0)
            slope = jnp.where(branch_unstable, slope_u, slope)
            
            offset = jnp.where(branch_unstable, offset_u, 0.0)
            
            # ERROR MASKING
            # If branch_unstable is False (because we forced active), this returns 0.0.
            error = jnp.where(branch_unstable, base_error_mag, 0.0)
            
            # Transform Center and Generators
            new_center = x.center * slope + offset
            new_gens = x.generators * slope[:, None, :]
            
            # Create NEW error terms for relaxation
            # Error is diagonal per neuron
            new_err_gens = jax.vmap(jnp.diag)(error)
            
            z_branch = Zonotope(new_center, new_gens, x.history)
            return z_branch.append_error_terms(
                new_err_gens, 
                source_type='relu', 
                meta={'layer': layer_name}
            )

        # 6. Execute Branching
        if not branch_candidates:
            # Just return the single relaxed output
            return [apply_branch_config([], [])]
        
        # Generate configurations for split logic (usually 0, 1, or 2 branches if splitting 1 neuron)
        # Current logic supports multi-neuron splitting via product (be careful of explosion)
        options = [[(idx, 'active'), (idx, 'inactive')] 
                   for idx in branch_candidates if jnp.any(is_unstable[:, idx])]
                   
        if not options: 
            return [apply_branch_config([], [])]
        
        results = []
        for combination in itertools.product(*options):
            e_act = [idx for idx, s in combination if s == 'active']
            e_inact = [idx for idx, s in combination if s == 'inactive']
            results.append(apply_branch_config(e_act, e_inact))
            
        return results

def merge_independent_sources(x: Zonotope, z: Zonotope) -> Tuple[Zonotope, Zonotope]:
    """
    Prepares independent inputs x and z for joint processing.
    Block-diagonalizes their generators so they don't interfere.
    Returns aligned versions of x and z with compatible generator shapes.
    """
    return x.stack_independent(z)


class AbstractENN(nnx.Module):
    def __init__(
        self, x_dim: int, a_dim: int, z_dim: int, hidden_dim: int, *, rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        # Replace standard layers with Abstract versions
        self.base_fc1 = AbstractLinear(x_dim + a_dim, hidden_dim, rngs=rngs)
        self.base_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)

        in_dim = hidden_dim + x_dim + a_dim + z_dim
        self.epi_fc1 = AbstractLinear(in_dim, hidden_dim, rngs=rngs)
        self.epi_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)
        self.prior_fc1 = AbstractLinear(in_dim, hidden_dim, rngs=rngs)
        self.prior_out = AbstractLinear(hidden_dim, x_dim, rngs=rngs)

        # Abstract Activations
        self.act_base = AbstractReLU()
        self.act_epi = AbstractReLU()
        self.act_prior = AbstractReLU()

    def copy_weights_from(self, source_model: ENN) -> None:
        """
        Copies parameters from a trained concrete ENN to this abstract ENN.
        """
        # Helper to copy Linear -> AbstractLinear
        def copy_linear(src: nnx.Linear, dst: AbstractLinear) -> None:
            # AbstractLinear wraps the real layer in '.linear'
            dst.linear.kernel.value = src.kernel.value
            if src.bias is not None and dst.linear.bias is not None:
                dst.linear.bias.value = src.bias.value

        # Copy Base Network
        copy_linear(source_model.base_fc1, self.base_fc1)
        copy_linear(source_model.base_out, self.base_out)

        # Copy Epistemic Network
        copy_linear(source_model.epi_fc1, self.epi_fc1)
        copy_linear(source_model.epi_out, self.epi_out)

        # Copy Prior Network
        copy_linear(source_model.prior_fc1, self.prior_fc1)
        copy_linear(source_model.prior_out, self.prior_out)

    @classmethod
    def from_concrete(cls, concrete_enn: ENN) -> "AbstractENN":
        """
        Factory method to create an AbstractENN directly from a trained ENN.
        """
        rngs = nnx.Rngs(params=0)
        # Instantiate abstract model with same dims
        abstract_model = cls(
            x_dim=concrete_enn.x_dim,
            a_dim=concrete_enn.a_dim,
            z_dim=concrete_enn.z_dim,
            # Infer hidden dim from the first layer output features
            hidden_dim=concrete_enn.base_fc1.out_features,
            rngs=rngs,
        )

        # Copy weights
        abstract_model.copy_weights_from(concrete_enn)

        return abstract_model

    def __call__(
        self,
        x: Zonotope,
        z: Zonotope,
        error_scales: Optional[Dict[str, Float[Array, "any"]]] = None,
        split_idxs: Optional[Dict[str, list[int]]] = None,
    ) -> list[Zonotope]:
        """
        Args:
            x: Zonotope representing state-action inputs [batch, x_dim + a_dim]
            z: Zonotope representing epistemic indices [batch, z_dim]
            error_scales: Optional dict for sensitivity analysis
            split_idxs: Optional dict mapping layer names to lists of neuron indices to split.

        Returns:
            A list of Zonotopes representing the union of all verification branches.
        """
        scales = error_scales or {}
        sm_split = split_idxs or {}

        # --- Base Path ---
        # 1. Linear
        phi_pre = self.base_fc1(x)

        # 2. ReLU (Now returns a LIST of branches)
        phi_branches = self.act_base(
            phi_pre,
            error_scale=scales.get("base"),
            layer_name="base_fc1",
            split_idxs=sm_split.get("base_fc1", []), # type: ignore
        )

        # We must process every branch generated by the Base network
        final_outputs = []
        for phi in phi_branches:
            base_out = self.base_out(phi)

            # Construct Joint Input for this specific branch
            # Note: .concatenate() automatically pads 'x' and 'z' to match
            # the history of 'phi' (which might differ per branch due to splits)
            full_phi = phi.concatenate([x], axis=-1)
            joint_input = full_phi.concatenate([z], axis=-1)

            # --- Epistemic Path ---
            epi_h_pre = self.epi_fc1(joint_input)
            epi_branches = self.act_epi(
                epi_h_pre,
                error_scale=scales.get("epi"),
                layer_name="epi_fc1",
                split_idxs=sm_split.get("epi_fc1", []), # type: ignore
            )
            # Transform all epi branches
            epi_outs = [self.epi_out(h) for h in epi_branches]

            # --- Prior Path ---
            prior_h_pre = self.prior_fc1(joint_input)
            prior_branches = self.act_prior(
                prior_h_pre,
                error_scale=scales.get("prior"),
                layer_name="prior_fc1",
                split_idxs=sm_split.get("prior_fc1", []), # type: ignore
            )
            # Transform all prior branches
            prior_outs = [self.prior_out(h) for h in prior_branches]

            # --- Final Combination ---
            # Cartesian Product: Base + Epi_i + Prior_j
            # Since splits in Epi and Prior are independent, and they add generators starting 
            # from the SAME index (end of joint_input), we must ALIGN them to valid overlap.
            # We want Final Generators: [Shared, Epi_Unique, Prior_Unique]
            n_shared = joint_input.generators.shape[1]

            for e_out in epi_outs:
                for p_out in prior_outs:
                    # e_out: [Shared, Epi_Unique]
                    # p_out: [Shared, Prior_Unique]
                    # We need to construct:
                    # E_aligned: [Shared, Epi_Unique, 0]
                    # P_aligned: [Shared, 0, Prior_Unique]
                    
                    # 1. Align E
                    e_gens = e_out.generators
                    Ke = e_gens.shape[1] - n_shared
                    
                    # 2. Align P
                    p_gens = p_out.generators
                    Kp = p_gens.shape[1] - n_shared
                    
                    # P padding for E [Batch, Kp, Feat]
                    pad_e = jnp.zeros((e_gens.shape[0], Kp, e_gens.shape[2]))
                    e_gens_aligned = jnp.concatenate([e_gens, pad_e], axis=1)
                    
                    # E padding for P [Batch, Ke, Feat]
                    # We must insert it in the middle -> [Shared, Pad, Unique]
                    p_shared = p_gens[:, :n_shared, :]
                    p_unique = p_gens[:, n_shared:, :]
                    pad_p = jnp.zeros((p_gens.shape[0], Ke, p_gens.shape[2]))
                    p_gens_aligned = jnp.concatenate([p_shared, pad_p, p_unique], axis=1)
                    
                    # 3. Update History for P (Shift unique parts by Ke)
                    p_hist_aligned = []
                    for g in p_out.history:
                        if g.start_idx >= n_shared:
                            new_g = GeneratorGroup(g.source_type, g.start_idx + Ke, g.count, g.metadata)
                            p_hist_aligned.append(new_g)
                        else:
                            p_hist_aligned.append(g)
                            
                    # Construct Aligned Zonotopes
                    # Note: We don't strictly need to update e_out history for the sum to work, 
                    # but for completeness we keep e_out history as is.
                    e_zono_aligned = Zonotope(e_out.center, e_gens_aligned, e_out.history)
                    p_zono_aligned = Zonotope(p_out.center, p_gens_aligned, tuple(p_hist_aligned))

                    # Combine: Base (small) + E_aligned (medium) -> returns history of E
                    # Then + P_aligned -> returns history of P??
                    # We explicitly want the union of histories.
                    
                    temp_sum = base_out.add(e_zono_aligned)
                    
                    final_gens = temp_sum.generators + p_zono_aligned.generators
                    final_center = temp_sum.center + p_zono_aligned.center
                    
                    # Construct final history manually to ensure Union
                    final_history = e_out.history + tuple(
                        g for g in p_hist_aligned if g.start_idx >= n_shared
                    )
                    
                    final_out = Zonotope(final_center, final_gens, final_history)
                    final_outputs.append(final_out)

        return final_outputs


def rollout_trajectory(
    model: AbstractENN, x_init: Zonotope, z_sample: Zonotope, steps: int
) -> List[Zonotope]:
    # 1. SETUP INDEPENDENCE (Time t=0)
    x_curr, z_fixed = merge_independent_sources(x_init, z_sample)

    trajectory = [x_curr]

    for t in range(steps):
        # 2. STEP (Time t > 0)
        # Model returns a list of branches. 
        # For simple rollout, assuming no splitting? 
        # Or do we handle branching? 
        # Current logic implies SINGLE output or we just take the first?
        # WARNING: This assumes NO splitting for rollout!
        branches = model(x_curr, z_fixed)
        if len(branches) > 1:
            # We must handle branching. For now, this simple rollout might fail or need expansion.
            # But usually rollout is used with 0 splits.
            pass
            
        x_next = branches[0] 
        x_curr = x_next
        trajectory.append(x_curr)

    return trajectory


def box_to_zonotope(
    min_vals: Float[Array, "dim"],
    max_vals: Float[Array, "dim"],
    source_type: str = "input",
    meta: Optional[Dict[str, Any]] = None,
) -> Zonotope:
    """Creates a diagonal (axis-aligned) Zonotope from a hyper-rectangle."""
    center_val = (max_vals + min_vals) / 2.0
    radii_val = (max_vals - min_vals) / 2.0

    # Generators: Diagonal matrix [Dim, Dim] -> Expanded to [1, Dim, Dim]
    generators = jnp.expand_dims(jnp.diag(radii_val), axis=0)
    center = jnp.expand_dims(center_val, axis=0)

    # Initial history
    n_dims = len(min_vals)
    history = (
        GeneratorGroup(
            source_type=source_type, start_idx=0, count=n_dims, metadata=meta or {}
        ),
    )

    return Zonotope(center, generators, history)
