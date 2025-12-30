from __future__ import annotations

import functools
from typing import Tuple, Iterator

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from tqdm import tqdm

from src.affine import Affine
from src.star import Star


class ENN(nnx.Module):
    def __init__(
        self, x_dim: int, a_dim: int, z_dim: int, hidden_dim: int, *, rngs: nnx.Rngs
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
        self.z_dim = z_dim

        self.base_fc1 = nnx.Linear(x_dim + a_dim, hidden_dim, rngs=rngs)
        self.base_out = nnx.Linear(hidden_dim, x_dim, rngs=rngs)

        in_dim = hidden_dim + x_dim + a_dim + z_dim
        self.epi_fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.epi_out = nnx.Linear(hidden_dim, x_dim, rngs=rngs)
        self.prior_fc1 = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.prior_out = nnx.Linear(hidden_dim, x_dim, rngs=rngs)

    def __call__(self, x: Float[Array, "b xa"], z: Float[Array, "b z"]) -> Float[Array, "b x"]:
        phi = jax.nn.relu(self.base_fc1(x))

        base_out = self.base_out(phi)

        full_phi = jax.lax.stop_gradient(jnp.concatenate([phi, x], axis=-1))
        
        # Concatenate inputs for epi/prior: [phi, x, a, z]
        # x here contains both state and action [x, a]
        epi_in = jnp.concatenate([full_phi, z], axis=-1)
        epi_hidden = jax.nn.relu(self.epi_fc1(epi_in))
        epi_out = self.epi_out(epi_hidden)
        
        prior_in = jnp.concatenate([full_phi, z], axis=-1)
        prior_hidden = jax.nn.relu(self.prior_fc1(prior_in))
        prior_out = self.prior_out(prior_hidden)

        return base_out + epi_out + prior_out

    def components(
        self, x: jax.Array, z: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        phi = jax.nn.relu(self.base_fc1(x))
        base_out = self.base_out(phi)

        full_phi = jax.lax.stop_gradient(jnp.concatenate([phi, x], axis=-1))
        epi_hidden = jax.nn.relu(self.epi_fc1(jnp.concatenate([full_phi, z], axis=-1)))
        epi_out = self.epi_out(epi_hidden)
        prior_hidden = jax.nn.relu(
            self.prior_fc1(jnp.concatenate([full_phi, z], axis=-1))
        )
        prior_out = self.prior_out(prior_hidden)

        return base_out, epi_out, prior_out

    # --- Verification Helpers ---
    @property
    def prior_scale(self) -> float:
        return 1.0 # Default scale if not optimized

    def base(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        phi = jax.nn.relu(self.base_fc1(x))
        base_out = self.base_out(phi)
        # Hack: Pass [phi, x] as 'phi' for downstream use
        full_phi = jnp.concatenate([phi, x], axis=-1)
        return full_phi, base_out

    def epinet(self, full_phi: jax.Array, z: jax.Array) -> jax.Array:
        epi_hidden = jax.nn.relu(self.epi_fc1(jnp.concatenate([full_phi, z], axis=-1)))
        return self.epi_out(epi_hidden)

    def prior(self, full_phi: jax.Array, z: jax.Array) -> jax.Array:
        prior_hidden = jax.nn.relu(self.prior_fc1(jnp.concatenate([full_phi, z], axis=-1)))
        return self.prior_out(prior_hidden)

    def propagate_star_set(self, star: Star) -> Iterator[Star]:
        """
        Propagates a Star set through the ENN.
        Yields Star sets representing the output reachable set pieces.
        """
        # star represents the joint input set [x, a, z]
        xa_dim = self.x_dim + self.a_dim

        # 1. Project to [x, a] for the base network
        # [x, a] corresponds to the first (x_dim + a_dim) outputs of star.transform (assuming identity/slice)
        # Actually star.transform maps input_poly -> y. y = [x,a,z] presumably.
        aff_xa = Affine(star.transform.A[:xa_dim], star.transform.b[:xa_dim])

        # 2. Compute phi stars
        # phi = relu(base_fc1([x, a]))
        aff_base_fc1 = Affine(self.base_fc1.kernel.value.T, self.base_fc1.bias.value)
        star_xa = Star(star.input_set, aff_xa)
        star_phi_pre = star_xa.map_affine(aff_base_fc1)
        stars_phi, _ = star_phi_pre.map_relu()

        # 3. Compute final output stars
        # output = base_out(phi) + epi_out(phi, x, a, z) + prior_out(phi, x, a, z)
        aff_base_out = Affine(self.base_out.kernel.value.T, self.base_out.bias.value)
        aff_epi_fc1 = Affine(self.epi_fc1.kernel.value.T, self.epi_fc1.bias.value)
        aff_epi_out = Affine(self.epi_out.kernel.value.T, self.epi_out.bias.value)
        aff_prior_fc1 = Affine(self.prior_fc1.kernel.value.T, self.prior_fc1.bias.value)
        aff_prior_out = Affine(self.prior_out.kernel.value.T, self.prior_out.bias.value)

        for s_phi in tqdm(stars_phi, desc="Propagating Stars"):
            # s_phi.transform is T_phi
            T_phi = s_phi.transform

            # Construct input for epi/prior: [phi, x, a, z]
            # star.transform is [x, a, z]
            A_input = jnp.concatenate([T_phi.A, star.transform.A], axis=0)
            b_input = jnp.concatenate([T_phi.b, star.transform.b], axis=0)
            T_input = Affine(A_input, b_input)

            # Precompute base term
            term_base = aff_base_out.map(T_phi)

            # Epi branch
            # epi_hidden = relu(epi_fc1([phi, x, a, z]))
            T_pre_epi = aff_epi_fc1.map(T_input)
            stars_epi, _ = Star(s_phi.input_set, T_pre_epi).map_relu()

            # Precompute prior input transform (same for all epi branches)
            T_pre_prior = aff_prior_fc1.map(T_input)

            for s_epi in stars_epi:
                T_epi_h = s_epi.transform
                term_epi = aff_epi_out.map(T_epi_h)

                # Prior branch
                # prior_hidden = relu(prior_fc1([phi, x, a, z]))
                # Use s_epi.input_set to maintain constraints
                stars_prior, _ = Star(s_epi.input_set, T_pre_prior).map_relu()

                for s_prior in stars_prior:
                    T_prior_h = s_prior.transform
                    term_prior = aff_prior_out.map(T_prior_h)

                    # Combine
                    A_final = term_base.A + term_epi.A + term_prior.A
                    b_final = term_base.b + term_epi.b + term_prior.b

                    yield Star(s_prior.input_set, Affine(A_final, b_final))


def enn_bootstrap_mse(
    model: ENN,
    x: jax.Array,  # (B, in_dim)
    y: jax.Array,  # (B, out_dim)
    *,
    key: jax.Array,
    num_heads: int,
    bootstrap_p: float = 0.8,
) -> jax.Array:
    """
    Sample num_heads head indices, give each a bootstrap mask over the batch,
    and minimize masked MSE averaged over heads.
    """
    B = x.shape[0]
    key_h, key_m = jax.random.split(key)

    # head_idx = jax.random.randint(key_h, (num_heads,), 0, model.z_dim)  # (H,)
    # zH = one_hot_z(head_idx, model.z_dim)  # (H, Z)
    zH = jax.random.normal(key_h, (num_heads, model.z_dim))  # (H, Z)

    # Bootstrap mask: (H, B) Bernoulli(p)
    mask = jax.random.bernoulli(key_m, p=bootstrap_p, shape=(num_heads, B)).astype(
        jnp.float32
    )

    def head_loss(z_vec: jax.Array, w: jax.Array) -> jax.Array:
        # broadcast (Z,) -> (B, Z)
        z = jnp.broadcast_to(z_vec[None, :], (B, model.z_dim))
        pred = model(x, z)  # (B, D)
        se = jnp.mean((pred - y) ** 2, axis=-1)  # (B,)
        # avoid divide-by-zero if mask is all zeros
        denom = jnp.maximum(jnp.sum(w), 1.0)
        return jnp.sum(w * se) / denom

    losses = jax.vmap(head_loss)(zH, mask)
    return jnp.mean(losses)


@functools.partial(nnx.jit, static_argnames=("num_heads",))
def train_step(
    model: ENN,
    optimizer: nnx.Optimizer | None,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    num_heads: int = 8,
    bootstrap_p: float = 0.8,
) -> jax.Array:
    def loss_fn(m: ENN):
        return enn_bootstrap_mse(
            m, x, y, key=key, num_heads=num_heads, bootstrap_p=bootstrap_p
        )

    if optimizer is not None:
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
    else:
        loss = loss_fn(model)
    return loss
