from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
import tree

from affine import Affine
from polytope import Polytope
from star import Star


# -----------------------------
# Utilities
# -----------------------------


def one_hot_z(head_idx: jax.Array, z_dim: int) -> jax.Array:
    """head_idx: (H,) or (B,) -> one-hot (..., z_dim)"""
    return jax.nn.one_hot(head_idx, z_dim, dtype=jnp.float32)


# -----------------------------
# Base network: mu(x) + features phi(x)
# -----------------------------


class BaseNet(nnx.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, feat_dim, rngs=rngs)
        self.fc2 = nnx.Linear(feat_dim, feat_dim, rngs=rngs)
        self.out = nnx.Linear(feat_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns (phi, mu)."""
        h = jax.nn.relu(self.fc1(x))
        phi = jax.nn.relu(self.fc2(h))
        mu = self.out(phi)
        return phi, mu
    
    def propagate_star_set(self, star: Star) -> tuple[list[Star], list[Star]]:
        stars = star.map_affine(Affine(self.fc1.kernel.value, self.fc1.bias.value)).map_relu()
        stars = tree.flatten([s.map_affine(Affine(self.fc2.kernel.value, self.fc2.bias.value)).map_relu() for s in stars])
        phi_stars = tree.flatten([s.map_affine(Affine(self.out.kernel.value, self.fc2.bias.value)).map_relu() for s in stars])
        mu_stars = [s.map_affine(Affine(self.out.kernel.value, self.out.bias.value)) for s in phi_stars]
        return phi_stars, mu_stars

class EpiNet(nnx.Module):
    def __init__(self, in_dim: int, feat_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.fc1 = nnx.Linear(in_dim, feat_dim, rngs=rngs)
        self.out = nnx.Linear(feat_dim, out_dim, rngs=rngs)

    def __call__(self, phi: jax.Array, z: jax.Array) -> jax.Array:
        h = jax.nn.relu(self.fc1(jnp.concatenate([phi, z], axis=-1)))
        return self.out(h)

    def propagate_star_set(self, phi_star: Star, z_star: Star) -> list[Star]:
        input_set_A = jnp.concatenate([phi_star.input_set.A, z_star.input_set.A], axis=-1)
        input_set_b = jnp.concatenate([phi_star.input_set.b, z_star.input_set.b], axis=-1)
        input_set = Polytope(input_set_A, input_set_b)
        transform_A = jnp.concatenate([phi_star.transform.A, z_star.transform.A], axis=-1)
        transform_b = jnp.concatenate([phi_star.transform.b, z_star.transform.b], axis=-1)
        transform = Affine(transform_A, transform_b)
        star = Star(input_set, transform)

        stars = star.map_affine(Affine(self.fc1.kernel.value, self.fc1.bias.value)).map_relu()
        return [s.map_affine(Affine(self.out.kernel.value, self.out.bias.value)) for s in stars]

# -----------------------------
# Epinet: linear-in-z head  A(phi) z + b(phi)
# -----------------------------


class LinearInZEpinet(nnx.Module):
    """
    Given features phi (B, F) and z (B, Z), output epistemic term e (B, D):
        e(phi, z) = A(phi) @ z + b(phi)
    where A(phi) is (B, D, Z) and b(phi) is (B, D).
    """

    def __init__(
        self,
        feat_dim: int,
        z_dim: int,
        out_dim: int,
        hidden: int,
        layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.z_dim = z_dim
        self.out_dim = out_dim

        mods = []
        in_d = feat_dim
        for _ in range(layers):
            mods.append(nnx.Linear(in_d, hidden, rngs=rngs))
            mods.append(jax.nn.relu)
            in_d = hidden

        # Produce A_flat and b from phi
        self.body = nnx.Sequential(*mods)
        self.A = nnx.Linear(in_d, out_dim * z_dim, rngs=rngs)
        self.b = nnx.Linear(in_d, out_dim, rngs=rngs)

    def __call__(self, phi: jax.Array, z: jax.Array) -> jax.Array:
        h = self.body(phi)
        A_flat = self.A(h)  # (B, D*Z)
        b = self.b(h)  # (B, D)
        A = A_flat.reshape(phi.shape[0], self.out_dim, self.z_dim)  # (B, D, Z)
        # (B, D, Z) @ (B, Z, 1) -> (B, D, 1) -> (B, D)
        return (A @ z[..., None]).squeeze(-1) + b

# -----------------------------
# Frozen randomized prior: same form, but weights are NOT trainable
# -----------------------------


class FixedLinear(nnx.Module):
    """A linear layer with fixed (non-Param) weights so the optimizer won't touch it."""

    def __init__(
        self, in_dim: int, out_dim: int, *, key: jax.Array, scale: float = 1.0
    ):
        kW, kb = jax.random.split(key)
        W = (
            scale
            * jax.random.normal(kW, (in_dim, out_dim), dtype=jnp.float32)
            / jnp.sqrt(in_dim)
        )
        b = scale * jax.random.normal(kb, (out_dim,), dtype=jnp.float32)
        self.W = nnx.Variable(W)  # NOTE: Variable, not Param
        self.b = nnx.Variable(b)

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.W.value + self.b.value

class FixedPrior(nnx.Module):
    def __init__(self, in_dim: int, out_dim: int, *, key: jax.Array):
        self.fc1 = FixedLinear(in_dim, out_dim, key=key)
        self.out = FixedLinear(out_dim, out_dim, key=key)

    def __call__(self, phi: jax.Array, z: jax.Array) -> jax.Array:
        h = jax.nn.relu(self.fc1(jnp.concatenate([phi, z], axis=-1)))
        return self.out(h)

class FixedLinearInZPrior(nnx.Module):
    """
    Frozen prior with the same structure:
        p(phi, z) = A_p(phi) z + b_p(phi)
    using fixed weights (nnx.Variable).
    """

    def __init__(
        self,
        feat_dim: int,
        z_dim: int,
        out_dim: int,
        hidden: int,
        layers: int,
        *,
        key: jax.Array,
    ):
        self.z_dim = z_dim
        self.out_dim = out_dim

        keys = jax.random.split(key, 2 * (layers + 2))
        ki = 0

        self.fcs = nnx.List()
        in_d = feat_dim
        for _ in range(layers):
            self.fcs.append(FixedLinear(in_d, hidden, key=keys[ki], scale=1.0))
            ki += 1
            in_d = hidden

        self.A = FixedLinear(in_d, out_dim * z_dim, key=keys[ki], scale=1.0)
        ki += 1
        self.b = FixedLinear(in_d, out_dim, key=keys[ki], scale=1.0)
        ki += 1

    def __call__(self, phi: jax.Array, z: jax.Array) -> jax.Array:
        h = phi
        for fc in self.fcs:
            h = jax.nn.relu(fc(h))
        A_flat = self.A(h)
        b = self.b(h)
        A = A_flat.reshape(phi.shape[0], self.out_dim, self.z_dim)
        return (A @ z[..., None]).squeeze(-1) + b

# -----------------------------
# Full ENN
# -----------------------------

class ENN(nnx.Module):
    def __init__(self. x_dim) -> None:
        super().__init__()
        self.base = BaseNet()
        self.epinet = EpiNet()
        self.prior = FixedPrior()


class OldENN(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        feat_dim: int,
        z_dim: int,
        epi_hidden: int = 128,
        epi_layers: int = 2,
        prior_hidden: int = 128,
        prior_layers: int = 2,
        prior_scale: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.feat_dim = feat_dim
        self.z_dim = z_dim
        self.prior_scale = prior_scale

        self.base = BaseNet(in_dim, feat_dim, out_dim, rngs=rngs)
        self.epinet = LinearInZEpinet(
            feat_dim, z_dim, out_dim, epi_hidden, epi_layers, rngs=rngs
        )

        # Frozen prior uses a separate key (not from optimizer params)
        prior_key = (
            rngs.params()
        )  # fine to use a params key; weights are nnx.Variable anyway
        self.prior = FixedLinearInZPrior(
            feat_dim, z_dim, out_dim, prior_hidden, prior_layers, key=prior_key
        )

    def __call__(self, x: jax.Array, z: jax.Array) -> jax.Array:
        phi, mu = self.base(x)
        phi_sg = jax.lax.stop_gradient(phi)  # <-- the key stop-grad placement
        epi = self.epinet(phi_sg, z)
        prior = self.prior(phi_sg, z)  # frozen randomized prior
        return mu + self.prior_scale * (epi + prior)

    def propagate_star_set(self, star: Star) -> list[Star]:
        phi_stars, mu_stars = self.base.propagate_star_set(star)
        epi_stars = self.epinet.propagate_star_set(star)


# -----------------------------
# Loss: bootstrap masking across heads (prevents collapse)
# -----------------------------


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

    head_idx = jax.random.randint(key_h, (num_heads,), 0, model.z_dim)  # (H,)
    zH = one_hot_z(head_idx, model.z_dim)  # (H, Z)

    # Bootstrap mask: (H, B) Bernoulli(p)
    mask = jax.random.bernoulli(key_m, p=bootstrap_p, shape=(num_heads, B)).astype(
        jnp.float32
    )

    def head_loss(z_vec, w):
        # broadcast (Z,) -> (B, Z)
        z = jnp.broadcast_to(z_vec[None, :], (B, model.z_dim))
        pred = model(x, z)  # (B, D)
        se = jnp.mean((pred - y) ** 2, axis=-1)  # (B,)
        # avoid divide-by-zero if mask is all zeros
        denom = jnp.maximum(jnp.sum(w), 1.0)
        return jnp.sum(w * se) / denom

    losses = jax.vmap(head_loss)(zH, mask)
    return jnp.mean(losses)


# -----------------------------
# Training step (NNX + Optax)
# -----------------------------


@functools.partial(nnx.jit, static_argnames=("num_heads",))
def train_step(
    model: ENN,
    optimizer: nnx.Optimizer | None,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    num_heads: int = 8,
    bootstrap_p: float = 0.8,
):
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


# -----------------------------
# Evaluation: mean + epistemic std over head samples
# -----------------------------


@nnx.jit
def enn_predict_stats(model: ENN, x: jax.Array, key: jax.Array, n_samples: int = 32):
    head_idx = jax.random.randint(key, (n_samples,), 0, model.z_dim)  # (S,)
    zS = one_hot_z(head_idx, model.z_dim)  # (S, Z)

    def one(z_vec):
        z = jnp.broadcast_to(z_vec[None, :], (x.shape[0], model.z_dim))
        return model(x, z)  # (B, D)

    preds = jax.vmap(one)(zS)  # (S, B, D)
    mean = jnp.mean(preds, axis=0)
    std = jnp.std(preds, axis=0)
    # often you log a scalar sigma per point:
    sigma = jnp.mean(std, axis=-1)  # (B,)
    return mean, sigma
