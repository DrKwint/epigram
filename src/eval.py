from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx


# -------------------------
# Small numpy helpers
# -------------------------


def _pearsonr_np(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).mean()) * np.sqrt((y * y).mean()) + 1e-12
    return float((x * y).mean() / denom)


def _auroc_np(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    AUROC from scratch (no sklearn). y_true in {0,1}.
    Returns 0.5 if degenerate.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Rank-based AUROC (Mann–Whitney U)
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    # Handle ties: average ranks for equal scores
    # (simple tie handling)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            avg = ranks[order[i:j]].mean()
            ranks[order[i:j]] = avg
        i = j

    sum_pos_ranks = ranks[pos].sum()
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _ece_np(errors: np.ndarray, sigmas: np.ndarray, n_bins: int = 10) -> float:
    """
    Very simple calibration proxy:
      - bucket by sigma quantiles
      - compare mean(error) vs mean(sigma) per bin
      - weighted average absolute gap
    """
    e = np.asarray(errors, dtype=np.float64)
    s = np.asarray(sigmas, dtype=np.float64)
    if e.size == 0:
        return 0.0

    qs = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    # make bins robust to duplicates
    qs[0] -= 1e-12
    qs[-1] += 1e-12

    ece = 0.0
    for b in range(n_bins):
        mask = (s >= qs[b]) & (s < qs[b + 1])
        if not np.any(mask):
            continue
        w = mask.mean()
        gap = abs(e[mask].mean() - s[mask].mean())
        ece += w * gap
    return float(ece)


# -------------------------
# Jitted model-side utilities
# -------------------------


@nnx.jit
def _ensemble_stats(
    model, x: jax.Array, keys: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """
    keys: (S, 2) PRNGKeys
    Returns:
      mean_pred: (B, D)
      sigma:     (B,) scalar uncertainty per sample = mean over dims of std
    """

    def single_pass(k):
        z = jax.random.normal(k, (x.shape[0], model.z_dim))
        return model(x, z)  # (B, D) denormalized output

    preds = jax.vmap(single_pass)(keys)  # (S, B, D)
    mean_pred = jnp.mean(preds, axis=0)  # (B, D)
    std_pred = jnp.std(preds, axis=0)  # (B, D)
    sigma = jnp.mean(std_pred, axis=-1)  # (B,)
    return mean_pred, sigma


@nnx.jit
def _epinet_additive_stats(model, x: jax.Array, keys: jax.Array):
    """
    For Osband-style ENN:
      pred(x,z) = mu(x) + prior_scale * (epi(phi_sg,z) + prior(phi_sg,z))

    We diagnose:
      - Var_z of total epistemic term (epi+prior)
      - Var_z of epi-only term (learned uncertainty)
      - Var_z of prior-only term (frozen randomness)
      - norms / ratios to detect 'epi collapsed to ~0' or 'epi << prior'
    """
    B = x.shape[0]

    # IMPORTANT: base() is deterministic; compute phi/mu once.
    phi, mu = model.base(x)  # phi: (B,F), mu: (B,D)
    phi_sg = jax.lax.stop_gradient(phi)

    def per_key(k):
        z = jax.random.normal(k, (B, model.z_dim))
        epi = model.epinet(phi_sg, z)  # (B,D)
        prior = model.prior(phi_sg, z)  # (B,D)

        epi_term = model.prior_scale * epi
        prior_term = model.prior_scale * prior
        total_term = epi_term + prior_term  # (B,D)

        return epi_term, prior_term, total_term

    epi_terms, prior_terms, total_terms = jax.vmap(per_key)(keys)  # (S,B,D)

    # Variance across z samples (mean over batch+dim)
    var_total = jnp.mean(jnp.var(total_terms, axis=0))
    var_epi = jnp.mean(jnp.var(epi_terms, axis=0))
    var_prior = jnp.mean(jnp.var(prior_terms, axis=0))

    # Mean norms (averaged over samples and batch)
    norm_epi = jnp.mean(jnp.linalg.norm(epi_terms, axis=-1))
    norm_prior = jnp.mean(jnp.linalg.norm(prior_terms, axis=-1))
    norm_total = jnp.mean(jnp.linalg.norm(total_terms, axis=-1))
    norm_mu = jnp.mean(jnp.linalg.norm(mu, axis=-1))

    # Useful ratios
    epi_over_prior = norm_epi / (norm_prior + 1e-8)
    total_over_mu = norm_total / (norm_mu + 1e-8)

    return {
        "var_total": var_total,
        "var_epi": var_epi,
        "var_prior": var_prior,
        "norm_epi": norm_epi,
        "norm_prior": norm_prior,
        "norm_total": norm_total,
        "norm_mu": norm_mu,
        "epi_over_prior": epi_over_prior,
        "total_over_mu": total_over_mu,
    }


# -------------------------
# Public API
# -------------------------


def compute_val_diagnostics(
    model,
    val_ds,
    rngs: nnx.Rngs,
    *,
    id_batch: int = 1024,
    ood_batch: int = 1024,
    n_samples: int = 32,
    ood_expand: float = 0.5,
    ece_bins: int = 10,
) -> Dict[str, float]:
    """
    Returns dict[str,float] diagnostics.

    ID batch uses real transitions from val_ds.
    OOD batch samples x uniformly in an expanded bounding box around ID x.

    Requires val_ds.sample_transitions(batch_size) (which you already have).
    """
    # ---- 1) Sample ID transitions ----
    s, a, s_next, done = val_ds.sample_transitions(id_batch)
    x_id = np.concatenate([s, a], axis=-1).astype(np.float32)
    y_id = s_next.astype(np.float32)

    # ---- 2) Sample OOD x by expanding bounds around ID ----
    bmin = x_id.min(axis=0)
    bmax = x_id.max(axis=0)
    expand = (bmax - bmin) * ood_expand
    x_ood = np.random.uniform(
        low=bmin - expand, high=bmax + expand, size=(ood_batch, x_id.shape[1])
    ).astype(np.float32)

    # ---- 3) Make keys OUTSIDE jit (shape must be static) ----
    k_id = rngs.epistemic()
    k_ood = rngs.epistemic()
    keys_id = jax.random.split(k_id, n_samples)
    keys_ood = jax.random.split(k_ood, n_samples)

    # ---- 4) Ensemble uncertainty on ID + OOD ----
    mean_id, sigma_id = _ensemble_stats(model, jnp.asarray(x_id), keys_id)
    _, sigma_ood = _ensemble_stats(model, jnp.asarray(x_ood), keys_ood)

    mean_id_np = np.asarray(mean_id)
    sigma_id_np = np.asarray(sigma_id)
    sigma_ood_np = np.asarray(sigma_ood)

    # ID error (L2)
    err_id = np.linalg.norm(mean_id_np - y_id, axis=-1)

    # AUROC: can sigma separate ID vs OOD?
    y_true = np.concatenate([np.zeros_like(sigma_id_np), np.ones_like(sigma_ood_np)])
    y_score = np.concatenate([sigma_id_np, sigma_ood_np])
    auroc = _auroc_np(y_true, y_score)

    # Calibration proxies
    corr = _pearsonr_np(err_id, sigma_id_np)
    ece = _ece_np(err_id, sigma_id_np, n_bins=ece_bins)

    # ---- 5) Collapse-style additive-term stats (ID and OOD) ----
    add_id = _epinet_additive_stats(model, jnp.asarray(x_id), keys_id)
    add_ood = _epinet_additive_stats(model, jnp.asarray(x_ood), keys_ood)

    add_id_np = {k: float(np.asarray(v)) for k, v in add_id.items()}
    add_ood_np = {k: float(np.asarray(v)) for k, v in add_ood.items()}

    # ---- 6) Summarize ----
    eps = 1e-12
    mean_sigma_id = float(sigma_id_np.mean())
    mean_sigma_ood = float(sigma_ood_np.mean())
    sigma_ratio = mean_sigma_ood / (mean_sigma_id + eps)  # collapse warning if ~1.0

    metrics: Dict[str, float] = {
        # ID/OOD uncertainty
        "val/mean_sigma_id": mean_sigma_id,
        "val/mean_sigma_ood": mean_sigma_ood,
        "val/sigma_ood_over_id": sigma_ratio,
        "val/auroc_ood": float(auroc),
        # calibration
        "val/calib_corr_sigma_vs_error": float(corr),
        "val/calib_ece_sigma_vs_error": float(ece),
        "val/mean_id_error_l2": float(err_id.mean()),
        # collapse diagnostics (ID)
        "var_total_id": add_id_np["var_total"],
        "var_epi_id": add_id_np["var_epi"],
        "var_prior_id": add_id_np["var_prior"],
        "norm_epi_id": add_id_np["norm_epi"],
        "norm_prior_id": add_id_np["norm_prior"],
        "norm_total_id": add_id_np["norm_total"],
        "norm_mu_id": add_id_np["norm_mu"],
        "epi_over_prior_id": add_id_np["epi_over_prior"],
        "total_over_mu_id": add_id_np["total_over_mu"],
        # collapse diagnostics (OOD)
        "var_total_ood": add_ood_np["var_total"],
        "var_epi_ood": add_ood_np["var_epi"],
        "var_prior_ood": add_ood_np["var_prior"],
        "norm_epi_ood": add_ood_np["norm_epi"],
        "norm_prior_ood": add_ood_np["norm_prior"],
        "norm_total_ood": add_ood_np["norm_total"],
        "norm_mu_ood": add_ood_np["norm_mu"],
        "epi_over_prior_ood": add_ood_np["epi_over_prior"],
        "total_over_mu_ood": add_ood_np["total_over_mu"],
    }

    return metrics
