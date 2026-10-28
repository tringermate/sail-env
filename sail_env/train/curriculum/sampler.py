"""Which starting situation should the next episode use? Uniform, SFL or SGS sampling.

A fixed set of N task configurations (starting situations) is drawn before training. Each one keeps
a circular buffer of its last H boolean outcomes; its success rate p is the buffer mean (an empty
buffer counts as p = 0, so all methods start uniform). The methods only differ in the weights:

    uniform   P(i) = 1 / N
    SFL       P(i) ~ p_i (1 - p_i) + eps                      (Sampling For Learnability, Rutherford et al. 2024)
    SGS       w_i = (p_i + eps)^(kappa t) (1 - p_i + eps)^(kappa (1 - t))
              l_i = log(max(w_i, eps) + eps)
              P(i) = softmax_i(l_i / max(T, 1))               (Success-Guided Sampling, Zhang et al. 2026, Eq. 1-2)

SGS favours configurations the policy succeeds on sometimes but not always: with t = 0.5 the weight
peaks at p = 0.5; kappa sharpens the peak; the temperature T flattens the distribution.

Pure JAX; the sampler state is part of the training state. API (as in the brief):
    init(N, H) -> state;  update(state, config_ids, successes, mask) -> state
    probs(state, cfg) -> (N,);  sample(probs, key, n) -> ids
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import struct


@dataclass(frozen=True)
class SamplerConfig:
    method: str = "sgs"  # "uniform", "sfl" or "sgs"
    num_configs: int = 4096
    history: int = 100  # H: outcomes remembered per configuration
    target: float = 0.5  # SGS t
    kappa: float = 1.0  # SGS concentration
    temperature: float = 2.0  # SGS T
    eps: float = 1e-4
    num_levels: int = 1  # > 1: joint curriculum over (situation, rule level); num_configs = situations x levels


@struct.dataclass
class SamplerState:
    outcomes: jax.Array  # (N, H) float32, 1 = success
    count: jax.Array  # (N,) int32, filled entries (<= H)
    ptr: jax.Array  # (N,) int32, next write position in the circular buffer


def init(n: int, h: int) -> SamplerState:
    return SamplerState(
        outcomes=jnp.zeros((n, h), jnp.float32), count=jnp.zeros(n, jnp.int32), ptr=jnp.zeros(n, jnp.int32)
    )


def success_rate(state: SamplerState) -> jax.Array:
    """p_i = mean of the buffer; 0 for an empty buffer."""
    return jnp.where(state.count > 0, state.outcomes.sum(axis=1) / jnp.maximum(state.count, 1), 0.0)


def update(state: SamplerState, config_ids, successes, mask) -> SamplerState:
    """Push the outcome of every finished episode (mask) into its configuration's buffer.

    Vectorised: episodes are sorted by configuration, each gets its rank within its configuration's
    group, and is written at slot (ptr + rank) mod H, so several episodes of the same configuration in
    one batch all count (if one configuration finishes more than H times in a batch, some overwrite).
    """
    n, h = state.outcomes.shape
    ids = jnp.where(mask.ravel(), config_ids.ravel(), n)  # masked entries -> out-of-range id n
    succ = successes.ravel().astype(jnp.float32)
    order = jnp.argsort(ids, stable=True)
    sorted_ids = ids[order]
    first = jnp.searchsorted(sorted_ids, sorted_ids, side="left")  # start of each id's group
    rank = jnp.zeros_like(ids).at[order].set(jnp.arange(ids.shape[0]) - first)
    pos = (state.ptr[jnp.minimum(ids, n - 1)] + rank) % h
    outcomes = state.outcomes.at[ids, pos].set(succ, mode="drop")  # id n is dropped
    finished = jnp.bincount(ids, length=n + 1)[:n]
    return SamplerState(
        outcomes=outcomes,
        count=jnp.minimum(state.count + finished, h),
        ptr=(state.ptr + finished) % h,
    )


def probs(state: SamplerState, cfg: SamplerConfig) -> jax.Array:
    p = success_rate(state)
    n = p.shape[0]
    if cfg.method == "uniform":
        return jnp.full(n, 1.0 / n)
    if cfg.method == "sfl":
        w = p * (1.0 - p) + cfg.eps
        return w / w.sum()
    if cfg.method == "sgs":
        e, k, t = cfg.eps, cfg.kappa, cfg.target
        w = (p + e) ** (k * t) * (1.0 - p + e) ** (k * (1.0 - t))
        logits = jnp.log(jnp.maximum(w, e) + e)
        return jax.nn.softmax(logits / max(cfg.temperature, 1.0))
    raise ValueError(f"unknown sampling method {cfg.method!r}")


def sample(cdf: jax.Array, key: jax.Array, shape=()) -> jax.Array:
    """Draw configuration ids from a precomputed CDF (cumsum of probs): O(log N) per draw."""
    u = jax.random.uniform(key, shape) * cdf[-1]
    return jnp.clip(jnp.searchsorted(cdf, u, side="right"), 0, cdf.shape[0] - 1)
