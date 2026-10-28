"""Proximal Policy Optimisation (PPO), single file, PureJaxRL style.

Reading guide (the order things happen in one training *update*):

  1. ROLLOUT   Run the current policy for `num_steps` steps in `num_envs` parallel environments
               (lax.scan over time, vmap over envs). Store every transition.
  2. GAE       Turn rewards + critic values into advantages A_t ("how much better than expected
               was this action?") and return targets R_t for the critic.
  3. UPDATE    For `update_epochs` passes over the batch, shuffle it into `num_minibatches`
               minibatches and take one gradient step per minibatch on the PPO loss.
  4. METRICS   Return diagnostics (return, entropy, KL, clip fraction, explained variance, ...).

Everything inside one update is pure JAX and compiled with jit. Seeds are batched with vmap, so
one call trains several independent agents at once.

Notation used in the comments:
    s_t, a_t, r_t   state (observation), action, reward at step t
    d_t             1 if the episode ended at step t (the next obs belongs to a new episode)
    pi(a|s)         policy (actor) probability;   V(s)  critic's value estimate
    gamma           discount factor;   lambda  GAE parameter;   eps  clip range
"""

from dataclasses import dataclass
from typing import Any, NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal


# =====================================================================================================
# Configuration
# =====================================================================================================


@dataclass(frozen=True)
class PPOConfig:
    # --- Data collection ---
    num_envs: int = 4  # parallel environments per seed
    num_steps: int = 128  # rollout length per env per update -> batch = num_envs * num_steps
    total_timesteps: int = 500_000  # per seed
    # --- Optimisation ---
    lr: float = 3e-4
    anneal_lr: bool = True  # linear decay of the learning rate to 0 over training
    update_epochs: int = 4  # passes over each batch
    num_minibatches: int = 4
    max_grad_norm: float = 0.5  # gradient-norm clipping
    # --- PPO objective ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    # Optional: decay the entropy bonus linearly to this value over training, so the policy can
    # sharpen at the end instead of staying random (and hedging against its own randomness).
    ent_coef_final: float | None = None
    vf_coef: float = 0.5
    # Optional kickstarting (Schmitt et al. 2018): with a teacher agent (see train(teacher=...)), add
    # kick * cross-entropy(teacher's action, policy) to the loss, so the policy first learns to act like the
    # teacher in the states it visits itself. kick decays linearly from kick_coef to 0 at kick_until
    # (fraction of training); from then on only the PPO objective is left, free to improve on the teacher.
    kick_coef: float = 0.0
    kick_until: float = 0.3
    # Optional opponent pool (a small league; fleet envs only). In a share opp_frac of the envs only boats
    # 0..opp_learners-1 are sailed by the learning policy; the other boats by an opponent drawn per episode
    # from opp_snapshots frozen copies of the policy (refreshed every opp_snapshot_every updates, the newest
    # replacing the oldest) and the fixed agents passed as train(opponents=[...]), e.g. the digital racer.
    # Only learner boats enter the loss. The other envs stay plain self-play. 0 = off.
    opp_frac: float = 0.0
    opp_learners: int = 2
    opp_snapshots: int = 4
    opp_snapshot_every: int = 100
    # Mixed pool fleets: draw the opponent separately for every pool boat (e.g. a digital racer next to an
    # earlier self) instead of one opponent for all pool boats of an env.
    opp_mix: bool = False
    # --- DECISION POINTS (see the walkthrough) ---
    normalize_advantages: bool = True  # standardise A within each minibatch
    clip_value_loss: bool = True  # PPO-style clipping on the critic too (PureJaxRL default)
    hidden: int = 64  # units per hidden layer (64 or 128)
    # --- Data collection detail ---
    # Before training, each env sails a random number of warm-up steps in [0, stagger_steps) with
    # random actions, so fixed-length episodes don't all end on the same step. 0 = off.
    stagger_steps: int = 0
    # --- Bookkeeping ---
    num_seeds: int = 5
    updates_per_chunk: int = 10  # updates compiled together between two logging points

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_chunks(self) -> int:
        updates = max(self.total_timesteps // self.batch_size, 1)
        return -(-updates // self.updates_per_chunk)  # ceil

    @property
    def num_updates(self) -> int:
        return self.num_chunks * self.updates_per_chunk  # rounded up to whole chunks


# =====================================================================================================
# Networks: separate actor and critic MLPs (2 hidden layers, tanh, orthogonal init)
# =====================================================================================================
# Orthogonal init with gain sqrt(2) on hidden layers keeps activations well scaled. The actor's
# output layer uses a tiny gain (0.01) so the initial policy is close to uniform (maximum entropy,
# maximum exploration); the critic's output uses gain 1.


class Actor(nn.Module):
    num_actions: int
    hidden: int = 64

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(2):
            x = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)  # logits


class Critic(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(2):
            x = nn.tanh(nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        return jnp.squeeze(nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x), -1)  # V(s)


# =====================================================================================================
# Categorical distribution, by hand
# =====================================================================================================
# The actor outputs logits z. pi(a|s) = softmax(z)_a, so log pi(a|s) = log_softmax(z)_a.


def categorical_sample(key, logits):
    """Sample a ~ softmax(logits). (jax.random.categorical uses the Gumbel-max trick.)"""
    return jax.random.categorical(key, logits)


def categorical_log_prob(logits, actions):
    """log pi(a|s) for the taken actions."""
    log_p = jax.nn.log_softmax(logits)
    return jnp.take_along_axis(log_p, actions[..., None], axis=-1)[..., 0]


def categorical_entropy(logits):
    """H = -sum_a pi(a) log pi(a). Maximal (= log n_actions) for a uniform policy."""
    log_p = jax.nn.log_softmax(logits)
    return -jnp.sum(jnp.exp(log_p) * log_p, axis=-1)


# =====================================================================================================
# Containers
# =====================================================================================================


class Transition(NamedTuple):
    obs: jax.Array  # s_t
    action: jax.Array  # a_t
    log_prob: jax.Array  # log pi_old(a_t|s_t), from the policy that collected the data
    value: jax.Array  # V_old(s_t)
    reward: jax.Array  # r_t
    done: jax.Array  # d_t
    info: Any
    teacher_action: Any = None  # the teacher's action in s_t (kickstarting), None without a teacher
    learner: Any = None  # bool: this boat is sailed by the learning policy (opponent pool), None = all are


class RunnerState(NamedTuple):
    params: Any  # {"actor": ..., "critic": ...}
    opt_state: Any
    env_state: Any
    last_obs: jax.Array
    key: jax.Array
    update_idx: jax.Array
    sampler_state: Any = None  # curriculum sampler (which starting situations to train on), if any
    league: Any = None  # opponent pool: {"snaps": frozen actor params (K, ...), "choice": opponent per env}


# =====================================================================================================
# GAE: Generalised Advantage Estimation (Schulman et al. 2016)
# =====================================================================================================


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Advantages A_t and return targets R_t = A_t + V(s_t). Arrays are (T, num_envs).

    One-step TD error:      delta_t = r_t + gamma * V(s_{t+1}) * (1 - d_t) - V(s_t)
    GAE (backwards in t):   A_t     = delta_t + gamma * lambda * (1 - d_t) * A_{t+1}

    (1 - d_t) cuts the sum at episode boundaries: with auto-reset, s_{t+1} after a done belongs to a
    new episode, so its value must not leak into this one. lambda trades bias (lambda = 0: pure TD)
    against variance (lambda = 1: Monte-Carlo returns).
    """

    def backward(carry, x):
        next_advantage, next_value = carry
        reward, value, done = x
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        advantage = delta + gamma * lam * not_done * next_advantage
        return (advantage, value), advantage

    _, advantages = jax.lax.scan(
        backward,
        (jnp.zeros_like(last_value), last_value),
        (rewards, values, dones.astype(jnp.float32)),
        reverse=True,  # walk from t = T-1 back to t = 0
    )
    return advantages, advantages + values


# =====================================================================================================
# PPO loss for one minibatch
# =====================================================================================================


def ppo_loss(params, actor, critic, batch: Transition, advantages, targets, cfg: PPOConfig, ent_coef, kick=0.0):
    logits = actor.apply(params["actor"], batch.obs)
    value = critic.apply(params["critic"], batch.obs)
    log_prob = categorical_log_prob(logits, batch.action)
    # With an opponent pool only the learner boats count: every mean below is over them. (Without one,
    # static branch: plain means, so default training is unchanged bit for bit.)
    if batch.learner is None:
        mean = jnp.mean
    else:
        w = batch.learner.astype(jnp.float32)
        n_w = jnp.maximum(w.sum(), 1.0)

        def mean(x):
            return jnp.sum(w * x) / n_w

    # --- Policy (actor) loss: the clipped surrogate objective -----------------------------------
    # ratio r = pi_new(a|s) / pi_old(a|s) = exp(log pi_new - log pi_old)
    # L_clip = E[ min( r * A,  clip(r, 1 - eps, 1 + eps) * A ) ]
    # Taking the min removes any incentive to push r outside [1-eps, 1+eps]: once a good action
    # (A > 0) is already eps more likely, or a bad one (A < 0) eps less likely, the gradient is 0.
    # That keeps each update "proximal" to the policy that collected the data.
    ratio = jnp.exp(log_prob - batch.log_prob)
    if cfg.normalize_advantages:  # DECISION: zero mean, unit std per minibatch
        if batch.learner is None:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            adv_mean = mean(advantages)
            advantages = (advantages - adv_mean) / (jnp.sqrt(mean(jnp.square(advantages - adv_mean))) + 1e-8)
    surrogate = jnp.minimum(ratio * advantages, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages)
    policy_loss = -mean(surrogate)  # we minimise, so negate the objective

    # --- Value (critic) loss: regress V(s) onto the GAE return targets ------------------------
    value_err = jnp.square(value - targets)
    if cfg.clip_value_loss:  # DECISION: also keep V close to V_old, taking the worse of the two errors
        value_clipped = batch.value + jnp.clip(value - batch.value, -cfg.clip_eps, cfg.clip_eps)
        value_err = jnp.maximum(value_err, jnp.square(value_clipped - targets))
    value_loss = 0.5 * mean(value_err)

    # --- Entropy bonus: rewards keeping the policy random enough to keep exploring -------------
    entropy = mean(categorical_entropy(logits))

    total = policy_loss + cfg.vf_coef * value_loss - ent_coef * entropy
    stats = {}
    if batch.teacher_action is not None:  # --- Kickstarting: imitate the teacher, weight `kick` ---------
        # Cross-entropy -log pi(a_teacher|s): the supervised loss of behaviour cloning, but on the states
        # the student visits itself (so it also learns what the teacher would do after its own mistakes).
        teacher_ce = -mean(categorical_log_prob(logits, batch.teacher_action))
        total = total + kick * teacher_ce
        stats["teacher_ce"] = teacher_ce

    # Diagnostics (not part of the gradient):
    log_ratio = log_prob - batch.log_prob
    approx_kl = mean((ratio - 1.0) - log_ratio)  # low-variance KL(old || new) estimator
    clip_frac = mean(jnp.abs(ratio - 1.0) > cfg.clip_eps)  # share of samples hitting the clip
    return total, {
        "policy_loss": policy_loss, "value_loss": value_loss, "entropy": entropy,
        "approx_kl": approx_kl, "clip_frac": clip_frac, **stats,
    }


# =====================================================================================================
# Training
# =====================================================================================================


def make_train(cfg: PPOConfig, env, env_params, env_params_fn=None, curriculum=None, teacher=None, opponents=None):
    """Returns (init_fn, chunk_fn).

    init_fn(key)            -> RunnerState for one seed
    chunk_fn(runner_state)  -> (runner_state, metrics) after cfg.updates_per_chunk updates

    Both are written for a single seed; vmap them over seeds and jit them (see `train`).

    env_params_fn(progress) -> (env_params, logged_values), optional: lets the environment's
    (non-static) parameters change with training progress in [0, 1], e.g. a curriculum that starts
    with easy rules and tightens them. Without it, env_params is used throughout.

    curriculum, optional: chooses which starting situation each episode resets to (see
    sail_env/train/curriculum/config_env.py). Its state lives in RunnerState.sampler_state; before each
    rollout it turns that state into env params (sampling probabilities), after it records which
    situations succeeded.

    teacher(key, env_state, obs) -> actions, optional: an agent whose actions the policy is pulled
    towards (kickstarting, cfg.kick_coef / kick_until), shaped like the policy's actions (num_envs[, N]).
    It sees the same (wrapped) env state and observations the policy acts in.

    opponents, optional: fixed agents for the opponent pool (cfg.opp_frac), each with the teacher's
    signature (key, env_state, obs) -> actions (num_envs, N). The pool also holds cfg.opp_snapshots frozen
    copies of the policy itself.
    """
    use_teacher = teacher is not None and cfg.kick_coef > 0.0
    use_league = cfg.opp_frac > 0.0
    opponents = list(opponents or [])
    if env_params_fn is None:
        env_params_fn = lambda progress: (env_params, {})  # noqa: E731
    actor = Actor(env.num_actions, cfg.hidden)
    critic = Critic(cfg.hidden)
    # Multi-agent envs (the fleet) add a boat axis after the env axis: every boat is one more batch
    # entry for the shared policy. lead = number of leading (batch) axes in rollout arrays.
    agent_shape = (env.num_agents,) if hasattr(env, "num_agents") else ()
    lead = 2 + len(agent_shape)
    total_batch = cfg.batch_size * (env.num_agents if agent_shape else 1)
    minibatch = total_batch // cfg.num_minibatches
    if use_league:  # static
        assert agent_shape, "the opponent pool needs a fleet environment"
        assert 0 < cfg.opp_learners < env.num_agents, "opp_learners: some boats learn, the others are opponents"
        n_opp_envs = max(1, int(round(cfg.opp_frac * cfg.num_envs)))  # envs 0..n_opp_envs-1 race the pool
        n_choices = cfg.opp_snapshots + len(opponents)
        # learner[e, i]: boat i of env e is sailed by the learning policy
        opp_env = jnp.arange(cfg.num_envs) < n_opp_envs
        learner_mask = ~opp_env[:, None] | (jnp.arange(env.num_agents)[None, :] < cfg.opp_learners)

    # Learning-rate schedule: linear decay from lr to 0 over all optimiser steps.
    n_opt_steps = cfg.num_updates * cfg.update_epochs * cfg.num_minibatches
    lr = optax.linear_schedule(cfg.lr, 0.0, n_opt_steps) if cfg.anneal_lr else cfg.lr
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(lr, eps=1e-5))

    def draw_choice(key):  # opponent index per (env, boat): one per env, or one per boat (cfg.opp_mix)
        if cfg.opp_mix:
            return jax.random.randint(key, (cfg.num_envs, env.num_agents), 0, n_choices)
        c = jax.random.randint(key, (cfg.num_envs,), 0, n_choices)
        return jnp.broadcast_to(c[:, None], (cfg.num_envs, env.num_agents))

    def init_fn(key, init_params=None):
        k_actor, k_critic, k_env, k_stagger, k_run = jax.random.split(key, 5)
        dummy = jnp.zeros((env.obs_dim,))
        params = {"actor": actor.init(k_actor, dummy), "critic": critic.init(k_critic, dummy)}
        if init_params is not None:  # warm start, e.g. the fleet from the trained solo policy
            params = init_params
        sampler_state = curriculum.init() if curriculum is not None else None
        league = None
        if use_league:  # every snapshot starts as the initial policy; a random opponent per env
            snaps = jax.tree.map(lambda x: jnp.broadcast_to(x, (cfg.opp_snapshots,) + x.shape), params["actor"])
            league = {"snaps": snaps, "choice": draw_choice(jax.random.fold_in(k_run, 7))}
        first_params = env_params_fn(0.0)[0]
        if curriculum is not None:
            first_params = curriculum.make_params(first_params, sampler_state)
        obs, env_state = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(k_env, cfg.num_envs), first_params)
        if cfg.stagger_steps > 0:
            obs, env_state = stagger(k_stagger, obs, env_state, first_params)
        return RunnerState(params, optimizer.init(params), env_state, obs, k_run, jnp.int32(0), sampler_state, league)

    def stagger(key, obs, env_state, env_params):
        """Desynchronise the envs: env e takes n_e ~ U{0, ..., stagger_steps-1} random steps (discarded).

        With fixed-length episodes (the sailing env: always 70 steps), envs that start together
        finish together forever, so most rollouts would contain no episode end (no reward at all)
        and the rest would contain all of them. Staggering once makes every rollout a mix.
        """
        k_n, k_steps = jax.random.split(key)
        n_warm = jax.random.randint(k_n, (cfg.num_envs,), 0, cfg.stagger_steps)

        def warm_step(carry, inp):
            obs, state = carry
            i, k = inp
            k_act, k_env = jax.random.split(k)
            action = jax.random.randint(k_act, (cfg.num_envs,) + agent_shape, 0, env.num_actions)
            new_obs, new_state, *_ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(k_env, cfg.num_envs), state, action, env_params
            )
            active = i < n_warm  # env e only moves during its first n_e warm-up steps

            def pick(new, old):
                return jnp.where(active.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

            return (pick(new_obs, obs), jax.tree.map(pick, new_state, state)), None

        steps = (jnp.arange(cfg.stagger_steps), jax.random.split(k_steps, cfg.stagger_steps))
        (obs, env_state), _ = jax.lax.scan(warm_step, (obs, env_state), steps)
        return obs, env_state

    def env_step(runner: RunnerState, params):
        """One step of the rollout in all envs: act with the current policy, step the envs."""
        key, k_act, k_env = jax.random.split(runner.key, 3)
        logits = actor.apply(runner.params["actor"], runner.last_obs)
        value = critic.apply(runner.params["critic"], runner.last_obs)
        action = categorical_sample(k_act, logits)
        log_prob = categorical_log_prob(logits, action)
        learner, league = None, runner.league
        if use_league:  # the pool's boats: a frozen snapshot of the policy (sampled) or a fixed agent
            k_opp = jax.random.fold_in(k_act, 2)
            snap_logits = jax.vmap(lambda p: actor.apply(p, runner.last_obs))(league["snaps"])  # (K, E, N, A)
            cand = [categorical_sample(jax.random.fold_in(k_opp, j), snap_logits[j]) for j in range(cfg.opp_snapshots)]
            cand += [f(jax.random.fold_in(k_opp, 100 + j), runner.env_state, runner.last_obs).astype(action.dtype)
                     for j, f in enumerate(opponents)]
            opp_action = jnp.take_along_axis(jnp.stack(cand), league["choice"][None], axis=0)[0]  # (E, N)
            learner = learner_mask
            action = jnp.where(learner, action, opp_action)
        obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(k_env, cfg.num_envs), runner.env_state, action, params
        )
        if use_league:  # a new opponent for every env whose race just ended
            # opp_fixed: the env's pool boats include a fixed agent (else only past selves)
            fixed = jnp.any(~learner & (league["choice"] >= cfg.opp_snapshots), axis=-1, keepdims=True)
            info = {**info, "learner": learner, "opp_fixed": jnp.broadcast_to(fixed, learner.shape)}
            fresh = draw_choice(jax.random.fold_in(k_opp, 999))
            league = {**league, "choice": jnp.where(done[:, :1], fresh, league["choice"])}
        teacher_action = None
        if use_teacher:  # what the teacher would do here (its own key: the student's random stream is unchanged)
            teacher_action = teacher(jax.random.fold_in(k_act, 1), runner.env_state, runner.last_obs)
            info = {**info, "teacher_agree": (jnp.argmax(logits, axis=-1) == teacher_action).astype(jnp.float32)}
        transition = Transition(runner.last_obs, action, log_prob, value, reward, done, info, teacher_action,
                                learner if learner is None else jnp.broadcast_to(learner, action.shape))
        return runner._replace(env_state=env_state, last_obs=obs, key=key, league=league), transition

    def update_step(runner: RunnerState, _):
        # ---- 1. ROLLOUT: (num_steps, num_envs, ...) transitions ----------------------------------
        progress = runner.update_idx / cfg.num_updates
        params_now, curriculum_values = env_params_fn(progress)
        if curriculum is not None:  # sampling probabilities over starting situations
            params_now = curriculum.make_params(params_now, runner.sampler_state)
        if cfg.ent_coef_final is None:
            ent_coef = cfg.ent_coef
        else:  # linear decay of the entropy bonus over training
            ent_coef = cfg.ent_coef + (cfg.ent_coef_final - cfg.ent_coef) * progress
        kick = cfg.kick_coef * jnp.clip(1.0 - progress / cfg.kick_until, 0.0, 1.0)  # linear decay to 0
        runner, traj = jax.lax.scan(lambda r, _: env_step(r, params_now), runner, None, length=cfg.num_steps)

        # ---- 2. GAE --------------------------------------------------------------------------------
        last_value = critic.apply(runner.params["critic"], runner.last_obs)  # V(s_T) to bootstrap
        advantages, targets = compute_gae(traj.reward, traj.value, traj.done, last_value, cfg.gamma, cfg.gae_lambda)

        # ---- 3. UPDATE: epochs x shuffled minibatches ------------------------------------------------
        def flatten(x):  # (T, E[, N], ...) -> (T*E[*N], ...)
            return x.reshape((total_batch,) + x.shape[lead:])

        batch = jax.tree.map(flatten, (traj._replace(info=None), advantages, targets))

        def epoch(carry, _):
            params, opt_state, key = carry
            key, k_perm = jax.random.split(key)
            perm = jax.random.permutation(k_perm, total_batch)
            # Shuffle, then cut into (num_minibatches, minibatch_size, ...).
            minibatches = jax.tree.map(
                lambda x: x[perm].reshape((cfg.num_minibatches, minibatch) + x.shape[1:]), batch
            )

            def minibatch_step(carry, mb):
                params, opt_state = carry
                mb_traj, mb_adv, mb_targets = mb
                grads, stats = jax.grad(ppo_loss, has_aux=True)(
                    params, actor, critic, mb_traj, mb_adv, mb_targets, cfg, ent_coef, kick
                )
                updates, opt_state = optimizer.update(grads, opt_state, params)
                return (optax.apply_updates(params, updates), opt_state), stats

            (params, opt_state), stats = jax.lax.scan(minibatch_step, (params, opt_state), minibatches)
            return (params, opt_state, key), stats

        key, k_epochs = jax.random.split(runner.key)
        (params, opt_state, _), stats = jax.lax.scan(
            epoch, (runner.params, runner.opt_state, k_epochs), None, length=cfg.update_epochs
        )

        # ---- 4. METRICS ------------------------------------------------------------------------------
        info = traj.info
        n_done = info["returned_episode"].sum()

        def per_episode(x):
            # Mean over the episodes that finished during this rollout. NaN if none finished
            # (long episodes can span several updates): "no data", not a return of 0.
            total = jnp.sum(jnp.where(info["returned_episode"], x, 0.0))
            return jnp.where(n_done > 0, total / jnp.maximum(n_done, 1), jnp.nan)

        # Explained variance: 1 = critic predicts the return targets perfectly, 0 = no better than
        # predicting the mean, < 0 = worse than that.
        explained_var = 1.0 - jnp.var(targets - traj.value) / (jnp.var(targets) + 1e-8)
        metrics = {
            "episode_return": per_episode(info["returned_episode_returns"]),
            "episode_length": per_episode(info["returned_episode_lengths"]),
            "episodes": info["returned_episode"].sum(),
            "explained_variance": explained_var,
            **{k: v.mean() for k, v in stats.items()},
        }
        if "success" in info:
            metrics["success_rate"] = per_episode(info["success"])  # under the current rules
            metrics["ocs_rate"] = per_episode(info["ocs"])
        if "success_strict" in info:
            metrics["success_strict_rate"] = per_episode(info["success_strict"])  # under the full rules
        if "good_start_on_line" in info:
            metrics["good_start_on_line_rate"] = per_episode(info["good_start_on_line"])  # + 0-3 m at the gun
            metrics["on_line_at_gun_rate"] = per_episode(info["on_line_at_gun"])
        if "good_start" in info:
            metrics["good_start_rate"] = per_episode(info["good_start"])  # full rules + starboard
            metrics["starboard_start_rate"] = per_episode(info["start_on_starboard"])
            ok = info["returned_episode"] & info["success_strict"]  # efficiency of successful starts
            n_ok = ok.sum()
            metrics["upwind_efficiency"] = jnp.where(
                n_ok > 0, jnp.sum(jnp.where(ok, info["upwind_efficiency"], 0.0)) / jnp.maximum(n_ok, 1), jnp.nan
            )
        if "fouls" in info:  # fleet
            metrics["fouls_per_start"] = per_episode(info["fouls"])
            metrics["unserved_foul_rate"] = per_episode(info["unserved_foul"])
            metrics["clear_air_rate"] = per_episode(info["clear_air"])
        if "contacts" in info:
            metrics["contacts_per_boat"] = per_episode(info["contacts"])
            metrics["mark_hits_per_boat"] = per_episode(info["mark_hits"])
        if "race_finished" in info:  # the race: finished, how fast, how far the others got
            metrics["race_finished_rate"] = per_episode(info["race_finished"])
            metrics["legs_completed"] = per_episode(info["legs_completed"])
            fin = info["returned_episode"] & info["race_finished"]
            n_fin = fin.sum()
            metrics["finish_time"] = jnp.where(  # mean over the boats that finished (s after the gun)
                n_fin > 0, jnp.sum(jnp.where(fin, info["finish_time"], 0.0)) / jnp.maximum(n_fin, 1), jnp.nan)
            if "shadow_time" in info:  # fleet racing: s in another boat's wind shadow; how much the order
                metrics["shadow_time"] = per_episode(info["shadow_time"])  # changes after the first mark
                rounded = info["returned_episode"] & (info["mark1_place"] > 0)  # (the mean signed change is 0)
                moved = jnp.where(rounded, jnp.abs(info["mark1_place"] - (info["race_rank"] + 1.0)), 0.0)
                metrics["place_changes_after_mark1"] = jnp.where(
                    rounded.sum() > 0, moved.sum() / jnp.maximum(rounded.sum(), 1), jnp.nan)
            if "team_win" in info:  # team racing: winning team's mean sum of places (2v2: 3 = 1st+2nd ... 5)
                win = info["returned_episode"] & info["team_win"]
                metrics["winning_team_score"] = jnp.where(
                    win.sum() > 0, jnp.sum(jnp.where(win, info["team_score"], 0.0)) / jnp.maximum(win.sum(), 1), jnp.nan)
        metrics.update({f"curriculum_{k}": v for k, v in curriculum_values.items()})
        if cfg.ent_coef_final is not None:
            metrics["ent_coef"] = jnp.float32(ent_coef)
        if use_teacher:
            metrics["kick_coef"] = jnp.float32(kick)
            metrics["teacher_agreement"] = info["teacher_agree"].mean()  # share of steps: greedy action = teacher's
        if use_league and "race_rank" in info:  # races against the pool: the learner boats' mean place, against
            ended = info["returned_episode"] & ~info["learner"].all(axis=-1, keepdims=True) & info["learner"]
            for name, pick in [("pool", ended), ("pool_snapshots", ended & ~info["opp_fixed"]),
                               ("pool_fixed", ended & info["opp_fixed"])]:  # all, only past selves, with a fixed agent
                metrics[f"{name}_place"] = jnp.where(
                    pick.sum() > 0, jnp.sum(jnp.where(pick, info["race_rank"] + 1.0, 0.0)) / jnp.maximum(pick.sum(), 1),
                    jnp.nan)
            if "team_win" in info:
                metrics["pool_team_win_rate"] = jnp.where(
                    ended.sum() > 0, jnp.sum(jnp.where(ended, info["team_win"], False)) / jnp.maximum(ended.sum(), 1),
                    jnp.nan)

        if curriculum is not None:  # record which starting situations succeeded in this rollout
            sampler_state = curriculum.update(runner.sampler_state, info)
            metrics.update(curriculum.log(sampler_state))
            runner = runner._replace(sampler_state=sampler_state)

        if use_league:  # every opp_snapshot_every updates the newest policy replaces the oldest snapshot
            refresh = (runner.update_idx + 1) % cfg.opp_snapshot_every == 0
            snaps = jax.tree.map(lambda s, p: jnp.where(refresh, jnp.concatenate([p[None], s[:-1]]), s),
                                 runner.league["snaps"], params["actor"])
            runner = runner._replace(league={**runner.league, "snaps": snaps})
        runner = runner._replace(params=params, opt_state=opt_state, key=key, update_idx=runner.update_idx + 1)
        return runner, metrics

    def chunk_fn(runner: RunnerState):
        return jax.lax.scan(update_step, runner, None, length=cfg.updates_per_chunk)

    return init_fn, chunk_fn


def train(cfg: PPOConfig, env, env_params, key, log_fn=None, chunk_fn_callback=None, env_params_fn=None,
          curriculum=None, init_params=None, teacher=None, opponents=None):
    """Train cfg.num_seeds agents in parallel. Returns (final RunnerState, metrics dict of (seeds, updates)).

    log_fn(update_idx, env_steps, metrics_at_update) is called after every chunk for each update
    in it, with arrays of shape (num_seeds,), so a logger can stream live curves.
    chunk_fn_callback(last_update_idx, runner) is called after every chunk (e.g. to render a GIF).
    teacher: see make_train (kickstarting). opponents: see make_train (opponent pool, cfg.opp_frac).
    """
    init_fn, chunk_fn = make_train(cfg, env, env_params, env_params_fn, curriculum, teacher, opponents)
    keys = jax.random.split(key, cfg.num_seeds)
    if init_params is None:
        runner = jax.jit(jax.vmap(init_fn))(keys)
    else:  # init_params has a leading seed axis
        runner = jax.jit(jax.vmap(init_fn))(keys, init_params)
    chunk = jax.jit(jax.vmap(chunk_fn))

    history = []
    for c in range(cfg.num_chunks):
        runner, metrics = chunk(runner)  # each leaf: (num_seeds, updates_per_chunk)
        metrics = jax.tree.map(np.asarray, metrics)
        history.append(metrics)
        if log_fn is not None:
            for u in range(cfg.updates_per_chunk):
                idx = c * cfg.updates_per_chunk + u
                log_fn(idx, (idx + 1) * cfg.batch_size, {k: v[:, u] for k, v in metrics.items()})
        if chunk_fn_callback is not None:
            chunk_fn_callback((c + 1) * cfg.updates_per_chunk - 1, runner)
    all_metrics = {k: np.concatenate([h[k] for h in history], axis=1) for k in history[0]}
    return runner, all_metrics
