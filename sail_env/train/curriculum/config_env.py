"""Reset to configurations from a fixed set, chosen by a sampler (uniform, SFL or SGS).

    configs = make_config_set(env, env_params, key, n)      # N complete starting states, drawn once
    env = ConfigResetEnv(SoloStartEnv(...), configs)         # resets to configs[sampled id]
    curriculum = Curriculum(SamplerConfig(...), ...)         # the PPO hook: sampler state, weights, logs

The environment's params become ConfigParams(env=EnvParams, cdf=cumulative sampling probabilities);
PPO recomputes the cdf from the sampler state at every update, and after every rollout pushes each
finished episode's outcome into its configuration's buffer.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from sail_env.train.curriculum import sampler as smp
from sail_env.polar import beat_vmg
from sail_env.rules import signed_distance


@struct.dataclass
class ConfigParams:
    env: object  # EnvParams
    cdf: jax.Array  # (N,) cumulative sampling probabilities


@struct.dataclass
class ConfigState:
    env_state: object  # EnvState
    config_id: jax.Array


def make_config_set(env, params, key, n, reach_frac=1.2, dist_min=2.0):
    """N starting states from a wider distribution than training: random start times (5-60 s), from
    2 m below the line (trivial) to beyond what best VMG reaches by the gun (too hard)."""
    wide = params.replace(random_start_time=True, init_reach_frac=reach_frac, init_dist_min=dist_min)
    _, states = jax.vmap(env.reset, in_axes=(0, None))(jax.random.split(key, n), wide)
    return states


def make_race_start_bank(env, params, key, n, start_policy="fleet"):
    """N race states just after the start, for training the racing alone (the start is a solved task).

    Each of the N episodes is reset as usual (T-90) and every boat sails the start with a bundled start
    policy (most likely action; it reads the first 60 observation features, the start observation).
    The state is kept at the step when every boat has started, or at T+10 if some have not (OCS or
    late: they still have to start, now under the learner's control)."""
    from pathlib import Path

    from sail_env.agents.policy import load_policy
    from sail_env.start_env import OBS_DIM

    path = Path(start_policy)
    if not path.exists():
        path = Path(__file__).resolve().parents[2] / "pretrained" / start_policy
    _, _, actor, actor_params_all, _ = load_policy(path)
    actor_params = jax.tree.map(lambda x: x[0], actor_params_all["actor"])
    steps = int(round((10.0 - params.t_start) / params.dt))  # T-90 .. T+10

    def one(k):
        obs, s = env.reset(k, params)

        def body(carry, _):
            obs, s, kept, have = carry
            a = jnp.argmax(actor.apply(actor_params, obs[..., :OBS_DIM]), axis=-1)
            obs, s, _, _, _ = env.step_env(jax.random.PRNGKey(0), s, a, params)
            now = jnp.all(s.boats.status.started) & ~have
            kept = jax.tree.map(lambda new, old: jnp.where(now, new, old), s, kept)
            return (obs, s, kept, have | now), None

        (_, last, kept, have), _ = jax.lax.scan(body, (obs, s, s, jnp.bool_(False)), None, length=steps)
        return jax.tree.map(lambda a, b: jnp.where(have, a, b), kept, last)

    return jax.jit(jax.vmap(one))(jax.random.split(key, n))


def make_race_situation_bank(env, params, key, n, lateral_beat=150.0, lateral_run=60.0, leg_seconds=170.0):
    """N race situations anywhere on the course, sampled directly (no agent sails them).

    Each starts from an ordinary race reset (so wind, weather, line and course are consistent); every
    boat is then given, independently:
      leg       uniform over the four legs (0 beat, 1 run to the gate, 2 beat, 3 run to the finish)
      position  a uniform fraction of the way along that leg, up to `lateral_beat` m to either side on
                a beat (the boats spread out across the course) or `lateral_run` m on a run
      heading   any heading the boat can sail (outside the no-go zone), speed 50-100% of the polar
      status    started legally, no penalty; clock = (legs done + fraction) x leg_seconds, roughly
                where a boat would be in the race (the race clock is in the observation)
    The time limit (t_end) still applies; it is far enough away from every situation."""
    from sail_env.geometry import signed_twa, wrap_angle
    from sail_env.polar import polar_speed
    from sail_env.weather import any_weather, water_wind
    from sail_env.wind import wind_at

    n_boats = env.num_agents

    def one(k):
        k_reset, k_leg, k_frac, k_lat, k_head, k_speed = jax.random.split(k, 6)
        _, fs = env.reset(k_reset, params)
        c = env.course(fs, params)
        leg = jax.random.randint(k_leg, (n_boats,), 0, 4)
        frac = jax.random.uniform(k_frac, (n_boats,))
        start = jnp.where((leg == 0)[:, None], jnp.stack(c["c"]), jnp.where((leg == 1)[:, None] | (leg == 3)[:, None],
                                                                            jnp.stack(c["w"]), jnp.stack(c["g"])))
        end = jnp.where(((leg == 0) | (leg == 2))[:, None], jnp.stack(c["w"]),
                        jnp.where((leg == 1)[:, None], jnp.stack(c["g"]), jnp.stack(c["c"])))
        beat = (leg == 0) | (leg == 2)
        lat_max = jnp.where(beat, lateral_beat, lateral_run)
        lat = jax.random.uniform(k_lat, (n_boats,), minval=-1.0, maxval=1.0) * lat_max * jnp.sin(jnp.pi * frac)
        rx, ry = c["r"]
        x = start[:, 0] + frac * (end[:, 0] - start[:, 0]) + lat * rx
        y = start[:, 1] + frac * (end[:, 1] - start[:, 1]) + lat * ry
        st = fs.boats
        t = (leg + frac) * leg_seconds
        if any_weather(params):  # static
            wd, tws = jax.vmap(lambda w, we, tt, xx, yy: water_wind(w, we, tt, xx, yy, params))(st.wind, st.weather, t, x, y)
        else:
            wd, tws = jax.vmap(wind_at)(st.wind, t)
        # A heading outside the no-go zone: TWA uniform in [beat-ish, 180] on either tack.
        twa = jax.random.uniform(k_head, (n_boats,), minval=jnp.radians(40.0), maxval=jnp.pi)
        twa = twa * jnp.where(jax.random.bernoulli(jax.random.fold_in(k_head, 1), 0.5, (n_boats,)), 1.0, -1.0)
        heading = wrap_angle(wd - twa)
        v = jax.random.uniform(k_speed, (n_boats,), minval=0.5, maxval=1.0) * polar_speed(env.polar, signed_twa(heading, wd), tws)
        boat = st.boat.replace(x=x, y=y, heading=heading, target_heading=heading, speed=v,
                               manoeuvring=jnp.zeros(n_boats, bool))
        status = st.status.replace(started=jnp.ones(n_boats, bool), ocs=jnp.zeros(n_boats, bool),
                                   start_time=jnp.zeros(n_boats))
        t_all = jnp.full(n_boats, jnp.max(t))  # one clock for the fleet: the furthest-on boat's
        st = st.replace(boat=boat, status=status, t=t_all)
        if params.observe_weather:  # static: the instrument log starts here
            st = st.replace(log_origin=jnp.stack([x, y, t_all], axis=1))
        return fs.replace(boats=st, leg=leg)

    return jax.jit(jax.vmap(one))(jax.random.split(key, n))


def mix_banks(a, b, frac_b, key):
    """A bank of the same size as `a`, with a fraction `frac_b` of its entries taken from `b` instead."""
    n = jax.tree.leaves(a)[0].shape[0]
    take_b = jax.random.uniform(key, (n,)) < frac_b
    idx_b = jax.random.randint(jax.random.fold_in(key, 1), (n,), 0, jax.tree.leaves(b)[0].shape[0])

    def pick(x, y):
        yb = y[idx_b]
        return jnp.where(take_b.reshape((n,) + (1,) * (x.ndim - 1)), yb, x)

    return jax.tree.map(pick, a, b)


def reach_fraction(env, configs):
    """Difficulty proxy: distance below the line / (best VMG x time to gun); > 1 = cannot make it."""
    below = -jax.vmap(signed_distance)(configs.line, configs.boat.x, configs.boat.y)
    vmg = jax.vmap(lambda tws: beat_vmg(env.polar, tws))(configs.wind.tws)
    return below / (vmg * jnp.maximum(-configs.t, 1.0))


class ConfigResetEnv:
    """Resets to sampler-chosen configurations. With num_levels > 1 a configuration is a pair
    (situation, rule level): id = situation x num_levels + level, and the episode's rule strictness
    is level / (num_levels - 1) (needs per_episode_rules in the env params)."""

    def __init__(self, env, configs, num_levels: int = 1):
        self.env = env
        self.configs = configs
        self.num_levels = num_levels
        self.num_actions = env.num_actions
        self.obs_dim = env.obs_dim
        if hasattr(env, "num_agents"):  # fleet and race: one batch entry per boat
            self.num_agents = env.num_agents

    def reset(self, key, params: ConfigParams):
        cid = smp.sample(params.cdf, key)
        situation, level = cid // self.num_levels, cid % self.num_levels
        state = jax.tree.map(lambda x: x[situation], self.configs)
        if self.num_levels > 1:
            state = state.replace(rule_w=level.astype(jnp.float32) / (self.num_levels - 1))
        return self.env.get_obs(state, params.env), ConfigState(state, cid)

    def step(self, key, state: ConfigState, action, params: ConfigParams):
        """step_env + auto-reset to a freshly sampled configuration on done."""
        k_step, k_reset = jax.random.split(key)
        obs, env_state, reward, done, info = self.env.step_env(k_step, state.env_state, action, params.env)
        info = dict(info)
        info["config_id"] = jnp.broadcast_to(state.config_id, done.shape)  # one per boat in a fleet
        obs_re, state_re = self.reset(k_reset, params)
        end = done.reshape(-1)[0]  # the whole fleet ends together
        new = ConfigState(env_state, state.config_id)
        new = jax.tree.map(lambda a, b: jnp.where(end, a, b), state_re, new)
        obs = jnp.where(end, obs_re, obs)
        return obs, new, reward, done, info


class Curriculum:
    """Hook for make_train: owns the sampler and decides what gets logged."""

    def __init__(self, cfg: smp.SamplerConfig, success_key: str = "good_start_on_line", tracked_ids=()):
        self.cfg = cfg
        self.success_key = success_key
        self.tracked_ids = np.asarray(tracked_ids, dtype=np.int32)

    def init(self):
        return smp.init(self.cfg.num_configs, self.cfg.history)

    def make_params(self, env_params, state):
        return ConfigParams(env=env_params, cdf=jnp.cumsum(smp.probs(state, self.cfg)))

    def update(self, state, info):
        return smp.update(state, info["config_id"], info[self.success_key], info["returned_episode"])

    def log(self, state):
        p = smp.success_rate(state)
        prob = smp.probs(state, self.cfg)
        n = self.cfg.num_configs
        out = {
            "sampler_mean_p": p.mean(),
            "sampler_frac_learnable": jnp.mean((p > 0.1) & (p < 0.9)),  # sometimes but not always
            "sampler_frac_solved": jnp.mean(p >= 0.9),
            "sampler_frac_unseen": jnp.mean(state.count == 0),
            "sampler_effective_configs": 1.0 / jnp.sum(prob**2),  # how many configs the sampling really uses
        }
        for k, cid in enumerate(self.tracked_ids):
            out[f"tracked{k}_p"] = p[cid]
            out[f"tracked{k}_weight"] = prob[cid] * n  # 1 = uniform share
        return out


def pick_tracked(env, configs, quantiles=(0.02, 0.2, 0.4, 0.6, 0.8, 0.98)):
    """Config ids at difficulty quantiles (reach fraction), from trivial to too hard, for logging."""
    r = np.asarray(reach_fraction(env, configs))
    order = np.argsort(r)
    ids = [int(order[int(q * (len(r) - 1))]) for q in quantiles]
    return ids, [float(r[i]) for i in ids]
