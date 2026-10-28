"""LogWrapper: track each episode's total return and length, for logging (as in PureJaxRL).

The wrapped env's info dict gains:
    returned_episode          bool, an episode finished at this step
    returned_episode_returns  total reward of that finished episode
    returned_episode_lengths  its length in steps
Everything else (e.g. the sailing env's "success") passes through unchanged.
"""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class LogState:
    env_state: object
    episode_return: jax.Array
    episode_length: jax.Array


class LogWrapper:
    def __init__(self, env):
        self.env = env
        self.num_actions = env.num_actions
        self.obs_dim = env.obs_dim
        if hasattr(env, "num_agents"):
            self.num_agents = env.num_agents

    def reset(self, key, params):
        obs, env_state = self.env.reset(key, params)
        shape = (self.env.num_agents,) if hasattr(self.env, "num_agents") else ()  # one counter per boat
        return obs, LogState(env_state, jnp.zeros(shape, jnp.float32), jnp.zeros(shape, jnp.int32))

    def step(self, key, state: LogState, action, params):
        obs, env_state, reward, done, info = self.env.step(key, state.env_state, action, params)
        ep_return = state.episode_return + reward
        ep_length = state.episode_length + 1
        info = dict(info)
        info["returned_episode"] = done
        info["returned_episode_returns"] = ep_return
        info["returned_episode_lengths"] = ep_length
        # Counters restart when the episode ends (the env has already auto-reset).
        state = LogState(env_state, ep_return * (1 - done), ep_length * (1 - done))
        return obs, state, reward, done, info
