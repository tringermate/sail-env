"""Fleet-racing measures in info: seconds in another boat's wind shadow, and the order at the first windward mark."""

import jax
import jax.numpy as jnp
import numpy as np

import sail_env as se
from sail_env.agents import load_agent


def race_to_end(env, p, spec, key=0):
    policy = load_agent(spec)(env, p)
    obs, s = env.reset(jax.random.PRNGKey(key), p)
    n = env.num_agents

    def body(c, i):
        o, s, fin, last = c
        a = jax.vmap(policy)(jax.random.split(jax.random.PRNGKey(i), n), o, s.boats, jnp.arange(n))
        o2, s2, _, d, info = env.step_env(jax.random.PRNGKey(0), s, a, p)
        last = jax.tree.map(lambda new, old: jnp.where(d[0] & ~fin, new, old), info, last)
        s2 = jax.tree.map(lambda new, old: jnp.where(fin, old, new), s2, s)
        return (o2, s2, fin | d[0], last), None

    _, _, _, _, info0 = env.step_env(jax.random.PRNGKey(0), s, jnp.zeros(n, jnp.int32), p)
    init = (obs, s, jnp.bool_(False), jax.tree.map(jnp.zeros_like, info0))
    (_, s_end, done, last), _ = jax.jit(lambda: jax.lax.scan(body, init, jnp.arange(p.n_steps)))()
    return jax.tree.map(np.asarray, last), s_end, bool(done)


def test_first_mark_order_and_shadow_time_in_a_fleet_race():
    env, p = se.make("race", fleet_size=4, rrs16=True)
    info, s, done = race_to_end(env, p, "digital")
    assert done
    places = info["mark1_place"]
    rounded = places > 0
    assert rounded.sum() >= 3  # the digital racers get round
    assert set(places[rounded]) <= set(range(1, 5)) and 1 in places
    assert np.all(info["shadow_time"] >= 0) and np.all(info["shadow_time"] <= p.t_end)
    assert info["shadow_time"].sum() > 0  # four boats on one course: someone sails in someone's shadow


def test_solo_race_has_no_shadow():
    env, p = se.make("race-solo")
    info, _, done = race_to_end(env, p, "digital")
    assert done and info["shadow_time"][0] == 0.0 and info["mark1_place"][0] == 1
