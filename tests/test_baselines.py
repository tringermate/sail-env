import jax
import jax.numpy as jnp
import numpy as np

from sail_env.agents.digital_sailor import beat_angle, digital_expert, digital_sailor_action, rollout_solo
from sail_env.config import EnvParams, kn
from sail_env.polar import read_orc_csv
from sail_env.start_env import SoloStartEnv

P = EnvParams(gun_distance_max=3.0, port_penalty=1.0, upwind_bonus=1.0, efficiency_tail_seconds=10.0)
ENV = SoloStartEnv(P)


def test_beat_angle_matches_orc():
    orc = read_orc_csv()
    k = orc["tws"].index(10.0)
    assert abs(np.degrees(float(beat_angle(ENV.polar, jnp.float32(kn(10))))) - orc["beat_angle"][k]) < 1.0


def test_digital_sailor_runs_and_expert_is_at_least_as_good():
    _, states = jax.vmap(ENV.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(0), 32), P)
    a = jax.vmap(lambda s: digital_sailor_action(s, ENV.polar, P))(states)
    assert a.shape == (32,) and bool(jnp.all((a >= 0) & (a < 7)))
    sailor, r_sailor = jax.jit(jax.vmap(lambda s: rollout_solo(ENV, P, s, 0.3, -3.0)))(states)
    expert, r_expert, _ = jax.jit(jax.vmap(lambda s: digital_expert(ENV, P, s, spots=(0.3, 0.5), timings=(-3.0, 0.0))))(states)
    assert bool(jnp.all(r_expert >= r_sailor - 1e-5))  # the expert's grid contains the sailor's setting


def test_agents_never_stay_in_irons():
    """Regression: in irons the agents bear away; a tack waits until there is steerage way."""
    import jax.numpy as jnp

    from sail_env.agents.digital_sailor import unstick
    from sail_env.config import EnvParams, deg, kn
    from sail_env.physics import DOWN5, DOWN15, HOLD, TACK

    p, beat = EnvParams(), deg(40)
    head_to_wind = jnp.float32(deg(10))  # TWA -10 deg in a northerly
    assert int(unstick(TACK, jnp.float32(0.0), head_to_wind, 0.0, beat, p)) == DOWN15  # stopped: back the jib
    assert int(unstick(HOLD, jnp.float32(0.0), head_to_wind, 0.0, beat, p)) == DOWN15
    pinching = jnp.float32(deg(30))
    assert int(unstick(TACK, jnp.float32(kn(1.5)), pinching, 0.0, beat, p)) == DOWN5  # too slow: build speed first
    close_hauled = jnp.float32(deg(42))
    assert int(unstick(TACK, jnp.float32(kn(2.0)), close_hauled, 0.0, beat, p)) == HOLD
    assert int(unstick(TACK, jnp.float32(kn(5.0)), close_hauled, 0.0, beat, p)) == TACK  # fast enough: tack
