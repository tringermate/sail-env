"""The full race: course, roundings, gate, finish, ranking, mark-room, penalties, digital racer, tools."""

import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sail_env as se
from sail_env.agents import load_agent
from sail_env.config import deg, kn
from sail_env.evaluate import evaluate
from sail_env.physics import HOLD
from sail_env.play import KEYS, PlaySession
from sail_env.race_env import N_LEGS, PARK_X, RACE_FEATURES, RaceEnv, course_geometry
from sail_env.runs import load_run, render_replay, run_episode, save_run
from sail_env.start_env import OBS_DIM
from sail_env.wind import Wind

ENV, P0 = se.make("race")
# Steady wind from the north, square line centred on (0, 0): up = +y, right (looking upwind) = +x.
P = P0.replace(shift_amp_max=0.0, line_bias_max=0.0)
STEP = jax.jit(lambda s, a: ENV.step_env(jax.random.PRNGKey(0), s, a, P))


def race_state(specs, legs=None, t=300.0, tws_kn=10.0):
    """A 5-boat race state: boats placed by specs [(x, y, heading_deg, speed_kn)], the rest far apart
    (but on the course); all started, legs as given (default 0)."""
    _, fs = ENV.reset(jax.random.PRNGKey(0), P)
    n = 5
    specs = list(specs) + [(-300.0 - 100.0 * i, 200.0, 0.0, 0.0) for i in range(n - len(specs))]
    x, y, h, v = (jnp.array([s[k] for s in specs], jnp.float32) for k in range(4))
    h, v = jnp.radians(h), v * kn(1.0)
    boat = fs.boats.boat.replace(x=x, y=y, heading=h, target_heading=h, speed=v, manoeuvring=jnp.zeros(n, bool))
    wind = Wind(*(jnp.full(n, v_, jnp.float32) for v_ in (0.0, kn(tws_kn), 0.0, 120.0, 0.0)))
    status = fs.boats.status.replace(started=jnp.ones(n, bool), ocs=jnp.zeros(n, bool))
    boats = fs.boats.replace(boat=boat, wind=wind, status=status, t=jnp.full(n, t))
    legs = jnp.zeros(n, jnp.int32) if legs is None else jnp.asarray(legs, jnp.int32)
    return fs.replace(boats=boats, leg=legs)


def hold():
    return jnp.full(5, HOLD, jnp.int32)


def test_course_geometry_and_observation():
    _, fs = ENV.reset(jax.random.PRNGKey(0), P)
    k = ENV.course(fs, P)
    assert np.allclose(k["c"], (0.0, 0.0), atol=1e-4) and np.allclose(k["w"], (0.0, P.course_beat_length), atol=1e-3)
    assert np.allclose(k["g1"], (-k["hw"], P.gate_offset), atol=1e-3) and np.allclose(k["g2"], (k["hw"], P.gate_offset), atol=1e-3)
    assert k["hw"] == pytest.approx(0.5 * P.gate_width_bl * P.boat_length)
    assert ENV.obs_dim == OBS_DIM + RACE_FEATURES and ENV.n_marks == 5
    obs, fs = ENV.reset(jax.random.PRNGKey(1), P0)
    assert obs.shape == (5, ENV.obs_dim)
    assert np.all(np.asarray(obs[:, -RACE_FEATURES]) == 1.0)  # leg one-hot: everyone on beat 1
    env1, p1 = se.make("race-solo")
    assert isinstance(env1, RaceEnv) and env1.num_agents == 1 and env1.reset(jax.random.PRNGKey(0), p1)[0].shape == (1, env1.obs_dim)


def test_windward_mark_counts_only_a_port_rounding():
    w = P.course_beat_length
    half = 0.5 * P.boat_length
    # Boat 0: just right of the mark, above it, sailing west -> crosses the upwind ray right to left.
    # Boat 1: left of the mark, above it, sailing east (left to right): not a port rounding.
    # Boat 2: right of the mark but below it, sailing west: passes below, not a rounding.
    # (Centres are half a hull behind the bows; boat 1 sails higher up so the two do not meet.)
    fs = race_state([(1.0 + half, w + 6.0, -90.0, 6.0), (-1.0 - half, w + 30.0, 90.0, 6.0),
                     (1.0 + half, w - 8.0, -90.0, 6.0)])
    obs, fs2, _, _, info = STEP(fs, hold())
    # Across the half-line is only the first part: boat 0 still has to get back down past the mark.
    assert list(np.asarray(fs2.leg[:3])) == [0, 0, 0] and not np.asarray(info["rounded_now"][:3]).any()
    assert list(np.asarray(fs2.mark_through[:3])) == [True, False, False]
    # ... but its observation already shows the run (leg one-hot) and the gate marks.
    onehot = np.asarray(obs[:, -RACE_FEATURES:-RACE_FEATURES + N_LEGS + 1])
    assert onehot[0].argmax() == 1 and onehot[1].argmax() == 0


def test_windward_mark_must_be_rounded():
    w = P.course_beat_length
    half = 0.5 * P.boat_length
    # All three are across the half-line (mark_through), heading south: boat 0 left of the mark (rounded it),
    # boat 1 right of it, above (came back over: undone first), boat 2 still above the mark's level.
    fs = race_state([(-8.0, w + half + 1.0, 180.0, 6.0), (-1.0 - half, w + 25.0, 90.0, 6.0),
                     (-8.0, w + 20.0, 180.0, 6.0)])
    fs = fs.replace(mark_through=jnp.array([True, True, True, False, False]))
    _, fs2, _, _, info = STEP(fs, hold())
    assert list(np.asarray(fs2.leg[:3])) == [1, 0, 0]
    assert list(np.asarray(fs2.mark_through[:3])) == [False, False, True]
    assert bool(info["rounded_now"][0]) and not bool(info["rounded_now"][1])
    # Touch the half-line and turn back down the right side: no rounding. Over it to the left, then back
    # to the right (each a crossing within one step), then turned south well right of it and down past the mark.
    fs = race_state([(1.0 + half, w + 6.0, -90.0, 6.0)])
    _, fs = STEP(fs, hold())[:2]
    assert bool(fs.mark_through[0])
    turn = lambda fs, deg_: fs.replace(boats=fs.boats.replace(boat=fs.boats.boat.replace(
        heading=fs.boats.boat.heading.at[0].set(math.radians(deg_)),
        target_heading=fs.boats.boat.target_heading.at[0].set(math.radians(deg_)))))
    fs = turn(fs.replace(boats=fs.boats.replace(boat=fs.boats.boat.replace(x=fs.boats.boat.x.at[0].set(-1.0 - half)))), 90.0)
    _, fs = STEP(fs, hold())[:2]
    assert not bool(fs.mark_through[0])  # back over to the right: undone
    for _ in range(4):
        _, fs = STEP(fs, hold())[:2]
    fs = turn(fs, 180.0)
    for _ in range(8):
        _, fs = STEP(fs, hold())[:2]
    assert int(fs.leg[0]) == 0 and not bool(fs.mark_through[0])
    assert float(fs.boats.boat.y[0]) < w and float(fs.boats.boat.x[0]) > 0.0  # below the mark, on its right


def test_gate_and_finish():
    g = P.gate_offset
    hw = 0.5 * P.gate_width_bl * P.boat_length
    half = 0.5 * P.boat_length
    # Boat 0 runs through the gate; boat 1 passes outside it; boat 2 (leg 3) crosses the finish line.
    fs = race_state([(0.0, g + half + 1.0, 180.0, 6.0), (hw + 15.0, g + half + 1.0, 180.0, 6.0),
                     (5.0, half + 1.0, 180.0, 6.0)], legs=[1, 1, 3, 2, 2])
    _, fs2, r, done, info = STEP(fs, hold())
    # Through the gate is not the end of the run: a gate mark still has to be rounded.
    assert list(np.asarray(fs2.leg[:3])) == [1, 1, 4] and list(np.asarray(fs2.gate_through[:2])) == [True, False]
    assert float(fs2.finish_time[2]) == pytest.approx(300.0, abs=1.0)
    assert float(fs2.boats.boat.x[2]) > 0.5 * PARK_X  # finished boats leave the course
    assert not bool(done[0])


def test_gate_mark_must_be_rounded():
    g = P.gate_offset
    hw = 0.5 * P.boat_length * P.gate_width_bl
    half = 0.5 * P.boat_length
    # All three are through the gate, heading back up (north): boat 0 outside the left mark (rounded it),
    # boat 1 between the marks (turned up in the middle: no rounding, the pass is undone), boat 2 outside the
    # right mark but never went through (no rounding).
    y = g - half - 1.0
    fs = race_state([(-hw - 6.0, y, 0.0, 6.0), (0.0, y, 0.0, 6.0), (hw + 6.0, y, 0.0, 6.0)], legs=[1, 1, 1, 2, 2])
    fs = fs.replace(gate_through=jnp.array([True, True, False, False, False]))
    _, fs2, _, _, info = STEP(fs, hold())
    assert list(np.asarray(fs2.leg[:3])) == [2, 1, 1]
    assert list(np.asarray(fs2.gate_through[:3])) == [False, False, False]
    assert bool(info["rounded_now"][0]) and not bool(info["rounded_now"][1])


def test_ranking_by_finishing_order_and_end_when_all_finished():
    half = 0.5 * P.boat_length
    fs = race_state([(5.0, half + 1.0, 180.0, 6.0)], legs=[3, 4, 4, 4, 4])
    park = PARK_X + 100.0 * jnp.arange(5)
    b = fs.boats.boat
    fs = fs.replace(boats=fs.boats.replace(boat=b.replace(x=b.x.at[1:].set(park[1:]), y=b.y.at[1:].set(0.0))),
                    finish_time=jnp.array([jnp.inf, 100.0, 400.0, 200.0, 350.0]))
    _, fs2, r, done, info = STEP(fs, hold())
    assert bool(jnp.all(done)) and bool(jnp.all(info["race_finished"]))
    assert list(np.asarray(info["race_rank"])) == [2.0, 0.0, 4.0, 1.0, 3.0]  # 0 = winner
    # Boat 0, third of five: +1 for finishing, 0 for the middle place (winner +1 ... last -1).
    assert float(r[0]) == pytest.approx(P.race_finish_reward, abs=0.05)


def test_mark_room_inside_overlapped_boat_is_entitled():
    w = P.course_beat_length
    # Two port-tack boats overlapped, heading NE toward the windward mark. The inside boat A (closer to
    # the mark) is to windward, so without mark-room it would have to keep clear (RRS 11).
    h = 45.0
    fx, fy = math.sin(deg(h)), math.cos(deg(h))
    port = (-fy, fx)  # left of the heading = windward side on port tack
    bx, by = -6.0, w - 16.0
    ax_, ay_ = bx + 4.0 * port[0], by + 4.0 * port[1]
    fs = race_state([(ax_, ay_, h, 6.0), (bx, by, h, 6.0)])
    _, fs2, _, _, _ = STEP(fs, hold())
    assert bool(fs2.in_zone[0]) and bool(fs2.in_zone[1])
    assert bool(fs2.room[1, 0]) and not bool(fs2.room[0, 1])  # B (outside) owes A room
    k = np.asarray(ENV.keep_clear_matrix(fs2, fs2.boats.boat, 0.0, P))
    assert k[1, 0] and not k[0, 1]  # ... so B keeps clear, although A is to windward


def test_penalty_turn_waits_until_clear():
    # Boat 0 owes a penalty with boat 1 alongside: it keeps control (no forced turn) until clear.
    fs = race_state([(0.0, 200.0, 45.0, 6.0), (6.0, 200.0, 45.0, 6.0)])
    fs = fs.replace(penalty_left=fs.penalty_left.at[0].set(2 * math.pi))
    _, fs2, _, _, _ = STEP(fs, hold())
    assert not bool(fs2.penalty_turning[0]) and float(fs2.penalty_wait[0]) == pytest.approx(P.dt)
    assert float(fs2.penalty_left[0]) == pytest.approx(2 * math.pi)
    # Clear of everything: the turn starts.
    b = fs2.boats.boat
    fs3 = fs2.replace(boats=fs2.boats.replace(boat=b.replace(x=b.x.at[1].set(-150.0))))
    _, fs4, _, _, _ = STEP(fs3, hold())
    assert bool(fs4.penalty_turning[0]) and float(fs4.penalty_left[0]) < 2 * math.pi
    # Still not clear after the timeout: it turns anyway.
    fs5 = fs.replace(penalty_wait=fs.penalty_wait.at[0].set(P.penalty_clear_timeout))
    _, fs6, _, _, _ = STEP(fs5, hold())
    assert bool(fs6.penalty_turning[0])


def test_contact_while_turning_adds_no_new_penalty():
    w = P.course_beat_length
    # Boat 0 is turning its circle right next to the windward mark and touches it.
    fs = race_state([(0.0, w - 2.0, 90.0, 3.0)])  # touching the mark
    fs = fs.replace(penalty_left=fs.penalty_left.at[0].set(math.pi), penalty_turning=fs.penalty_turning.at[0].set(True))
    hit = False
    for _ in range(4):
        _, fs, _, _, _ = STEP(fs, hold())
        hit = hit or int(fs.mark_hits[0]) > 0
    assert hit and int(fs.fouls[0]) == 0


def test_digital_racer_sails_the_whole_race():
    env, p = se.make("race-solo")
    pol = load_agent("digital")(env, p)
    run = jax.jit(jax.vmap(lambda k: run_episode(env, p, k, [pol])))
    record, info = run(jax.random.split(jax.random.PRNGKey(0), 4))
    assert bool(jnp.all(info["race_finished"])) and bool(jnp.all(info["legs_completed"] == N_LEGS))
    assert bool(jnp.all((info["finish_time"] > 300.0) & (info["finish_time"] < p.t_end)))
    assert record["marks"].shape == (4, p.n_steps + 1, 3, 2) and record["leg"].shape == (4, p.n_steps + 1, 1)


def test_bundled_rl_policy_is_rejected_in_the_race():
    with pytest.raises(ValueError):
        load_agent("rl")(ENV, P0)


def test_race_run_save_replay_and_evaluate(tmp_path):
    env, p = se.make("race")
    pol = load_agent("digital")(env, p)
    record, info = jax.jit(lambda k: run_episode(env, p, k, [load_agent("random")(env, p), pol]))(jax.random.PRNGKey(2))
    path = save_run(tmp_path, record, info, p, meta={"mode": "race-vs", "agent": "random", "opponents": "digital"})
    run = load_run(path)
    assert {"race_finished", "finish_time", "race_rank", "legs_completed"} <= set(run["outcome"])
    board = (tmp_path / "leaderboard.csv").read_text(encoding="utf-8").splitlines()
    assert "finish_time" in board[0] and len(board) == 2
    render_replay(run, tmp_path / "race.gif", stride=60)
    assert (tmp_path / "race.gif").stat().st_size > 1000
    s = evaluate("digital", mode="race-vs", episodes=4)
    assert s["opponents"] == "digital"
    assert {"race_finished", "finish_time", "race_rank", "legs_completed", "fouls"} <= set(s["metrics"])
    assert "good_start_on_line" not in s["metrics"]


def test_play_session_race(tmp_path):
    session = PlaySession(mode="race-solo", player="tester", seed=3)
    assert session.race and session.opponents_name == ""
    session.press("up")
    session.tick()
    assert "leg: start" in session.hud() and session.last_action == KEYS["up"]
    while not session.tick():
        pass
    r = session.result()
    assert "race_finished" in r and "legs_completed" in r
    path = session.save(tmp_path)
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["meta"]["mode"] == "race-solo" and "marks" in np.load(path)


def test_race_time_cost_and_potential_shaping():
    """Discounted, the shaping adds up to -Phi(start) whatever the boat does (so it cannot change the best
    policy), and the time cost is paid every second until the finish."""
    from sail_env.agents import digital_sailor_agent
    from sail_env.race_env import rounding_to_go

    g = 0.997
    env, p = se.make("race-solo", race_time_cost=0.01, race_shaping=1.0, shaping_gamma=g, race_finish_reward=0.0)
    env0, p0 = se.make("race-solo", race_finish_reward=0.0)
    pol = digital_sailor_agent(env, p)
    obs, s = env.reset(jax.random.PRNGKey(3), p)
    k = env.course(s, p)
    b = s.boats.boat
    bx, by = b.x + 0.5 * p.boat_length * jnp.sin(b.heading), b.y + 0.5 * p.boat_length * jnp.cos(b.heading)
    phi0 = -float(rounding_to_go(s.leg[0], bx[0], by[0], k, 2.0 * p.boat_length)) / 100.0
    step = jax.jit(lambda s, a: env.step_env(jax.random.PRNGKey(0), s, a, p))
    step0 = jax.jit(lambda s, a: env0.step_env(jax.random.PRNGKey(0), s, a, p0))
    act = jax.jit(lambda o, s: jax.vmap(pol, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(0), o, s.boats, jnp.arange(1)))
    shaped, racing_steps, disc = 0.0, 0, 1.0
    for _ in range(p.n_steps):
        a = act(obs, s)
        _, _, r0, _, _ = step0(s, a)  # the same step without the new terms
        racing = int(s.leg[0] < 4)
        obs, s, r, done, info = step(s, a)
        shaped += disc * (float(r[0]) - float(r0[0]) + 0.01 * racing)  # the shaping part of the reward
        racing_steps += racing
        disc *= g
        if bool(done[0]):
            break
    assert bool(info["race_finished"][0])
    assert shaped == pytest.approx(-phi0, rel=1e-3)
    assert racing_steps == int(np.ceil(float(info["finish_time"][0]) - p.t_start))


def test_rounding_route_distance():
    """The shaping's distance to go: on a beat, being level with the windward mark on its left (the
    wrong side for a port rounding) is further to go than being level with it on its right."""
    from sail_env.race_env import rounding_to_go

    _, fs = ENV.reset(jax.random.PRNGKey(0), P)
    k = ENV.course(fs, P)
    (wx, wy), (rx, ry), (ux, uy) = k["w"], k["r"], k["u"]
    at = lambda right, up: (wx + right * rx + up * ux, wy + right * ry + up * uy)  # noqa: E731
    d = lambda p: float(rounding_to_go(jnp.int32(0), *p, k, 13.8))  # noqa: E731
    assert d(at(-20.0, 0.0)) > d(at(20.0, 0.0)) + 20.0  # left of the mark: has to go round
    assert d(at(20.0, -100.0)) < d(at(20.0, -200.0))  # further down the beat: further to go
    assert d(at(-20.0, 30.0)) > d(at(20.0, 30.0))  # overstood on the left: still has to come round


@pytest.mark.parametrize("legs", [1, 2])
def test_shorter_races_finish_at_the_last_mark(legs):
    """race_legs = 1: rounding the windward mark finishes the race; 2: passing the gate does."""
    from sail_env.agents import digital_sailor_agent
    from sail_env.runs import run_episode

    env, p = se.make("race-solo", race_legs=legs, race_finish_reward=1.0)
    pol = digital_sailor_agent(env, p)
    rec, info = jax.jit(lambda k: run_episode(env, p, k, [pol]))(jax.random.PRNGKey(2))
    assert bool(info["race_finished"][0]) and int(info["legs_completed"][0]) == legs
    full_env, full_p = se.make("race-solo")
    _, full_info = jax.jit(lambda k: run_episode(full_env, full_p, k, [digital_sailor_agent(full_env, full_p)]))(
        jax.random.PRNGKey(2))
    assert float(info["finish_time"][0]) < float(full_info["finish_time"][0])  # a shorter race


def test_a_boat_racing_alone_is_not_ranked():
    """Solo: finishing pays race_finish_reward, with no ranking penalty (it used to be -1 for 'last of 1')."""
    from sail_env.agents import digital_sailor_agent

    env, p = se.make("race-solo", race_legs=1, race_finish_reward=1.0)
    pol = digital_sailor_agent(env, p)
    obs, s = env.reset(jax.random.PRNGKey(2), p)
    step = jax.jit(lambda s, a: env.step_env(jax.random.PRNGKey(0), s, a, p))
    act = jax.jit(lambda o, s: jax.vmap(pol, in_axes=(None, 0, 0, 0))(jax.random.PRNGKey(0), o, s.boats, jnp.arange(1)))
    for _ in range(p.n_steps):
        obs, s, r, done, info = step(s, act(obs, s))
        if bool(done[0]):
            break
    assert bool(info["race_finished"][0]) and float(r[0]) == pytest.approx(1.0, abs=1e-6)
