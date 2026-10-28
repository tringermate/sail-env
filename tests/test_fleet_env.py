import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sail_env.config import EnvParams, deg, kn
from sail_env.fleet_env import FleetStartEnv, rollout_fleet
from sail_env.fleet_rules import contacts
from sail_env.physics import HOLD
from sail_env.start_env import OBS_DIM, OWN_FEATURES
from sail_env.wind import Wind

P = EnvParams(fleet_size=5, line_fleet_size=5, line_length_factor=2.0, t_start=-90.0)
ENV = FleetStartEnv(P)


def test_reset_shapes_line_and_no_overlap():
    obs, fs = jax.jit(lambda k: ENV.reset(k, P))(jax.random.PRNGKey(0))
    assert obs.shape == (5, OBS_DIM)
    assert P.line_length == pytest.approx(2 * 5 * 6.9)
    assert float(jnp.ptp(fs.boats.t)) == 0.0 and float(jnp.ptp(fs.boats.wind.tws)) == 0.0  # shared clock and wind
    for k in range(50):
        _, fs = ENV.reset(jax.random.PRNGKey(k), P)
        assert not bool(jnp.any(contacts(fs.boats.boat, P.boat_length, P.boat_beam)))


def test_neighbour_slots():
    obs, _ = ENV.reset(jax.random.PRNGKey(3), P)
    nb = np.asarray(obs[:, OWN_FEATURES:]).reshape(5, 5, 9)
    assert np.all(nb[:, :4, 0] == 1.0)  # 4 other boats present
    assert np.all(nb[:, 4] == 0.0)  # fifth slot empty (zero-padded)
    d = np.hypot(nb[:, :4, 1], nb[:, :4, 2])
    assert np.all(np.diff(d, axis=1) >= -1e-6)  # nearest first


def two_boat_state(specs, tws_kn=10.0):
    """A 5-boat state with boats 0 and 1 placed by specs [(x, y, heading_deg, speed_kn)] and the
    other three parked far away."""
    _, fs = ENV.reset(jax.random.PRNGKey(0), P)
    x = np.array([s[0] for s in specs] + [300.0, 400.0, 500.0], np.float32)
    y = np.array([s[1] for s in specs] + [-300.0, -300.0, -300.0], np.float32)
    h = np.array([deg(s[2]) for s in specs] + [0.0, 0.0, 0.0], np.float32)
    v = np.array([kn(s[3]) for s in specs] + [0.0, 0.0, 0.0], np.float32)
    boat = fs.boats.boat.replace(x=jnp.asarray(x), y=jnp.asarray(y), heading=jnp.asarray(h), target_heading=jnp.asarray(h),
                                 speed=jnp.asarray(v), manoeuvring=jnp.zeros(5, bool))
    wind = Wind(*(jnp.full(5, v_, jnp.float32) for v_ in (0.0, kn(tws_kn), 0.0, 120.0, 0.0)))
    return fs.replace(boats=fs.boats.replace(boat=boat, wind=wind, t=jnp.full(5, -30.0)))


def test_port_starboard_collision_penalises_port():
    # Starboard boat 0 heading NW, port boat 1 heading NE, on a collision course.
    fs = two_boat_state([(0.0, 0.0, -45.0, 6.0), (-12.0, 0.0, 45.0, 6.0)])
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), P))
    hit = False
    for _ in range(6):
        _, fs, _, _, info = step(fs)
        hit = hit or bool(info["contact_now"][1])
        if hit:
            break
    assert hit and bool(fs.penalty_left[1] > 0) and float(fs.penalty_left[0]) == 0.0
    assert int(fs.fouls[1]) == 1 and int(fs.fouls[0]) == 0


def test_wind_shadow_slows_the_boat_behind():
    # Two boats close-hauled on starboard; boat 1 sits 2 boat lengths straight downwind of boat 0.
    fs = two_boat_state([(0.0, 0.0, -42.0, 5.0), (0.0, -2 * 6.9, -42.0, 5.0)])
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), P))
    _, fs1, _, _, _ = step(fs)
    assert float(fs1.boats.boat.speed[1]) < float(fs1.boats.boat.speed[0])


def test_batched_steps_and_episode_length():
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    obs, fs = jax.vmap(ENV.reset, in_axes=(0, None))(keys, P)
    acts = jnp.zeros((8, 5), jnp.int32)
    obs, fs, r, d, info = jax.jit(jax.vmap(ENV.step, in_axes=(0, 0, 0, None)))(keys, fs, acts, P)
    assert obs.shape == (8, 5, OBS_DIM) and r.shape == (8, 5) and d.shape == (8, 5)
    traj, acts, rewards, info = rollout_fleet(ENV, P, jax.random.PRNGKey(2), lambda k, o: jnp.zeros(5, jnp.int32))
    assert traj.boats.t.shape == (P.n_steps + 1, 5) and float(traj.boats.t.max()) == pytest.approx(10.0)


def test_hitting_the_committee_boat_is_penalised_and_costs():
    p = P.replace(contact_cost=1.0)
    fs = two_boat_state([(0.0, 0.0, 0.0, 5.0), (300.0, 0.0, 0.0, 0.0)])
    cb_x, cb_y = float(fs.boats.line.cb_x[0]), float(fs.boats.line.cb_y[0])
    # Boat 0 sails straight north into the committee boat from 8 m below it.
    boat = fs.boats.boat
    fs = fs.replace(boats=fs.boats.replace(boat=boat.replace(x=boat.x.at[0].set(cb_x), y=boat.y.at[0].set(cb_y - 10.0))))
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), p))
    total, hit = 0.0, False
    for _ in range(4):
        _, fs, r, _, info = step(fs)
        total += float(r[0])
        hit = hit or bool(info["contact_now"][0])
    assert hit and int(fs.mark_hits[0]) == 1 and float(fs.penalty_left[0]) > 0
    assert total <= -1.0 + 1e-6  # contact cost


def test_right_of_way_boat_also_pays_for_contact_but_serves_no_penalty():
    p = P.replace(contact_cost=1.0)
    fs = two_boat_state([(0.0, 0.0, -45.0, 6.0), (-12.0, 0.0, 45.0, 6.0)])  # stbd 0, port 1
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), p))
    for _ in range(6):
        _, fs, r, _, info = step(fs)
        if bool(info["contact_now"][1]):
            break
    assert int(fs.contacts[0]) == 1 and int(fs.contacts[1]) == 1  # both pay contact_cost ...
    assert float(fs.penalty_left[0]) == 0.0 and float(fs.penalty_left[1]) > 0  # ... only port serves


def test_passing_a_mark_at_a_distance_is_fine():
    fs = two_boat_state([(0.0, 0.0, 0.0, 5.0), (300.0, 0.0, 0.0, 0.0)])
    cb_x, cb_y = float(fs.boats.line.cb_x[0]), float(fs.boats.line.cb_y[0])
    boat = fs.boats.boat
    fs = fs.replace(boats=fs.boats.replace(boat=boat.replace(x=boat.x.at[0].set(cb_x + 12.0), y=boat.y.at[0].set(cb_y - 10.0))))
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), P))
    for _ in range(6):
        _, fs, _, _, _ = step(fs)
    assert int(fs.mark_hits[0]) == 0


def test_rank_reward_is_zero_sum_and_orders_by_height():
    p = P.replace(rank_reward=1.0)
    _, fs = ENV.reset(jax.random.PRNGKey(0), p)
    st = fs.boats
    # At T+9: boats 0, 1, 2 started legally at 30, 20, 10 m above the line; boats 3, 4 never started.
    started = jnp.array([True, True, True, False, False])
    heights = jnp.array([30.0, 20.0, 10.0, -20.0, -40.0])
    boat = st.boat.replace(x=jnp.linspace(-30, 30, 5), y=heights - 3.45, heading=jnp.zeros(5), target_heading=jnp.zeros(5),
                           speed=jnp.zeros(5), manoeuvring=jnp.zeros(5, bool))
    status = st.status.replace(started=started, ocs=jnp.zeros(5, bool), start_time=jnp.where(started, 1.0, jnp.inf))
    fs = fs.replace(boats=st.replace(boat=boat, status=status, t=jnp.full(5, 9.0)))
    p_rank = p.replace(contact_cost=0.0, foul_penalty=0.0)
    _, _, r_rank, done, info = ENV.step_env(jax.random.PRNGKey(0), fs, jnp.full(5, HOLD), p_rank)
    _, _, r_base, _, _ = ENV.step_env(jax.random.PRNGKey(0), fs, jnp.full(5, HOLD), p_rank.replace(rank_reward=0.0))
    assert bool(done[0])
    np.testing.assert_allclose(np.asarray(info["rank"]), [0, 1, 2, 3.5, 3.5])
    diff = np.asarray(r_rank - r_base)
    np.testing.assert_allclose(diff, [0.5, 0.25, 0.0, -0.375, -0.375], atol=1e-5)
    assert abs(diff.sum()) < 1e-5  # zero-sum


def test_right_of_way_boat_pays_reduced_contact_cost():
    p = P.replace(contact_cost=2.0, contact_cost_row=0.5, foul_penalty=0.0)
    fs = two_boat_state([(0.0, 0.0, -45.0, 6.0), (-12.0, 0.0, 45.0, 6.0)])  # stbd 0 (right of way), port 1
    p0 = p.replace(contact_cost=0.0, contact_cost_row=0.0)
    step = jax.jit(lambda s, pp: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), pp))
    for _ in range(6):
        _, fs_next, r, _, info = step(fs, p)
        _, _, r0, _, _ = step(fs, p0)
        if bool(info["contact_now"][1]):
            break
        fs = fs_next
    cost = np.asarray(r0 - r)
    assert cost[0] == pytest.approx(0.5) and cost[1] == pytest.approx(2.0)


def test_penalty_is_a_full_circle_and_the_turning_boat_keeps_clear():
    import math

    p = P.replace(penalty_turn=True, penalty_turn_rate=deg(30.0))
    fs = two_boat_state([(0.0, 0.0, -45.0, 6.0), (-12.0, 0.0, 45.0, 6.0)])  # stbd 0, port 1: port fouls
    step = jax.jit(lambda s: ENV.step_env(jax.random.PRNGKey(0), s, jnp.full(5, HOLD), p))
    for _ in range(6):
        _, fs, _, _, info = step(fs)
        if bool(info["contact_now"][1]):
            break
    assert float(fs.penalty_left[1]) == pytest.approx(2 * math.pi, abs=1e-5)
    # Separate the boats, then let boat 1 turn: 12 decisions at 30 deg/s = 360 deg.
    b = fs.boats.boat
    fs = fs.replace(boats=fs.boats.replace(boat=b.replace(x=b.x.at[0].set(200.0))), in_contact=jnp.zeros((5, 5), bool))
    total = 0.0
    prev = float(fs.boats.boat.heading[1])
    for k in range(12):
        _, fs, _, _, _ = step(fs)
        h = float(fs.boats.boat.heading[1])
        total += abs((h - prev + math.pi) % (2 * math.pi) - math.pi)
        prev = h
    assert total == pytest.approx(2 * math.pi, abs=0.05) and float(fs.penalty_left[1]) == pytest.approx(0.0, abs=1e-5)
    assert float(fs.boats.boat.speed[1]) < kn(6.0)  # the circle costs speed


def test_boat_taking_a_penalty_must_keep_clear():
    from sail_env.fleet_rules import must_keep_clear

    p = P.replace(penalty_turn=True)
    fs = two_boat_state([(0.0, 0.0, -45.0, 6.0), (-12.0, 0.0, 45.0, 6.0)])  # boat 1 on port
    fs = fs.replace(penalty_left=fs.penalty_left.at[0].set(1.0))  # the starboard boat is taking a penalty
    obs = ENV.get_obs(fs, p)
    b = fs.boats.boat.replace(manoeuvring=fs.penalty_left > 0)
    k = np.asarray(must_keep_clear(b, 0.0, P.boat_length))
    assert k[0, 1] and not k[1, 0]  # starboard boat loses its right of way while taking the penalty


def test_rank_uses_height_along_the_wind_axis_with_a_shift():
    import math

    # Wind shifted 20 deg right (from 020): boat 1 is further upwind along the new wind axis although
    # boat 0 is further "north".
    p = P.replace(rank_reward=1.0, contact_cost=0.0, foul_penalty=0.0)
    _, fs = ENV.reset(jax.random.PRNGKey(0), p)
    st = fs.boats
    w = math.radians(20.0)
    xs = jnp.array([0.0, 30.0, -30.0, 60.0, -60.0])
    ys = jnp.array([20.0, 15.0, -40.0, -40.0, -40.0])
    boat = st.boat.replace(x=xs, y=ys - 3.45, heading=jnp.full(5, w), target_heading=jnp.full(5, w), speed=jnp.zeros(5),
                           manoeuvring=jnp.zeros(5, bool))
    started = jnp.array([True, True, False, False, False])
    status = st.status.replace(started=started, ocs=jnp.zeros(5, bool), start_time=jnp.where(started, 1.0, jnp.inf))
    wind = st.wind.replace(base_dir=jnp.full(5, w), shift_amp=jnp.zeros(5))
    fs = fs.replace(boats=st.replace(boat=boat, status=status, wind=wind, t=jnp.full(5, 9.0)))
    _, _, _, _, info = ENV.step_env(jax.random.PRNGKey(0), fs, jnp.full(5, HOLD), p)
    h0 = 0.0 * math.sin(w) + 20.0 * math.cos(w)
    h1 = 30.0 * math.sin(w) + 15.0 * math.cos(w)
    assert h1 > h0 and float(info["rank"][1]) == 0.0 and float(info["rank"][0]) == 1.0


def luff_into_windward_boat(rrs16, leeward_action, windward_action):
    """Boats 0 (leeward, right of way) and 1 (windward, 4 m to its starboard side), both close-hauled on
    starboard. Returns (fouls of boat 0, fouls of boat 1) after the first contact."""
    p = P.replace(rrs16=rrs16)
    h = -42.0
    rx, ry = np.cos(np.radians(h)), -np.sin(np.radians(h))  # starboard unit vector of a boat heading h
    fs = two_boat_state([(0.0, 0.0, h, 5.0), (4.0 * rx, 4.0 * ry, h, 5.0)])
    step = jax.jit(lambda s, a: FleetStartEnv(p).step_env(jax.random.PRNGKey(0), s, a, p))
    acts = jnp.full(5, HOLD).at[0].set(leeward_action).at[1].set(windward_action)
    for _ in range(10):
        _, fs, _, _, info = step(fs, acts)
        if bool(info["contact_now"].any()):
            break
    return int(fs.fouls[0]), int(fs.fouls[1])


def test_rrs16_right_of_way_boat_changing_course():
    from sail_env.physics import DOWN15, UP15

    # Leeward boat luffs hard into the windward boat, which holds its course.
    assert luff_into_windward_boat(False, UP15, HOLD) == (0, 1)  # without RRS 16: the windward boat pays
    assert luff_into_windward_boat(True, UP15, HOLD) == (1, 0)  # RRS 16: the luffing boat did not give room
    # The windward (keep-clear) boat bears away into a leeward boat holding its course: its fault either way.
    assert luff_into_windward_boat(False, HOLD, DOWN15) == (0, 1)
    assert luff_into_windward_boat(True, HOLD, DOWN15) == (0, 1)
