"""Full race: start, then a windward-leeward course of 2 laps with a leeward gate, finish downwind.

    env, params = sail_env.make("race")            # 5 boats ("race-solo": 1 boat)
    obs, state = env.reset(key, params)              # obs: (N, obs_dim)
    obs, state, reward, done, info = env.step(key, state, actions, params)

Course (set to the mean wind direction, so wind shifts skew it, as on the water):

    windward mark W   course_beat_length upwind of the start line centre
    leeward gate      two marks gate_offset upwind of the line centre, gate_width_bl boat lengths apart
    finish            the start line, crossed downwind between its ends

Legs, after a legal start: 0 beat to W -> 1 run through the gate -> 2 beat to W -> 3 run to the finish
-> 4 finished. Roundings:
    windward mark   leave it to port: the bow crosses the half-line straight upwind of the mark from right
                    to left (seen looking upwind), then comes back down past the mark's level on its left
                    side: the leg counts then (FleetState.mark_through records the first part; crossing the
                    half-line back to the right undoes it). From the first part on, the observation shows
                    the next leg.
    gate            pass between the gate marks going downwind, then round either mark: the leg counts when
                    the boat comes back up across the gate line outside a mark (FleetState.gate_through
                    records the pass; coming back up between the marks undoes it)
    finish          cross the start line downwind between the pin and the committee boat

Everything from the fleet start applies all race long: physics, wind shadow, right of way, contacts,
marks as obstacles (touching one = penalty), penalty turns. Mark-room (simplified RRS 18): when the
first of two boats rounding the same mark reaches the zone (zone_bl boat lengths), an inside boat
that is overlapped, or a boat clear ahead, is entitled to room; while that lasts, the boat owing room
must keep clear (it is at fault in a contact). With EnvParams.rrs18_opposite_tacks, mark-room does not apply
between boats on opposite tacks at the windward mark (RRS 18.1(a)), and a boat tacking onto the other's tack
in the zone owes her room (simplified RRS 18.3). Finished boats leave the course.

Reward (all optional, set in EnvParams): race_finish_reward on finishing, a zero-sum ranking by
finishing order at the end (race_rank_reward; unfinished boats are ranked by legs done and distance to
go), race_progress_reward per 100 m of progress, plus the fleet's contact and penalty costs. The
episode ends when every boat has finished or at t_end.

Team racing (team_size > 0, e.g. make("race", fleet_size=4, team_size=2) for 2v2): boats 0..team_size-1
against the rest. The team with the lower sum of places wins, a tie goes to the team without first place
(RRS Appendix D), and race_rank_reward pays the team result (+x/2 to every winner, -x/2 to every loser)
instead of the individual place. The observation gets TEAM_FEATURES more inputs (after the race features,
before any weather block): for each neighbour slot, 1 if that boat is a teammate.

Observation: the fleet start observation, then RACE_FEATURES numbers: current leg (one-hot over 5:
beat 1, run 1, beat 2, run 2, finished), the next mark(s) in the boat's frame (forward, starboard; /100
m; on the run through the gate both gate marks, otherwise the next mark and zeros), race clock / 600 s,
penalty circles still owed, penalty turn under way (0/1).

Penalties (race presets): RRS 44.2, get clear first. The penalised boat keeps control and its normal
rights until no boat and no mark is within penalty_clear_distance (or penalty_clear_timeout has
passed); then it turns the circle, keeping clear of everyone. A contact during the circle costs reward
but adds no further penalty.
"""

import jax
import jax.numpy as jnp

from sail_env.fleet_env import FleetStartEnv, FleetState
from sail_env.fleet_rules import clear_astern
from sail_env.geometry import heading_vector, signed_twa
from sail_env.physics import bow_position
from sail_env.rules import crossing
from sail_env.start_env import N_NEIGHBOURS, NEIGHBOUR_FEATURES, OWN_FEATURES
from sail_env.weather import WEATHER_FEATURES

RACE_FEATURES = 12
TEAM_FEATURES = 5  # team racing: teammate flag per neighbour slot (N_NEIGHBOURS)


def team_block_start(params, obs_dim):
    """Index of the first teammate flag in a team-race observation: after the race features, before the
    weather block (if observe_weather)."""
    return obs_dim - TEAM_FEATURES - (WEATHER_FEATURES if params.observe_weather else 0)
N_LEGS = 4
PARK_X = 1.0e5  # finished boats are moved this far away so they no longer interact
LEG_NAMES = ["beat 1", "run 1", "beat 2", "run 2", "finished"]


def team_result(rank, team_size):
    """(N,) bool team win and (N,) team score for two teams (boats 0..team_size-1, then the rest), from the
    individual ranks (0 = winner, ties 0.5). Lower sum of places wins; a tie goes to the team that did not
    have first place (RRS Appendix D)."""
    n = rank.shape[0]
    team = jnp.arange(n) // team_size
    place = rank + 1.0
    score = jnp.stack([jnp.sum(jnp.where(team == t, place, 0.0)) for t in (0, 1)])
    first_team = team[jnp.argmin(rank)]
    win0 = (score[0] < score[1]) | ((score[0] == score[1]) & (first_team == 1))
    win = jnp.where(team == 0, win0, ~win0)
    return win, score[team]


def course_geometry(line, base_dir, params):
    """Dict of course points for a line (pin/committee) and mean wind direction (base_dir)."""
    ux, uy = jnp.sin(base_dir), jnp.cos(base_dir)  # upwind
    rx, ry = jnp.cos(base_dir), -jnp.sin(base_dir)  # right, looking upwind
    cx, cy = 0.5 * (line.pin_x + line.cb_x), 0.5 * (line.pin_y + line.cb_y)
    wx, wy = cx + params.course_beat_length * ux, cy + params.course_beat_length * uy
    gx, gy = cx + params.gate_offset * ux, cy + params.gate_offset * uy
    hw = 0.5 * params.gate_width_bl * params.boat_length
    return {
        "u": (ux, uy), "r": (rx, ry), "c": (cx, cy), "w": (wx, wy), "g": (gx, gy), "hw": hw,
        "g1": (gx - hw * rx, gy - hw * ry), "g2": (gx + hw * rx, gy + hw * ry),
    }


def _dist(ax, ay, bx, by):
    return jnp.hypot(ax - bx, ay - by)


GATE_ROUND_OFFSET = 10.0  # m: the run ends at a point this far outside a gate mark (rounding it)


def gate_round_points(k):
    """Rounding points of the two gate marks: GATE_ROUND_OFFSET m outside each mark, on the gate line."""
    (rx, ry), off = k["r"], GATE_ROUND_OFFSET
    return (k["g1"][0] - off * rx, k["g1"][1] - off * ry), (k["g2"][0] + off * rx, k["g2"][1] + off * ry)


def distance_to_go(leg, px, py, k, legs=N_LEGS, through=False):
    """Distance still to sail (m) from point p in leg `leg` (array-friendly), in a race of `legs` legs
    (static): to the next mark, then the remaining legs mark to mark. The run ends by rounding a gate mark:
    to the gate's centre, out to the rounding point of a mark (GATE_ROUND_OFFSET m outside it), then up to the
    windward mark; once through the gate (`through`, FleetState.gate_through) straight to the nearer rounding point."""
    (wx, wy), (gx, gy), (cx, cy) = k["w"], k["g"], k["c"]
    (p1x, p1y), (p2x, p2y) = gate_round_points(k)
    w_g, w_c = _dist(wx, wy, gx, gy), _dist(wx, wy, cx, cy)
    g_p, p_w = _dist(gx, gy, p1x, p1y), _dist(p1x, p1y, wx, wy)  # the same for both marks (symmetric)
    through_d = jnp.minimum(_dist(px, py, p1x, p1y), _dist(px, py, p2x, p2y))
    to_round = jnp.where(through, through_d, _dist(px, py, gx, gy) + g_p)  # leg 1: to a rounding point
    after = [w_g + g_p, p_w, w_c]  # lengths of legs 1, 2, 3 (mark to mark; leg 1 ends at a rounding point)
    rest = [sum(after[j - 1] for j in range(i + 1, legs)) if i + 1 < legs else 0.0 for i in range(N_LEGS)]
    to_mark = [_dist(px, py, wx, wy), to_round, _dist(px, py, wx, wy), _dist(px, py, cx, cy)]
    return jnp.select([leg == i for i in range(legs)], [to_mark[i] + rest[i] for i in range(legs)], 0.0)


def rounding_to_go(leg, px, py, k, offset, legs=N_LEGS, through=False, past=False):
    """Distance still to sail (m) that follows the route a port rounding needs (for the reward shaping).

    distance_to_go measures straight to the windward mark. But the mark only counts when the boat crosses
    the line straight upwind of it from right to left (looking upwind), so a boat level with the mark on
    its left side is "close" and yet has to go round. On a beat this distance goes via the mark's
    correct side instead:
        right of the mark:  to Q = `offset` m straight upwind of the mark (reaching it from the right
                            is the rounding)
        left of the mark:   to S = `offset` m to the right of the mark, then on to Q
    Passing the mark on the wrong side makes it jump up; getting round to the right side, down. Once past
    the half-line (`past`, FleetState.mark_through) and on other legs: as distance_to_go."""
    (wx, wy), (ux, uy), (rx, ry) = k["w"], k["u"], k["r"]
    qx, qy = wx + offset * ux, wy + offset * uy
    sx, sy = wx + offset * rx, wy + offset * ry
    right = (px - wx) * rx + (py - wy) * ry > 0.0
    to_mark = jnp.where(right, _dist(px, py, qx, qy), _dist(px, py, sx, sy) + _dist(sx, sy, qx, qy)) + offset
    d = distance_to_go(leg, px, py, k, legs, through)
    beat = ((leg == 0) | (leg == 2)) & (leg < legs) & jnp.logical_not(past)
    return jnp.where(beat, d - _dist(px, py, *k["w"]) + to_mark, d)


class RaceEnv(FleetStartEnv):
    def __init__(self, params):
        super().__init__(params)
        self.n_marks = 5  # pin, committee boat, windward mark, gate mark (left), gate mark (right)
        self.obs_dim = self.obs_dim + RACE_FEATURES
        if params.team_size:  # static
            if params.fleet_size != 2 * params.team_size:
                raise ValueError(f"team racing is two teams: fleet_size ({params.fleet_size}) must be 2 x team_size "
                                 f"({params.team_size})")
            self.obs_dim = self.obs_dim + TEAM_FEATURES

    def course(self, fs: FleetState, params):
        line = jax.tree.map(lambda x: x[0], fs.boats.line)
        return course_geometry(line, fs.boats.wind.base_dir[0], params)

    # --- hooks -----------------------------------------------------------------------------------

    def marks(self, fs: FleetState, params):
        centres, radii = super().marks(fs, params)
        k = self.course(fs, params)
        course = jnp.stack([jnp.stack(k["w"]), jnp.stack(k["g1"]), jnp.stack(k["g2"])])
        return jnp.concatenate([centres, course]), jnp.concatenate([radii, jnp.full(3, params.course_mark_radius)])

    def keep_clear_matrix(self, fs: FleetState, boats, wind_dir, params):
        base = super().keep_clear_matrix(fs, boats, wind_dir, params)
        n = base.shape[0]
        # Mark-room: the boat owing room keeps clear; the boat entitled to room does not.
        k = jnp.where(fs.room, True, jnp.where(fs.room.T, False, base))
        return k & ~jnp.eye(n, dtype=bool)

    def post_step(self, fs_prev: FleetState, fs: FleetState, reward, done, info, params):
        n = self.num_agents
        k = self.course(fs, params)
        (ux, uy), (rx, ry) = k["u"], k["r"]
        (wx, wy), (gx, gy), hw = k["w"], k["g"], k["hw"]
        bpx, bpy = bow_position(fs_prev.boats.boat, params)
        bnx, bny = bow_position(fs.boats.boat, params)
        leg = fs_prev.leg
        started = fs.boats.status.started
        t0, t1 = fs_prev.boats.t[0], fs.boats.t[0]
        line = jax.tree.map(lambda x: x[0], fs.boats.line)

        # Windward mark to port: across the half-line straight upwind of it, right -> left, then back down
        # past the mark's level on its left side. Back across the half-line, left -> right, undoes the first part.
        beat_leg = ((leg == 0) | (leg == 2)) & started
        s0 = (bpx - wx) * rx + (bpy - wy) * ry
        s1 = (bnx - wx) * rx + (bny - wy) * ry
        f = s0 / jnp.where(s0 != s1, s0 - s1, 1.0)
        along = (bpx + f * (bnx - bpx) - wx) * ux + (bpy + f * (bny - bpy) - wy) * uy
        over = beat_leg & (s0 > 0.0) & (s1 <= 0.0) & (along > 0.0)
        back = beat_leg & (s0 <= 0.0) & (s1 > 0.0) & (along > 0.0)
        a0 = (bpx - wx) * ux + (bpy - wy) * uy
        a1 = (bnx - wx) * ux + (bny - wy) * uy
        fa = a0 / jnp.where(a0 != a1, a0 - a1, 1.0)
        side = (bpx + fa * (bnx - bpx) - wx) * rx + (bpy + fa * (bny - bpy) - wy) * ry
        down_left = beat_leg & (a0 > 0.0) & (a1 <= 0.0) & (side < 0.0)
        through_w = (fs_prev.mark_through | over) & ~back
        round_w = through_w & down_left
        mark_through = through_w & ~round_w
        # Gate: pass between its marks going downwind.
        g0 = (bpx - gx) * ux + (bpy - gy) * uy
        g1 = (bnx - gx) * ux + (bny - gy) * uy
        fg = g0 / jnp.where(g0 != g1, g0 - g1, 1.0)
        lat = (bpx + fg * (bnx - bpx) - gx) * rx + (bpy + fg * (bny - bpy) - gy) * ry
        # Gate: through it going downwind between the marks, then round either mark: back up across the
        # gate line outside a mark. Back up between the marks undoes the pass.
        down_through = (leg == 1) & (g0 > 0.0) & (g1 <= 0.0) & (jnp.abs(lat) <= hw)
        back_up = (leg == 1) & (g0 <= 0.0) & (g1 > 0.0)
        pass_gate = fs_prev.gate_through & back_up & (jnp.abs(lat) > hw)
        gate_through = (fs_prev.gate_through | down_through) & ~(back_up & (jnp.abs(lat) <= hw)) & ~pass_gate
        # Finish: cross the start line downwind between its ends.
        _, down, ff = jax.vmap(crossing, in_axes=(None, 0, 0, 0, 0))(line, bpx, bpy, bnx, bny)
        legs = params.race_legs  # static: 4 = the full race
        if legs == N_LEGS:
            finish = (leg == 3) & down
            advance = round_w | pass_gate | finish
            leg_new = leg + advance.astype(jnp.int32)
            finish_time = jnp.where(finish, t0 + ff * (t1 - t0), fs_prev.finish_time)
        else:  # a shorter race (training in stages): completing leg `legs` finishes it
            advance = (round_w | pass_gate) & (leg < legs)
            leg_new = leg + advance.astype(jnp.int32)
            finish = advance & (leg_new >= legs)
            frac = jnp.where(leg == 1, fg, fa)  # when, within the second, the gate or the mark was passed
            finish_time = jnp.where(finish, t0 + frac * (t1 - t0), fs_prev.finish_time)
        finished = leg_new >= legs
        was_finished = leg >= legs

        # Rewards: progress (optional), finishing.
        progress = (distance_to_go(leg, bpx, bpy, k, legs, fs_prev.gate_through)
                    - distance_to_go(leg_new, bnx, bny, k, legs, gate_through))
        offset = 2.0 * params.boat_length  # rounding route: 2 boat lengths off the windward mark
        to_go_before = rounding_to_go(leg, bpx, bpy, k, offset, legs, fs_prev.gate_through, fs_prev.mark_through)
        to_go_after = jnp.where(finished, 0.0, rounding_to_go(leg_new, bnx, bny, k, offset, legs, gate_through,
                                                              mark_through))
        reward = reward + jnp.where(was_finished, 0.0, params.race_progress_reward * progress / 100.0)
        reward = reward + params.race_finish_reward * finish
        # Learning to race (both 0 in the presets): a cost per second still racing, and potential-based
        # shaping on the distance still to sail by the rounding route, Phi = -to_go / 100 m (Phi = 0 once
        # finished; the episode's last state counts as terminal).
        racing = ~was_finished
        reward = reward - params.race_time_cost * params.dt * racing
        phi0, phi1 = -to_go_before / 100.0, -to_go_after / 100.0
        episode_over = done | jnp.all(finished)
        shaping = params.shaping_gamma * phi1 * (1.0 - episode_over) - phi0
        reward = reward + params.race_shaping * jnp.where(racing, shaping, 0.0)

        # Mark-room zones (simplified RRS 18) for the mark each boat is rounding.
        g_left_nearer = _dist(bnx, bny, *k["g1"]) < _dist(bnx, bny, *k["g2"])
        target = jnp.select([((leg_new == 0) | (leg_new == 2)) & started, (leg_new == 1) & g_left_nearer, leg_new == 1],
                            [2, 3, 4], -1)
        tx = jnp.select([target == 2, target == 3, target == 4], [wx, k["g1"][0], k["g2"][0]], 0.0)
        ty = jnp.select([target == 2, target == 3, target == 4], [wy, k["g1"][1], k["g2"][1]], 0.0)
        d_t = _dist(bnx, bny, tx, ty)
        in_zone = (target >= 0) & (d_t < params.zone_bl * params.boat_length) & ~finished
        same = (target[:, None] == target[None, :]) & (target[:, None] >= 0) & ~jnp.eye(n, dtype=bool)
        entered = in_zone & ~fs_prev.in_zone
        first = (entered[:, None] & ~fs_prev.in_zone[None, :]) | (entered[None, :] & ~fs_prev.in_zone[:, None])
        astern = clear_astern(fs.boats.boat, params.boat_length)  # [i, j]: i clear astern of j
        overlapped = ~astern & ~astern.T
        owes = jnp.where(overlapped, d_t[:, None] > d_t[None, :], astern)  # outside boat / boat astern owes room
        room = jnp.where(same & first, owes, fs_prev.room)
        keep = same & (in_zone[:, None] | in_zone[None, :]) & ~advance[:, None] & ~advance[None, :]
        if params.rrs18_opposite_tacks:  # static. RRS 18.1(a): not between opposite tacks at the windward mark
            stbd = signed_twa(fs.boats.boat.heading, self.local_winds(fs.boats, params)[0]) > 0.0
            stbd_prev = signed_twa(fs_prev.boats.boat.heading, self.local_winds(fs_prev.boats, params)[0]) > 0.0
            w_pair = (target[:, None] == 2) & (target[None, :] == 2)
            applies = ~(w_pair & (stbd[:, None] != stbd[None, :]))
            applies_prev = ~(w_pair & (stbd_prev[:, None] != stbd_prev[None, :]))
            # RRS 18.3 (simplified): a boat that passes head to wind in the zone onto the other's tack owes her room.
            became = same & applies & ~applies_prev & (in_zone[:, None] | in_zone[None, :])
            tacked = stbd != stbd_prev
            room = jnp.where(became, tacked[:, None] & ~tacked[None, :], room)
            keep = keep & applies
        room = room & keep

        # Finished boats leave the course.
        b = fs.boats.boat
        park_x = PARK_X + 100.0 * jnp.arange(n)
        b = b.replace(x=jnp.where(finished, park_x, b.x), y=jnp.where(finished, 0.0, b.y),
                      speed=jnp.where(finished, 0.0, b.speed))

        # End of the race: everyone finished, or the time limit; rank by finishing order.
        done = done | jnp.all(finished)
        to_go = distance_to_go(leg_new, bnx, bny, k, legs, gate_through)
        # Compared exactly, pair by pair: finished boats by finish time, then the others by legs done and
        # distance to go. (A single float32 score, 1e7 - finish_time, only resolves whole seconds near 1e7: boats
        # finishing within ~1 s tied and shared places.)
        fi, fj = finished[:, None], finished[None, :]
        ti, tj = finish_time[:, None], finish_time[None, :]
        li, lj = leg_new[:, None], leg_new[None, :]
        gi, gj = to_go[:, None], to_go[None, :]
        ahead = jnp.where(fi & fj, tj < ti, jnp.where(fi | fj, fj, (lj > li) | ((lj == li) & (gj < gi))))  # [i, j]: j ahead of i
        level = jnp.where(fi & fj, tj == ti, ~fi & ~fj & (lj == li) & (gj == gi)) & ~jnp.eye(n, dtype=bool)
        higher = ahead.sum(axis=1)
        ties = level.sum(axis=1)
        rank = higher + 0.5 * ties  # 0 = winner
        if params.team_size:  # static: team racing, the team result instead of the individual place
            team_win, team_score = team_result(rank, params.team_size)
            reward = reward + jnp.where(done, params.race_rank_reward * jnp.where(team_win, 0.5, -0.5), 0.0)
        elif n > 1:  # static: a boat racing alone is not ranked (it used to get -race_rank_reward / 2)
            reward = reward + jnp.where(done, params.race_rank_reward * ((n - 1 - rank) / (n - 1) - 0.5), 0.0)

        info = dict(info)
        info["race_finished"] = finished & done
        info["finish_time"] = jnp.where(done & finished, finish_time, jnp.inf)
        info["race_rank"] = jnp.where(done, rank, 0.0)
        if params.team_size:  # static
            info["team_win"] = team_win & done
            info["team_score"] = jnp.where(done, team_score, 0.0)
        info["legs_completed"] = jnp.where(done, leg_new, 0)
        info["rounded_now"] = advance
        info["started"] = fs.boats.status.started & done
        # The fleet's end-of-episode counters, also when the race ends early.
        info["fouls"] = jnp.where(done, fs.fouls, 0)
        info["contacts"] = jnp.where(done, fs.contacts, 0)
        info["mark_hits"] = jnp.where(done, fs.mark_hits, 0)
        info["unserved_foul"] = (fs.penalty_left > 0.0) & done
        # Order of rounding the first windward mark (boats rounding in the same second share a place).
        first_round = advance & (leg == 0) & (fs.mark1_order < 0)
        mark1_order = jnp.where(first_round, (fs.mark1_order >= 0).sum(), fs.mark1_order)
        info["mark1_place"] = jnp.where(done, mark1_order + 1, 0)  # 0 = never rounded it
        info["shadow_time"] = jnp.where(done, fs.shadow_time, 0.0)  # (again: the race can end here)

        fs = fs.replace(boats=fs.boats.replace(boat=b), leg=leg_new, finish_time=finish_time, in_zone=in_zone, room=room,
                        gate_through=gate_through, mark_through=mark_through,
                        mark1_order=mark1_order)
        return fs, reward, done, info

    # --- observation -----------------------------------------------------------------------------

    def get_obs(self, fs: FleetState, params):
        base = super().get_obs(fs, params)
        n = self.num_agents
        # Neighbours that have finished (parked far away) are not "near": blank their slots.
        lo, hi = OWN_FEATURES, OWN_FEATURES + N_NEIGHBOURS * NEIGHBOUR_FEATURES
        nb = base[:, lo:hi].reshape(n, N_NEIGHBOURS, NEIGHBOUR_FEATURES)
        far = (jnp.abs(nb[..., 1]) > 50.0) | (jnp.abs(nb[..., 2]) > 50.0)
        nb = jnp.where(far[..., None], 0.0, nb)
        base = base.at[:, lo:hi].set(nb.reshape(n, -1))

        k = self.course(fs, params)
        b = fs.boats.boat
        fx, fy = heading_vector(b.heading)
        rx, ry = fy, -fx

        def rel(px, py):
            dx, dy = px - b.x, py - b.y
            return (dx * fx + dy * fy) / 100.0, (dx * rx + dy * ry) / 100.0

        leg = fs.leg + fs.mark_through.astype(jnp.int32)  # past the windward mark's half-line: on to the run
        w_f, w_s = rel(*k["w"])
        g1_f, g1_s = rel(*k["g1"])
        g2_f, g2_s = rel(*k["g2"])
        c_f, c_s = rel(*k["c"])
        beat = (leg == 0) | (leg == 2)
        first_f = jnp.select([beat, leg == 1, leg == 3], [w_f, g1_f, c_f], 0.0)
        first_s = jnp.select([beat, leg == 1, leg == 3], [w_s, g1_s, c_s], 0.0)
        second_f = jnp.where(leg == 1, g2_f, 0.0)
        second_s = jnp.where(leg == 1, g2_s, 0.0)
        onehot = jax.nn.one_hot(jnp.clip(leg, 0, N_LEGS), N_LEGS + 1)
        clock = jnp.broadcast_to(fs.boats.t[0] / 600.0, (n,))
        owed = fs.penalty_left / (2.0 * jnp.pi) if params.penalty_turn else fs.penalty_left / params.penalty_seconds
        turning = fs.penalty_turning.astype(jnp.float32)
        race = jnp.concatenate([onehot, jnp.stack([first_f, first_s, second_f, second_s, clock, owed, turning], axis=1)],
                               axis=1)
        if params.team_size:  # static: one teammate flag per neighbour slot, in the slots' order (nearest first)
            dist = jnp.hypot(b.x[None, :] - b.x[:, None], b.y[None, :] - b.y[:, None])
            dist = dist + jnp.where(jnp.eye(n, dtype=bool), jnp.inf, 0.0)
            kk = min(N_NEIGHBOURS, n - 1)
            order = jnp.argsort(dist, axis=1)[:, :kk]  # as FleetStartEnv.get_obs orders the slots
            team = jnp.arange(n) // params.team_size
            mate = (team[order] == team[:, None]) & ~far[:, :kk]  # a blanked (far) slot is not a teammate
            mate = jnp.concatenate([mate, jnp.zeros((n, N_NEIGHBOURS - kk), bool)], axis=1)
            race = jnp.concatenate([race, mate.astype(jnp.float32)], axis=1)
        if params.observe_weather:  # static: the weather block stays last, after the race features
            return jnp.concatenate([base[:, :-WEATHER_FEATURES], race.astype(jnp.float32), base[:, -WEATHER_FEATURES:]],
                                   axis=1)
        return jnp.concatenate([base, race.astype(jnp.float32)], axis=1)
