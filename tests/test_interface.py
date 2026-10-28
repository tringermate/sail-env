"""The public interface: presets, agents, episode runner, saved runs and replays, evaluation, play logic."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sail_env as se
from sail_env.agents import load_agent
from sail_env.evaluate import evaluate
from sail_env.play import KEYS, PlaySession
from sail_env.runs import load_run, render_replay, run_episode, save_run


def test_presets_match_the_pretrained_policies():
    env, p = se.make("solo")
    assert isinstance(env, se.SoloStartEnv) and p.t_start == -60.0 and not p.random_start_time
    env, p = se.make("fleet")
    assert isinstance(env, se.FleetStartEnv) and env.num_agents == 5 and p.t_start == -90.0
    assert p.line_length == pytest.approx(1.6 * 5 * 6.9)
    assert float(p.shift_amp_max) > 0 and float(p.line_bias_max) > 0 and p.penalty_turn
    with pytest.raises(ValueError):
        se.make("nope")


@pytest.mark.parametrize("spec", ["random", "hold", "digital", "rl"])
def test_builtin_agents_act_in_the_fleet(spec):
    env, p = se.make("fleet")
    policy = load_agent(spec)(env, p)
    obs, fs = env.reset(jax.random.PRNGKey(0), p)
    acts = jax.jit(jax.vmap(policy))(jax.random.split(jax.random.PRNGKey(1), 5), obs, fs.boats, jnp.arange(5))
    assert acts.shape == (5,) and bool(jnp.all((acts >= 0) & (acts < 7)))


def test_custom_agent_from_module_path():
    policy_factory = load_agent("examples.my_agent:make_policy")
    env, p = se.make("solo")
    obs, s = env.reset(jax.random.PRNGKey(0), p)
    a = policy_factory(env, p)(jax.random.PRNGKey(1), obs, s, 0)
    assert 0 <= int(a) < 7


def test_run_save_load_replay(tmp_path):
    env, p = se.make("fleet")
    policies = [load_agent("random")(env, p), load_agent("rl")(env, p)]
    record, info = jax.jit(lambda k: run_episode(env, p, k, policies))(jax.random.PRNGKey(3))
    assert record["x"].shape == (p.n_steps + 1, 5) and record["actions"].shape == (p.n_steps, 5)
    path = save_run(tmp_path, record, info, p, meta={"mode": "fleet-vs", "agent": "random", "opponents": "rl"})
    run = load_run(path)
    assert run["meta"]["agent"] == "random" and len(run["outcome"]["good_start_on_line"]) == 5
    assert run["params"]["fleet_size"] == 5
    board = (tmp_path / "leaderboard.csv").read_text(encoding="utf-8").splitlines()
    assert len(board) == 2 and board[0].startswith("time,run,mode")
    render_replay(run, tmp_path / "replay.gif")
    assert (tmp_path / "replay.gif").stat().st_size > 1000


def test_solo_episode_runner():
    env, p = se.make("solo")
    record, info = run_episode(env, p, jax.random.PRNGKey(0), [load_agent("digital")(env, p)])
    assert record["x"].shape == (p.n_steps + 1, 1)


def test_evaluate_reports_metrics(tmp_path):
    s = evaluate("digital", mode="fleet-vs", opponents="rl", episodes=8, out=tmp_path, save_replays=2)
    assert {"good_start_on_line", "started", "ocs", "rank"} <= set(s["metrics"])
    assert len(list(tmp_path.glob("*.npz"))) == 2 and len(list(tmp_path.glob("eval_*.json"))) == 1
    s_solo = evaluate("rl-solo", mode="solo", episodes=16)
    assert 0.0 <= s_solo["metrics"]["good_start_on_line"]["mean"] <= 1.0


def test_play_session_logic(tmp_path):
    session = PlaySession(mode="fleet", opponents="digital", player="tester", seed=1)
    assert session.press("up") == KEYS["up"] and session.press("x") is None
    session.tick()
    assert session.last_action == KEYS["up"] and session.pending == 0  # one press = one action
    assert "kn" in session.hud()
    steps = 1  # the tick above
    while True:
        steps += 1
        if session.tick():
            break
    assert steps == session.params.n_steps and session.result() is not None
    path = session.save(tmp_path)
    side = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert side["meta"]["player"] == "tester" and side["meta"]["human"]
    assert np.load(path)["x"].shape == (session.params.n_steps + 1, 5)
