"""Exported agents (sail_env/agents/export.py): the JSON network is the trained actor, and the kind is right."""

import json

import jax
import jax.numpy as jnp
import numpy as np

import sail_env as se
from sail_env.agents import PRETRAINED
from sail_env.agents.export import export_agent, main
from sail_env.agents.policy import load_policy


def mlp(layers, x):
    for i, layer in enumerate(layers):
        x = x @ np.asarray(layer["W"]) + np.asarray(layer["b"])
        if i < len(layers) - 1:
            x = np.tanh(x)
    return x


def test_exported_racer_gives_the_actors_logits():
    agent = export_agent("racer", seed=1)
    assert agent["format"] == "sail_env-agent" and agent["kind"] == "race" and agent["obs_dim"] == 72
    assert not agent["weather"] and agent["actions"] == 7
    _, _, actor, params, _ = load_policy(PRETRAINED / "racer")
    env, p = se.make("race")
    obs, _ = env.reset(jax.random.PRNGKey(0), p)
    want = actor.apply(jax.tree.map(lambda x: x[1], params["actor"]), obs)
    np.testing.assert_allclose(mlp(agent["layers"], np.asarray(obs, np.float64)), np.asarray(want), atol=1e-3)
    assert bool(jnp.all(jnp.argmax(want, -1) == np.argmax(mlp(agent["layers"], np.asarray(obs)), -1)))


def test_kinds_of_the_bundled_policies():
    assert export_agent("fleet")["kind"] == "start" and export_agent("fleet")["obs_dim"] == 60
    w = export_agent("racer-weather")
    assert w["kind"] == "race" and w["weather"] and w["obs_dim"] == 94


def test_command_line(tmp_path, monkeypatch, capsys):
    out = tmp_path / "agent.json"
    monkeypatch.setattr("sys.argv", ["export", "racer", "-o", str(out), "--name", "My racer"])
    main()
    agent = json.loads(out.read_text(encoding="utf-8"))
    assert agent["name"] == "My racer" and len(agent["layers"]) == 3
    assert "race agent" in capsys.readouterr().out
