# Browser version

`index.html` is a self-contained page: the simulator, the agents and the bundled RL policies (the fleet start
policy and the racer) are inlined. Players can also load their own agent: a JSON file from
`python -m sail_env.agents.export` (see the main README, *In the browser*). The repository's GitHub Pages workflow (`.github/workflows/pages.yml`) can publish it.

| File | Role |
|---|---|
| `sim.js` | A JavaScript port of the simulator: physics, start and fleet rules, race course, digital sailor and racer, and RL network inference |
| `game.template.html` | Interface, rendering and input |
| `weights.json` | The bundled RL policies (`fleet`, `racer`), exported by `tools/export_web_data.py` |
| `build.js` | Inlines the three files above into `index.html` |
| `check.js` | Parity test against the Python simulator; runs in CI |

```sh
node web/build.js                                             # rebuild index.html
python tools/export_web_data.py --traces web/ref/traces.json  # reference episodes from Python
node web/check.js web/ref/traces.json                         # replay them in sim.js
```
