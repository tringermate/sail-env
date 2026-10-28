// Parity test: the browser simulator (web/sim.js) against the Python simulator.
//
//   python tools/export_web_data.py --traces web/ref/traces.json
//   node web/check.js web/ref/traces.json
//
// Each reference episode is replayed from the same initial state with Python's actions. Every step
// starts from Python's recorded boats (position, heading, speed, target heading, manoeuvre), so the test
// checks each step of the physics on its own: Python runs in float32 and JavaScript in float64, and in a
// free replay a boat grazing the edge of a wind-shadow cone by a millimetre would tip one way in one and
// the other way in the other, then drift apart for the rest of the race.
// Passes if, at every step and for every boat, the position after the step agrees within 1 cm, the
// discrete state (started, OCS, penalties, legs, fouls) is identical, the JS agents (digital sailor,
// digital racer, RL network) choose the same action as Python in at least 99.9% of decisions (a decision
// sitting exactly on a steering threshold can still tip either way), and the final ranking is the same.
"use strict";
const path = require("path");
const S = require("./sim.js");
const weights = require("./weights.json");
const traces = require(path.resolve(process.argv[2] || "web/ref/traces.json"));

const POS_TOL = 0.01; // m, after one step from the same state
const ACTION_AGREEMENT = 0.999;
let failed = false;

for (const tr of traces) {
  const sim = new S.Sim(tr.preset);
  const ib = tr.init.boats;
  const boats = ib.x.map((_, i) => sim.newBoat(ib.x[i], ib.y[i], ib.heading[i], ib.speed[i], ib.target[i], ib.man[i]));
  const w = tr.init.wind, l = tr.init.line;
  let s = sim.initState({
    boats, wind: { base: w.base_dir, tws: w.tws, amp: w.shift_amp, period: w.shift_period, phase: w.shift_phase },
    line: { px: l.pin_x, py: l.pin_y, cx: l.cb_x, cy: l.cb_y }, bias: tr.init.bias, t: tr.init.t,
  });
  let obsErr = 0;
  if (!sim.p.solo) sim.obs(s).forEach((row, i) => row.forEach((v, k) => { obsErr = Math.max(obsErr, Math.abs(v - tr.obs0[i][k])); }));

  let maxPos = 0, discrete = 0, agree = 0, total = 0, firstBad = null;
  tr.steps.forEach((rec, k) => {
    const mine = S.opponentActions(sim, s, tr.agent.startsWith("rl") ? "rl" : "digital", weights);
    mine.forEach((a, i) => { total++; if (a === rec.a[i]) agree++; });
    s = sim.step(s, rec.a);
    s.boats.forEach((b, i) => {
      maxPos = Math.max(maxPos, Math.hypot(b.x - rec.x[i], b.y - rec.y[i]));
      // continue from Python's boat for the next step
      b.x = rec.x[i]; b.y = rec.y[i]; b.h = rec.heading[i]; b.v = rec.speed[i]; b.th = rec.target[i]; b.man = rec.man[i];
      const bad = b.started !== rec.started[i] || b.ocs !== rec.ocs[i] ||
        (rec.pen && Math.abs(s.penLeft[i] - rec.pen[i]) > 1e-3) || (rec.leg && s.leg[i] !== rec.leg[i]) ||
        (rec.fouls && s.fouls[i] !== rec.fouls[i]);
      if (bad) { discrete++; firstBad = firstBad || { step: k, boat: i }; }
    });
  });
  const pi = tr.steps[tr.steps.length - 1].info || {};
  const pyRank = tr.preset.startsWith("race") ? pi.race_rank : pi.rank;
  const sameRank = s.done && JSON.stringify((s.info.rank || []).map(Number)) === JSON.stringify((pyRank === undefined ? [0] : [].concat(pyRank)).map(Number));
  const ok = maxPos < POS_TOL && discrete === 0 && agree >= ACTION_AGREEMENT * total && sameRank && obsErr < 1e-4;
  failed = failed || !ok;
  console.log(`${ok ? "pass" : "FAIL"}  ${tr.preset.padEnd(9)} ${tr.agent.padEnd(7)} ${String(tr.steps.length).padStart(4)} steps  ` +
    `max one-step position error ${maxPos.toFixed(4)} m  observation error ${obsErr.toExponential(1)}  ` +
    `same action ${agree}/${total}  discrete mismatches ${discrete}  same ranking ${sameRank}` +
    (firstBad ? `  (first mismatch: step ${firstBad.step}, boat ${firstBad.boat})` : ""));
}

if (failed) { console.error("FAILED: the browser simulator differs from the Python simulator"); process.exit(1); }
console.log("OK: the browser simulator matches the Python simulator");
