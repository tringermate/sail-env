// sail_env in the browser: a line-by-line port of the Python simulator (sail_env/*.py).
// Physics (polar, speed lag, drag, turning, tacks), start rules (line crossing, OCS), fleet rules
// (right of way, contacts, marks, penalty turns, wind shadow), the full race (course, roundings,
// mark-room, finish), the digital sailor / racer, and the bundled RL fleet policy (MLP).
"use strict";

const KNOT = 0.5144;
const DEG = Math.PI / 180;
const [HOLD, UP5, DOWN5, UP15, DOWN15, TACK, LUFF] = [0, 1, 2, 3, 4, 5, 6];
const ACTION_TURN = [0, 5 * DEG, -5 * DEG, 15 * DEG, -15 * DEG, 0, 0];
const TURN_DONE_TOL = 0.5 * DEG;
const N_LEGS = 4;
const PARK_X = 1e5;
const LEG_NAMES = ["beat 1", "run 1", "beat 2", "run 2", "finished"];

const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
const stwa = (h, w) => wrap(w - h);
const clip = (x, a, b) => Math.min(Math.max(x, a), b);
const sign = (x) => (x > 0 ? 1 : x < 0 ? -1 : 0);

// ---------------------------------------------------------------- parameters (sail_env/presets.py)
function baseParams() {
  return {
    dt: 1, nSub: 4, tStart: -90, tEnd: 10, windDir: 0, twsMin: 6 * KNOT, twsMax: 16 * KNOT,
    shiftAmpMax: 0.0873, shiftPeriod: 120, L: 6.9, B: 2.25, tauAccel: 4, tauDecel: 8, mass: 1265,
    waterK: 37, cda: 5, rhoAir: 1.225, noGo: 30 * DEG, luffFactor: 0.12, maxTurn: 20 * DEG,
    fullSteer: 3 * KNOT, minSteer: 0.5 * KNOT, ironsRate: 5 * DEG, tackLoss: 0, gybeLoss: 0.1,
    n: 5, lineLength: 1.6 * 5 * 6.9, lineBiasMax: 0.1745,
    shadowLenBL: 6, shadowHalf: 12 * DEG, shadowMin: 0.75, shadowApparent: true,
    penaltyTurns: 1, penaltyRate: 0.5236, // preset value (a 360 in 12 s)
    clearDist: 0, clearTimeout: 10,
    pinR: 1, cbR: 4, beat: 400, gateOffset: 60, gateWidthBL: 8, markR: 1, zoneBL: 3,
    initDistMin: 20, initDistMax: 120, reachFrac: 0.75, latMax: 100, latSpeed: 1.5, speedFracMax: 1,
    successWindow: 3, successSpeed: 0.9, gunDistMax: 3,
    solo: false, race: false,
  };
}
function makeParams(mode) {
  const p = baseParams();
  if (mode === "solo") Object.assign(p, { solo: true, n: 1, tStart: -60, lineLength: 1.25 * 15 * 6.9, shiftAmpMax: 0, lineBiasMax: 0 });
  if (mode === "race" || mode === "race-solo") Object.assign(p, { race: true, tEnd: 1200, clearDist: 2 * 6.9 });
  if (mode === "race-solo") p.n = 1;
  p.mode = mode;
  p.nMarks = p.race ? 5 : 2;
  return p;
}

// ---------------------------------------------------------------- polar (sail_env/polar.py)
const ORC = {
  tws: [6, 8, 10, 12, 14, 16, 20, 24],
  angles: [52, 60, 75, 90, 110, 120, 135, 150],
  table: [
    [4.69, 5.58, 6.05, 6.33, 6.47, 6.54, 6.60, 6.57], [4.93, 5.78, 6.21, 6.55, 6.74, 6.85, 6.96, 6.97],
    [5.08, 5.89, 6.36, 6.78, 7.16, 7.39, 7.68, 7.80], [5.38, 6.20, 6.61, 6.88, 7.29, 7.82, 8.54, 8.94],
    [5.45, 6.33, 7.01, 7.58, 8.01, 8.44, 9.29, 10.40], [5.30, 6.22, 6.93, 7.72, 8.50, 9.12, 10.38, 11.76],
    [4.78, 5.84, 6.53, 7.21, 8.07, 9.13, 12.21, 13.90], [4.06, 5.14, 5.92, 6.47, 7.02, 7.78, 10.66, 13.80],
  ],
  beatAngle: [43.40, 41.60, 39.40, 37.60, 37.40, 37.50, 38.50, 40.40],
  beatVmg: [3.08, 3.77, 4.25, 4.52, 4.59, 4.62, 4.59, 4.48],
  runVmg: [3.52, 4.45, 5.12, 5.60, 6.08, 6.74, 9.23, 11.95],
};

function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  const n = xs.length;
  if (x >= xs[n - 1]) return ys[n - 1];
  let i = 0;
  while (x > xs[i + 1]) i++;
  // np.interp: exact knots return their value
  return ys[i] + ((x - xs[i]) * (ys[i + 1] - ys[i])) / (xs[i + 1] - xs[i]);
}

function f32(x) { return Math.fround(x); }

function buildPolar(noGo) {
  const step = 0.25;
  const nTwa = Math.round(180 / step) + 1;
  const twaDeg = Array.from({ length: nTwa }, (_, i) => (180 * i) / (nTwa - 1));
  const cols = ORC.tws.map((_, k) => {
    const beatSpeed = ORC.beatVmg[k] / Math.cos(ORC.beatAngle[k] * DEG);
    const xs = [0, noGo / DEG, ORC.beatAngle[k], ...ORC.angles, 180];
    const ys = [0, 0, beatSpeed, ...ORC.table.map((r) => r[k]), 0.9 * ORC.runVmg[k]];
    return twaDeg.map((a) => interp(a, xs, ys));
  });
  const twa = twaDeg.map((d) => f32(d * DEG));
  const tws = [0, ...ORC.tws].map((v) => f32(v * KNOT));
  const speed = twaDeg.map((_, i) => [0, ...cols.map((c) => f32(c[i] * KNOT))]);
  const cosT = twa.map(Math.cos);
  const vmgPerTws = tws.map((_, j) => Math.max(...speed.map((row, i) => row[j] * cosT[i])));
  return { twa, tws, speed, cosT, vmgPerTws, cache: new Map() };
}

function locate(grid, x) {
  let lo = 0, hi = grid.length; // searchsorted right: first index with grid[idx] > x
  while (lo < hi) { const m = (lo + hi) >> 1; if (grid[m] <= x) lo = m + 1; else hi = m; }
  const i = clip(lo - 1, 0, grid.length - 2);
  const f = clip((x - grid[i]) / (grid[i + 1] - grid[i]), 0, 1);
  return [i, f];
}

function polarSpeed(P, twa, tws) {
  const a = Math.abs(wrap(twa));
  const [i, fa] = locate(P.twa, a);
  const [j, fs] = locate(P.tws, tws);
  const s = P.speed;
  const lo = (1 - fs) * s[i][j] + fs * s[i][j + 1];
  const hi = (1 - fs) * s[i + 1][j] + fs * s[i + 1][j + 1];
  return (1 - fa) * lo + fa * hi;
}

const beatVmg = (P, tws) => interp(tws, P.tws, P.vmgPerTws);

function angles(P, tws) { // [beat angle, run angle] for this wind speed (argmax over the grid)
  let c = P.cache.get(tws);
  if (c) return c;
  let bi = 0, bv = -Infinity, ri = 0, rv = -Infinity;
  for (let i = 0; i < P.twa.length; i++) {
    const sp = interp(tws, P.tws, P.speed[i]);
    const up = sp * P.cosT[i], down = -sp * P.cosT[i];
    if (up > bv) { bv = up; bi = i; }
    if (down > rv) { rv = down; ri = i; }
  }
  c = [P.twa[bi], P.twa[ri]];
  P.cache.set(tws, c);
  return c;
}

// ---------------------------------------------------------------- wind, line (wind.py, rules.py)
const windAt = (w, t) => [w.base + w.amp * Math.sin((2 * Math.PI * t) / w.period + w.phase), w.tws];

function makeLine(cx, cy, windDir, bias, length) {
  const ux = Math.sin(windDir), uy = Math.cos(windDir), rx = Math.cos(windDir), ry = -Math.sin(windDir);
  const dx = rx * Math.cos(bias) + ux * Math.sin(bias), dy = ry * Math.cos(bias) + uy * Math.sin(bias);
  const h = 0.5 * length;
  return { px: cx - h * dx, py: cy - h * dy, cx: cx + h * dx, cy: cy + h * dy };
}
function lineFrame(l) {
  const vx = l.cx - l.px, vy = l.cy - l.py, len = Math.hypot(vx, vy);
  const dx = vx / len, dy = vy / len;
  return [dx, dy, -dy, dx, len];
}
function signedDist(l, x, y) { const [, , nx, ny] = lineFrame(l); return (x - l.px) * nx + (y - l.py) * ny; }
function crossing(l, x0, y0, x1, y1) {
  const [dx, dy, , , len] = lineFrame(l);
  const d0 = signedDist(l, x0, y0), d1 = signedDist(l, x1, y1);
  const frac = clip(d0 / (d0 === d1 ? 1 : d0 - d1), 0, 1);
  const px = x0 + frac * (x1 - x0), py = y0 + frac * (y1 - y0);
  const along = (px - l.px) * dx + (py - l.py) * dy;
  const on = along >= 0 && along <= len;
  return [d0 <= 0 && d1 > 0 && on, d0 > 0 && d1 <= 0 && on, frac];
}

// ---------------------------------------------------------------- physics (physics.py)
function turnRate(v, p) { return p.maxTurn * clip((v - p.minSteer) / (p.fullSteer - p.minSteer), 0, 1); }
function drag(v, twa, tws, p) {
  const u = tws * Math.cos(twa) + v;
  return p.waterK * v * Math.abs(v) + 0.5 * p.rhoAir * p.cda * u * Math.abs(u);
}
function updateSpeed(v, target, twa, tws, dt, p) {
  const tau = target > v ? p.tauAccel : p.tauDecel;
  const accel = v + (target - v) * (1 - Math.exp(-dt / tau));
  let drive = target > 0 ? drag(target, twa, tws, p) : 0;
  drive = Math.max(drive, 0);
  let decel = v + (dt * (drive - drag(v, twa, tws, p))) / p.mass;
  decel = Math.max(decel, Math.max(target, 0));
  return target > v ? accel : decel;
}
function applyAction(b, action, windDir, p) {
  const twa = stwa(b.h, windDir);
  let up = sign(twa); if (up === 0) up = 1;
  const free = !b.man;
  let luffing = false;
  if (free && action >= UP5 && action <= DOWN15) b.th = wrap(b.h + up * ACTION_TURN[action]);
  if (free && action === TACK) {
    b.th = wrap(windDir + twa);
    const gybe = Math.abs(twa) > Math.PI / 2;
    b.v = b.v * (1 - (gybe ? p.gybeLoss : p.tackLoss));
    b.man = true;
  }
  if (free && action === LUFF) luffing = true;
  return luffing;
}
function substep(b, P, windDir, tws, luffing, dt, p) {
  const err = wrap(b.th - b.h);
  const up = sign(stwa(b.h, windDir));
  const bearingAway = err * up < 0;
  let rate = turnRate(b.v, p);
  if (bearingAway) rate = Math.max(rate, p.ironsRate);
  const maxT = rate * dt;
  const h = wrap(b.h + clip(err, -maxT, maxT));
  const done = Math.abs(wrap(b.th - h)) < TURN_DONE_TOL;
  const stalled = b.v < p.minSteer;
  b.man = b.man && !done && !stalled;
  const twa = stwa(h, windDir);
  let target = polarSpeed(P, twa, tws);
  if (luffing) target *= p.luffFactor;
  let v = updateSpeed(b.v, target, twa, tws, dt, p);
  // Browser game only, no Python counterpart (b.fullLuff is never set in the parity replay): a held
  // luff lets the sails fly, so the boat drifts to a stop instead of settling at the luffed speed.
  if (luffing && b.fullLuff) {
    v = Math.min(updateSpeed(b.v, 0, twa, tws, dt, p), b.v * Math.exp(-dt / p.tauDecel));
    if (v < 0.05) v = 0;
  }
  b.x += v * Math.sin(h) * dt; b.y += v * Math.cos(h) * dt;
  b.h = h; b.v = v;
}
const bow = (b, p) => [b.x + 0.5 * p.L * Math.sin(b.h), b.y + 0.5 * p.L * Math.cos(b.h)];
function physicsStep(b, action, P, windDir, tws, p) {
  const luffing = applyAction(b, action, windDir, p);
  const [bx0, by0] = bow(b, p);
  for (let k = 0; k < p.nSub; k++) substep(b, P, windDir, tws, luffing, p.dt / p.nSub, p);
  const [bx1, by1] = bow(b, p);
  return [bx0, by0, bx1, by1];
}

// ---------------------------------------------------------------- fleet rules (fleet_rules.py)
function bowStern(b, L) {
  const fx = Math.sin(b.h), fy = Math.cos(b.h), h = 0.5 * L;
  return [b.x + h * fx, b.y + h * fy, b.x - h * fx, b.y - h * fy];
}
function clearAstern(boats, L) { // A[i][j]: i clear astern of j
  const n = boats.length, bs = boats.map((b) => bowStern(b, L));
  return boats.map((bi, i) => boats.map((bj, j) => {
    const fx = Math.sin(bj.h), fy = Math.cos(bj.h);
    return (bs[i][0] - bs[j][2]) * fx + (bs[i][1] - bs[j][3]) * fy < 0;
  }));
}
function mustKeepClear(boats, windDir, L, manoeuvring) {
  const n = boats.length;
  const stbd = boats.map((b) => stwa(b.h, windDir) > 0);
  const astern = clearAstern(boats, L);
  const K = [];
  for (let i = 0; i < n; i++) {
    K.push([]);
    for (let j = 0; j < n; j++) {
      if (i === j) { K[i].push(false); continue; }
      const overlapped = !astern[i][j] && !astern[j][i];
      const bj = boats[j];
      const rx = Math.cos(bj.h), ry = -Math.sin(bj.h);
      const side = (boats[i].x - bj.x) * rx + (boats[i].y - bj.y) * ry;
      const windward = stbd[j] ? side > 0 : side < 0;
      let rule = stbd[i] === stbd[j] ? (overlapped ? windward : astern[i][j]) : !stbd[i];
      const ti = manoeuvring[i], tj = manoeuvring[j];
      if (ti && !tj) rule = true; else if (tj && !ti) rule = false;
      if (ti && tj) rule = false;
      K[i].push(rule);
    }
  }
  return K;
}
function shadowFactors(boats, windDir, p, tws) { // S[i][j]: factor i gets from j's shadow
  const n = boats.length, len = p.shadowLenBL * p.L, tanH = Math.tan(p.shadowHalf);
  const dirs = boats.map((b) => {
    let dx = -Math.sin(windDir), dy = -Math.cos(windDir);
    if (p.shadowApparent) {
      const ax = tws * dx - b.v * Math.sin(b.h), ay = tws * dy - b.v * Math.cos(b.h);
      const nrm = Math.max(Math.hypot(ax, ay), 1e-6);
      dx = ax / nrm; dy = ay / nrm;
    }
    return [dx, dy];
  });
  return boats.map((bi, i) => boats.map((bj, j) => {
    if (i === j) return 1;
    const dx = bi.x - bj.x, dy = bi.y - bj.y, [ux, uy] = dirs[j];
    const along = dx * ux + dy * uy, across = Math.abs(dx * uy - dy * ux);
    const inside = along > 0 && along < len && across < along * tanH;
    return inside ? p.shadowMin + ((1 - p.shadowMin) * along) / len : 1;
  }));
}
function segDist(p1x, p1y, q1x, q1y, p2x, p2y, q2x, q2y) {
  const d1x = q1x - p1x, d1y = q1y - p1y, d2x = q2x - p2x, d2y = q2y - p2y, rx = p1x - p2x, ry = p1y - p2y;
  const a = d1x * d1x + d1y * d1y, e = d2x * d2x + d2y * d2y, f = d2x * rx + d2y * ry;
  const c = d1x * rx + d1y * ry, b = d1x * d2x + d1y * d2y;
  const denom = a * e - b * b;
  let s = denom > 1e-9 ? clip((b * f - c * e) / denom, 0, 1) : 0;
  let t = (b * s + f) / e;
  if (t < 0) s = clip(-c / a, 0, 1); else if (t > 1) s = clip((b - c) / a, 0, 1);
  t = clip(t, 0, 1);
  return Math.hypot(p1x + d1x * s - (p2x + d2x * t), p1y + d1y * s - (p2y + d2y * t));
}
function pointSegDist(px, py, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay;
  const t = clip(((px - ax) * vx + (py - ay) * vy) / Math.max(vx * vx + vy * vy, 1e-9), 0, 1);
  return Math.hypot(ax + t * vx - px, ay + t * vy - py);
}
function contactsMatrix(boats, L, B) {
  const bs = boats.map((b) => bowStern(b, L));
  return boats.map((_, i) => boats.map((_, j) => i !== j &&
    segDist(bs[i][2], bs[i][3], bs[i][0], bs[i][1], bs[j][2], bs[j][3], bs[j][0], bs[j][1]) < B));
}

// ---------------------------------------------------------------- course (race_env.py)
function course(line, baseDir, p) {
  const ux = Math.sin(baseDir), uy = Math.cos(baseDir), rx = Math.cos(baseDir), ry = -Math.sin(baseDir);
  const cx = 0.5 * (line.px + line.cx), cy = 0.5 * (line.py + line.cy);
  const wx = cx + p.beat * ux, wy = cy + p.beat * uy, gx = cx + p.gateOffset * ux, gy = cy + p.gateOffset * uy;
  const hw = 0.5 * p.gateWidthBL * p.L;
  return { u: [ux, uy], r: [rx, ry], c: [cx, cy], w: [wx, wy], g: [gx, gy], hw,
    g1: [gx - hw * rx, gy - hw * ry], g2: [gx + hw * rx, gy + hw * ry] };
}
const GATE_ROUND_OFFSET = 10; // m: the run ends at a point this far outside a gate mark (race_env.py)
function gateRoundPoints(k) {
  const [rx, ry] = k.r, o = GATE_ROUND_OFFSET;
  return [[k.g1[0] - o * rx, k.g1[1] - o * ry], [k.g2[0] + o * rx, k.g2[1] + o * ry]];
}
// race_env.distance_to_go: the run ends by rounding a gate mark (through: past the gate, still to round one)
function distanceToGo(leg, px, py, k, through = false) {
  const d = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const [p1, p2] = gateRoundPoints(k), pt = [px, py];
  const wg = d(k.w, k.g), wc = d(k.w, k.c), gp = d(k.g, p1), pw = d(p1, k.w);
  if (leg === 0) return d(pt, k.w) + wg + gp + pw + wc;
  if (leg === 1) return (through ? Math.min(d(pt, p1), d(pt, p2)) : d(pt, k.g) + gp) + pw + wc;
  if (leg === 2) return d(pt, k.w) + wc;
  if (leg === 3) return d(pt, k.c);
  return 0;
}

// ---------------------------------------------------------------- environment
class Sim {
  constructor(mode) {
    this.p = makeParams(mode);
    this.P = buildPolar(this.p.noGo);
  }

  marks(s) {
    const l = s.line, p = this.p;
    const m = [[l.px, l.py, p.pinR], [l.cx, l.cy, p.cbR]];
    if (p.race) {
      const k = course(l, s.wind.base, p);
      m.push([...k.w, p.markR], [...k.g1, p.markR], [...k.g2, p.markR]);
    }
    return m;
  }
  course(s) { return course(s.line, s.wind.base, this.p); }

  reset(rand = Math.random) {
    const p = this.p, n = p.n, U = (a, b) => a + (b - a) * rand();
    const wind = { base: p.windDir, tws: U(p.twsMin, p.twsMax), amp: U(0, p.shiftAmpMax), period: p.shiftPeriod, phase: U(0, 2 * Math.PI) };
    const bias = U(-p.lineBiasMax, p.lineBiasMax);
    const line = makeLine(0, 0, p.windDir, bias, p.lineLength);
    const ttg = -p.tStart;
    const vmg = beatVmg(this.P, wind.tws);
    const distMax = Math.min(p.initDistMax, p.reachFrac * vmg * ttg);
    const distMin = Math.min(p.initDistMin, 0.5 * distMax);
    const latMax = Math.min(p.latMax, 0.5 * p.lineLength + p.latSpeed * ttg);
    let lats;
    if (p.solo) lats = [U(-latMax, latMax)];
    else {
      const slots = n > 1 ? Array.from({ length: n }, (_, i) => (-1 + (2 * i) / (n - 1)) * latMax) : [0];
      for (let i = slots.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [slots[i], slots[j]] = [slots[j], slots[i]]; }
      const spacing = (2 * latMax) / Math.max(n - 1, 1);
      lats = slots.map((s) => s + U(-1, 1) * (spacing / 3));
    }
    const [dx, dy, nx, ny] = lineFrame(line);
    const [wd, tws] = windAt(wind, -ttg);
    const boats = lats.map((lat) => {
      const dist = U(distMin, distMax), h = U(-Math.PI, Math.PI);
      const v = U(0, p.speedFracMax) * polarSpeed(this.P, stwa(h, wd), tws);
      return this.newBoat(lat * dx - dist * nx, lat * dy - dist * ny, h, v);
    });
    return this.initState({ boats, wind, line, bias, t: -ttg });
  }

  newBoat(x, y, h, v, th = h, man = false) {
    return { x, y, h, v, th, man, started: false, ocs: false, startTime: Infinity, ssr: 0, sos: false, gunBelow: NaN, gunInside: false };
  }

  initState({ boats, wind, line, bias, t }) {
    const n = boats.length, M = this.p.nMarks;
    const z = (v) => Array(n).fill(v), zz = (v, m = n) => Array.from({ length: n }, () => Array(m).fill(v));
    return { boats, wind, line, bias, t, penLeft: z(0), penDir: z(1), penTurning: z(false), penWait: z(0),
      inContact: zz(false), fouls: z(0), markContact: zz(false, M), contacts: z(0), markHits: z(0),
      leg: z(0), gateThrough: z(false), markThrough: z(false), finishTime: z(Infinity), inZone: z(false), room: zz(false), done: false, info: null };
  }

  clone(s) { return structuredClone(s); }

  keepClear(s, boats, windDir, manoeuvring) {
    const K = mustKeepClear(boats, windDir, this.p.L, manoeuvring);
    if (!this.p.race) return K;
    return K.map((row, i) => row.map((k, j) => (i === j ? false : s.room[i][j] ? true : s.room[j][i] ? false : k)));
  }

  // one boat, solo start env step (start_env.py step_env) with shadow scale and extra success condition
  soloStep(s, b, action, scale) {
    const p = this.p, t0 = s.t, t1 = t0 + p.dt;
    const [wd, tws] = windAt(s.wind, t0);
    const path = physicsStep(b, action, this.P, wd, tws * scale, p);
    const [up, , frac] = crossing(s.line, ...path);
    const d1 = signedDist(s.line, path[2], path[3]);
    const gunNow = t0 < 0 && t1 >= 0, afterGun = t0 >= 0;
    const wasOcs = b.ocs;
    let ocs = b.ocs || (gunNow && d1 > 0);
    ocs = ocs && !(afterGun && d1 <= 0);
    const startsNow = afterGun && up && !wasOcs && !b.started;
    if (startsNow) {
      b.started = true; b.startTime = t0 + frac * (t1 - t0);
      b.ssr = b.v / Math.max(polarSpeed(this.P, stwa(b.h, wd), tws), 1e-3);
      b.sos = stwa(b.h, wd) > 0;
    }
    b.ocs = ocs;
    if (gunNow) {
      const [bx, by] = bow(b, p), [ldx, ldy, , , llen] = lineFrame(s.line);
      const along = (bx - s.line.px) * ldx + (by - s.line.py) * ldy;
      b.gunBelow = -signedDist(s.line, bx, by); b.gunInside = along >= 0 && along <= llen;
    }
  }

  goodStart(b, extraOk) {
    const strict = b.started && b.startTime >= 0 && b.startTime <= this.p.successWindow && b.ssr >= this.p.successSpeed && extraOk;
    const onLine = b.gunBelow >= 0 && b.gunBelow <= this.p.gunDistMax && b.gunInside;
    return strict && b.sos && onLine;
  }

  step(s, actions) {
    const p = this.p, n = p.n;
    const [wd, tws] = windAt(s.wind, s.t);
    const t1 = s.t + p.dt;
    const done0 = t1 >= p.tEnd - 1e-6;
    if (p.solo) {
      this.soloStep(s, s.boats[0], actions[0], 1);
      s.t = t1;
      if (done0) { s.done = true; s.info = { good: [this.goodStart(s.boats[0], true)], rank: [0] }; }
      return s;
    }
    const before = this.clone(s);
    const sh = shadowFactors(s.boats, wd, p, tws);
    const scale = sh.map((row) => Math.min(...row));
    const pending = s.penLeft.map((l) => l > 0);
    const marks = this.marks(s);
    const serving = [], dir0 = [], wait = [], left = [], acts = actions.slice();
    for (let i = 0; i < n; i++) {
      const b = s.boats[i];
      let gapB = Infinity;
      for (let j = 0; j < n; j++) if (j !== i) gapB = Math.min(gapB, Math.hypot(s.boats[j].x - b.x, s.boats[j].y - b.y));
      const gapM = Math.min(...marks.map((m) => Math.hypot(m[0] - b.x, m[1] - b.y) - m[2]));
      const clear = gapB > p.clearDist && gapM > p.clearDist;
      serving[i] = pending[i] && (s.penTurning[i] || clear || s.penWait[i] >= p.clearTimeout || p.clearDist <= 0);
      let away0 = -sign(stwa(b.h, wd)); if (away0 === 0) away0 = 1;
      dir0[i] = serving[i] && !s.penTurning[i] && p.clearDist > 0 ? away0 : s.penDir[i];
      wait[i] = pending[i] && !serving[i] ? s.penWait[i] + p.dt : 0;
      const turn = Math.min(p.penaltyRate * p.dt, s.penLeft[i]);
      if (serving[i]) { const hn = wrap(b.h + dir0[i] * turn); b.h = hn; b.th = hn; b.man = false; acts[i] = HOLD; }
      left[i] = s.penLeft[i] - (serving[i] ? turn : 0);
      if (left[i] < 1e-9) left[i] = 0; // float residue: the circle is complete
    }
    const willServe = left.map((l) => l <= 0);
    const clearAir = scale.map((f) => f >= 1);
    for (let i = 0; i < n; i++) this.soloStep(s, s.boats[i], acts[i], scale[i]);
    s.t = t1;

    const touching = contactsMatrix(s.boats, p.L, p.B);
    const taking = serving.map((sv, i) => sv && left[i] > 0);
    const K = this.keepClear(before, s.boats, wd, s.boats.map((b, i) => b.man || taking[i]));
    const onMark = s.boats.map((b) => {
      const [bx, by, sx, sy] = bowStern(b, p.L);
      return marks.map((m) => pointSegDist(m[0], m[1], sx, sy, bx, by) < 0.5 * p.B + m[2]);
    });
    const penalised = [];
    for (let i = 0; i < n; i++) {
      let boatFoul = false, fault = 0, row = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const nw = touching[i][j] && !s.inContact[i][j];
        const atFault = K[i][j] && !K[j][i], undecided = K[i][j] === K[j][i];
        if (nw && (atFault || undecided)) { boatFoul = true; fault++; } else if (nw) row++;
      }
      const newMark = onMark[i].some((m, k) => m && !s.markContact[i][k]);
      penalised[i] = (boatFoul || newMark) && !serving[i];
      s.contacts[i] += fault + row + (newMark ? 1 : 0);
      if (newMark) s.markHits[i]++;
      const b = s.boats[i];
      let away = -sign(stwa(b.h, wd)); if (away === 0) away = 1;
      s.penLeft[i] = penalised[i] ? left[i] + p.penaltyTurns * 2 * Math.PI : left[i];
      s.penDir[i] = penalised[i] && left[i] <= 0 ? away : dir0[i];
      s.penTurning[i] = serving[i] && left[i] > 0;
      s.penWait[i] = wait[i];
      if (penalised[i]) s.fouls[i]++;
    }
    s.inContact = touching;
    s.markContact = onMark;
    s.lastPenalised = penalised;

    if (p.race) return this.postStep(before, s, done0);
    if (done0) {
      const [wEnd] = windAt(s.wind, s.t);
      const cx = 0.5 * (s.line.px + s.line.cx), cy = 0.5 * (s.line.py + s.line.cy);
      const score = s.boats.map((b, i) => {
        const [bx, by] = bow(b, p);
        const valid = b.started && !b.ocs && !(s.penLeft[i] > 0);
        return valid ? (bx - cx) * Math.sin(wEnd) + (by - cy) * Math.cos(wEnd) : -1e6;
      });
      const rank = score.map((sc) => score.filter((o) => o > sc).length + 0.5 * (score.filter((o) => o === sc).length - 1));
      s.done = true;
      s.info = { good: s.boats.map((b, i) => this.goodStart(b, clearAir[i] && willServe[i])), rank, clearAir,
        owed: s.penLeft.map((l) => l > 0) };
    }
    return s;
  }

  postStep(prev, s, done0) {
    const p = this.p, n = p.n, k = this.course(s);
    const [ux, uy] = k.u, [rx, ry] = k.r, [wx, wy] = k.w, [gx, gy] = k.g, hw = k.hw;
    const t0 = prev.t, t1 = s.t;
    const advance = [], legNew = [], dT = [], target = [];
    for (let i = 0; i < n; i++) {
      const [bpx, bpy] = bow(prev.boats[i], p), [bnx, bny] = bow(s.boats[i], p);
      const leg = prev.leg[i], started = s.boats[i].started;
      const s0 = (bpx - wx) * rx + (bpy - wy) * ry, s1 = (bnx - wx) * rx + (bny - wy) * ry;
      const f = s0 / (s0 !== s1 ? s0 - s1 : 1);
      const along = (bpx + f * (bnx - bpx) - wx) * ux + (bpy + f * (bny - bpy) - wy) * uy;
      // windward mark: across the half-line upwind of it right -> left, then back down past it on its left side
      // (back across the half-line left -> right undoes the first part)
      const beatLeg = (leg === 0 || leg === 2) && started;
      const over = beatLeg && s0 > 0 && s1 <= 0 && along > 0, back = beatLeg && s0 <= 0 && s1 > 0 && along > 0;
      const a0 = (bpx - wx) * ux + (bpy - wy) * uy, a1 = (bnx - wx) * ux + (bny - wy) * uy;
      const fa = a0 / (a0 !== a1 ? a0 - a1 : 1);
      const side = (bpx + fa * (bnx - bpx) - wx) * rx + (bpy + fa * (bny - bpy) - wy) * ry;
      const throughW = (prev.markThrough[i] || over) && !back;
      const roundW = throughW && beatLeg && a0 > 0 && a1 <= 0 && side < 0;
      s.markThrough[i] = throughW && !roundW;
      const g0 = (bpx - gx) * ux + (bpy - gy) * uy, g1 = (bnx - gx) * ux + (bny - gy) * uy;
      const fg = g0 / (g0 !== g1 ? g0 - g1 : 1);
      const lat = (bpx + fg * (bnx - bpx) - gx) * rx + (bpy + fg * (bny - bpy) - gy) * ry;
      // gate: through between the marks going downwind, then round either mark (back up outside it)
      const downThrough = leg === 1 && g0 > 0 && g1 <= 0 && Math.abs(lat) <= hw;
      const backUp = leg === 1 && g0 <= 0 && g1 > 0;
      const passGate = prev.gateThrough[i] && backUp && Math.abs(lat) > hw;
      s.gateThrough[i] = (prev.gateThrough[i] || downThrough) && !(backUp && Math.abs(lat) <= hw) && !passGate;
      const [, down, ff] = crossing(s.line, bpx, bpy, bnx, bny);
      const finish = leg === 3 && down;
      advance[i] = roundW || passGate || finish;
      legNew[i] = leg + (advance[i] ? 1 : 0);
      if (finish) s.finishTime[i] = t0 + ff * (t1 - t0);
      const finished = legNew[i] >= N_LEGS;
      const gLeft = Math.hypot(bnx - k.g1[0], bny - k.g1[1]) < Math.hypot(bnx - k.g2[0], bny - k.g2[1]);
      target[i] = (legNew[i] === 0 || legNew[i] === 2) && started ? 2 : legNew[i] === 1 && gLeft ? 3 : legNew[i] === 1 ? 4 : -1;
      const tp = target[i] === 2 ? k.w : target[i] === 3 ? k.g1 : target[i] === 4 ? k.g2 : [0, 0];
      dT[i] = Math.hypot(bnx - tp[0], bny - tp[1]);
      s.inZone[i] = target[i] >= 0 && dT[i] < p.zoneBL * p.L && !finished;
    }
    const astern = clearAstern(s.boats, p.L);
    const room = [];
    for (let i = 0; i < n; i++) {
      room.push([]);
      for (let j = 0; j < n; j++) {
        const same = i !== j && target[i] === target[j] && target[i] >= 0;
        const ei = s.inZone[i] && !prev.inZone[i], ej = s.inZone[j] && !prev.inZone[j];
        const first = (ei && !prev.inZone[j]) || (ej && !prev.inZone[i]);
        const overlapped = !astern[i][j] && !astern[j][i];
        const owes = overlapped ? dT[i] > dT[j] : astern[i][j];
        let r = same && first ? owes : prev.room[i][j];
        const keep = same && (s.inZone[i] || s.inZone[j]) && !advance[i] && !advance[j];
        room[i].push(r && keep);
      }
    }
    s.room = room;
    s.leg = legNew;
    for (let i = 0; i < n; i++) if (legNew[i] >= N_LEGS) {
      const b = s.boats[i]; b.x = PARK_X + 100 * i; b.y = 0; b.v = 0;
    }
    s.lastRounded = advance;
    const finished = legNew.map((l) => l >= N_LEGS);
    const done = done0 || finished.every(Boolean);
    if (done) {
      const score = s.boats.map((b, i) => {
        const [bnx, bny] = bow(b, p); // (parked boats: finished, scored by time)
        return finished[i] ? 1e7 - s.finishTime[i] : legNew[i] * 1e5 - distanceToGo(legNew[i], bnx, bny, k, s.gateThrough[i]);
      });
      s.done = true;
      s.info = { rank: score.map((sc) => score.filter((o) => o > sc).length + 0.5 * (score.filter((o) => o === sc).length - 1)),
        finished, finishTime: s.finishTime.slice() };
    }
    return s;
  }

  // ---------------------------------------------------------------- observation (start_env / fleet_env)
  obs(s) {
    const p = this.p, n = p.n, [wd, tws] = windAt(s.wind, s.t);
    const taking = s.penTurning.map((t, i) => t || (s.penLeft[i] > 0 && p.clearDist <= 0));
    const K = this.keepClear(s, s.boats, wd, s.boats.map((b, i) => b.man || taking[i]));
    const sh = shadowFactors(s.boats, wd, p, tws);
    return s.boats.map((b, i) => {
      const twa = stwa(b.h, wd), fx = Math.sin(b.h), fy = Math.cos(b.h), rx = fy, ry = -fx;
      const fr = (px, py) => [((px - b.x) * fx + (py - b.y) * fy) / 100, ((px - b.x) * rx + (py - b.y) * ry) / 100];
      const [pf, pr] = fr(s.line.px, s.line.py), [cf, cr] = fr(s.line.cx, s.line.cy);
      const own = [b.v / 5, Math.sin(twa), Math.cos(twa), twa >= 0 ? 1 : -1, -s.t / 60, pf, pr, cf, cr,
        signedDist(s.line, b.x, b.y) / 100, s.bias / Math.max(p.lineBiasMax, 1e-6), b.man ? 1 : 0,
        b.started ? 1 : 0, b.ocs ? 1 : 0, s.wind.tws / 8];
      const others = [];
      for (let j = 0; j < n; j++) if (j !== i) others.push([Math.hypot(s.boats[j].x - b.x, s.boats[j].y - b.y), j]);
      others.sort((a, c) => a[0] - c[0]);
      const nb = [];
      for (let q = 0; q < 5; q++) {
        if (q >= others.length) { nb.push(0, 0, 0, 0, 0, 0, 0, 0, 0); continue; }
        const j = others[q][1], bj = s.boats[j], [f, st] = fr(bj.x, bj.y), relh = bj.h - b.h;
        let feats = [1, f, st, Math.sin(relh), Math.cos(relh), bj.v / 5, stwa(bj.h, wd) >= 0 ? 1 : -1, K[j][i] ? 1 : 0, sh[i][j] < 1 ? 1 : 0];
        if (p.race && (Math.abs(f) > 50 || Math.abs(st) > 50)) feats = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        nb.push(...feats);
      }
      if (!p.race) return own.concat(nb);
      // race_env.get_obs: leg one-hot, the next mark(s) in the boat's frame (/100 m), clock, penalty owed, turning
      const k = this.course(s), leg = s.leg[i] + (s.markThrough[i] ? 1 : 0); // past the half-line: the run
      const [wf, ws] = fr(...k.w), [g1f, g1s] = fr(...k.g1), [g2f, g2s] = fr(...k.g2), [lcf, lcs] = fr(...k.c);
      const beat = leg === 0 || leg === 2;
      const first = beat ? [wf, ws] : leg === 1 ? [g1f, g1s] : leg === 3 ? [lcf, lcs] : [0, 0];
      const second = leg === 1 ? [g2f, g2s] : [0, 0];
      const onehot = [0, 1, 2, 3, 4].map((l) => (Math.min(Math.max(leg, 0), N_LEGS) === l ? 1 : 0));
      const race = [...onehot, ...first, ...second, s.t / 600, s.penLeft[i] / (2 * Math.PI), s.penTurning[i] ? 1 : 0];
      return own.concat(nb, race);
    });
  }
}

// ---------------------------------------------------------------- agents (agents/*.py)
function steer(heading, desired, windDir) {
  const twaNow = stwa(heading, windDir), twaDes = stwa(desired, windDir);
  const other = sign(twaNow) !== sign(twaDes);
  const up = twaNow >= 0 ? 1 : -1;
  const kd = (wrap(desired - heading) / DEG) * up;
  const act = kd > 10 ? UP15 : kd > 2.5 ? UP5 : kd < -10 ? DOWN15 : kd < -2.5 ? DOWN5 : HOLD;
  return other && Math.abs(wrap(desired - heading) / DEG) > 20 ? TACK : act;
}

// Never stuck head to wind (agents/digital_sailor.py unstick): in irons bear away; a tack below full
// steerage speed waits, sailing close-hauled on this tack until there is speed for it.
function unstick(act, b, wd, beat, p) {
  const twa = Math.abs(stwa(b.h, wd));
  if (b.v < p.minSteer && twa < beat) return DOWN15;
  if (act === TACK && b.v < p.fullSteer && twa < 0.5 * Math.PI) return twa < beat ? DOWN5 : HOLD;
  return act;
}

function digitalSailor(sim, s, i, spot, timing = -3, finalWindow = 6) {
  const p = sim.p, b = s.boats[i], [wd, tws] = windAt(s.wind, s.t);
  const [beat] = angles(sim.P, tws);
  const vcl = Math.max(polarSpeed(sim.P, beat, tws), 0.5);
  const hcl = wrap(wd - beat), ux = Math.sin(hcl), uy = Math.cos(hcl);
  const tx = s.line.px + spot * (s.line.cx - s.line.px), ty = s.line.py + spot * (s.line.cy - s.line.py);
  const ttg = -s.t, lead = Math.max(ttg + timing, 0), half = 0.5 * p.L;
  const px = tx - (vcl * lead + half) * ux, py = ty - (vcl * lead + half) * uy;
  const dx = px - b.x, dy = py - b.y, dist = Math.hypot(dx, dy);
  const lateral = Math.abs(dx * uy - dy * ux);
  const onApproach = lateral < 6 && Math.abs(wrap(b.h - hcl)) < 20 * DEG;
  const bearing = Math.atan2(dx, dy);
  const twaB = stwa(bearing, wd), onStbd = stwa(b.h, wd) >= 0;
  const cl = onStbd ? wd - beat : wd + beat;
  let desired = Math.abs(twaB) < beat ? cl : bearing;
  if (dist < 8) desired = hcl;
  if (ttg < finalWindow) desired = hcl;
  const act = steer(b.h, desired, wd);
  const bx = b.x + half * Math.sin(b.h), by = b.y + half * Math.cos(b.h);
  const dAlong = Math.max((tx - bx) * ux + (ty - by) * uy, 0);
  const tNeed = dAlong / vcl + p.tauAccel * clip(1 - b.v / vcl, 0, 1);
  const burn = ttg + timing - tNeed;
  return unstick(onApproach && burn > 1 && ttg > 2 ? LUFF : act, b, wd, beat, p);
}

const AVOID = [0, 20, -20, 40, -40, 70, -70, 110, -110].map((d) => d * DEG);
function avoid(b, desired, wd, beat, obst, p, margin = 1.5) {
  const look = 0.5 * p.L + 4 + 3 * b.v;
  const clearOf = (h, len) => {
    const dx = Math.sin(h), dy = Math.cos(h);
    return obst.every(([ox, oy, r]) => {
      const mx = ox - b.x, my = oy - b.y, along = clip(mx * dx + my * dy, 0, len);
      return Math.hypot(mx - along * dx, my - along * dy) > r + 0.5 * p.B + margin;
    });
  };
  for (let c = 0; c < AVOID.length; c++) {
    const h = wrap(desired + AVOID[c]);
    const way = wrap(b.h + clip(wrap(h - b.h), -15 * DEG, 15 * DEG));
    const safe = clearOf(h, look) && clearOf(way, 0.6 * look);
    const sailable = Math.abs(stwa(h, wd)) >= beat - 5 * DEG;
    if (safe && (sailable || c === 0)) return h;
  }
  return desired;
}

function raceSailor(sim, s, i, obs, spot, timing = -3) {
  const p = sim.p, b = s.boats[i], leg = Math.min(Math.max(s.leg[i] + (s.markThrough[i] ? 1 : 0), 0), N_LEGS);
  const [wd, tws] = windAt(s.wind, s.t), k = sim.course(s);
  const [ux, uy] = k.u, [rx, ry] = k.r, [wx, wy] = k.w;
  const [beat, run] = angles(sim.P, tws);
  const lane = clip((spot - 0.15) / 0.7, 0, 1), line = s.line;
  const lx = line.px + spot * (line.cx - line.px), ly = line.py + spot * (line.cy - line.py);
  const upW = (b.x - wx) * ux + (b.y - wy) * uy, rightW = (b.x - wx) * rx + (b.y - wy) * ry;
  const off = 8 + 12 * lane;
  const atMark = upW > -2 && rightW > 0;
  let tx = atMark ? wx + 10 * ux - 10 * rx : wx + off * rx, ty = atMark ? wy + 10 * uy - 10 * ry : wy + off * ry;
  const gateOff = (lane - 0.5) * 1.2 * k.hw;
  if (leg === 1) { // through the gate at the lane's spot, then round the mark on that side (agents/race_sailor.py)
    tx = k.g[0] + gateOff * rx; ty = k.g[1] + gateOff * ry;
    const side = lane < 0.5 ? -1 : 1, gmx = k.g[0] + side * k.hw * rx, gmy = k.g[1] + side * k.hw * ry;
    const below = (b.x - k.g[0]) * ux + (b.y - k.g[1]) * uy < 0;
    const outside = side * ((b.x - k.g[0]) * rx + (b.y - k.g[1]) * ry) > k.hw + 3;
    if (below) {
      tx = outside ? gmx + side * 8 * rx + 15 * ux : gmx + side * 10 * rx - 8 * ux;
      ty = outside ? gmy + side * 8 * ry + 15 * uy : gmy + side * 10 * ry - 8 * uy;
    }
  } else if (leg === 3) { tx = lx; ty = ly; }
  // just past the windward mark: bear away down its left side (the rounding counts once below it there)
  if ((leg === 1 || leg === 3) && upW > -3 && Math.hypot(b.x - wx, b.y - wy) < 60) { tx = wx - 15 * rx - 20 * ux; ty = wy - 15 * ry - 20 * uy; }
  const nx = -(line.cy - line.py), ny = line.cx - line.px, nrm = Math.hypot(nx, ny);
  const above = ((b.x - line.px) * nx + (b.y - line.py) * ny) / nrm > -1;
  const late = !b.started && s.t > 0;
  if (late) { tx = above ? lx - 20 * ux : lx + 15 * ux; ty = above ? ly - 20 * uy : ly + 15 * uy; }
  const bearing = Math.atan2(tx - b.x, ty - b.y), twaB = stwa(bearing, wd), onStbd = stwa(b.h, wd) >= 0;
  const cl = onStbd ? wd - beat : wd + beat, vmgDown = onStbd ? wd - run : wd + run;
  let desired = Math.abs(twaB) < beat ? cl : Math.abs(twaB) > run ? vmgDown : bearing;
  const beating = (leg === 0 || leg === 2) && !atMark && !late;
  const near = Math.hypot(b.x - wx, b.y - wy) < 50;
  const nX = wx + 6 * rx + 2 * ux, nY = wy + 6 * ry + 2 * uy, brN = Math.atan2(nX - b.x, nY - b.y), twaN = stwa(brN, wd);
  const fetch = twaN >= beat - 2 * DEG || Math.abs(twaN) > 0.5 * Math.PI;
  if (beating && near) desired = fetch ? brN : wd + beat;
  const obst = sim.marks(s).map((m) => [m[0], m[1], m[2]]);
  const fx = Math.sin(b.h), fy = Math.cos(b.h), rbx = fy, rby = -fx;
  for (let q = 0; q < 5; q++) { // give-way neighbours where they will be in 2 s (obs slots)
    const o = obs.slice(15 + 9 * q, 24 + 9 * q).map(Math.fround); // Python's agents read the float32 observation
    const fwd = 100 * o[1], st = 100 * o[2];
    if (!(o[0] > 0.5 && o[7] < 0.5 && Math.hypot(fwd, st) < 40)) continue;
    const hj = b.h + Math.atan2(o[3], o[4]), vj = 5 * o[5];
    obst.push([b.x + fwd * fx + st * rbx + 2 * vj * Math.sin(hj), b.y + fwd * fy + st * rby + 2 * vj * Math.cos(hj), 0.6 * p.L]);
  }
  desired = avoid(b, wrap(desired), wd, beat, obst, p);
  const act = unstick(steer(b.h, desired, wd), b, wd, beat, p);
  const racing = (b.started && leg < N_LEGS) || late;
  return racing ? act : digitalSailor(sim, s, i, spot, timing);
}

function spots(n) { return n > 1 ? Array.from({ length: n }, (_, i) => 0.15 + (0.7 * i) / (n - 1)) : [0.3]; }

function mlp(layers, x) {
  let h = x;
  layers.forEach((layer, li) => {
    const out = layer.b.slice();
    for (let i = 0; i < h.length; i++) { const hi = h[i], Wi = layer.W[i]; for (let j = 0; j < out.length; j++) out[j] += hi * Wi[j]; }
    h = li < layers.length - 1 ? out.map(Math.tanh) : out;
  });
  return h;
}
const argmax = (a) => a.reduce((bi, v, i) => (v > a[bi] ? i : bi), 0);

const START_OBS = 60, RACE_OBS = 72; // start_env.OBS_DIM; + race_env's 12 race features

// Is `agent` (a parsed sail_env-agent JSON, see sail_env/agents/export.py) playable here? null if yes, else why not.
function agentProblem(agent, race) {
  if (!agent || agent.format !== "sail_env-agent" || !Array.isArray(agent.layers) || !agent.layers.length) return "not a sail_env agent file";
  if (agent.weather) return "it was trained with weather, which the browser version does not have";
  const nIn = agent.layers[0].W.length, nOut = agent.layers[agent.layers.length - 1].b.length;
  if (nOut !== 7) return `it has ${nOut} actions instead of 7`;
  if (race && nIn !== RACE_OBS) return nIn === START_OBS ? "it is a start agent: choose a start" : `it expects ${nIn} inputs, a race gives ${RACE_OBS}`;
  if (!race && nIn !== START_OBS) return nIn === RACE_OBS ? "it is a race agent: choose a race" : `it expects ${nIn} inputs, a start gives ${START_OBS}`;
  return null;
}

// A network's most likely action. In a race, a racer leaves the start to the start policy until its boat has
// started (or T+10), as sail_env.agents.pretrained_agent does: racers are trained from just after the start.
function rlAction(layers, startLayers, sim, s, i, o) {
  if (sim.p.race && startLayers && !s.boats[i].started && s.t < 10) return argmax(mlp(startLayers, o.slice(0, START_OBS).map(Math.fround)));
  return argmax(mlp(layers, o.map(Math.fround)));
}

// actions for every boat from the chosen opponents: "digital" (hand-written sailors and racers), "rl" (the
// bundled RL policies: the fleet start policy in starts, the racer in races) or "custom" (W.custom, an
// uploaded agent). W: { fleet, racer, custom } networks (weights.json; custom: agent.layers). obs computed once
function opponentActions(sim, s, kind, W) {
  const n = sim.p.n, sp = spots(n), race = sim.p.race;
  if (sim.p.solo) return [digitalSailor(sim, s, 0, 0.3)];
  const net = !W ? null : kind === "custom" ? W.custom : kind === "rl" ? (race ? W.racer : W.fleet) : null;
  const obs = sim.obs(s);
  return obs.map((o, i) => {
    if (net) return rlAction(net, W.fleet, sim, s, i, o);
    return race ? raceSailor(sim, s, i, o, sp[i]) : digitalSailor(sim, s, i, sp[i]);
  });
}

const SailSim = { Sim, makeParams, opponentActions, agentProblem, rlAction, START_OBS, RACE_OBS, digitalSailor, raceSailor, mlp, argmax, windAt, stwa, wrap, bow,
  polarSpeed, beatVmg, angles, course, lineFrame, signedDist, distanceToGo, spots, KNOT, DEG, N_LEGS, PARK_X, LEG_NAMES,
  HOLD, UP5, DOWN5, UP15, DOWN15, TACK, LUFF };
if (typeof module !== "undefined") module.exports = SailSim;
