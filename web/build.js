// Builds web/index.html (the playable page served on GitHub Pages) from the template, the simulator
// and the RL fleet policy:   node web/build.js
"use strict";
const fs = require("fs");
const path = require("path");
const dir = __dirname;
const sim = fs.readFileSync(path.join(dir, "sim.js"), "utf8");
const weights = fs.readFileSync(path.join(dir, "weights.json"), "utf8").trim();
const html = fs.readFileSync(path.join(dir, "game.template.html"), "utf8")
  .replace("/*SIM*/", () => sim)
  .replace("/*WEIGHTS*/", () => weights);
fs.writeFileSync(path.join(dir, "index.html"), html);
console.log(`wrote web/index.html (${(html.length / 1024).toFixed(0)} KB)`);
