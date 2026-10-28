"""Rebuild data/polar_j70_orc.csv from public ORC certificate data.

Source: https://github.com/jieter/orc-data (MIT licence), file ALL2024.json, which holds the ORC VPP
output of every 2024 certificate. We take every J/70 certificate and use the per-cell median, the same
method boatpolars.com uses for its J/70 page.

Usage:  python tools/extract_orc_polar.py
"""

import ast
import re
import statistics
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/jieter/orc-data/master/ALL2024.json"
OUT = Path(__file__).resolve().parents[1] / "sail_env" / "data" / "polar_j70_orc.csv"
J70 = re.compile(r"J[\s\-/]*70(\s|$)", re.I)  # matches "J-70", "J-70 ONE DESIGN", "J/70", "J 70"


def main() -> None:
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    rows = ast.literal_eval(raw)  # the file is a Python literal (single quotes), not strict JSON
    certs = [r["vpp"] for r in rows if J70.match((r["boat"]["type"] or "").strip())]
    angles, speeds = certs[0]["angles"], certs[0]["speeds"]
    assert all(c["angles"] == angles and c["speeds"] == speeds for c in certs)

    def median_row(key):
        return [round(statistics.median(col), 2) for col in zip(*(c[key] for c in certs))]

    lines = [
        "# J/70 polar: median of ORC VPP output over %d 2024 certificates." % len(certs),
        "# Source: github.com/jieter/orc-data ALL2024.json (MIT). Rebuild with tools/extract_orc_polar.py.",
        "# Numeric rows: boat speed (kn) at true wind angle (deg) x true wind speed (kn, header).",
        "# beat_angle/beat_vmg: optimal upwind angle (deg) and VMG (kn); run_angle/run_vmg: same downwind.",
        "twa," + ",".join(str(s) for s in speeds),
    ]
    for a in angles:
        lines.append(f"{a}," + ",".join(f"{v:.2f}" for v in median_row(a)))
    for key in ["beat_angle", "beat_vmg", "run_angle", "run_vmg"]:
        lines.append(f"{key}," + ",".join(f"{v:.2f}" for v in median_row(key)))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} from {len(certs)} certificates")


if __name__ == "__main__":
    main()
