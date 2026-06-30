"""Debug escape-sector metrics on logged trials."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from metrics.escape_sector_metrics import (
    _ray_blocked_by_pursuer,
    _ray_feasible,
    compute_escape_sector_metrics,
)


def analyze_trial(json_path: Path) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    cap_step = meta.get("capture_step")
    bounds = (0.0, 40.0, 0.0, 40.0)
    ray_length = 6.0
    pbr = 1.0
    n = 720

    print(f"\n=== {json_path.name} capture_step={cap_step} ===")
    if cap_step is None:
        return

    rec = next(r for r in data["records"] if r["step"] == cap_step)
    ev = np.array(rec["evader"], dtype=float)
    ps = np.array(rec["pursuers"], dtype=float)
    wall = min(ev[0], 40.0 - ev[0], ev[1], 40.0 - ev[1])
    print(f"evader={ev}, min_wall_dist={wall:.3f}")
    print(f"pursuer dists={np.linalg.norm(ps - ev, axis=1)}")

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    feasible_idx: list[int] = []
    blocked_by = [0] * len(ps)
    for i, a in enumerate(angles):
        if _ray_feasible(ev, float(a), ray_length, 40, *bounds, 0.0, [], 0.0):
            feasible_idx.append(i)
            for j, p in enumerate(ps):
                if _ray_blocked_by_pursuer(ev, float(a), ray_length, p, pbr):
                    blocked_by[j] += 1
                    break

    print(f"feasible (no pursuers): {len(feasible_idx)*360/n:.1f} deg")
    for j, p in enumerate(ps):
        b = math.degrees(math.atan2(p[1] - ev[1], p[0] - ev[0]))
        d = float(np.linalg.norm(p - ev))
        print(f"  pursuer {j}: bearing={b:.1f} deg dist={d:.3f} blocks {blocked_by[j]*360/n:.1f} deg")

    for test_pbr in (0.3, 0.5, 0.8, 1.0, 1.5):
        r = compute_escape_sector_metrics(
            ev, ps, [], bounds, pursuer_block_radius=test_pbr
        )
        print(
            f"  pbr={test_pbr}: free={r['free_escape_angle_deg']:.1f} "
            f"blocked={r['blocked_escape_angle_deg']:.1f} "
            f"unblocked={r['unblocked_escape_angle_deg']:.1f} "
            f"G_esc={r['G_esc_deg']:.1f}"
        )

    print("Interior vs corner along trajectory:")
    for step in (0, 100, 200, 400, 600, cap_step):
        srec = data["records"][step]
        sev = np.array(srec["evader"], dtype=float)
        swall = min(sev[0], 40.0 - sev[0], sev[1], 40.0 - sev[1])
        r = compute_escape_sector_metrics(
            sev, np.array(srec["pursuers"], dtype=float), [], bounds
        )
        print(
            f"  step {step:4d} wall={swall:5.2f} "
            f"free={r['free_escape_angle_deg']:6.1f} "
            f"blocked={r['blocked_escape_angle_deg']:6.1f} "
            f"unblocked={r['unblocked_escape_angle_deg']:6.1f}"
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--scan":
        root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
            "results/20260629_154700_comparison/fcem/free"
        )
        bounds = (0.0, 40.0, 0.0, 40.0)
        for p in sorted(root.glob("*.json"))[:25]:
            data = json.loads(p.read_text(encoding="utf-8"))
            cap = data["metadata"].get("capture_step")
            if cap is None:
                continue
            rec = next(r for r in data["records"] if r["step"] == cap)
            ev = np.array(rec["evader"], dtype=float)
            wall = min(ev[0], 40.0 - ev[0], ev[1], 40.0 - ev[1])
            r = compute_escape_sector_metrics(
                ev, np.array(rec["pursuers"], dtype=float), [], bounds
            )
            print(
                f"{p.name} wall={wall:5.2f} free={r['free_escape_angle_deg']:6.1f} "
                f"blocked={r['blocked_escape_angle_deg']:6.1f} "
                f"unblocked={r['unblocked_escape_angle_deg']:6.1f}"
            )
    else:
        path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
            "results/20260629_161502_comparison/fcem/free/fcem_free_t000.json"
        )
        analyze_trial(path)
