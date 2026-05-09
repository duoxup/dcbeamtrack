"""Static field comparison: CircularGreenSpaceChargeSolver vs analytic ground truth.

Three baseline particle configurations + n_theta sweep on the random uniform-disk
case. Diagnostic only — no hard pass/fail. Outputs JSON summary, NPZ arrays, and
a loglog convergence PNG to artifacts/analytic_vs_green/.

Memory note: the n_theta sweep precomputes Green kernels at n_r=200, n_theta up
to 1440. Peak memory for the largest entry is ~900 MB (200**2 * 1440 * 16 bytes).
Drop the largest n_theta if your machine is tight.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from partdist import SliceDistribution

from dcbeamtrack import (
    AnalyticMirrorChargeSpaceChargeSolver,
    CircularGreenSpaceChargeSolver,
)
from dcbeamtrack.tracking import velocity_to_momentum_evc


RADIUS = 150e-6
N_R_FIXED = 200
N_THETA_BASELINE = 360
N_THETA_SWEEP = (90, 180, 360, 720, 1440)


def _slice(x: np.ndarray, y: np.ndarray, lam: np.ndarray) -> SliceDistribution:
    n = x.size
    px, py, pz = velocity_to_momentum_evc(np.zeros(n), np.zeros(n), np.zeros(n))
    return SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=np.zeros(n), lam=lam
    )


def _case_on_axis_chain(rng: np.random.Generator):
    n = 200
    x = np.linspace(0.05 * RADIUS, 0.85 * RADIUS, n)
    y = np.zeros(n)
    lam = np.full(n, 1.0e-12)
    return "on_axis_chain", x, y, lam


def _case_two_cluster(rng: np.random.Generator):
    n_per = 100
    sigma = 0.05 * RADIUS
    cx1, cy1 = +0.4 * RADIUS, 0.0
    cx2, cy2 = -0.3 * RADIUS, +0.2 * RADIUS
    x1 = cx1 + sigma * rng.standard_normal(n_per)
    y1 = cy1 + sigma * rng.standard_normal(n_per)
    x2 = cx2 + sigma * rng.standard_normal(n_per)
    y2 = cy2 + sigma * rng.standard_normal(n_per)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    lam = np.full(x.size, 1.0e-12)
    return "two_cluster_off_axis", x, y, lam


def _case_random_disk(rng: np.random.Generator):
    n = 200
    r = RADIUS * 0.8 * np.sqrt(rng.uniform(0.0, 1.0, size=n))
    th = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    lam = np.full(n, 1.0e-12)
    return "random_uniform_disk", x, y, lam


def _metrics(green_ex, green_ey, ref_ex, ref_ey):
    scale = max(np.max(np.abs(ref_ex)), np.max(np.abs(ref_ey)))
    return {
        "max_abs_ex": float(np.max(np.abs(green_ex - ref_ex))),
        "max_abs_ey": float(np.max(np.abs(green_ey - ref_ey))),
        "max_rel_ex": float(np.max(np.abs(green_ex - ref_ex)) / scale),
        "max_rel_ey": float(np.max(np.abs(green_ey - ref_ey)) / scale),
        "ref_scale": float(scale),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/analytic_vs_green")
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    analytic = AnalyticMirrorChargeSpaceChargeSolver(radius=RADIUS)
    green_baseline = CircularGreenSpaceChargeSolver(
        radius=RADIUS, n_r=N_R_FIXED, n_theta=N_THETA_BASELINE
    )

    cases = [_case_on_axis_chain(rng), _case_two_cluster(rng), _case_random_disk(rng)]

    summary = {
        "radius_m": RADIUS,
        "n_r": N_R_FIXED,
        "n_theta_baseline": N_THETA_BASELINE,
        "n_theta_sweep": list(N_THETA_SWEEP),
        "seed": int(args.seed),
        "cases": {},
        "n_theta_convergence": [],
    }
    arrays_out = {}

    for name, x, y, lam in cases:
        dist = _slice(x, y, lam)
        fm_ref = analytic.solve(dist)
        fm_green = green_baseline.solve(dist)
        m = _metrics(fm_green.ex, fm_green.ey, fm_ref.ex, fm_ref.ey)
        summary["cases"][name] = m
        arrays_out[f"{name}_x"] = x
        arrays_out[f"{name}_y"] = y
        arrays_out[f"{name}_lam"] = lam
        arrays_out[f"{name}_ex_ref"] = fm_ref.ex
        arrays_out[f"{name}_ey_ref"] = fm_ref.ey
        arrays_out[f"{name}_ex_green"] = fm_green.ex
        arrays_out[f"{name}_ey_green"] = fm_green.ey
        print(
            f"[{name}] max rel: Ex={m['max_rel_ex']:.3e}  Ey={m['max_rel_ey']:.3e}  "
            f"scale={m['ref_scale']:.3e} V/m"
        )

    # n_theta sweep on case #3 (random uniform disk)
    name, x, y, lam = cases[2]
    dist = _slice(x, y, lam)
    fm_ref = analytic.solve(dist)
    sweep_max_rel = []
    for n_theta in N_THETA_SWEEP:
        green = CircularGreenSpaceChargeSolver(
            radius=RADIUS, n_r=N_R_FIXED, n_theta=n_theta
        )
        fm_g = green.solve(dist)
        m = _metrics(fm_g.ex, fm_g.ey, fm_ref.ex, fm_ref.ey)
        sweep_max_rel.append(max(m["max_rel_ex"], m["max_rel_ey"]))
        summary["n_theta_convergence"].append(
            {"n_theta": int(n_theta), "max_rel": sweep_max_rel[-1]}
        )
        print(f"[sweep] n_theta={n_theta:>4d}  max rel={sweep_max_rel[-1]:.3e}")

    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.loglog(N_THETA_SWEEP, sweep_max_rel, "o-")
    ax.set_xlabel("n_theta")
    ax.set_ylabel("max relative error vs analytic")
    ax.set_title(f"CircularGreen Eu/Er error on random uniform disk (n_r={N_R_FIXED})")
    ax.grid(True, which="both", alpha=0.4)
    fig_path = args.output_dir / "n_theta_convergence.png"
    fig.savefig(fig_path, dpi=160)

    summary_path = args.output_dir / "static_summary.json"
    arrays_path = args.output_dir / "static_arrays.npz"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(arrays_path, **arrays_out)

    print(f"summary={summary_path}")
    print(f"arrays={arrays_path}")
    print(f"convergence_plot={fig_path}")


if __name__ == "__main__":
    main()
