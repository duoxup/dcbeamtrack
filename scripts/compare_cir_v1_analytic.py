"""End-to-end tracking: analytic mirror-charge vs CircularGreen at t2d_cir_V1 params.

Uses the same legacy initial-particle construction path as
``scripts/compare_t2d_cir_v1.py`` (TriGrid + UniformDisk +
randomize_directions(2 mrad, rebuild_transverse, seed)) so the three
comparison scripts in this directory share one definition of the initial slice.
Diagnostic only — no hard pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

OLD_CODE_DIR = Path("/mnt/d/OneDrive/pys")


def _load_legacy_t2dcore():
    if str(OLD_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(OLD_CODE_DIR))
    import t2dcore as legacy

    return legacy


def _envelope(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    r = np.hypot(x, y)
    return np.max(r, axis=1), np.average(r, axis=1, weights=weights)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--n-r", type=int, default=200)
    parser.add_argument("--n-theta", type=int, default=360)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cir_v1_analytic")
    )
    args = parser.parse_args()

    legacy = _load_legacy_t2dcore()

    from partdist import SliceDistribution

    from dcbeamtrack import (
        AnalyticMirrorChargeSpaceChargeSolver,
        CircularBoundary,
        DcBeamSimulation,
        TrackConfig,
        UniformMagneticField,
    )
    from dcbeamtrack.tracking import velocity_to_momentum_evc

    current = -0.015
    voltage = 19000.0
    lz = 30e-3
    dt = 1e-13
    q_over_m_abs = 1.76e11
    z0 = 0.0
    rb = 35e-6
    sample_seed = 2025
    pipe_radius = 150e-6
    bz_uniform = 0.6

    eps0_script = 8.85418782e-12
    mu0_script = 1.25663706e-6
    c_script = 1.0 / np.sqrt(eps0_script * mu0_script)
    vz0 = c_script * np.sqrt(1.0 - 1.0 / (1.0 + voltage / 511000.0) ** 2)

    source_legacy = legacy.UniformDisk(rb, current, vz0)
    part_grid_legacy = legacy.TriGrid(((-rb, rb), (-rb, rb)), hmax=24e-7)
    initial_legacy = legacy.sample_particles_from_distribution_on_grid(
        source_legacy,
        part_grid_legacy,
        vz0,
        rng=np.random.default_rng(sample_seed),
    )
    initial_legacy.randomize_directions(2e-3, mode="rebuild_transverse", seed=args.seed)

    px, py, pz = velocity_to_momentum_evc(
        initial_legacy.vxs, initial_legacy.vys, initial_legacy.vzs
    )
    initial_slice = SliceDistribution(
        z=z0,
        x=initial_legacy.xs,
        y=initial_legacy.ys,
        px=px,
        py=py,
        pz=pz,
        t=np.zeros(initial_legacy.num_particles),
        lam=initial_legacy.macro_charges,
    )

    magnetic = UniformMagneticField(bz=bz_uniform)
    config = TrackConfig(
        dt=dt,
        steps=args.steps,
        z0=z0,
        vz0=vz0,
        Lz=lz,
        q_over_m=-q_over_m_abs,
        deposition="CIC",
        snapshot_stride=args.snapshot_stride,
        precision=args.precision,
    )

    analytic_solver = AnalyticMirrorChargeSpaceChargeSolver(radius=pipe_radius)
    boundary = CircularBoundary(radius=pipe_radius)
    sim_analytic = DcBeamSimulation.from_solver(
        spacecharge_solver=analytic_solver,
        magnetic_field=magnetic,
        config=config,
        boundary=boundary,
    )
    sim_green = DcBeamSimulation.circular_green(
        radius=pipe_radius,
        magnetic_field=magnetic,
        config=config,
        n_r=args.n_r,
        n_theta=args.n_theta,
    )

    res_analytic = sim_analytic.run(initial_slice)
    res_green = sim_green.run(initial_slice)

    n_a = len(res_analytic)
    n_g = len(res_green)
    n = min(n_a, n_g)

    weights = np.abs(initial_legacy.macro_charges)
    x_a = res_analytic.array("x")[:n]
    y_a = res_analytic.array("y")[:n]
    vx_a = res_analytic.array("vx")[:n]
    vy_a = res_analytic.array("vy")[:n]
    vz_a = res_analytic.array("vz")[:n]
    z_a = res_analytic.z[:n]
    x_g = res_green.array("x")[:n]
    y_g = res_green.array("y")[:n]
    vx_g = res_green.array("vx")[:n]
    vy_g = res_green.array("vy")[:n]
    vz_g = res_green.array("vz")[:n]
    z_g = res_green.z[:n]

    r_max_a, r_mean_a = _envelope(x_a, y_a, weights)
    r_max_g, r_mean_g = _envelope(x_g, y_g, weights)

    wall_a = int(np.sum(np.asarray(res_analytic.final.get_data("wall_hit"), dtype=bool)))
    wall_g = int(np.sum(np.asarray(res_green.final.get_data("wall_hit"), dtype=bool)))

    summary = {
        "steps_requested": args.steps,
        "snapshots_analytic": int(n_a),
        "snapshots_green": int(n_g),
        "snapshots_compared": int(n),
        "particles": int(initial_legacy.num_particles),
        "precision": args.precision,
        "seed": int(args.seed),
        "n_r_green": int(args.n_r),
        "n_theta_green": int(args.n_theta),
        "pipe_radius_m": pipe_radius,
        "bz_uniform_T": bz_uniform,
        "vz0_m_per_s": float(vz0),
        "analytic_final_z_m": float(z_a[-1]),
        "green_final_z_m": float(z_g[-1]),
        "analytic_final_r_max_m": float(r_max_a[-1]),
        "green_final_r_max_m": float(r_max_g[-1]),
        "analytic_final_r_mean_m": float(r_mean_a[-1]),
        "green_final_r_mean_m": float(r_mean_g[-1]),
        "r_max_max_abs_diff": float(np.max(np.abs(r_max_g - r_max_a))),
        "r_max_max_rel_diff": float(
            np.max(np.abs(r_max_g - r_max_a) / np.maximum(np.abs(r_max_a), 1e-30))
        ),
        "r_mean_max_abs_diff": float(np.max(np.abs(r_mean_g - r_mean_a))),
        "r_mean_max_rel_diff": float(
            np.max(np.abs(r_mean_g - r_mean_a) / np.maximum(np.abs(r_mean_a), 1e-30))
        ),
        "wall_hit_analytic": wall_a,
        "wall_hit_green": wall_g,
        "note": (
            "Analytic mirror-charge solver is the ground truth (exact within FP). "
            "CircularGreen uses polar grid + radial snap + angular linear interp."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"steps{args.steps}_seed{args.seed}_{args.precision}"
    summary_path = args.output_dir / f"summary_{suffix}.json"
    npz_path = args.output_dir / f"arrays_{suffix}.npz"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        z_analytic=z_a,
        z_green=z_g,
        r_max_analytic=r_max_a,
        r_max_green=r_max_g,
        r_mean_analytic=r_mean_a,
        r_mean_green=r_mean_g,
        x_analytic_final=x_a[-1],
        y_analytic_final=y_a[-1],
        x_green_final=x_g[-1],
        y_green_final=y_g[-1],
        vx_analytic_final=vx_a[-1],
        vy_analytic_final=vy_a[-1],
        vz_analytic_final=vz_a[-1],
        vx_green_final=vx_g[-1],
        vy_green_final=vy_g[-1],
        vz_green_final=vz_g[-1],
        weights=weights,
    )

    print(f"particles={summary['particles']}")
    print(f"snapshots compared={summary['snapshots_compared']}")
    print(f"analytic final r_max [um]={r_max_a[-1] * 1e6:.6f}")
    print(f"green    final r_max [um]={r_max_g[-1] * 1e6:.6f}")
    print(f"analytic final r_mean [um]={r_mean_a[-1] * 1e6:.6f}")
    print(f"green    final r_mean [um]={r_mean_g[-1] * 1e6:.6f}")
    print(f"r_max max abs diff [um]={summary['r_max_max_abs_diff'] * 1e6:.6f}")
    print(f"r_max max rel diff     ={summary['r_max_max_rel_diff']:.4e}")
    print(f"r_mean max abs diff [um]={summary['r_mean_max_abs_diff'] * 1e6:.6f}")
    print(f"r_mean max rel diff    ={summary['r_mean_max_rel_diff']:.4e}")
    print(f"wall_hit analytic={wall_a}  green={wall_g}")
    print(f"summary={summary_path}")
    print(f"arrays={npz_path}")


if __name__ == "__main__":
    main()
