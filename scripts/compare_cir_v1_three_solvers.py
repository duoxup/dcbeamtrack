"""Three-solver end-to-end comparison at t2d_cir_V1 params.

Same legacy initial-particle construction path as scripts/compare_t2d_cir_v1.py
and scripts/compare_cir_v1_analytic.py. Runs three solver paths in turn and
reports per-snapshot envelope agreement and per-step wall-clock timing:

    1. RectFdmCircularBoundarySolver (new)
    2. CircularGreenSpaceChargeSolver
    3. AnalyticMirrorChargeSpaceChargeSolver (ground truth)

Diagnostic only — no hard pass/fail.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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


def _diff_metrics(label_a: str, arr_a: np.ndarray, arr_b: np.ndarray) -> dict:
    abs_diff = np.abs(arr_a - arr_b)
    rel_diff = abs_diff / np.maximum(np.abs(arr_b), 1e-30)
    return {
        f"{label_a}_max_abs_diff": float(np.max(abs_diff)),
        f"{label_a}_max_rel_diff": float(np.max(rel_diff)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--n-r", type=int, default=200)
    parser.add_argument("--n-theta", type=int, default=360)
    parser.add_argument("--fdm-spacing", type=float, default=1.5e-6)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/cir_v1_three_solvers")
    )
    args = parser.parse_args()

    legacy = _load_legacy_t2dcore()

    from partdist import SliceDistribution

    from dcbeamtrack import (
        AnalyticMirrorChargeSpaceChargeSolver,
        CircularBoundary,
        DcBeamSimulation,
        RectGrid,
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

    # Solver 1: RectFdmCircularBoundarySolver
    fdm_grid_extent = 2.0 * pipe_radius + 2.0 * args.fdm_spacing
    fdm_grid = RectGrid.from_size(
        width=fdm_grid_extent, height=fdm_grid_extent, spacing=args.fdm_spacing
    )
    sim_fdm = DcBeamSimulation.rect_fdm_circular(
        grid=fdm_grid,
        radius=pipe_radius,
        magnetic_field=magnetic,
        config=config,
    )

    # Solver 2: CircularGreenSpaceChargeSolver
    sim_green = DcBeamSimulation.circular_green(
        radius=pipe_radius,
        magnetic_field=magnetic,
        config=config,
        n_r=args.n_r,
        n_theta=args.n_theta,
    )

    # Solver 3: AnalyticMirrorChargeSpaceChargeSolver (ground truth)
    sim_analytic = DcBeamSimulation.from_solver(
        spacecharge_solver=AnalyticMirrorChargeSpaceChargeSolver(radius=pipe_radius),
        magnetic_field=magnetic,
        config=config,
        boundary=CircularBoundary(radius=pipe_radius),
    )

    def _run_timed(sim, label):
        t0 = time.perf_counter()
        result = sim.run(initial_slice)
        wall_s = time.perf_counter() - t0
        n_snap = len(result)
        per_snap_ms = 1e3 * wall_s / max(n_snap, 1)
        print(f"[{label}] wall={wall_s:.2f}s  snapshots={n_snap}  per_snap={per_snap_ms:.2f} ms")
        return result, wall_s, n_snap, per_snap_ms

    res_fdm, wall_fdm, n_fdm, ps_fdm = _run_timed(sim_fdm, "fdm")
    res_green, wall_green, n_green, ps_green = _run_timed(sim_green, "green")
    res_analytic, wall_analytic, n_analytic, ps_analytic = _run_timed(sim_analytic, "analytic")

    n = min(n_fdm, n_green, n_analytic)
    weights = np.abs(initial_legacy.macro_charges)

    def _slice_to_n(result):
        return (
            result.array("x")[:n],
            result.array("y")[:n],
            result.z[:n],
        )

    x_f, y_f, z_f = _slice_to_n(res_fdm)
    x_g, y_g, z_g = _slice_to_n(res_green)
    x_a, y_a, z_a = _slice_to_n(res_analytic)

    r_max_f, r_mean_f = _envelope(x_f, y_f, weights)
    r_max_g, r_mean_g = _envelope(x_g, y_g, weights)
    r_max_a, r_mean_a = _envelope(x_a, y_a, weights)

    wall_count_f = int(
        np.sum(np.asarray(res_fdm.final.get_data("wall_hit"), dtype=bool))
    )
    wall_count_g = int(
        np.sum(np.asarray(res_green.final.get_data("wall_hit"), dtype=bool))
    )
    wall_count_a = int(
        np.sum(np.asarray(res_analytic.final.get_data("wall_hit"), dtype=bool))
    )

    summary = {
        "steps_requested": args.steps,
        "snapshots_compared": int(n),
        "particles": int(initial_legacy.num_particles),
        "precision": args.precision,
        "seed": int(args.seed),
        "n_r_green": int(args.n_r),
        "n_theta_green": int(args.n_theta),
        "fdm_spacing_m": float(args.fdm_spacing),
        "pipe_radius_m": pipe_radius,
        "bz_uniform_T": bz_uniform,
        "vz0_m_per_s": float(vz0),
        "wall_seconds": {
            "fdm": float(wall_fdm),
            "green": float(wall_green),
            "analytic": float(wall_analytic),
        },
        "per_snapshot_ms": {
            "fdm": float(ps_fdm),
            "green": float(ps_green),
            "analytic": float(ps_analytic),
        },
        "wall_hit": {
            "fdm": wall_count_f,
            "green": wall_count_g,
            "analytic": wall_count_a,
        },
        "final_r_max_um": {
            "fdm": float(r_max_f[-1] * 1e6),
            "green": float(r_max_g[-1] * 1e6),
            "analytic": float(r_max_a[-1] * 1e6),
        },
        "final_r_mean_um": {
            "fdm": float(r_mean_f[-1] * 1e6),
            "green": float(r_mean_g[-1] * 1e6),
            "analytic": float(r_mean_a[-1] * 1e6),
        },
    }
    summary["fdm_vs_analytic_r_max"] = _diff_metrics("fdm_vs_analytic_r_max", r_max_f, r_max_a)
    summary["fdm_vs_analytic_r_mean"] = _diff_metrics("fdm_vs_analytic_r_mean", r_mean_f, r_mean_a)
    summary["green_vs_analytic_r_max"] = _diff_metrics("green_vs_analytic_r_max", r_max_g, r_max_a)
    summary["green_vs_analytic_r_mean"] = _diff_metrics("green_vs_analytic_r_mean", r_mean_g, r_mean_a)
    summary["note"] = (
        "Analytic mirror-charge solver is ground truth (exact within FP). "
        "FDM uses staircase circular Dirichlet on rect grid; Green uses polar "
        "snap+linear interp on a (n_r, n_theta) grid."
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"steps{args.steps}_seed{args.seed}_{args.precision}"
    summary_path = args.output_dir / f"summary_{suffix}.json"
    npz_path = args.output_dir / f"arrays_{suffix}.npz"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        z_fdm=z_f,
        z_green=z_g,
        z_analytic=z_a,
        r_max_fdm=r_max_f,
        r_max_green=r_max_g,
        r_max_analytic=r_max_a,
        r_mean_fdm=r_mean_f,
        r_mean_green=r_mean_g,
        r_mean_analytic=r_mean_a,
        x_fdm_final=x_f[-1],
        y_fdm_final=y_f[-1],
        x_green_final=x_g[-1],
        y_green_final=y_g[-1],
        x_analytic_final=x_a[-1],
        y_analytic_final=y_a[-1],
        weights=weights,
    )

    print()
    print(f"particles               = {summary['particles']}")
    print(f"snapshots compared      = {n}")
    print(f"FDM    final r_max [um] = {r_max_f[-1] * 1e6:.6f}")
    print(f"Green  final r_max [um] = {r_max_g[-1] * 1e6:.6f}")
    print(f"Analyt final r_max [um] = {r_max_a[-1] * 1e6:.6f}")
    print(f"FDM   vs Analytic r_max max rel diff = {summary['fdm_vs_analytic_r_max']['fdm_vs_analytic_r_max_max_rel_diff']:.4e}")
    print(f"Green vs Analytic r_max max rel diff = {summary['green_vs_analytic_r_max']['green_vs_analytic_r_max_max_rel_diff']:.4e}")
    print(f"per-snapshot timing [ms]: fdm={ps_fdm:.2f}  green={ps_green:.2f}  analytic={ps_analytic:.2f}")
    print(f"wall_hit counts: fdm={wall_count_f}  green={wall_count_g}  analytic={wall_count_a}")
    print(f"summary={summary_path}")
    print(f"arrays={npz_path}")


if __name__ == "__main__":
    main()
