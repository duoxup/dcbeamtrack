"""Compare new circular Green solver against legacy t2d_cir_V1.py parameters.

The legacy ``t2dcore.RUGrid`` carries a known bug in the ``Eu`` mirror-charge
formula (extra ``ra**2`` factor, see ``docs/status_and_plan.md``). The new
``CircularGreenSpaceChargeSolver`` implements the corrected derivation, so the
two paths are not expected to match point-for-point. Legacy results are
reported here as a reference only.
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


def _legacy_position_arrays(history):
    return (
        np.vstack([snapshot.xs for snapshot in history]),
        np.vstack([snapshot.ys for snapshot in history]),
        np.asarray([snapshot.zs[0] for snapshot in history], dtype=float),
    )


def _legacy_velocity_arrays(history):
    return (
        np.vstack([snapshot.vxs for snapshot in history]),
        np.vstack([snapshot.vys for snapshot in history]),
        np.vstack([snapshot.vzs for snapshot in history]),
    )


def _prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare new circular Green solver against legacy t2d_cir_V1 parameters."
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument(
        "--seed",
        type=int,
        default=2000,
        help="Seed used by ParticleDistribution2D.randomize_directions (matches t2d_cir_V1 loop).",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
    )
    parser.add_argument(
        "--n-r",
        type=int,
        default=200,
        help="Polar grid radial bin count (matches legacy RUGrid gnr=200).",
    )
    parser.add_argument(
        "--n-theta",
        type=int,
        default=360,
        help="Polar grid angular bin count (matches legacy RUGrid gnu=360).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/t2d_cir_v1_compare"),
    )
    args = parser.parse_args()

    legacy = _load_legacy_t2dcore()

    from partdist import SliceDistribution

    from dcbeamtrack import (
        CircularBoundary,
        CircularGreenSpaceChargeSolver,
        DcBeamSimulation,
        TrackConfig,
        UniformMagneticField,
    )
    from dcbeamtrack.tracking import velocity_to_momentum_evc

    # --- t2d_cir_V1 parameters ---
    current = -0.015
    voltage = 19000.0
    la = 0.3e-3
    lb = 0.3e-3
    lz = 30e-3
    dt = 1e-13
    dxy = 30e-7
    q_over_m_abs = 1.76e11
    z0 = 0.0
    rb = 35e-6
    sample_seed = 2025

    eps0_script = 8.85418782e-12
    mu0_script = 1.25663706e-6
    c_script = 1.0 / np.sqrt(eps0_script * mu0_script)
    vz0 = c_script * np.sqrt(1.0 - 1.0 / (1.0 + voltage / 511000.0) ** 2)

    pipe_radius = 150e-6
    bz_uniform = 0.6

    # --- legacy initial particles + transverse randomization ---
    source_legacy = legacy.UniformDisk(rb, current, vz0)
    part_grid_legacy = legacy.TriGrid(((-rb, rb), (-rb, rb)), hmax=24e-7)
    initial_legacy = legacy.sample_particles_from_distribution_on_grid(
        source_legacy,
        part_grid_legacy,
        vz0,
        rng=np.random.default_rng(sample_seed),
    )
    initial_legacy.randomize_directions(2e-3, mode="rebuild_transverse", seed=args.seed)

    field_grid_legacy = legacy.RUGrid(pipe_radius, gnr=args.n_r, gnu=args.n_theta)

    bz_zps = np.linspace(-10.0, 40.0, 50)
    bz_table = np.full_like(bz_zps, bz_uniform)
    bdata_legacy = np.vstack([bz_zps, bz_table]).T

    legacy_ctrl = legacy.SimCtrl(
        dt=dt,
        steps=args.steps,
        z0=z0,
        vz0=vz0,
        Lz=lz,
        q_over_m=-q_over_m_abs,
        magnetic_field_data=bdata_legacy,
        backend="cpu",
        gpu_interp=True,
        push_on_gpu=True,
        snapshot_stride=args.snapshot_stride,
        precision=args.precision,
        pusher="boris_bz",
        show_progress=False,
    )
    legacy_history = legacy.run_simulation(initial_legacy, field_grid_legacy, legacy_ctrl)

    # --- new circular Green setup with the same initial particles ---
    px, py, pz = velocity_to_momentum_evc(
        initial_legacy.vxs,
        initial_legacy.vys,
        initial_legacy.vzs,
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
    solver = CircularGreenSpaceChargeSolver(
        radius=pipe_radius, n_r=args.n_r, n_theta=args.n_theta
    )
    boundary = CircularBoundary(radius=pipe_radius)
    new_simulation = DcBeamSimulation.from_solver(
        spacecharge_solver=solver,
        magnetic_field=magnetic,
        config=config,
        boundary=boundary,
    )
    new_result = new_simulation.run(initial_slice)

    # --- align snapshots ---
    n_old = len(legacy_history)
    n_new = len(new_result)
    n_common = min(n_old, n_new)

    weights = np.abs(initial_legacy.macro_charges)

    x_old, y_old, z_old = _legacy_position_arrays(legacy_history)
    vx_old, vy_old, vz_old = _legacy_velocity_arrays(legacy_history)
    x_old, y_old, vx_old, vy_old, vz_old = (
        x_old[:n_common],
        y_old[:n_common],
        vx_old[:n_common],
        vy_old[:n_common],
        vz_old[:n_common],
    )
    z_old = z_old[:n_common]

    x_new = new_result.array("x")[:n_common]
    y_new = new_result.array("y")[:n_common]
    vx_new = new_result.array("vx")[:n_common]
    vy_new = new_result.array("vy")[:n_common]
    vz_new = new_result.array("vz")[:n_common]
    z_new = new_result.z[:n_common]

    # Legacy envelope kept inline: legacy_history is a list of legacy
    # snapshot objects, not a SliceTrackResult, so the new API does not apply.
    r_old = np.hypot(x_old, y_old)
    r_max_old = np.max(r_old, axis=1)
    r_mean_old = np.average(r_old, axis=1, weights=weights)

    # New-side envelope arrays for cross-comparison and npz dump.
    z_new_full, r_max_new_full = new_result.radial_envelope("max")
    _, r_mean_new_full = new_result.radial_envelope("mean")
    r_max_new = r_max_new_full[:n_common]
    r_mean_new = r_mean_new_full[:n_common]

    wall_hit_final = np.asarray(
        new_result.final.get_data("wall_hit"), dtype=bool
    )

    summary = {
        "steps_requested": args.steps,
        "snapshots_legacy": int(n_old),
        "snapshots_new": int(n_new),
        "snapshots_compared": int(n_common),
        "particles": int(initial_legacy.num_particles),
        "precision": args.precision,
        "seed": int(args.seed),
        "n_r": int(args.n_r),
        "n_theta": int(args.n_theta),
        "pipe_radius_m": pipe_radius,
        "bz_uniform_T": bz_uniform,
        "vz0_m_per_s": float(vz0),
        "legacy_final_z_m": float(z_old[-1]),
        "legacy_final_r_max_m": float(r_max_old[-1]),
        "legacy_final_r_mean_m": float(r_mean_old[-1]),
        "r_max_max_abs_diff": float(np.max(np.abs(r_max_new - r_max_old))),
        "r_max_max_rel_diff": float(
            np.max(np.abs(r_max_new - r_max_old) / np.maximum(np.abs(r_max_old), 1e-30))
        ),
        "r_mean_max_abs_diff": float(np.max(np.abs(r_mean_new - r_mean_old))),
        "r_mean_max_rel_diff": float(
            np.max(np.abs(r_mean_new - r_mean_old) / np.maximum(np.abs(r_mean_old), 1e-30))
        ),
        "legacy_buggy_eu": True,
        "note": (
            "Legacy uses t2dcore.RUGrid which carries the Eu formula bug; "
            "results are reported for reference only and are NOT ground truth."
        ),
    }
    summary.update(_prefix(new_result.envelope_summary(), "new"))
    summary.update(_prefix(new_result.final_summary(), "new"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"steps{args.steps}_seed{args.seed}_{args.precision}"
    summary_path = args.output_dir / f"summary_{suffix}.json"
    npz_path = args.output_dir / f"arrays_{suffix}.npz"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez(
        npz_path,
        z_old=z_old,
        z_new=z_new,
        r_max_old=r_max_old,
        r_max_new=r_max_new,
        r_mean_old=r_mean_old,
        r_mean_new=r_mean_new,
        x_old_final=x_old[-1],
        y_old_final=y_old[-1],
        x_new_final=x_new[-1],
        y_new_final=y_new[-1],
        vx_old_final=vx_old[-1],
        vy_old_final=vy_old[-1],
        vz_old_final=vz_old[-1],
        vx_new_final=vx_new[-1],
        vy_new_final=vy_new[-1],
        vz_new_final=vz_new[-1],
        wall_hit_new=wall_hit_final,
        weights=weights,
    )

    print(f"particles={summary['particles']}")
    print(f"snapshots compared={summary['snapshots_compared']}")
    print(f"new   final r_max [um]={summary['new_r_max_final'] * 1e6:.6f}")
    print(f"legacy final r_max [um]={r_max_old[-1] * 1e6:.6f}  (buggy Eu, reference)")
    print(f"new   final r_mean [um]={summary['new_r_mean_final'] * 1e6:.6f}")
    print(f"legacy final r_mean [um]={r_mean_old[-1] * 1e6:.6f}  (buggy Eu, reference)")
    print(f"r_max max abs diff [um]={summary['r_max_max_abs_diff'] * 1e6:.6f}")
    print(f"r_max max rel diff     ={summary['r_max_max_rel_diff']:.4e}")
    print(f"r_mean max abs diff [um]={summary['r_mean_max_abs_diff'] * 1e6:.6f}")
    print(f"r_mean max rel diff    ={summary['r_mean_max_rel_diff']:.4e}")
    print(f"new wall_hit count     ={summary['new_n_wall_hit_final']}")
    print(f"summary={summary_path}")
    print(f"arrays={npz_path}")


if __name__ == "__main__":
    main()
