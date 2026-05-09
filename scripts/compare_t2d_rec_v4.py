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


def _velocity_arrays(history):
    return (
        np.vstack([snapshot.vxs for snapshot in history]),
        np.vstack([snapshot.vys for snapshot in history]),
        np.vstack([snapshot.vzs for snapshot in history]),
    )


def _position_arrays(history):
    return (
        np.vstack([snapshot.xs for snapshot in history]),
        np.vstack([snapshot.ys for snapshot in history]),
        np.asarray([snapshot.zs[0] for snapshot in history], dtype=float),
    )


def _slice_velocity_arrays(result):
    return (
        result.array("vx"),
        result.array("vy"),
        result.array("vz"),
    )


def _slice_position_arrays(result):
    return result.array("x"), result.array("y"), result.z


def _metrics(name, ref, test):
    diff = np.asarray(test) - np.asarray(ref)
    return {
        f"{name}_max_abs": float(np.max(np.abs(diff))),
        f"{name}_rms": float(np.sqrt(np.mean(diff * diff))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the refactored dcbeamtrack kernel against /mnt/d/OneDrive/pys/t2d_rec_V4.py parameters."
    )
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--snapshot-stride", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/t2d_rec_v4_compare"))
    parser.add_argument(
        "--match-legacy-eps0",
        action="store_true",
        help="Use the old t2dcore EPS0 value inside the new FDM solver for a same-constant comparison.",
    )
    args = parser.parse_args()

    legacy = _load_legacy_t2dcore()

    from partdist import SliceDistribution

    import dcbeamtrack.spacecharge.solver as solver_module
    from dcbeamtrack import AxialMagneticField, DcBeamSimulation, RectGrid, TrackConfig
    from dcbeamtrack.tracking.boris import velocity_to_momentum_evc

    if args.match_legacy_eps0:
        solver_module.EPS0 = legacy.EPS0

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
    seed = 2025

    eps0_script = 8.85418782e-12
    mu0_script = 1.25663706e-6
    c_script = 1.0 / np.sqrt(eps0_script * mu0_script)
    vz0 = c_script * np.sqrt(1.0 - 1.0 / (1.0 + voltage / 511000.0) ** 2)

    gnx = int(la / dxy)
    gny = int(lb / dxy)

    source = legacy.UniformDisk(rb, current, vz0)
    part_grid = legacy.TriGrid(((-rb, rb), (-rb, rb)), hmax=30e-7)
    initial_legacy = legacy.sample_particles_from_distribution_on_grid(
        source,
        part_grid,
        vz0,
        rng=np.random.default_rng(seed),
    )
    field_grid_legacy = legacy.RectGrid(((-la / 2.0, la / 2.0), (-lb / 2.0, lb / 2.0)), gnx, gny)

    bzm = 0.35
    lp_mm = 4.93
    z_mm = np.linspace(-3.0, 33.0, 10001)
    bz_t = bzm * np.sin((2.0 * np.pi / lp_mm) * z_mm)
    bdata = np.column_stack([z_mm, bz_t])

    legacy_ctrl = legacy.SimCtrl(
        dt=dt,
        steps=args.steps,
        z0=z0,
        vz0=vz0,
        Lz=lz,
        q_over_m=-q_over_m_abs,
        magnetic_field_data=bdata,
        backend="cpu",
        gpu_interp=True,
        push_on_gpu=True,
        snapshot_stride=10,
        precision=args.precision,
        pusher="boris_bzbr",
        show_progress=False,
    )
    legacy_history = legacy.run_simulation(initial_legacy, field_grid_legacy, legacy_ctrl)

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
    field_grid = RectGrid(
        x_nodes=np.asarray(field_grid_legacy.x_nodes, dtype=float),
        y_nodes=np.asarray(field_grid_legacy.y_nodes, dtype=float),
    )
    field = AxialMagneticField.from_array(bdata, z_unit="mm", include_radial=True)
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
    new_simulation = DcBeamSimulation.rect_fdm(
        grid=field_grid,
        magnetic_field=field,
        config=config,
    )
    new_result = new_simulation.run(initial_slice)

    if len(legacy_history) != len(new_result):
        raise RuntimeError(
            f"Snapshot count differs: legacy={len(legacy_history)}, new={len(new_result)}. "
            "Use --snapshot-stride 1 for the CPU legacy branch."
        )

    x_old, y_old, z_old = _position_arrays(legacy_history)
    vx_old, vy_old, vz_old = _velocity_arrays(legacy_history)
    x_new, y_new, z_new = _slice_position_arrays(new_result)
    vx_new, vy_new, vz_new = _slice_velocity_arrays(new_result)

    r_old = np.hypot(x_old, y_old)
    r_new = np.hypot(x_new, y_new)
    r_max_old = np.max(r_old, axis=1)
    r_max_new = np.max(r_new, axis=1)
    r_mean_old = np.average(r_old, axis=1, weights=np.abs(initial_legacy.macro_charges))
    r_mean_new = np.average(r_new, axis=1, weights=np.abs(initial_legacy.macro_charges))

    summary = {
        "steps_requested": args.steps,
        "effective_steps": len(legacy_history) - 1,
        "precision": args.precision,
        "match_legacy_eps0": bool(args.match_legacy_eps0),
        "particles": int(initial_legacy.num_particles),
        "snapshots": int(len(legacy_history)),
        "legacy_final_z_m": float(z_old[-1]),
        "new_final_z_m": float(z_new[-1]),
        "legacy_final_r_max_m": float(r_max_old[-1]),
        "new_final_r_max_m": float(r_max_new[-1]),
        "legacy_final_r_mean_m": float(r_mean_old[-1]),
        "new_final_r_mean_m": float(r_mean_new[-1]),
        "total_lam_legacy": float(np.sum(initial_legacy.macro_charges)),
        "total_lam_new": float(np.sum(new_result.initial.lam)),
    }
    for key, old_arr, new_arr in (
        ("x", x_old, x_new),
        ("y", y_old, y_new),
        ("z", z_old, z_new),
        ("vx", vx_old, vx_new),
        ("vy", vy_old, vy_new),
        ("vz", vz_old, vz_new),
        ("r_max", r_max_old, r_max_new),
        ("r_mean", r_mean_old, r_mean_new),
    ):
        summary.update(_metrics(key, old_arr, new_arr))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{args.precision}_{'legacy_eps0' if args.match_legacy_eps0 else 'default_eps0'}"
    summary_path = args.output_dir / f"compare_summary_{suffix}.json"
    npz_path = args.output_dir / f"compare_arrays_{suffix}.npz"

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
        lam=initial_legacy.macro_charges,
    )

    print(f"particles={summary['particles']}")
    print(f"snapshots={summary['snapshots']}")
    print(f"effective_steps={summary['effective_steps']}")
    print(f"legacy_final_z_m={summary['legacy_final_z_m']:.12e}")
    print(f"new_final_z_m={summary['new_final_z_m']:.12e}")
    print(f"x_max_abs={summary['x_max_abs']:.12e}")
    print(f"y_max_abs={summary['y_max_abs']:.12e}")
    print(f"vx_max_abs={summary['vx_max_abs']:.12e}")
    print(f"vy_max_abs={summary['vy_max_abs']:.12e}")
    print(f"vz_max_abs={summary['vz_max_abs']:.12e}")
    print(f"r_max_max_abs={summary['r_max_max_abs']:.12e}")
    print(f"r_mean_max_abs={summary['r_mean_max_abs']:.12e}")
    print(f"summary={summary_path}")
    print(f"arrays={npz_path}")


if __name__ == "__main__":
    main()
