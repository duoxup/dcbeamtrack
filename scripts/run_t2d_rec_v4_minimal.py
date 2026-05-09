from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dcbeamtrack import (
    AxialMagneticField,
    DcBeamSimulation,
    RectGrid,
    TrackConfig,
    TriGrid,
    UniformDisk,
    sample_slice_from_density_on_grid,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--snapshot-stride", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/t2d_rec_v4"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    e0 = 8.85418782e-12
    u0 = 1.25663706e-6
    c = 1.0 / np.sqrt(e0 * u0)
    yita = 1.76e11

    current = -0.015
    voltage = 19000.0
    la = 0.3e-3
    lb = 0.3e-3
    lz = 30e-3
    dt = 1e-13
    dxy = 30e-7
    z0 = 0.0
    rb = 35e-6
    bzm = 0.35
    lp_mm = 4.93

    vz0 = c * np.sqrt(1.0 - 1.0 / (1.0 + voltage / 511000.0) ** 2)

    source = UniformDisk(radius=rb, current=current, vz=vz0)
    n_tri_axis = 24
    x_nodes = np.linspace(-rb, rb, n_tri_axis + 1)
    y_nodes = np.linspace(-rb, rb, n_tri_axis + 1)
    particle_grid = TriGrid(x_nodes=x_nodes, y_nodes=y_nodes)
    initial = sample_slice_from_density_on_grid(
        source,
        grid=particle_grid,
        z0=z0,
        vz0=vz0,
        remove_small_charge=True,
        min_lam_fraction=1e-3,
        renormalize=True,
    )

    field_grid = RectGrid.from_size(width=la, height=lb, spacing=dxy)
    z_mm = np.linspace(-3.0, 33.0, 10001)
    bz_t = bzm * np.sin((2.0 * np.pi / lp_mm) * z_mm)
    magnetic = AxialMagneticField.from_z_bz(z_mm, bz_t, z_unit="mm", include_radial=True)

    config = TrackConfig(
        dt=dt,
        steps=args.steps,
        z0=z0,
        vz0=vz0,
        Lz=lz,
        q_over_m=-yita,
        deposition="CIC",
        snapshot_stride=args.snapshot_stride,
    )

    simulation = DcBeamSimulation.rect_fdm(
        grid=field_grid,
        magnetic_field=magnetic,
        config=config,
    )
    result = simulation.run(initial)

    z_env, r_max = result.radial_envelope("max")
    _, r_mean = result.radial_envelope("mean")
    _, r_rms = result.radial_envelope("rms")

    npz_path = args.output_dir / "t2d_rec_v4_minimal_result.npz"
    np.savez(
        npz_path,
        z=z_env,
        r_max=r_max,
        r_mean=r_mean,
        r_rms=r_rms,
        initial_x=initial.x,
        initial_y=initial.y,
        final_x=result.final.x,
        final_y=result.final.y,
        final_px=result.final.px,
        final_py=result.final.py,
        final_pz=result.final.pz,
        lam=initial.lam,
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=False, constrained_layout=True)
    axes[0].plot(z_env * 1e3, r_max * 1e6, label="max")
    axes[0].plot(z_env * 1e3, r_mean * 1e6, label="mean")
    axes[0].plot(z_env * 1e3, r_rms * 1e6, label="rms")
    axes[0].set_xlabel("z [mm]")
    axes[0].set_ylabel("r [um]")
    axes[0].grid(True)
    axes[0].legend()

    sample = np.linspace(0, result.n_particles - 1, min(40, result.n_particles), dtype=int)
    z_all = result.z * 1e3
    x_all = result.array("x")[:, sample]
    y_all = result.array("y")[:, sample]
    r_all = np.sqrt(x_all * x_all + y_all * y_all)
    axes[1].plot(z_all, r_all * 1e6, color="0.25", linewidth=0.8, alpha=0.65)
    axes[1].set_xlabel("z [mm]")
    axes[1].set_ylabel("sampled particle r [um]")
    axes[1].grid(True)

    fig_path = args.output_dir / "t2d_rec_v4_minimal_envelope.png"
    fig.savefig(fig_path, dpi=180)

    print(f"particles={result.n_particles}")
    print(f"snapshots={result.n_snapshots}")
    print(f"z_final_m={result.final.z:.9e}")
    print(f"r_max_final_m={r_max[-1]:.9e}")
    print(f"r_mean_final_m={r_mean[-1]:.9e}")
    print(f"r_rms_final_m={r_rms[-1]:.9e}")
    print(f"total_lam_initial={np.sum(initial.lam):.9e}")
    print(f"total_lam_final={np.sum(result.final.lam):.9e}")
    print(npz_path.resolve())
    print(fig_path.resolve())


if __name__ == "__main__":
    main()
