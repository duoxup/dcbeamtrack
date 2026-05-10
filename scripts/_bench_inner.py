"""Bench inner script: time one (solver, n_part, spacing) configuration.

Run as a subprocess from bench_fdm_vs_analytic_threads.py controller.
Reads MKL_NUM_THREADS / NUMBA_NUM_THREADS / OMP_NUM_THREADS from env (set by
controller before spawning) so MKL and numba both honor the requested thread
count. Prints one JSON line to stdout with timing results.

Standalone usage (for debugging a single config):

    MKL_NUM_THREADS=8 NUMBA_NUM_THREADS=8 OMP_NUM_THREADS=8 \\
      PYTHONPATH=src python scripts/_bench_inner.py \\
      --solver fdm_pardiso --n-part 2000 --spacing 1.5e-6 --n-iter 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


PIPE_RADIUS = 150e-6  # m, fixed for circular-pipe geometry


def _build_dist(n_part: int, seed: int = 1234):
    from partdist import SliceDistribution
    from dcbeamtrack.tracking import velocity_to_momentum_evc

    rng = np.random.default_rng(seed)
    r = PIPE_RADIUS * 0.7 * np.sqrt(rng.uniform(0.05, 1.0, size=n_part))
    th = rng.uniform(0.0, 2.0 * np.pi, size=n_part)
    x = r * np.cos(th)
    y = r * np.sin(th)
    lam = np.full(n_part, 1.0e-12)
    vx = vy = vz = np.zeros(n_part)
    px, py, pz = velocity_to_momentum_evc(vx, vy, vz)
    return SliceDistribution(
        z=0.0, x=x, y=y, px=px, py=py, pz=pz, t=np.zeros(n_part), lam=lam
    )


def _build_solver(solver_name: str, spacing: float):
    from dcbeamtrack import (
        AnalyticMirrorChargeSpaceChargeSolver,
        RectFdmCircularBoundarySolver,
        RectGrid,
    )

    if solver_name == "analytic":
        return AnalyticMirrorChargeSpaceChargeSolver(radius=PIPE_RADIUS)

    extent = 2.0 * PIPE_RADIUS + 2.0 * spacing
    grid = RectGrid.from_size(width=extent, height=extent, spacing=spacing)

    if solver_name == "fdm_pardiso":
        return RectFdmCircularBoundarySolver(
            grid, radius=PIPE_RADIUS, backend="pypardiso"
        )
    if solver_name == "fdm_scipy":
        return RectFdmCircularBoundarySolver(
            grid, radius=PIPE_RADIUS, backend="scipy"
        )
    raise ValueError(f"unknown solver: {solver_name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--solver", choices=("fdm_pardiso", "fdm_scipy", "analytic"), required=True
    )
    parser.add_argument("--n-part", type=int, required=True)
    parser.add_argument("--spacing", type=float, required=True)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument("--n-warmup", type=int, default=5)
    args = parser.parse_args()

    threads = int(os.environ.get("MKL_NUM_THREADS", "0")) or int(
        os.environ.get("NUMBA_NUM_THREADS", "0")
    ) or 0

    solver = _build_solver(args.solver, args.spacing)
    dist = _build_dist(args.n_part)

    for _ in range(args.n_warmup):
        solver.solve(dist)

    times = np.empty(args.n_iter, dtype=float)
    for i in range(args.n_iter):
        t0 = time.perf_counter()
        solver.solve(dist)
        times[i] = time.perf_counter() - t0

    record = {
        "solver": args.solver,
        "n_part": int(args.n_part),
        "spacing": float(args.spacing),
        "threads": int(threads),
        "n_iter": int(args.n_iter),
        "mean_ms": float(np.mean(times) * 1e3),
        "std_ms": float(np.std(times) * 1e3),
        "min_ms": float(np.min(times) * 1e3),
    }
    sys.stdout.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
