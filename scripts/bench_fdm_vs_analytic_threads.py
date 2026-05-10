"""Sweep FDM (scipy/pypardiso) vs analytic numba over thread count, N_part, grid.

Spawns one subprocess per (threads, n_part, spacing, solver) combination — MKL
and numba honor MKL_NUM_THREADS / NUMBA_NUM_THREADS at import time only, so
subprocess isolation is the only reliable way to vary thread count.

Outputs:
  artifacts/bench_fdm_vs_analytic_threads/results.csv
  artifacts/bench_fdm_vs_analytic_threads/results.json
  artifacts/bench_fdm_vs_analytic_threads/speedup_vs_threads.png
  artifacts/bench_fdm_vs_analytic_threads/summary_threads_8.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parent.parent
INNER = REPO_ROOT / "scripts" / "_bench_inner.py"

SOLVERS = ("fdm_pardiso", "fdm_scipy", "analytic")
THREADS_FULL = (1, 2, 4, 8, 16)
N_PART_FULL = (1000, 2000, 5000, 10000)
SPACINGS_FULL = (3e-6, 1.5e-6)  # → grid 100×100 / 200×200

THREADS_QUICK = (2, 8)
N_PART_QUICK = (1000, 5000)
SPACINGS_QUICK = (3e-6,)


def _run_one(threads: int, n_part: int, spacing: float, solver: str, n_iter: int) -> dict:
    env = os.environ.copy()
    env["MKL_NUM_THREADS"] = str(threads)
    env["NUMBA_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    cmd = [
        sys.executable,
        str(INNER),
        "--solver",
        solver,
        "--n-part",
        str(n_part),
        "--spacing",
        str(spacing),
        "--n-iter",
        str(n_iter),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
    last_line = proc.stdout.strip().splitlines()[-1]
    record = json.loads(last_line)
    record["requested_threads"] = threads
    return record


def _plot_threads_sweep(records: list[dict], n_parts: tuple, spacings: tuple, out_path: Path) -> None:
    n_panels = len(n_parts) * len(spacings)
    n_cols = max(1, len(spacings))
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False, constrained_layout=True
    )

    for i, n_part in enumerate(n_parts):
        for j, spacing in enumerate(spacings):
            ax = axes[i][j]
            for solver in SOLVERS:
                xs, ys = [], []
                for r in records:
                    if (
                        r["n_part"] == n_part
                        and abs(r["spacing"] - spacing) < 1e-12
                        and r["solver"] == solver
                    ):
                        xs.append(r["requested_threads"])
                        ys.append(r["mean_ms"])
                if xs:
                    order = np.argsort(xs)
                    ax.loglog(np.array(xs)[order], np.array(ys)[order], "o-", label=solver)
            ax.set_xlabel("threads")
            ax.set_ylabel("mean ms / solve")
            ax.set_title(f"N={n_part}, spacing={spacing:g}")
            ax.grid(True, which="both", alpha=0.4)
            ax.legend(fontsize=8)
    fig.savefig(out_path, dpi=160)


def _summary_threads_8(records: list[dict], threads_target: int = 8) -> str:
    lines = [f"# All configurations at threads={threads_target}, sorted by mean_ms"]
    lines.append("solver           n_part  spacing      mean_ms  std_ms")
    matched = [r for r in records if r["requested_threads"] == threads_target]
    matched.sort(key=lambda r: r["mean_ms"])
    for r in matched:
        lines.append(
            f"{r['solver']:<16} {r['n_part']:>6}  {r['spacing']:>9.2e}  "
            f"{r['mean_ms']:>8.3f}  {r['std_ms']:>6.3f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-iter", type=int, default=50)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Reduced sweep: 2 threads × 2 N × 1 spacing × 3 solvers = 12 runs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/bench_fdm_vs_analytic_threads"),
    )
    args = parser.parse_args()

    threads_list = THREADS_QUICK if args.quick else THREADS_FULL
    n_parts = N_PART_QUICK if args.quick else N_PART_FULL
    spacings = SPACINGS_QUICK if args.quick else SPACINGS_FULL

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    total = len(threads_list) * len(n_parts) * len(spacings) * len(SOLVERS)
    counter = 0
    for threads in threads_list:
        for n_part in n_parts:
            for spacing in spacings:
                for solver in SOLVERS:
                    counter += 1
                    print(
                        f"[{counter}/{total}] threads={threads} n_part={n_part} "
                        f"spacing={spacing:g} solver={solver}",
                        flush=True,
                    )
                    rec = _run_one(threads, n_part, spacing, solver, args.n_iter)
                    print(
                        f"    mean_ms={rec['mean_ms']:.3f}  std_ms={rec['std_ms']:.3f}  "
                        f"reported_threads={rec['threads']}",
                        flush=True,
                    )
                    records.append(rec)

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    png_path = args.output_dir / "speedup_vs_threads.png"
    txt_path = args.output_dir / "summary_threads_8.txt"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    if records:
        keys = list(records[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)

    _plot_threads_sweep(records, n_parts, spacings, png_path)

    txt_path.write_text(_summary_threads_8(records), encoding="utf-8")

    print()
    print(f"results = {json_path}")
    print(f"csv     = {csv_path}")
    print(f"plot    = {png_path}")
    print(f"summary = {txt_path}")


if __name__ == "__main__":
    main()
