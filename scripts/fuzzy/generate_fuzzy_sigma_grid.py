"""
generate_fuzzy_sigma_grid.py -- Generate fuzzy graphs for all sigma combinations

Sweeps all 6×6 = 36 combinations of (σ_spatial, σ_morpho) derived from the
retained-weight criterion, computed per task with calculate_sigma.py:

    exp(-median² / 2σ²) = r  →  σ = median / sqrt(2·ln(1/r))

r = 0.1 → median edge nearly suppressed; r = 0.9 → median edge nearly full
weight; med → σ = median distance (weight = e^-½ ≈ 0.607). The per-task
values are in SIGMA_SPATIAL / SIGMA_MORPHO below.

Output layout:
    results_graphs_november_23_fuzzy/
      sigmas_{r_s}_{r_m}/       e.g. sigmas_0.1_0.7, sigmas_med_med
        {task}/k_19/
          *.pt

Total runs: 36 combinations × 3 tasks = 108

Usage:
    # Preview all commands without running
    python scripts/fuzzy/generate_fuzzy_sigma_grid.py --dry-run

    # Run all combinations for all tasks
    python scripts/fuzzy/generate_fuzzy_sigma_grid.py

    # Run a single task only
    python scripts/fuzzy/generate_fuzzy_sigma_grid.py --task 3class

    # Skip combinations whose output dir already contains .pt files
    python scripts/fuzzy/generate_fuzzy_sigma_grid.py --skip-existing
"""

import argparse
import subprocess
import sys
from itertools import product
from pathlib import Path

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

# ---------------------------------------------------------------------------
# Sigma grid per task  (values from calculate_sigma.py, k=19)
# ---------------------------------------------------------------------------

# Keys are the retention-weight labels used in folder names
SIGMA_SPATIAL = {
    "2class": {"0.1": 0.0453, "0.3": 0.0626, "0.5": 0.0825, "0.7": 0.1151, "0.9": 0.2117, "med": 0.0972},
    "3class": {"0.1": 0.0447, "0.3": 0.0618, "0.5": 0.0815, "0.7": 0.1136, "0.9": 0.2089, "med": 0.0959},
    "4class": {"0.1": 0.0442, "0.3": 0.0611, "0.5": 0.0806, "0.7": 0.1123, "0.9": 0.2067, "med": 0.0949},
}

SIGMA_MORPHO = {
    "2class": {"0.1": 0.1241, "0.3": 0.1716, "0.5": 0.2261, "0.7": 0.3153, "0.9": 0.5800, "med": 0.2663},
    "3class": {"0.1": 0.1053, "0.3": 0.1456, "0.5": 0.1919, "0.7": 0.2675, "0.9": 0.4922, "med": 0.2259},
    "4class": {"0.1": 0.1057, "0.3": 0.1462, "0.5": 0.1927, "0.7": 0.2687, "0.9": 0.4944, "med": 0.2269},
}

# ---------------------------------------------------------------------------
# Task → input directory mapping
# ---------------------------------------------------------------------------

_BASE_IN = os.path.join(MOLSUB, "data", "BCNB", "results_graphs_november_23")

TASK_INPUT_DIRS = {
    "2class": os.path.join(_BASE_IN,
        "graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x", "graphs_k_19"),
    "3class": os.path.join(_BASE_IN,
        "graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x", "graphs_k_19"),
    "4class": os.path.join(_BASE_IN,
        "graphs_PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x", "graphs_k_19"),
}

_BASE_OUT = os.path.join(MOLSUB, "data", "BCNB", "results_graphs_november_23_fuzzy")

GENERATE_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "generate_fuzzy_graphs.py"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def output_dir(r_s_label: str, r_m_label: str, task: str) -> Path:
    return Path(_BASE_OUT) / f"sigmas_{r_s_label}_{r_m_label}" / task / "k_19"


def already_done(out: Path) -> bool:
    """True if the output dir exists and contains at least one .pt file."""
    return out.exists() and any(out.glob("*.pt"))


def build_command(r_s_label: str, r_m_label: str, task: str) -> list[str]:
    out = output_dir(r_s_label, r_m_label, task)
    return [
        sys.executable, GENERATE_SCRIPT,
        "--input-dir",     TASK_INPUT_DIRS[task],
        "--output-dir",    str(out),
        "--sigma-spatial", str(SIGMA_SPATIAL[task][r_s_label]),
        "--sigma-morpho",  str(SIGMA_MORPHO[task][r_m_label]),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate fuzzy graphs for all 36 sigma combinations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task", choices=list(TASK_INPUT_DIRS), default=None,
                        help="Run a single task only (default: all three).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip combinations whose output dir already has .pt files.")
    args = parser.parse_args()

    tasks = [args.task] if args.task else list(TASK_INPUT_DIRS)
    labels = ["0.1", "0.3", "0.5", "0.7", "0.9", "med"]
    combos = list(product(labels, labels)) # 36 (r_s, r_m) pairs

    total = len(combos) * len(tasks)
    print(f"Sigma grid: {len(labels)} spatial × {len(labels)} morpho = {len(combos)} combos")
    print(f"Tasks     : {tasks}")
    print(f"Total runs: {total}")
    if args.dry_run:
        print("(dry-run — no commands will be executed)\n")

    n_run = n_skip = n_fail = 0

    for task in tasks:
        for r_s, r_m in combos:
            out = output_dir(r_s, r_m, task)
            cmd = build_command(r_s, r_m, task)
            tag = f"sigmas_{r_s}_{r_m}/{task}"

            if args.skip_existing and already_done(out):
                print(f"  SKIP  {tag}  (already exists)")
                n_skip += 1
                continue

            cmd_str = " ".join(cmd)
            if args.dry_run:
                print(f"  CMD   {tag}\n        {cmd_str}\n")
                n_run += 1
                continue

            print(f"  RUN   {tag} ...", flush=True)
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"  ERROR {tag} — exit code {result.returncode}", file=sys.stderr)
                n_fail += 1
            else:
                n_run += 1

    print(f"\nDone.  run={n_run}  skipped={n_skip}  failed={n_fail}")


if __name__ == "__main__":
    main()
