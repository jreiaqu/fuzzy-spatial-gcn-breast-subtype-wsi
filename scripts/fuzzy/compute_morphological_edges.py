"""
compute_morphological_edges.py

For each graph .pt under --input-dir (k_19 subdirs only, searched recursively):
  - Adds x_norm: L2-normalized node features (original x preserved)
  - Adds edge_feat_dist: normalized morphological distance in [0,1] for each
    edge in edge_index, computed as ||x_norm[i] - x_norm[j]|| / 2

Saves modified graphs mirroring the folder structure under --output-dir
(originals untouched). Defaults to the original BCNB paths for backward
compatibility; pass --input-dir/--output-dir to target another dataset (e.g.
CLARIFY/SBC k=19 graphs) -- the function only needs graph.x/graph.edge_index,
so it works unchanged on any dataset with that schema.

Usage:
    python scripts/fuzzy/compute_morphological_edges.py
    python scripts/fuzzy/compute_morphological_edges.py --dry-run
    python scripts/fuzzy/compute_morphological_edges.py \\
        --input-dir data/SBC/results_graphs_january_25 \\
        --output-dir data/SBC/results_graphs_january_25_morph
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_ROOT = REPO_ROOT / "data" / "BCNB" / "results_graphs_november_23"
DEFAULT_OUT_ROOT   = REPO_ROOT / "data" / "BCNB" / "results_graphs_november_23_morph"


def find_k19_files(graph_root: Path) -> list[Path]:
    # os.walk(followlinks=True) instead of Path.rglob, which does not
    # traverse symlinked directories (e.g. data/SBC/.../graphs_k_19 symlinks
    # into data/CLARIFY/...) in Python 3.10.
    import os
    found = []
    for dirpath, _, filenames in os.walk(graph_root, followlinks=True):
        if "k_19" in Path(dirpath).name:
            found.extend(Path(dirpath) / fn for fn in filenames if fn.endswith(".pt"))
    return sorted(
        f for f in found
        if "k_19" in f.parent.name
    )


def process_graph(graph) -> np.ndarray:
    x = graph.x.float()
    x_norm = x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)
    graph.x_norm = x_norm

    src, dst = graph.edge_index[0], graph.edge_index[1]
    graph.edge_feat_dist = (x_norm[src] - x_norm[dst]).norm(dim=1) / 2.0

    return graph.edge_feat_dist.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Process only the first graph per subdir, print stats, do not save.",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help=f"Root to search for k_19 graphs (default: {DEFAULT_GRAPH_ROOT})",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Root to mirror the modified graphs into (default: {DEFAULT_OUT_ROOT})",
    )
    args = parser.parse_args()

    GRAPH_ROOT = Path(args.input_dir) if args.input_dir else DEFAULT_GRAPH_ROOT
    OUT_ROOT = Path(args.output_dir) if args.output_dir else DEFAULT_OUT_ROOT

    all_files = find_k19_files(GRAPH_ROOT)
    if not all_files:
        print(f"No .pt files found under {GRAPH_ROOT}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        seen: set[Path] = set()
        files = []
        for f in all_files:
            if f.parent not in seen:
                seen.add(f.parent)
                files.append(f)
        print(f"[dry-run] {len(files)} graph(s) (1 per subdir) — no files will be written")
    else:
        files = all_files
        print(f"Found {len(files)} graphs under k_19 subdirs")
        print(f"Output root: {OUT_ROOT}")

    # group_name = top-level task subdir (e.g. graphs_PM_OTHERvsTNBC_...)
    group_morph:   dict[str, list[np.ndarray]] = {}
    group_spatial: dict[str, list[np.ndarray]] = {}

    for pt_file in tqdm(files, desc="Graphs"):
        graph = torch.load(pt_file, weights_only=False)
        dists = process_graph(graph)

        group = pt_file.relative_to(GRAPH_ROOT).parts[0]
        group_morph.setdefault(group, []).append(dists)
        group_spatial.setdefault(group, []).append(graph.edge_features.numpy())

        if not args.dry_run:
            out_path = OUT_ROOT / pt_file.relative_to(GRAPH_ROOT)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(graph, out_path)

    total      = sum(len(v) for v in group_morph.values())
    flat_morph = np.concatenate([d for v in group_morph.values()   for d in v])
    flat_spat  = np.concatenate([d for v in group_spatial.values() for d in v])

    print(f"\n--- Summary ---")
    print(f"Graphs processed : {total}")
    if not args.dry_run:
        print(f"Output           : {OUT_ROOT}")
    print(f"edge_feat_dist (global) : mean={flat_morph.mean():.4f}  std={flat_morph.std():.4f}  "
          f"min={flat_morph.min():.4f}  max={flat_morph.max():.4f}")
    print(f"edge_features  (global) : mean={flat_spat.mean():.4f}  std={flat_spat.std():.4f}  "
          f"min={flat_spat.min():.4f}  max={flat_spat.max():.4f}")

    print(f"\n--- Median per group (suggested sigma values) ---")
    for group in sorted(group_morph):
        arr_m  = np.concatenate(group_morph[group])
        arr_s  = np.concatenate(group_spatial[group])
        task   = group.split("graphs_PM_")[-1].split("_BB_")[0]
        print(f"  {task}")
        print(f"    graphs={len(group_morph[group])}"
              f"  sigma_morphological={np.median(arr_m):.4f}"
              f"  sigma_spatial={np.median(arr_s):.4f}")


if __name__ == "__main__":
    main()
