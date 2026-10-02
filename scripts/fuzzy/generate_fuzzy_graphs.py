"""
generate_fuzzy_graphs.py

Generates .pt graph files with fuzzy-combined edge structure.
For each input graph the script:
  1. Computes ALL pairwise distances (O(N²) per graph):
       d_s : anisotropic per-WSI  sqrt((Δrow/rows_span)²+(Δcol/cols_span)²) ∈ [0,√2]
             (matches original edge_features values)
             NOTE: spatial KNN topology uses raw L2 (torch.cdist) to match original HNSW,
             but stored distances use the anisotropic formula above.
       d_m : ||x_norm_i - x_norm_j|| / 2  (unit-norm features → max L2=2) ∈ [0,1]
  2. Derives exact KNN topologies:
       edge_index  — spatial   top-k (smallest d_s)
       edge_latent — morpholog. top-k (smallest d_m)
  3. Selects fuzzy topology WITHOUT sigma:
       combined linear weight  w_ij = (1-d_s)·(1-d_m)
       (only used to rank neighbours; negative for very distant pairs, since d_s ≤ √2)
       edge_index_fuzzy = top-k by w_ij over ALL pairs
       This preserves distance ordering with no calibration circularity.
  4. Computes Gaussian edge attributes ONLY for the selected k edges:
       mu_s = exp(-d_s²/2σ_s²),  mu_m = exp(-d_m²/2σ_m²)
       edge_mu_fuzzy = mu_s · mu_m
       σ is calibrated consistently on the selected edges (auto = median of their distances).

Output fields added/updated in each .pt:
    x_norm                  L2-normalised node features  [N, feat_dim]
    edge_index              exact spatial KNN  [2, k·N]
    edge_features           spatial distances for edge_index  [k·N]
    edge_latent             exact morpholog. KNN  [2, k·N]
    edge_features_latent    morpholog. distances for edge_latent  [k·N]
    edge_index_fuzzy        combined fuzzy topology  [2, k·N]
    edge_features_fuzzy_s   spatial distances for fuzzy edges  [k·N]
    edge_features_fuzzy_m   morpholog. distances for fuzzy edges  [k·N]
    edge_mu_fuzzy           Gaussian weight mu_ij for fuzzy edges  [k·N]

Output directory layout:
    results_graphs_november_23_fuzzy/
      <sigma-variant>/          e.g. sigmas_med_med  or  sigmas_0.1_0.7
        <task>/k_<knn>/         e.g. 3class/k_19/
          *.pt

    Pass the full leaf path as --output-dir; the script saves graphs there directly.

Usage:
    # Sigma suggestions from selected-edge distribution (no output written)
    python scripts/fuzzy/generate_fuzzy_graphs.py --stats-only \\
        --input-dir data/BCNB/results_graphs_november_23/<task>/graphs_k_19

    # Generate graphs (auto sigma = global median of selected edges)
    python scripts/fuzzy/generate_fuzzy_graphs.py \\
        --input-dir  data/BCNB/results_graphs_november_23/<task>/graphs_k_19 \\
        --output-dir data/BCNB/results_graphs_november_23_fuzzy/sigmas_auto/<task>/k_19

    # Manual sigma
    python scripts/fuzzy/generate_fuzzy_graphs.py \\
        --input-dir  data/BCNB/results_graphs_november_23/<task>/graphs_k_19 \\
        --output-dir data/BCNB/results_graphs_november_23_fuzzy/sigmas_med_0.9/<task>/k_19 \\
        --sigma-spatial 0.0959 --sigma-morpho 0.4922

    # Dry run (first 5 graphs, no save)
    python scripts/fuzzy/generate_fuzzy_graphs.py \\
        --input-dir  data/BCNB/results_graphs_november_23/<task>/graphs_k_19 \\
        --output-dir data/BCNB/results_graphs_november_23_fuzzy/sigmas_med_0.9/<task>/k_19 \\
        --dry-run
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# --- Repository path shim ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _norm_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _aniso_dist_matrix(centroid: torch.Tensor) -> torch.Tensor:
    """Full N×N anisotropic spatial distance matrix in [0,1] (per-WSI).
    sqrt((Δrow/rows_span)²+(Δcol/cols_span)²)
    Matches the distance values stored in edge_features by the original pipeline.
    Used for: stored edge_features, combined weight W, Gaussian weights.
    """
    c = centroid.float()
    rows_span = (c[:, 0].max() - c[:, 0].min()).clamp(min=1e-6)
    cols_span = (c[:, 1].max() - c[:, 1].min()).clamp(min=1e-6)
    dr = (c[:, 0].unsqueeze(1) - c[:, 0].unsqueeze(0)) / rows_span
    dc = (c[:, 1].unsqueeze(1) - c[:, 1].unsqueeze(0)) / cols_span
    return (dr ** 2 + dc ** 2).sqrt()


def _gaussian(d: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-d ** 2 / (2.0 * sigma ** 2))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_pt_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.pt"))


# ---------------------------------------------------------------------------
# Stats pass  (selects edges with linear weight, returns their distances)
# ---------------------------------------------------------------------------

def _stats_one(pt_path: Path, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Select top-k by linear weight (1-d_s)·(1-d_m), return their d_s and d_m."""
    graph = torch.load(pt_path, weights_only=False, map_location="cpu")
    N = graph.x.shape[0]
    if N < 2:
        return None

    xn  = _norm_features(graph.x)
    D_s = _aniso_dist_matrix(graph.centroid)
    D_m = torch.cdist(xn, xn) / 2.0

    W = (1.0 - D_s) * (1.0 - D_m)
    W.fill_diagonal_(0.0)

    k_act = min(k, N - 1)
    _, top_idx = torch.topk(W, k_act, dim=1, largest=True)

    src = torch.arange(N).unsqueeze(1).expand(-1, k_act).reshape(-1)
    dst = top_idx.reshape(-1)
    return D_s[src, dst].numpy(), D_m[src, dst].numpy()


def compute_stats(files: list[Path], k: int) -> tuple[np.ndarray, np.ndarray]:
    all_s, all_m = [], []
    for f in tqdm(files, desc="Stats pass"):
        r = _stats_one(f, k)
        if r is None:
            continue
        all_s.append(r[0])
        all_m.append(r[1])
    return np.concatenate(all_s), np.concatenate(all_m)


def print_stats(d_s: np.ndarray, d_m: np.ndarray, k: int,
                sigma_s_cur: float | None, sigma_m_cur: float | None) -> None:
    pct_levels = [10, 25, 50, 75, 90]
    med_s = float(np.median(d_s))
    med_m = float(np.median(d_m))

    print(f"\n--- Distance distribution of selected fuzzy edges (top-{k} per node by linear weight) ---")
    print(f"  Both distributions are for the SELECTED edges, so sigma is consistently calibrated.")
    print(f"  {'':5s}  {'spatial':>10s}  {'morpholog.':>11s}")
    for lvl in pct_levels:
        vs = np.percentile(d_s, lvl)
        vm = np.percentile(d_m, lvl)
        tag = "  ← median" if lvl == 50 else ""
        print(f"  p{lvl:<4d}  {vs:>10.4f}  {vm:>11.4f}{tag}")

    # Sigma by retained-weight: exp(-median²/2σ²) = r  →  σ = median/sqrt(2·ln(1/r))
    targets = [0.1, 0.3, 0.5, 0.7, 0.9]
    r_med = float(np.exp(-0.5))  # weight when σ = median exactly
    print(f"\n--- Sigma by retained-weight criterion: exp(-median²/2σ²) = r ---")
    print(f"  {'r':>6s}  {'σ_spatial':>10s}  {'σ_morpho':>10s}  interpretation")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*35}")
    for r in targets:
        sig_s = med_s / np.sqrt(2.0 * np.log(1.0 / r))
        sig_m = med_m / np.sqrt(2.0 * np.log(1.0 / r))
        print(f"  {r:>6.1f}  {sig_s:>10.4f}  {sig_m:>10.4f}  median edge → weight {r}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*35}")
    print(f"  {r_med:>6.3f}  {med_s:>10.4f}  {med_m:>10.4f}  σ = median  (weight = e^-½ ≈ 0.607)")

    if sigma_s_cur is not None:
        print(f"\n  --sigma-spatial provided: {sigma_s_cur}  (auto would be {med_s:.4f})")
    if sigma_m_cur is not None:
        print(f"  --sigma-morpho  provided: {sigma_m_cur}  (auto would be {med_m:.4f})")


# ---------------------------------------------------------------------------
# Process one graph  (module-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> tuple:
    pt_path, cfg = args
    k          = cfg["k"]
    sigma_s    = cfg["sigma_s"]
    sigma_m    = cfg["sigma_m"]
    input_dir  = Path(cfg["input_dir"])
    output_dir = Path(cfg["output_dir"])
    dry_run    = cfg["dry_run"]

    try:
        graph = torch.load(pt_path, weights_only=False, map_location="cpu")
        N     = graph.x.shape[0]
        if N < 2:
            return ("skip", str(pt_path), "N<2")

        xn      = _norm_features(graph.x)
        c       = graph.centroid.float()
        # raw L2 — for spatial KNN selection only (matches original HNSW space='l2')
        D_s_raw = torch.cdist(c, c)
        # anisotropic — for stored edge_features, W, and Gaussian (matches original values)
        D_s     = _aniso_dist_matrix(graph.centroid)
        D_m     = torch.cdist(xn, xn) / 2.0            # [N, N], in [0,1]

        inf_diag    = torch.eye(N, dtype=torch.bool)
        D_s_raw_knn = D_s_raw.masked_fill(inf_diag, float("inf"))
        D_m_knn     = D_m.masked_fill(inf_diag, float("inf"))
        k_act       = min(k, N - 1)
        src_knn     = torch.arange(N).unsqueeze(1).expand(-1, k_act).reshape(-1)

        # Exact spatial KNN: topology by raw L2 (matches HNSW), values anisotropic (matches original)
        _, s_idx = torch.topk(D_s_raw_knn, k_act, dim=1, largest=False)
        s_vals   = D_s[src_knn, s_idx.reshape(-1)]
        # Exact morphological KNN  (correctly computed from features)
        m_vals, m_idx = torch.topk(D_m_knn, k_act, dim=1, largest=False)

        # --- Fuzzy topology: top-k by linear combined weight over ALL pairs ---
        W = (1.0 - D_s) * (1.0 - D_m)
        W.fill_diagonal_(0.0)
        _, f_idx = torch.topk(W, k_act, dim=1, largest=True)

        src_f = src_knn
        dst_f = f_idx.reshape(-1)
        ds_f  = D_s[src_f, dst_f]
        dm_f  = D_m[src_f, dst_f]

        # --- Gaussian edge attributes ---
        mu_f = _gaussian(ds_f, sigma_s) * _gaussian(dm_f, sigma_m)

        # --- Build output graph ---
        out = deepcopy(graph)
        out.x_norm                = xn
        out.edge_index            = torch.stack([src_knn, s_idx.reshape(-1)], dim=0)
        out.edge_features         = s_vals
        out.edge_latent           = torch.stack([src_knn, m_idx.reshape(-1)], dim=0)
        out.edge_features_latent  = m_vals.reshape(-1)
        out.edge_index_fuzzy      = torch.stack([src_f, dst_f], dim=0)
        out.edge_features_fuzzy_s = ds_f
        out.edge_features_fuzzy_m = dm_f
        out.edge_mu_fuzzy         = mu_f

        if not dry_run:
            rel      = Path(pt_path).relative_to(input_dir)
            out_path = output_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(out, out_path)

        return ("ok", str(pt_path), None)

    except Exception as exc:
        return ("error", str(pt_path), str(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate .pt graphs with fuzzy-combined edge structure."
    )
    parser.add_argument("--input-dir",     type=str, required=True,
                        help="Input .pt graph directory (searched recursively)")
    parser.add_argument("--output-dir",    type=str, default=None,
                        help="Output directory. Required unless --stats-only.")
    parser.add_argument("--k",             type=int, default=19,
                        help="Number of neighbors for all KNN topologies (default: 19)")
    parser.add_argument("--sigma-spatial", type=float, default=None,
                        help="Gaussian σ for spatial distances applied to selected edges. "
                             "Default: auto (global median of selected-edge distances)")
    parser.add_argument("--sigma-morpho",  type=float, default=None,
                        help="Gaussian σ for morphological distances applied to selected edges. "
                             "Default: auto (global median of selected-edge distances)")
    parser.add_argument("--workers",       type=int, default=1,
                        help="Parallel workers (default: 1; >1 may hang with PyTorch multiprocessing)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Process first 5 graphs, do not save")
    parser.add_argument("--stats-only",    action="store_true",
                        help="Show distance distribution of selected edges and sigma "
                             "suggestions, then exit")
    args = parser.parse_args()

    if not args.stats_only and args.output_dir is None:
        parser.error("--output-dir is required unless --stats-only")

    input_dir = Path(args.input_dir)
    files = find_pt_files(input_dir)
    if not files:
        print(f"No .pt files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} graphs under {input_dir}")

    # --- Stats pass (runs when sigma not fully specified, or always in --stats-only) ---
    need_stats = args.stats_only or args.sigma_spatial is None or args.sigma_morpho is None
    d_s_sel = d_m_sel = None

    if need_stats:
        d_s_sel, d_m_sel = compute_stats(files, args.k)
        print_stats(d_s_sel, d_m_sel, args.k, args.sigma_spatial, args.sigma_morpho)

    if args.stats_only:
        return

    # --- Resolve sigmas ---
    sigma_s = args.sigma_spatial if args.sigma_spatial is not None \
              else float(np.median(d_s_sel))
    sigma_m = args.sigma_morpho  if args.sigma_morpho  is not None \
              else float(np.median(d_m_sel))

    src_s = "manual" if args.sigma_spatial is not None else "auto (global median of selected edges)"
    src_m = "manual" if args.sigma_morpho  is not None else "auto (global median of selected edges)"

    print(f"\nk={args.k}")
    print(f"  σ_spatial = {sigma_s:.4f}  [{src_s}]")
    print(f"  σ_morpho  = {sigma_m:.4f}  [{src_m}]")

    if args.dry_run:
        files = files[:5]
        print(f"\n[dry-run] Processing first {len(files)} graphs — nothing will be saved")

    output_dir = Path(args.output_dir)
    cfg = {
        "k":          args.k,
        "sigma_s":    sigma_s,
        "sigma_m":    sigma_m,
        "input_dir":  str(input_dir),
        "output_dir": str(output_dir),
        "dry_run":    args.dry_run,
    }

    t0 = time.time()
    task_args = [(f, cfg) for f in files]
    n_ok = n_skip = n_err = 0
    skipped: list[str] = []

    if args.workers > 1 and not args.dry_run:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(pool.imap_unordered(_process_one, task_args),
                                total=len(files), desc="Graphs"))
    else:
        results = [_process_one(a) for a in tqdm(task_args, desc="Graphs")]

    for status, path, msg in results:
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
            skipped.append(f"{path}: {msg}")
        else:
            n_err += 1
            print(f"  ERROR {path}: {msg}", file=sys.stderr)

    elapsed = time.time() - t0

    if not args.dry_run:
        if skipped:
            sp = output_dir / "skipped.txt"
            output_dir.mkdir(parents=True, exist_ok=True)
            sp.write_text("\n".join(skipped))

        prov = {
            "k":                       args.k,
            "selection":               "top-k by (1-d_s)*(1-d_m) over all pairs",
            "sigma_spatial":           sigma_s,
            "sigma_morpho":            sigma_m,
            "sigma_spatial_source":    src_s,
            "sigma_morpho_source":     src_m,
            "spatial_normalization":   "anisotropic_per_wsi",
            "morpho_normalization":    "l2_unit_norm_div2",
            "n_graphs_processed":      n_ok,
            "n_graphs_skipped":        n_skip,
            "timestamp":               datetime.now().isoformat(),
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "0_fuzzy_graph_config.json").write_text(json.dumps(prov, indent=2))
        print(f"  Provenance: {output_dir / '0_fuzzy_graph_config.json'}")

    print(f"\n--- Summary ---")
    print(f"Processed : {n_ok}  Skipped: {n_skip}  Errors: {n_err}  Time: {elapsed:.1f}s")
    if not args.dry_run:
        print(f"Output    : {output_dir}")


if __name__ == "__main__":
    main()
