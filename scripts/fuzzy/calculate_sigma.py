"""
calculate_sigma.py -- Compute interpretable sigma values for fuzzy edge weighting

For a given graph directory, extracts edge distance distributions and derives
sigma via the "retained-weight" criterion:

    exp(-median² / 2σ²) = r  →  σ = median / sqrt(2·ln(1/r))

Interpretation: with that sigma, the *typical* (median-distance) edge carries
weight r.  Asking for r=0.5 means the median edge is half-weight; r=0.1 means
it is nearly suppressed.  More interpretable than choosing sigma by raw percentile.

Targets evaluated: r ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

Three graph sources are supported (auto-detected from the first .pt, or forced
with --source):

  raw    -- original graphs; all-pairs distances computed on the fly, edges
            selected by fuzzy top-k (1-d_s)·(1-d_m). Expensive O(N²) per graph.
            Topology: fuzzy top-k

  fuzzy  -- already-generated fuzzy .pt files; reads edge_features_fuzzy_s /
            edge_features_fuzzy_m directly. Fast.
            Topology: fuzzy top-k  (same selection as raw)

  morph  -- _morph .pt files; reads edge_features (spatial distances) and
            edge_feat_dist (morphological distances) directly. Fast.
            Topology: spatial KNN  (suitable for Option 2 sigma calibration)

Usage:
    # Auto-detect source from graph fields
    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23_fuzzy/sigmas_med_med/3class/k_19

    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23_morph/<subdir>/graphs_k_19

    # Force source explicitly
    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23/<task>/graphs_k_19 \\
        --source raw

    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23_fuzzy/sigmas_med_med/3class/k_19 \\
        --source fuzzy

    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23_morph/<subdir>/graphs_k_19 \\
        --source morph

    # Quick estimate with a subset of graphs
    python scripts/fuzzy/calculate_sigma.py \\
        --input-dir data/BCNB/results_graphs_november_23/<task>/graphs_k_19 \\
        --max-graphs 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


TARGETS = [0.1, 0.3, 0.5, 0.7, 0.9]

# Topology labels shown in the report
_TOPOLOGY = {
    "raw":   "fuzzy top-k by (1-d_s)·(1-d_m)",
    "fuzzy": "fuzzy top-k by (1-d_s)·(1-d_m)  [pre-stored]",
    "morph": "spatial KNN  [pre-stored]",
}


# ---------------------------------------------------------------------------
# Distance helpers (identical to generate_fuzzy_graphs.py)
# ---------------------------------------------------------------------------

def _norm_features(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)


def _aniso_dist_matrix(centroid: torch.Tensor) -> torch.Tensor:
    c = centroid.float()
    rows_span = (c[:, 0].max() - c[:, 0].min()).clamp(min=1e-6)
    cols_span = (c[:, 1].max() - c[:, 1].min()).clamp(min=1e-6)
    dr = (c[:, 0].unsqueeze(1) - c[:, 0].unsqueeze(0)) / rows_span
    dc = (c[:, 1].unsqueeze(1) - c[:, 1].unsqueeze(0)) / cols_span
    return (dr ** 2 + dc ** 2).sqrt()


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def detect_source(pt_path: Path) -> str:
    """Inspect first .pt file and return 'fuzzy', 'morph', or 'raw'."""
    g = torch.load(pt_path, weights_only=False, map_location="cpu")
    if hasattr(g, 'edge_features_fuzzy_s') and g.edge_features_fuzzy_s is not None:
        return "fuzzy"
    if hasattr(g, 'edge_feat_dist') and g.edge_feat_dist is not None:
        return "morph"
    return "raw"


# ---------------------------------------------------------------------------
# Per-graph distance extraction  (one function per source type)
# ---------------------------------------------------------------------------

def _extract_raw(pt_path: Path, k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """All-pairs computation; select top-k fuzzy edges."""
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


def _extract_fuzzy(pt_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read pre-stored fuzzy edge distances."""
    graph = torch.load(pt_path, weights_only=False, map_location="cpu")
    if not hasattr(graph, 'edge_features_fuzzy_s') or graph.edge_features_fuzzy_s is None:
        return None
    return (
        graph.edge_features_fuzzy_s.float().numpy(),
        graph.edge_features_fuzzy_m.float().numpy(),
    )


def _extract_morph(pt_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Read pre-stored spatial and morphological distances from a _morph graph."""
    graph = torch.load(pt_path, weights_only=False, map_location="cpu")
    if not hasattr(graph, 'edge_feat_dist') or graph.edge_feat_dist is None:
        return None
    d_s = graph.edge_features.float()
    if d_s.dim() > 1:
        d_s = d_s.squeeze(-1)
    d_m = graph.edge_feat_dist.float()
    if d_m.dim() > 1:
        d_m = d_m.squeeze(-1)
    return d_s.numpy(), d_m.numpy()


# ---------------------------------------------------------------------------
# Aggregate over all graphs
# ---------------------------------------------------------------------------

def collect_distances(files: list[Path], source: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    all_s, all_m = [], []
    for f in tqdm(files, desc="Reading graphs"):
        if source == "raw":
            result = _extract_raw(f, k)
        elif source == "fuzzy":
            result = _extract_fuzzy(f)
        else:
            result = _extract_morph(f)
        if result is None:
            continue
        all_s.append(result[0])
        all_m.append(result[1])
    return np.concatenate(all_s), np.concatenate(all_m)


# ---------------------------------------------------------------------------
# Sigma calculation
# ---------------------------------------------------------------------------

def sigma_for_target(median: float, r: float) -> float:
    """σ such that exp(-median² / 2σ²) = r  →  σ = median / sqrt(2·ln(1/r))."""
    return median / np.sqrt(2.0 * np.log(1.0 / r))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(d_s: np.ndarray, d_m: np.ndarray, n_graphs: int, n_edges: int,
                 source: str, input_dir: Path) -> None:
    med_s = float(np.median(d_s))
    med_m = float(np.median(d_m))

    print(f"\n{'='*65}")
    print(f"  Fuzzy edge distance analysis")
    print(f"  Source dir : {input_dir}")
    print(f"  Graph type : {source}")
    print(f"  Topology   : {_TOPOLOGY[source]}")
    print(f"  Graphs     : {n_graphs}   Edges analysed: {n_edges:,}")
    print(f"{'='*65}")

    # Distance distribution
    pct_levels = [10, 25, 50, 75, 90]
    print(f"\n  Distance distribution")
    print(f"  {'':5s}  {'d_spatial':>10s}  {'d_morpho':>10s}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*10}")
    for lvl in pct_levels:
        tag = "  ← median" if lvl == 50 else ""
        print(f"  p{lvl:<4d}  {np.percentile(d_s, lvl):>10.4f}  {np.percentile(d_m, lvl):>10.4f}{tag}")

    # Sigma table
    r_median = float(np.exp(-0.5))  # weight when σ = median  → exp(-median²/2·median²) = exp(-½)
    print(f"\n  Sigma by retained-weight criterion")
    print(f"  exp(-median² / 2σ²) = r  →  σ = median / sqrt(2·ln(1/r))")
    print(f"\n  {'r':>6s}  {'σ_spatial':>10s}  {'σ_morpho':>10s}  interpretation")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*38}")
    for r in TARGETS:
        sig_s = sigma_for_target(med_s, r)
        sig_m = sigma_for_target(med_m, r)
        print(f"  {r:>6.1f}  {sig_s:>10.4f}  {sig_m:>10.4f}  median edge → weight {r}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*38}")
    print(f"  {r_median:>6.3f}  {med_s:>10.4f}  {med_m:>10.4f}  σ = median  (weight = e^-½ ≈ 0.607)")

    print(f"\n  Medians:  d_spatial={med_s:.4f}   d_morpho={med_m:.4f}")

    if source in ("raw", "fuzzy"):
        sig_s_50 = sigma_for_target(med_s, 0.5)
        sig_m_50 = sigma_for_target(med_m, 0.5)
        print(f"\n  Suggested generate_fuzzy_graphs.py flags:")
        print(f"    r=0.5  : --sigma-spatial {sig_s_50:.4f} --sigma-morpho {sig_m_50:.4f}")
        print(f"    σ=med  : --sigma-spatial {med_s:.4f} --sigma-morpho {med_m:.4f}")
    print(f"{'='*65}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute interpretable sigma values for fuzzy edge weighting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with .pt graph files (searched recursively)")
    parser.add_argument("--source", choices=["raw", "fuzzy", "morph"], default=None,
                        help="Graph type: raw (all-pairs computation), fuzzy (pre-stored fuzzy "
                             "distances), morph (pre-stored spatial+morpho distances). "
                             "Default: auto-detected from the first .pt file.")
    parser.add_argument("--k", type=int, default=19,
                        help="KNN k for fuzzy edge selection (raw source only, default: 19).")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Limit analysis to the first N graphs (quick estimate).")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    files = sorted(f for f in input_dir.rglob("*.pt") if not f.name.startswith("0_"))
    if not files:
        print(f"No .pt files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_graphs is not None:
        files = files[:args.max_graphs]
        print(f"Limiting to first {len(files)} graphs (--max-graphs).")

    # Resolve source
    source = args.source
    if source is None:
        source = detect_source(files[0])
        print(f"Auto-detected source: '{source}'")
    else:
        print(f"Source: '{source}' (forced via --source)")

    print(f"Found {len(files)} graphs under {input_dir}")

    d_s, d_m = collect_distances(files, source, args.k)
    print_report(d_s, d_m, len(files), len(d_s), source, input_dir)


if __name__ == "__main__":
    main()
