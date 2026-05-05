"""
compute_attention_concentration.py -- Attention concentration and spatial coverage metrics

Computes per-patient attention concentration and spatial coverage metrics for
both CA and NCA models from the raw attention .pkl files. Outputs a CSV with
all metrics that can be joined with the tissue composition CSV.

Concentration metrics per patient per model:
  - Gini coefficient (0=uniform, 1=maximally concentrated)
  - Normalized entropy (1=uniform, 0=maximally concentrated; inverse of Gini direction)
  - Effective number of patches (exp of entropy)
  - Patches for 50%/80% attention mass
  - Max single-patch weight

Spatial coverage metrics per patient per model:
  - Hull fraction: convex hull area of patches carrying 50% attention mass,
    as a fraction of the total biopsy convex hull area. Measures what
    proportion of the biopsy's spatial extent the model examines.
  - Normalized spatial spread: attention-weighted spatial variance divided
    by the biopsy's overall spatial variance (1 = attention spread matches
    biopsy spread; <1 = attention is spatially concentrated).

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/compute_attention_concentration.py
    python scripts/compute_attention_concentration.py --tasks 2class 3class 4class
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.stats import wilcoxon

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"


def gini_coefficient(attn):
    """Compute Gini coefficient of an attention distribution.

    The Gini coefficient measures inequality in a distribution:
      0 = perfectly uniform (every patch gets equal attention)
      1 = maximally concentrated (one patch gets all attention)

    For attention weights, high Gini means the model bases its decision
    on a small number of patches; low Gini means it distributes attention
    broadly across the graph.

    Formula (for sorted probabilities p_1 <= p_2 <= ... <= p_n):
      G = (2 * sum(i * p_i) / (n * sum(p_i))) - (n + 1) / n

    This is equivalent to the normalized area between the Lorenz curve
    and the line of equality.
    """
    p = attn / attn.sum()
    n = len(p)
    sorted_p = np.sort(p)
    return (2 * np.sum(np.arange(1, n + 1) * sorted_p) / (n * np.sum(sorted_p))) - (n + 1) / n


def attention_concentration_metrics(attn):
    """Full set of concentration metrics for an attention distribution."""
    p = attn / attn.sum()
    n = len(p)

    # Entropy (higher = more uniform)
    ent = -np.sum(p * np.log(p + 1e-10))
    max_ent = np.log(n)

    # Effective number of patches (exp of entropy)
    eff_n = np.exp(ent)

    # Gini (higher = more concentrated)
    g = gini_coefficient(attn)

    # Patches for 50% and 80% attention mass
    desc = np.sort(p)[::-1]
    cum = np.cumsum(desc)
    n_50 = int(np.searchsorted(cum, 0.5) + 1)
    n_80 = int(np.searchsorted(cum, 0.8) + 1)

    return {
        "gini": g,
        "norm_entropy": ent / max_ent if max_ent > 0 else 0.0,
        "effective_n": eff_n,
        "effective_pct": eff_n / n * 100,
        "n_for_50pct": n_50,
        "pct_for_50pct": n_50 / n * 100,
        "n_for_80pct": n_80,
        "pct_for_80pct": n_80 / n * 100,
        "max_weight": float(p.max()),
    }


def spatial_coverage_metrics(attn, coords, frac=0.5):
    """Compute spatial coverage of high-attention patches.

    Hull fraction: convex hull area of patches carrying `frac` of total
    attention mass, divided by convex hull area of all patches. Measures
    what proportion of the biopsy's spatial extent the model examines.
    A pathologist analogy: hull_fraction=0.66 means the model's decision
    draws from 66% of the biopsy area, not just one corner.

    Normalized spatial spread: attention-weighted variance of patch
    coordinates, divided by the unweighted variance of all patch
    coordinates. Values near 1.0 mean attention is spread across the
    full biopsy; values near 0 mean attention clusters in one spot.
    """
    p = attn / attn.sum()

    # Attention-weighted spatial variance
    weighted_center = (p[:, None] * coords).sum(axis=0)
    sq_dists = np.sum((coords - weighted_center) ** 2, axis=1)
    attn_spatial_var = (p * sq_dists).sum()

    biopsy_center = coords.mean(axis=0)
    biopsy_sq_dists = np.sum((coords - biopsy_center) ** 2, axis=1)
    biopsy_var = biopsy_sq_dists.mean()
    normalized_spread = attn_spatial_var / biopsy_var if biopsy_var > 0 else 0.0

    # Convex hull fraction
    desc_idx = np.argsort(p)[::-1]
    cum = np.cumsum(p[desc_idx])
    n_top = int(np.searchsorted(cum, frac) + 1)
    top_idx = desc_idx[:n_top]

    try:
        area_all = ConvexHull(coords).volume  # 2D: .volume = area
    except Exception:
        return {"hull_fraction": np.nan, "normalized_spread": normalized_spread}

    if len(top_idx) >= 3:
        try:
            area_top = ConvexHull(coords[top_idx]).volume
        except Exception:
            area_top = 0.0
    else:
        area_top = 0.0

    hull_frac = area_top / area_all if area_all > 0 else 0.0

    return {"hull_fraction": hull_frac, "normalized_spread": normalized_spread}


def main():
    parser = argparse.ArgumentParser(description="Compute attention concentration and spatial coverage metrics")
    parser.add_argument("--tasks", nargs="+", default=["2class", "3class", "4class"])
    args = parser.parse_args()

    rows = []
    for task in args.tasks:
        ca_path = f"{RESULTS}/attention/BCNB_{task}_CA_attention_direct.pkl"
        nca_path = f"{RESULTS}/attention/BCNB_{task}_NCA_node_data.pkl"

        if not os.path.exists(ca_path) or not os.path.exists(nca_path):
            print(f"  Skipping {task}: pkl files not found")
            continue

        with open(ca_path, "rb") as f:
            ca_data = pickle.load(f)
        with open(nca_path, "rb") as f:
            nca_data = pickle.load(f)

        # Build NCA lookup (keys may be str or int)
        nca_lookup = {}
        for k, v in nca_data.items():
            nca_lookup[str(k)] = v

        n_processed = 0
        for pid in ca_data:
            pid_str = str(pid)
            if pid_str not in nca_lookup:
                continue

            ca_attn = ca_data[pid]["attention_weights"].flatten()
            ca_coords = ca_data[pid]["centroid"]
            nca_entry = nca_lookup[pid_str]
            nca_attn = nca_entry["attention_weights"]
            if hasattr(nca_attn, "numpy"):
                import torch
                nca_attn = nca_attn.numpy()
            nca_attn = nca_attn.flatten()
            nca_coords = nca_entry["centroids"]
            if hasattr(nca_coords, "numpy"):
                nca_coords = nca_coords.numpy()

            if len(ca_attn) < 5 or len(nca_attn) < 5:
                continue
            if len(ca_attn) != len(nca_attn):
                continue

            # Concentration metrics
            ca_conc = attention_concentration_metrics(ca_attn)
            nca_conc = attention_concentration_metrics(nca_attn)

            # Spatial coverage metrics
            ca_spatial = spatial_coverage_metrics(ca_attn, ca_coords)
            nca_spatial = spatial_coverage_metrics(nca_attn, nca_coords)

            row = {"task": task, "patient_id": int(pid), "n_nodes": len(ca_attn)}
            for k, v in ca_conc.items():
                row[f"ca_{k}"] = v
            for k, v in nca_conc.items():
                row[f"nca_{k}"] = v
            for k, v in ca_spatial.items():
                row[f"ca_{k}"] = v
            for k, v in nca_spatial.items():
                row[f"nca_{k}"] = v

            rows.append(row)
            n_processed += 1

        print(f"  {task}: {n_processed} patients processed")

    df = pd.DataFrame(rows)

    # Summary table
    print(f"\n{'=' * 80}")
    print(f"  ATTENTION CONCENTRATION AND SPATIAL COVERAGE SUMMARY")
    print(f"{'=' * 80}")

    for task in args.tasks:
        t = df[df.task == task]
        if len(t) == 0:
            continue

        print(f"\n  --- {task} (n={len(t)}) ---")
        print(f"  {'Metric':<30} {'NCA':>10} {'CA':>10} {'Wilcoxon p':>12}")
        print(f"  {'-' * 62}")

        metrics = [
            ("Gini coefficient", "gini"),
            ("Patches for 50% attn", "n_for_50pct"),
            ("  (% of graph)", "pct_for_50pct"),
            ("Hull fraction (50% attn)", "hull_fraction"),
            ("Norm spatial spread", "normalized_spread"),
            ("Max single weight", "max_weight"),
        ]

        for label, col in metrics:
            nca_col = f"nca_{col}"
            ca_col = f"ca_{col}"
            nca_vals = t[nca_col].dropna()
            ca_vals = t[ca_col].dropna()
            # Align indices for paired test
            common = nca_vals.index.intersection(ca_vals.index)
            if len(common) > 10:
                _, p = wilcoxon(nca_vals.loc[common], ca_vals.loc[common])
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"  {label:<30} {nca_vals.mean():>10.3f} {ca_vals.mean():>10.3f} {p:>10.4f} {sig}")
            else:
                print(f"  {label:<30} {nca_vals.mean():>10.3f} {ca_vals.mean():>10.3f} {'n/a':>12}")

    # Convergence summary
    print(f"\n{'=' * 80}")
    print(f"  CONVERGENCE PATTERN")
    print(f"{'=' * 80}")
    print(f"\n  {'Task':<10} {'Gini gap':<12} {'Hull gap':<12} {'Tumor enrich gap':<18}")
    print(f"  {'-' * 52}")
    for task in args.tasks:
        t = df[df.task == task]
        if len(t) == 0:
            continue
        gini_gap = t.nca_gini.mean() - t.ca_gini.mean()
        hull_nca = t.nca_hull_fraction.dropna().mean()
        hull_ca = t.ca_hull_fraction.dropna().mean()
        hull_gap = hull_ca - hull_nca
        print(f"  {task:<10} {gini_gap:>+10.2f}   {hull_gap:>+10.1%}   (see tissue CSV)")

    # Save
    out_path = f"{RESULTS}/interpretability/attention_concentration_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df)} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
