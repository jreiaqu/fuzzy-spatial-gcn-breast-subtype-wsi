"""
analyze_attention_tissue_composition.py -- R1.1 Interpretability Analysis

Quantifies what tissue types each model (NCA, CA) focuses on by analyzing
the tissue composition of the most-attended patches.

Methodology:
  For each patient and model:
  1. Rank patches by attention weight (descending)
  2. Find the minimal set of patches that collectively carry X% of total
     attention mass (50% and 80% thresholds)
  3. Compute the mean tissue composition of those patches
  4. Compare with the overall (unweighted) biopsy composition
  5. The difference = "enrichment" (how much the model preferentially
     selects patches with that tissue type)

This approach is methodologically proper because:
  - It uses attention-mass thresholds (not fixed percentiles), making
    the comparison fair despite NCA concentrating attention on ~7% of
    patches while CA distributes it across 13-40%
  - Wilcoxon signed-rank test assesses significance of enrichment
    across 217 patients

Also reports attention concentration metrics:
  - Effective number of patches (exp of entropy)
  - Gini coefficient
  - Number of patches for 50%/80% attention mass

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/analyze_attention_tissue_composition.py
    python scripts/analyze_attention_tissue_composition.py --tasks 3class
"""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# [REPLACED by _paths.py] MOLSUB = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"
TISSUE_DIR = f"{MOLSUB}/data/BCNB/patches_paths_class_perc"
TISSUE_NAMES = ["other", "tumor", "stroma", "inflammation", "necrosis"]
TISSUE_ANALYSIS = ["tumor", "stroma", "inflammation", "necrosis"]


def load_tissue_composition():
    frames = []
    for s in ("train", "val", "test"):
        frames.append(pd.read_csv(f"{TISSUE_DIR}/{s}_patches_class_perc_0_tp.csv"))
    df = pd.concat(frames, ignore_index=True)
    tissue_map = {}
    for _, row in df.iterrows():
        fn = row["patch_path"].replace("\\", "/").split("/")[-1].replace(".jpg", "")
        parts = fn.split("_")
        if len(parts) < 3:
            continue
        try:
            tissue_map[(parts[0], int(parts[1]), int(parts[2]))] = np.array(
                [row[f"class_perc_{i}"] for i in range(5)]
            )
        except ValueError:
            continue
    return tissue_map


def attention_concentration_metrics(attn):
    """Compute concentration metrics for an attention distribution."""
    p = attn / attn.sum()
    n = len(p)
    # Entropy
    ent = -np.sum(p * np.log(p + 1e-10))
    max_ent = np.log(n)
    # Effective number of patches
    eff_n = np.exp(ent)
    # Gini
    sorted_p = np.sort(p)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_p) / (n * np.sum(sorted_p))) - (n + 1) / n
    # Patches for 50% and 80% mass
    desc = np.sort(p)[::-1]
    cum = np.cumsum(desc)
    n_50 = int(np.searchsorted(cum, 0.5) + 1)
    n_80 = int(np.searchsorted(cum, 0.8) + 1)
    return {
        "norm_entropy": ent / max_ent,
        "effective_n": eff_n,
        "effective_pct": eff_n / n * 100,
        "gini": gini,
        "n_for_50pct": n_50,
        "pct_for_50pct": n_50 / n * 100,
        "n_for_80pct": n_80,
        "pct_for_80pct": n_80 / n * 100,
        "max_weight": p.max(),
    }


def patches_for_attention_mass(attn, tissue, frac):
    """Tissue composition of patches carrying `frac` of total attention."""
    p = attn / attn.sum()
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    n_needed = int(np.searchsorted(cum, frac) + 1)
    top_idx = order[:n_needed]
    return tissue[top_idx].mean(axis=0), n_needed


def analyze_task(task, tissue_map):
    with open(f"{RESULTS}/attention/BCNB_{task}_CA_attention_direct.pkl", "rb") as f:
        ca_data = pickle.load(f)
    with open(f"{RESULTS}/attention/BCNB_{task}_NCA_node_data.pkl", "rb") as f:
        nca_data = pickle.load(f)

    rows = []
    for pid_int, ca_entry in ca_data.items():
        pid_str = str(pid_int)
        nca_entry = nca_data.get(pid_str)
        if nca_entry is None:
            continue

        ca_attn = ca_entry["attention_weights"].flatten()
        nca_attn = nca_entry["attention_weights"]
        if isinstance(nca_attn, torch.Tensor):
            nca_attn = nca_attn.numpy()
        nca_attn = nca_attn.flatten()
        centroid = ca_entry["centroid"]
        n = len(ca_attn)
        if len(nca_attn) != n:
            continue

        tissue = np.full((n, 5), np.nan)
        for ni in range(n):
            r, c = int(round(centroid[ni, 0])), int(round(centroid[ni, 1]))
            t = tissue_map.get((pid_str, r, c))
            if t is not None:
                tissue[ni] = t

        valid = ~np.isnan(tissue[:, 0])
        if valid.sum() < 10:
            continue

        ca_v, nca_v, tissue_v = ca_attn[valid], nca_attn[valid], tissue[valid]
        overall = tissue_v.mean(axis=0)

        row = {
            "task": task, "patient_id": pid_int,
            "y_true": ca_entry["y_true"], "y_pred": ca_entry["y_pred"],
            "n_nodes": int(valid.sum()),
        }

        # Overall tissue
        for ti, tn in enumerate(TISSUE_NAMES):
            row[f"overall_{tn}"] = overall[ti]

        # Per-method analysis
        for method, attn in [("nca", nca_v), ("ca", ca_v)]:
            # Concentration metrics
            conc = attention_concentration_metrics(attn)
            for k, v in conc.items():
                row[f"{method}_{k}"] = v

            # Attention-weighted mean
            w = attn / attn.sum()
            weighted = (w[:, None] * tissue_v).sum(axis=0)
            for ti, tn in enumerate(TISSUE_NAMES):
                row[f"{method}_weighted_{tn}"] = weighted[ti]
                row[f"{method}_enrichment_{tn}"] = weighted[ti] - overall[ti]

            # Patches carrying 50% and 80% attention
            for frac in [0.5, 0.8]:
                comp, n_used = patches_for_attention_mass(attn, tissue_v, frac)
                fl = f"{int(frac * 100)}pct"
                row[f"{method}_{fl}_n_patches"] = n_used
                row[f"{method}_{fl}_pct_graph"] = n_used / len(ca_v) * 100
                for ti, tn in enumerate(TISSUE_NAMES):
                    row[f"{method}_{fl}_{tn}"] = comp[ti]
                    row[f"{method}_{fl}_enrichment_{tn}"] = comp[ti] - overall[ti]

        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(df, task):
    n = len(df)
    print(f"\n{'=' * 70}", flush=True)
    print(f"  {task} (n={n})", flush=True)
    print(f"{'=' * 70}", flush=True)

    print(f"\n  ATTENTION CONCENTRATION:", flush=True)
    print(f"  {'': >25s} {'NCA': >15s} {'CA': >15s}", flush=True)
    n50_nca = df["nca_n_for_50pct"].mean()
    n50_ca = df["ca_n_for_50pct"].mean()
    p50_nca = df["nca_pct_for_50pct"].mean()
    p50_ca = df["ca_pct_for_50pct"].mean()
    gini_nca = df["nca_gini"].mean()
    gini_ca = df["ca_gini"].mean()
    print(f"  {'Patches for 50% attn': >25s} {n50_nca: >7.0f} ({p50_nca: >4.1f}%) {n50_ca: >7.0f} ({p50_ca: >4.1f}%)", flush=True)
    print(f"  {'Gini coefficient': >25s} {gini_nca: >15.3f} {gini_ca: >15.3f}", flush=True)

    for fl, label in [("weighted", "Attention-weighted mean"), ("50pct", "Patches carrying 50% attn")]:
        print(f"\n  {label}:", flush=True)
        print(f"  {'Tissue': >15s} {'Overall': >8s} {'NCA': >8s} {'CA': >8s} {'NCA enrich': >11s} {'CA enrich': >10s} {'NCA Wilcoxon': >13s} {'CA Wilcoxon': >12s}", flush=True)
        print(f"  {'-' * 90}", flush=True)
        for tn in TISSUE_ANALYSIS:
            ov = df[f"overall_{tn}"].mean() * 100
            nca_val = df[f"nca_{fl}_{tn}"].mean() * 100
            ca_val = df[f"ca_{fl}_{tn}"].mean() * 100
            nca_e = nca_val - ov
            ca_e = ca_val - ov
            _, p_nca = wilcoxon(df[f"nca_{fl}_enrichment_{tn}"] if f"nca_{fl}_enrichment_{tn}" in df.columns else df[f"nca_enrichment_{tn}"])
            _, p_ca = wilcoxon(df[f"ca_{fl}_enrichment_{tn}"] if f"ca_{fl}_enrichment_{tn}" in df.columns else df[f"ca_enrichment_{tn}"])
            sig_nca = "***" if p_nca < 0.001 else "**" if p_nca < 0.01 else "*" if p_nca < 0.05 else "ns"
            sig_ca = "***" if p_ca < 0.001 else "**" if p_ca < 0.01 else "*" if p_ca < 0.05 else "ns"
            print(f"  {tn: >15s} {ov: >7.1f}% {nca_val: >7.1f}% {ca_val: >7.1f}% {nca_e: >+10.1f}pp {ca_e: >+9.1f}pp  p={p_nca:.4f}{sig_nca: >4s}  p={p_ca:.4f}{sig_ca: >4s}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["2class", "3class", "4class"])
    args = parser.parse_args()

    print("Loading tissue composition...", flush=True)
    tissue_map = load_tissue_composition()
    print(f"  {len(tissue_map)} patches", flush=True)

    all_dfs = []
    for task in args.tasks:
        df = analyze_task(task, tissue_map)
        all_dfs.append(df)
        print_summary(df, task)

    # Save
    df_all = pd.concat(all_dfs, ignore_index=True)
    out_path = f"{RESULTS}/interpretability/attention_tissue_composition_corrected.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df_all)} rows)", flush=True)

    # Paper-ready summary table
    print(f"\n{'=' * 70}", flush=True)
    print(f"  PAPER TABLE: PATCHES CARRYING 50% OF ATTENTION MASS", flush=True)
    print(f"{'=' * 70}", flush=True)
    print(f"\n  {'Task': >8s} {'Method': >8s} {'N patches': >10s} {'Tumor%': >8s} {'Stroma%': >9s} {'Inflam%': >9s} {'Necros%': >9s}", flush=True)
    print(f"  {'-' * 60}", flush=True)
    for df in all_dfs:
        task = df["task"].iloc[0]
        vals_ov = [df[f"overall_{tn}"].mean() * 100 for tn in TISSUE_ANALYSIS]
        print(f"  {task: >8s} {'Overall': >8s} {df['n_nodes'].mean(): >9.0f}  {vals_ov[0]: >7.1f}% {vals_ov[1]: >8.1f}% {vals_ov[2]: >8.1f}% {vals_ov[3]: >8.1f}%", flush=True)
        for method, label in [("nca", "NCA"), ("ca", "CA")]:
            n_p = df[f"{method}_50pct_n_patches"].mean()
            pct = df[f"{method}_50pct_pct_graph"].mean()
            vals = [df[f"{method}_50pct_{tn}"].mean() * 100 for tn in TISSUE_ANALYSIS]
            print(f"  {'': >8s} {label: >8s} {n_p: >5.0f}({pct: >4.1f}%) {vals[0]: >7.1f}% {vals[1]: >8.1f}% {vals[2]: >8.1f}% {vals[3]: >8.1f}%", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
