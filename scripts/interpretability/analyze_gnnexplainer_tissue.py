"""
analyze_gnnexplainer_tissue.py -- R1.1 Interpretability Validation

Correlates GNNExplainer node importance (CA) and NCA attention weights
with TSM tissue composition (tumor, stroma, inflammation, necrosis)
across all BCNB test patients.

This is the VALIDATION analysis for the preliminary gradient-based results
from Session 2. GNNExplainer is saturation-proof (optimization-based, not
gradient-based), providing robust per-node importance scores.

Input:
  - results/interpretability/gnnexplainer_{task}_importance.pkl (from batch_gnnexplainer.py)
  - results/attention/BCNB_{task}_NCA_node_data.pkl (from generate_predictions.py)
  - data/BCNB/patches_paths_class_perc/*.csv (TSM tissue composition)

Output:
  - results/interpretability/gnnexplainer_tissue_correlation.csv (per-patient correlations)
  - Console summary with cross-task comparison

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/analyze_gnnexplainer_tissue.py
    python scripts/analyze_gnnexplainer_tissue.py --tasks 2class 3class
"""

import os, sys, argparse, pickle
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# [REPLACED by _paths.py] MOLSUB = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"
TISSUE_DIR = f"{MOLSUB}/data/BCNB/patches_paths_class_perc"

TISSUE_NAMES = {0: "other", 1: "tumor", 2: "stroma", 3: "inflammation", 4: "necrosis"}
TISSUE_ANALYSIS = ["tumor", "stroma", "inflammation", "necrosis"]  # skip "other"


def load_tissue_composition():
    """Load TSM tissue percentages from split CSVs.
    Returns dict keyed by (pid_str, row, col) -> np.array([perc_0..4]).
    """
    frames = []
    for split in ("train", "val", "test"):
        frames.append(pd.read_csv(os.path.join(TISSUE_DIR, f"{split}_patches_class_perc_0_tp.csv")))
    df = pd.concat(frames, ignore_index=True)

    tissue_map = {}
    for _, row in df.iterrows():
        path = row["patch_path"].replace("\\", "/")
        filename = path.split("/")[-1].replace(".jpg", "")
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        try:
            tissue_map[(parts[0], int(parts[1]), int(parts[2]))] = np.array([
                row["class_perc_0"], row["class_perc_1"], row["class_perc_2"],
                row["class_perc_3"], row["class_perc_4"]
            ])
        except ValueError:
            continue
    return tissue_map


def analyze_task(task, tissue_map):
    """Run tissue-type correlation for one task. Returns DataFrame of per-patient results."""
    print(f"\n{'='*70}\n  TASK: {task}\n{'='*70}", flush=True)

    # Load GNNExplainer importance
    gnn_path = f"{RESULTS}/interpretability/gnnexplainer_{task}_importance.pkl"
    if not os.path.exists(gnn_path):
        print(f"  SKIP: {gnn_path} not found", flush=True)
        return pd.DataFrame()
    with open(gnn_path, "rb") as f:
        gnn_data = pickle.load(f)

    # Load NCA attention (string keys)
    nca_path = f"{RESULTS}/attention/BCNB_{task}_NCA_node_data.pkl"
    with open(nca_path, "rb") as f:
        nca_data = pickle.load(f)

    patient_corrs = []

    for pid_raw in sorted(gnn_data.keys()):
        pid_str = str(int(pid_raw))

        gnn_imp = gnn_data[pid_raw]["importance"]
        centroid = gnn_data[pid_raw]["centroid"]
        y_true = gnn_data[pid_raw]["y_true"]
        y_pred = gnn_data[pid_raw]["y_pred"]
        n_nodes = len(gnn_imp)

        # NCA attention
        nca_entry = nca_data.get(pid_str)
        nca_attn = None
        if nca_entry is not None and "attention_weights" in nca_entry:
            aw = nca_entry["attention_weights"]
            if isinstance(aw, torch.Tensor):
                aw = aw.numpy()
            nca_attn = aw.flatten()

        # Match nodes to tissue composition
        tissue_percs = np.full((n_nodes, 5), np.nan)
        matched = 0
        for ni in range(n_nodes):
            r = int(round(centroid[ni, 0]))
            c = int(round(centroid[ni, 1]))
            key = (pid_str, r, c)
            if key in tissue_map:
                tissue_percs[ni] = tissue_map[key]
                matched += 1

        if matched < 10:
            continue

        # Filter: remove unmatched + background-dominated (>60% other)
        valid = (~np.isnan(tissue_percs[:, 0])) & (tissue_percs[:, 0] < 0.6)
        if valid.sum() < 10:
            continue

        gf = gnn_imp[valid]
        tf = tissue_percs[valid]
        nf = nca_attn[valid] if nca_attn is not None and len(nca_attn) == n_nodes else None

        row = {
            "patient_id": int(pid_raw), "task": task,
            "y_true": y_true, "y_pred": y_pred,
            "correct": int(y_true == y_pred),
            "n_nodes": n_nodes, "n_matched": matched,
            "n_tissue_nodes": int(valid.sum()),
        }

        # Correlations: CA importance vs each tissue type
        for ti, tn in TISSUE_NAMES.items():
            rc, pc = spearmanr(gf, tf[:, ti])
            row[f"ca_r_{tn}"] = rc
            row[f"ca_p_{tn}"] = pc
            if nf is not None:
                rn, pn = spearmanr(nf, tf[:, ti])
                row[f"nca_r_{tn}"] = rn
                row[f"nca_p_{tn}"] = pn

        # CA vs NCA spatial correlation
        if nf is not None:
            rcn, pcn = spearmanr(gf, nf)
            row["ca_vs_nca_r"] = rcn
            row["ca_vs_nca_p"] = pcn

        patient_corrs.append(row)

    df = pd.DataFrame(patient_corrs)
    n = len(df)
    print(f"\n  Patients analyzed: {n} / 218", flush=True)

    # Print summary
    for label, prefix in [("CA (GNNExplainer)", "ca"), ("NCA (attention)", "nca")]:
        col_check = f"{prefix}_r_tumor"
        if col_check not in df.columns:
            continue
        print(f"\n  {label} vs tissue type (mean Spearman r, n={n}):", flush=True)
        print(f"  {'Tissue':>15s} {'Mean r':>8s} {'95% CI':>16s} {'% sig':>7s}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for tn in TISSUE_ANALYSIS:
            col = f"{prefix}_r_{tn}"
            pcol = f"{prefix}_p_{tn}"
            mr = df[col].mean()
            se = df[col].std() / np.sqrt(n)
            ci = f"[{mr-1.96*se:+.3f}, {mr+1.96*se:+.3f}]"
            psig = (df[pcol] < 0.05).mean() * 100
            print(f"  {tn:>15s} {mr:>+8.3f} {ci:>16s} {psig:>6.1f}%", flush=True)

    if "ca_vs_nca_r" in df.columns:
        mr = df["ca_vs_nca_r"].mean()
        neg = ((df["ca_vs_nca_p"] < 0.05) & (df["ca_vs_nca_r"] < 0)).sum()
        pos = ((df["ca_vs_nca_p"] < 0.05) & (df["ca_vs_nca_r"] > 0)).sum()
        print(f"\n  CA vs NCA spatial: r={mr:+.3f} +/- {df['ca_vs_nca_r'].std():.3f}", flush=True)
        print(f"    Sig negative: {neg}/{n} ({neg/n*100:.1f}%)  Sig positive: {pos}/{n} ({pos/n*100:.1f}%)", flush=True)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["2class", "3class", "4class"])
    args = parser.parse_args()

    print("Loading tissue composition...", flush=True)
    tissue_map = load_tissue_composition()
    print(f"  {len(tissue_map)} patches loaded", flush=True)

    all_results = []
    for task in args.tasks:
        df = analyze_task(task, tissue_map)
        if len(df) > 0:
            all_results.append(df)

    if not all_results:
        print("No results to save.")
        return

    df_all = pd.concat(all_results, ignore_index=True)
    out_path = f"{RESULTS}/interpretability/gnnexplainer_tissue_correlation.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(df_all)} rows)", flush=True)

    # Cross-task summary
    print(f"\n{'='*70}\n  CROSS-TASK SUMMARY\n{'='*70}", flush=True)
    for label, prefix in [("CA (GNNExplainer)", "ca"), ("NCA (attention)", "nca")]:
        print(f"\n  {label}:", flush=True)
        print(f"  {'Task':>8s}", end="", flush=True)
        for tn in TISSUE_ANALYSIS:
            print(f" {tn:>12s}", end="", flush=True)
        print(flush=True)
        print(f"  {'-'*56}", flush=True)
        for df in all_results:
            t = df["task"].iloc[0]
            col_check = f"{prefix}_r_tumor"
            if col_check not in df.columns:
                continue
            print(f"  {t:>8s}", end="", flush=True)
            for tn in TISSUE_ANALYSIS:
                print(f" {df[f'{prefix}_r_{tn}'].mean():>+12.3f}", end="", flush=True)
            print(flush=True)


if __name__ == "__main__":
    main()
