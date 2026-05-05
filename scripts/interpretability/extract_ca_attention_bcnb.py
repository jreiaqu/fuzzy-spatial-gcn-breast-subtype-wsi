"""
extract_ca_attention_bcnb.py -- Extract direct attention weights from CA models
with attention pooling on BCNB test set.

For models with gated attention pooling (Attn_Net_Gated), extracts per-node
softmax-normalized attention weights via forward hook on path_attention_head.
Correlates with TSM tissue composition for interpretability analysis.

Currently supports: 3-class GENConv/attention (retrained model).
For 2-class (mean pooling) and 4-class (max pooling), attention extraction
is not applicable; use GNNExplainer or gradient methods instead.

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/extract_ca_attention_bcnb.py
    python scripts/extract_ca_attention_bcnb.py --tasks 3class
    python scripts/extract_ca_attention_bcnb.py --tasks 2class 3class 4class  # after retrain
"""

import sys, os, argparse, pickle, types, importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# Compat patches
def _apply_compat_patches():
    if not hasattr(torch._utils, '_rebuild_parameter_v2'):
        if hasattr(torch._utils, '_rebuild_parameter_with_state'):
            torch._utils._rebuild_parameter_v2 = torch._utils._rebuild_parameter_with_state
    _orig = nn.Module.__getattr__
    def _p(self, name):
        if name in ('_lazy_load_hook', 'decomposed_layers', 'explain'): return None
        return _orig(self, name)
    nn.Module.__getattr__ = _p
    import torch_geometric, torch_geometric.nn.conv
    try: importlib.import_module('torch_geometric.nn.conv.utils.inspector')
    except ModuleNotFoundError:
        import torch_geometric.inspector as ni
        if not hasattr(torch_geometric.nn.conv, 'utils'):
            torch_geometric.nn.conv.utils = types.ModuleType('torch_geometric.nn.conv.utils')
            sys.modules['torch_geometric.nn.conv.utils'] = torch_geometric.nn.conv.utils
        torch_geometric.nn.conv.utils.inspector = ni
        sys.modules['torch_geometric.nn.conv.utils.inspector'] = ni
    from torch_geometric.inspector import Inspector
    _oi = Inspector.implements
    def _si(self, fh):
        try: return _oi(self, fh)
        except AttributeError: return set()
    Inspector.implements = _si

_apply_compat_patches()

# [REPLACED by _paths.py] MOLSUB = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] sys.path.insert(0, f"{MOLSUB}/code")
# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"
SPLIT_DIR = f"{MOLSUB}/data/BCNB/patches_paths_class_perc"
GT_FILE = f"{MOLSUB}/data/BCNB/ground_truth/patient-clinical-data.xlsx"

# Models with attention pooling (can extract attention directly)
CA_MODELS = {
    "3class": {
        "weights_path": f"{RESULTS}/retrained_models/3class_GENConv_5L_attn_lr2e5_final.pth",
        "graph_dir": f"{MOLSUB}/data/BCNB/results_graphs_november_23/graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x/graphs_k_19",
        "n_classes": 3, "pooling": "attention",
        "label_map": {"Luminal A": 0, "Luminal B": 0, "HER2(+)": 1, "HER2 enriched": 1, "Triple negative": 2, "TNBC": 2},
    },
    "2class": {
        "weights_path": f"{RESULTS}/retrained_models/2class_GENConv_5L_attn_5L_attn_lr1e5_wd_final.pth",
        "graph_dir": f"{MOLSUB}/data/BCNB/results_graphs_november_23/graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x/graphs_k_19",
        "n_classes": 2, "pooling": "attention",
        "label_map": {"Luminal A": 0, "Luminal B": 0, "HER2(+)": 0, "HER2 enriched": 0, "Triple negative": 1, "TNBC": 1},
    },
    "4class": {
        "weights_path": f"{RESULTS}/retrained_models/4class_GENConv_5L_attn_5L_attn_lr1e5_final.pth",
        "graph_dir": f"{MOLSUB}/data/BCNB/results_graphs_november_23/graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x/graphs_k_19",
        "n_classes": 4, "pooling": "attention",
        "label_map": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "HER2 enriched": 2, "Triple negative": 3, "TNBC": 3},
    },
}

TISSUE_NAMES = {1: "tumor", 2: "stroma", 3: "inflammation", 4: "necrosis"}


def get_test_ids():
    df = pd.read_csv(f"{SPLIT_DIR}/test_patches_class_perc_0_tp.csv")
    return set(df["patch_path"].str.extract(r"patches_512_fullWSIs_0/(\d+)/", expand=False).dropna().astype(int).unique())


def load_gt(label_map):
    gt = pd.read_excel(GT_FILE).rename(columns={"Patient ID": "patient_id", "Molecular subtype": "mol_subtype"})
    gt["label"] = gt["mol_subtype"].map(label_map)
    gt = gt.dropna(subset=["label"])
    gt["label"] = gt["label"].astype(int)
    return gt


def load_tissue_composition():
    frames = []
    for s in ("train", "val", "test"):
        frames.append(pd.read_csv(f"{SPLIT_DIR}/{s}_patches_class_perc_0_tp.csv"))
    df = pd.concat(frames, ignore_index=True)
    tissue_map = {}
    for _, row in df.iterrows():
        fn = row["patch_path"].replace("\\", "/").split("/")[-1].replace(".jpg", "")
        parts = fn.split("_")
        if len(parts) < 3: continue
        try:
            tissue_map[(parts[0], int(parts[1]), int(parts[2]))] = np.array(
                [row[f"class_perc_{i}"] for i in range(5)]
            )
        except ValueError: continue
    return tissue_map


def extract_attention_for_task(task, tissue_map):
    """Extract CA attention weights and correlate with tissue for one task."""
    cfg = CA_MODELS[task]

    print(f"\n{'='*70}\n  TASK: {task}\n{'='*70}", flush=True)

    # Load model
    model = torch.load(cfg["weights_path"], map_location="cpu", weights_only=False)
    model.eval()
    assert model.pooling == "attention", f"Model pooling is '{model.pooling}', not 'attention'"
    assert hasattr(model, "path_attention_head"), "Model has no path_attention_head"
    print(f"  Model loaded: {model.pooling} pooling, has path_attention_head", flush=True)

    test_ids = get_test_ids()
    gt = load_gt(cfg["label_map"])

    # Also load NCA attention for comparison
    nca_path = f"{RESULTS}/attention/BCNB_{task}_NCA_node_data.pkl"
    nca_data = {}
    if os.path.exists(nca_path):
        with open(nca_path, "rb") as f:
            nca_data = pickle.load(f)

    patient_results = []
    attention_data = {}  # For saving raw attention per patient

    for pid in sorted(test_ids):
        gpath = os.path.join(cfg["graph_dir"], f"{pid}_graph.pt")
        if not os.path.exists(gpath): continue
        gt_row = gt[gt["patient_id"] == pid]
        if len(gt_row) == 0: continue
        label = gt_row["label"].values[0]

        graph = torch.load(gpath, map_location="cpu", weights_only=False)
        centroid = graph["centroid"].numpy()
        n_nodes = graph["x"].shape[0]

        # Forward with attention hook
        captured = {}
        def hook_fn(module, inp, out):
            A, x = out
            captured["raw_A"] = A.detach().clone()

        hook = model.path_attention_head.register_forward_hook(hook_fn)
        with torch.no_grad():
            Y_prob, Y_hat, logits, h = model(graph)
        hook.remove()

        attn = F.softmax(captured["raw_A"].squeeze(-1), dim=0).numpy()
        pred = Y_hat.item()

        # Save raw attention
        attention_data[pid] = {
            "attention_weights": attn,
            "centroid": centroid,
            "y_true": label, "y_pred": pred,
            "y_prob": Y_prob.squeeze().cpu().numpy(),
            "n_nodes": n_nodes,
        }

        # Match to tissue
        tissue_percs = np.full((n_nodes, 5), np.nan)
        for ni in range(n_nodes):
            r, c = int(round(centroid[ni, 0])), int(round(centroid[ni, 1]))
            t = tissue_map.get((str(pid), r, c))
            if t is not None: tissue_percs[ni] = t

        valid = (~np.isnan(tissue_percs[:, 0])) & (tissue_percs[:, 0] < 0.6)
        if valid.sum() < 10: continue

        af = attn[valid]
        tf = tissue_percs[valid]

        row = {"patient_id": pid, "task": task, "y_true": label, "y_pred": pred,
               "correct": int(label == pred), "n_nodes": n_nodes}

        for ti, tn in TISSUE_NAMES.items():
            r_val, p_val = spearmanr(af, tf[:, ti])
            row[f"ca_attn_r_{tn}"] = r_val
            row[f"ca_attn_p_{tn}"] = p_val

        # NCA comparison
        nca_entry = nca_data.get(str(pid))
        if nca_entry is not None:
            nca_attn = nca_entry["attention_weights"]
            if isinstance(nca_attn, torch.Tensor): nca_attn = nca_attn.numpy()
            nca_attn = nca_attn.flatten()
            if len(nca_attn) == n_nodes:
                nf = nca_attn[valid]
                for ti, tn in TISSUE_NAMES.items():
                    r_val, p_val = spearmanr(nf, tf[:, ti])
                    row[f"nca_r_{tn}"] = r_val
                    row[f"nca_p_{tn}"] = p_val
                # CA vs NCA
                r_cn, p_cn = spearmanr(af, nca_attn[valid])
                row["ca_vs_nca_r"] = r_cn
                row["ca_vs_nca_p"] = p_cn

        patient_results.append(row)

    df = pd.DataFrame(patient_results)
    n = len(df)
    print(f"\n  Patients: {n}", flush=True)

    # Summary
    for label, prefix in [("CA attention", "ca_attn"), ("NCA attention", "nca")]:
        if f"{prefix}_r_tumor" not in df.columns: continue
        print(f"\n  {label} vs tissue (mean Spearman r, n={n}):", flush=True)
        print(f"  {'Tissue':>15s} {'Mean r':>8s} {'95% CI':>16s} {'% sig':>7s}", flush=True)
        print(f"  {'-'*50}", flush=True)
        for tn in TISSUE_NAMES.values():
            col = f"{prefix}_r_{tn}"
            pcol = f"{prefix}_p_{tn}"
            mr = df[col].mean()
            se = df[col].std() / np.sqrt(n)
            psig = (df[pcol] < 0.05).mean() * 100
            print(f"  {tn:>15s} {mr:>+8.3f} [{mr-1.96*se:>+.3f}, {mr+1.96*se:>+.3f}] {psig:>6.1f}%", flush=True)

    if "ca_vs_nca_r" in df.columns:
        mr = df["ca_vs_nca_r"].mean()
        print(f"\n  CA vs NCA: r={mr:+.3f} +/- {df['ca_vs_nca_r'].std():.3f}", flush=True)

    return df, attention_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["3class"],
                        choices=list(CA_MODELS.keys()))
    args = parser.parse_args()

    tissue_map = load_tissue_composition()
    print(f"Tissue composition: {len(tissue_map)} patches", flush=True)

    os.makedirs(f"{RESULTS}/interpretability", exist_ok=True)

    for task in args.tasks:
        if task not in CA_MODELS:
            print(f"SKIP {task}: no attention-pooling model configured", flush=True)
            continue

        df, attention_data = extract_attention_for_task(task, tissue_map)

        # Save correlation results
        corr_path = f"{RESULTS}/interpretability/{task}_ca_direct_attention_tissue_correlation.csv"
        df.to_csv(corr_path, index=False)
        print(f"  Saved correlations: {corr_path}", flush=True)

        # Save raw attention data
        attn_path = f"{RESULTS}/attention/BCNB_{task}_CA_attention_direct.pkl"
        with open(attn_path, "wb") as f:
            pickle.dump(attention_data, f)
        print(f"  Saved attention data: {attn_path}", flush=True)


if __name__ == "__main__":
    main()
