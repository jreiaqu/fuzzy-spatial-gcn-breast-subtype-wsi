"""
retrain_3class_gcn_mccv.py -- Monte Carlo CV + Final Training for 3-class GENConv

Follows the paper's protocol (04_Experimental_Setting.tex):
  Phase 1: 5-fold CV x N repeats on the 840-patient pool (train+val) to select best config
  Phase 2: Train final model on 630 training patients, 210 val for early stopping
  Phase 3: Evaluate on 218 held-out test patients, save model + predictions

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/retrain_3class_gcn_mccv.py --device cpu
    python scripts/retrain_3class_gcn_mccv.py --device cpu --dry-run
    python scripts/retrain_3class_gcn_mccv.py --device cpu --cv-repeats 3  # faster
"""

import sys
import os
import argparse
import types
import importlib
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# ---------------------------------------------------------------------------
# Compat patches (from retrain_classifier_predictions.py)
# ---------------------------------------------------------------------------

def _apply_compat_patches():
    patches = []
    if not hasattr(torch._utils, '_rebuild_parameter_v2'):
        if hasattr(torch._utils, '_rebuild_parameter_with_state'):
            torch._utils._rebuild_parameter_v2 = torch._utils._rebuild_parameter_with_state
            patches.append("_rebuild_parameter_v2")
    _orig_getattr = nn.Module.__getattr__
    def _patched_getattr(self, name):
        if name in ('_lazy_load_hook', 'decomposed_layers', 'explain'):
            return None
        return _orig_getattr(self, name)
    nn.Module.__getattr__ = _patched_getattr
    patches.append("nn.Module.__getattr__")
    import torch_geometric, torch_geometric.nn.conv
    try:
        importlib.import_module('torch_geometric.nn.conv.utils.inspector')
    except ModuleNotFoundError:
        import torch_geometric.inspector as new_insp
        if not hasattr(torch_geometric.nn.conv, 'utils'):
            torch_geometric.nn.conv.utils = types.ModuleType('torch_geometric.nn.conv.utils')
            sys.modules['torch_geometric.nn.conv.utils'] = torch_geometric.nn.conv.utils
        torch_geometric.nn.conv.utils.inspector = new_insp
        sys.modules['torch_geometric.nn.conv.utils.inspector'] = new_insp
        patches.append("inspector_module_redirect")
    from torch_geometric.inspector import Inspector
    _orig_implements = Inspector.implements
    def _safe_implements(self, func_header):
        try:
            return _orig_implements(self, func_header)
        except AttributeError:
            return set()
    Inspector.implements = _safe_implements
    patches.append("Inspector.implements")
    return patches

_applied_patches = _apply_compat_patches()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB_ROOT = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] CODE_DIR = f"{MOLSUB_ROOT}/code"
GRAPH_DIR = (
    f"{MOLSUB_ROOT}/data/BCNB/results_graphs_november_23/"
    "graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x"
    "/graphs_k_19"
)
GT_FILE = f"{MOLSUB_ROOT}/data/BCNB/ground_truth/patient-clinical-data.xlsx"
SPLIT_DIR = f"{MOLSUB_ROOT}/data/BCNB/patches_paths_class_perc"
# [REPLACED by _paths.py] RESULTS_DIR = "/Users/kckj099/Documents/CMPB-Review/results"

# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)

LABEL_MAP = {"Luminal A": 0, "Luminal B": 0, "HER2(+)": 1, "TNBC": 2}
CLASS_NAMES = {0: "Luminals", 1: "HER2(+)", 2: "TNBC"}
N_CLASSES = 3

# Configs to evaluate (top 3 from single-run experiments)
CONFIGS = [
    {"name": "5L_mean_lr1e5",    "num_layers": 5, "pooling": "mean",      "lr": 1e-05, "wd": 0},
    {"name": "5L_mean_lr2e5_wd", "num_layers": 5, "pooling": "mean",      "lr": 2e-05, "wd": 1e-05},
    {"name": "5L_attn_lr2e5",    "num_layers": 5, "pooling": "attention",  "lr": 2e-05, "wd": 0},
]


def load_ground_truth():
    gt = pd.read_excel(GT_FILE)
    gt = gt.rename(columns={"Patient ID": "patient_id", "Molecular subtype": "mol_subtype"})
    subtype_map = {
        "Luminal A": "Luminal A", "Luminal B": "Luminal B",
        "HER2(+)": "HER2(+)", "HER2 enriched": "HER2(+)",
        "Triple negative": "TNBC", "TNBC": "TNBC",
    }
    gt["mol_subtype_clean"] = gt["mol_subtype"].map(subtype_map)
    gt["label"] = gt["mol_subtype_clean"].map(LABEL_MAP)
    gt = gt.dropna(subset=["label"])
    gt["label"] = gt["label"].astype(int)
    return gt


def get_patient_split():
    splits = {}
    for split_name in ["train", "val", "test"]:
        path = os.path.join(SPLIT_DIR, f"{split_name}_patches_class_perc_0_tp.csv")
        df = pd.read_csv(path)
        pids = df["patch_path"].str.extract(r"patches_512_fullWSIs_0/(\d+)/", expand=False)
        splits[split_name] = set(pids.dropna().astype(int).unique())
    return splits


def load_graphs(graph_dir, patient_ids, gt_df, device):
    graphs, labels, pids = [], [], []
    for pid in sorted(patient_ids):
        gpath = os.path.join(graph_dir, f"{pid}_graph.pt")
        if not os.path.exists(gpath):
            continue
        gt_row = gt_df[gt_df["patient_id"] == pid]
        if len(gt_row) == 0:
            continue
        label = gt_row["label"].values[0]
        graph = torch.load(gpath, map_location=device, weights_only=False)
        graphs.append(graph)
        labels.append(label)
        pids.append(pid)
    return graphs, labels, pids


def create_model(cfg, device):
    from MIL_models import PatchGCN_MeanMax_LSelec
    model = PatchGCN_MeanMax_LSelec(
        num_features=512, num_layers=cfg["num_layers"], hidden_dim=128,
        n_classes=N_CLASSES, pooling=cfg["pooling"], gnn_layer_type="GENConv",
    ).to(device)
    return model


def train_epoch(model, graphs, labels, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for idx in np.random.permutation(len(graphs)):
        graph = graphs[idx].to(device)
        label = torch.tensor([labels[idx]], dtype=torch.long, device=device)
        optimizer.zero_grad()
        _, Y_hat, logits, _ = model(graph)
        loss = F.cross_entropy(logits, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (Y_hat.item() == labels[idx])
        total += 1
    return total_loss / total, correct / total


def evaluate(model, graphs, labels, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for graph, label in zip(graphs, labels):
            graph = graph.to(device)
            Y_prob, Y_hat, _, _ = model(graph)
            all_preds.append(Y_hat.item())
            all_probs.append(Y_prob.squeeze().cpu().numpy())
            all_labels.append(label)
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    y_prob = np.array(all_probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0
    return acc, f1, auc, y_pred, y_prob


def train_with_patience(model, train_graphs, train_labels, val_graphs, val_labels,
                        lr, wd, epochs, patience, device, label=""):
    """Train model with early stopping on val F1. Returns best state dict and metrics."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val_f1, best_state, patience_counter = 0, None, 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_graphs, train_labels, optimizer, device)
        val_acc, val_f1, val_auc, _, _ = evaluate(model, val_graphs, val_labels, device)

        improved = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            improved = " *"
        else:
            patience_counter += 1

        if epoch <= 3 or epoch % 10 == 0 or improved or patience_counter >= patience:
            print(f"      {label}ep{epoch:>3d} loss={train_loss:.3f} val_f1={val_f1:.3f} pat={patience_counter}/{patience}{improved}", flush=True)

        if patience_counter >= patience:
            print(f"      {label}early stop at epoch {epoch}", flush=True)
            break

    return best_state, best_val_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--config-name", type=str, default=None,
                        help="Run only this config (by name). If set, only does Phase 1 CV for that config.")
    parser.add_argument("--final-only", type=str, default=None,
                        help="Skip CV, train final model with this config name directly.")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}", flush=True)
    print(f"MC-CV: {args.cv_folds}-fold x {args.cv_repeats} repeats", flush=True)

    # Load data
    gt_df = load_ground_truth()
    splits = get_patient_split()

    # Load ALL non-test graphs (train + val pool for CV)
    cv_pool_ids = splits["train"] | splits["val"]  # 840 patients
    print(f"\nLoading graphs for CV pool ({len(cv_pool_ids)} patients)...", flush=True)
    cv_graphs, cv_labels, cv_pids = load_graphs(GRAPH_DIR, cv_pool_ids, gt_df, "cpu")
    print(f"  Loaded: {len(cv_graphs)} graphs", flush=True)

    # Load test set separately (for final evaluation only)
    print(f"Loading test set ({len(splits['test'])} patients)...", flush=True)
    test_graphs, test_labels, test_pids = load_graphs(GRAPH_DIR, splits["test"], gt_df, "cpu")
    print(f"  Loaded: {len(test_graphs)} graphs", flush=True)

    # Class distributions
    for name, labs in [("CV pool", cv_labels), ("Test", test_labels)]:
        counts = np.bincount(labs, minlength=N_CLASSES)
        print(f"  {name}: {dict(zip(CLASS_NAMES.values(), counts))}", flush=True)

    if args.dry_run:
        print(f"\nDry run. Would evaluate {len(CONFIGS)} configs:", flush=True)
        for cfg in CONFIGS:
            print(f"  {cfg['name']}: {cfg['num_layers']}L/{cfg['pooling']}/lr={cfg['lr']}/wd={cfg['wd']}", flush=True)
        print(f"  {args.cv_folds}-fold x {args.cv_repeats} repeats = {args.cv_folds * args.cv_repeats} training runs per config", flush=True)
        print(f"  Total: {len(CONFIGS) * args.cv_folds * args.cv_repeats} training runs", flush=True)
        return

    from MIL_models import PatchGCN_MeanMax_LSelec

    cv_labels_arr = np.array(cv_labels)
    cv_pids_arr = np.array(cv_pids)

    # Filter configs if --config-name specified
    if args.config_name:
        configs_to_run = [c for c in CONFIGS if c["name"] == args.config_name]
        if not configs_to_run:
            print(f"ERROR: config '{args.config_name}' not found. Available: {[c['name'] for c in CONFIGS]}", flush=True)
            return
    else:
        configs_to_run = CONFIGS

    # Skip to final training if --final-only
    if args.final_only:
        best_cfg = [c for c in CONFIGS if c["name"] == args.final_only]
        if not best_cfg:
            print(f"ERROR: config '{args.final_only}' not found.", flush=True)
            return
        best_cfg = best_cfg[0]
        # Jump to Phase 2
        print(f"\n  SKIPPING CV. Final training with: {args.final_only}", flush=True)
        train_graphs, train_labels, train_pids = load_graphs(GRAPH_DIR, splits["train"], gt_df, "cpu")
        val_graphs, val_labels, val_pids = load_graphs(GRAPH_DIR, splits["val"], gt_df, "cpu")
        final_model = create_model(best_cfg, device)
        best_state, best_val_f1 = train_with_patience(
            final_model, train_graphs, train_labels, val_graphs, val_labels,
            best_cfg["lr"], best_cfg["wd"], args.epochs, args.patience, device
        )
        final_model.load_state_dict(best_state)
        test_acc, test_f1, test_auc, test_preds, test_probs = evaluate(final_model, test_graphs, test_labels, device)
        print(f"  Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}", flush=True)
        out_dir = os.path.join(RESULTS_DIR, "retrained_models")
        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"3class_GENConv_{args.final_only}_final.pth")
        torch.save(final_model, model_path)
        print(f"  Saved model: {model_path}", flush=True)
        pred_rows = []
        for pid, yt, yp, yprob in zip(test_pids, test_labels, test_preds, test_probs):
            row = {"patient_id": pid, "y_true": yt, "y_pred": int(yp)}
            for c in range(N_CLASSES):
                row[f"y_prob_{c}"] = float(yprob[c])
            pred_rows.append(row)
        pred_path = os.path.join(RESULTS_DIR, "predictions", "BCNB_3class_CA_GENConv_final_predictions.csv")
        pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
        print(f"  Saved predictions: {pred_path}", flush=True)
        return

    # ====================================================================
    # PHASE 1: Monte Carlo CV for config selection
    # ====================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 1: Monte Carlo CV ({args.cv_folds}-fold x {args.cv_repeats} repeats)", flush=True)
    print(f"{'='*70}", flush=True)

    all_cv_results = []

    for cfg in configs_to_run:
        print(f"\n  --- Config: {cfg['name']} ---", flush=True)
        fold_results = []

        for repeat in range(args.cv_repeats):
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=repeat * 42)

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(cv_labels_arr, cv_labels_arr)):
                t0 = time.time()
                train_g = [cv_graphs[i] for i in train_idx]
                train_l = [cv_labels[i] for i in train_idx]
                val_g = [cv_graphs[i] for i in val_idx]
                val_l = [cv_labels[i] for i in val_idx]

                model = create_model(cfg, device)
                fold_label = f"R{repeat+1}F{fold_idx+1} "
                best_state, best_val_f1 = train_with_patience(
                    model, train_g, train_l, val_g, val_l,
                    cfg["lr"], cfg["wd"], args.epochs, args.patience, device,
                    label=fold_label
                )

                # Evaluate best model on the val fold
                model.load_state_dict(best_state)
                val_acc, val_f1, val_auc, _, _ = evaluate(model, val_g, val_l, device)
                elapsed = time.time() - t0

                fold_results.append({
                    "config": cfg["name"], "repeat": repeat, "fold": fold_idx,
                    "val_acc": val_acc, "val_f1": val_f1, "val_auc": val_auc,
                    "elapsed_s": elapsed,
                })
                print(f"    R{repeat+1}F{fold_idx+1}: val_acc={val_acc:.3f} val_f1={val_f1:.3f} ({elapsed:.0f}s)", flush=True)

                del model, best_state, train_g, val_g

        # Summary for this config
        df_cfg = pd.DataFrame(fold_results)
        mean_f1 = df_cfg["val_f1"].mean()
        std_f1 = df_cfg["val_f1"].std()
        mean_acc = df_cfg["val_acc"].mean()
        std_acc = df_cfg["val_acc"].std()
        print(f"  >> {cfg['name']}: F1={mean_f1:.4f}+/-{std_f1:.4f}  Acc={mean_acc:.4f}+/-{std_acc:.4f}", flush=True)

        all_cv_results.extend(fold_results)

        # Save per-config CV results immediately (safe for parallel)
        out_dir = os.path.join(RESULTS_DIR, "retrained_models")
        os.makedirs(out_dir, exist_ok=True)
        cfg_cv_path = os.path.join(out_dir, f"3class_mccv_{cfg['name']}.csv")
        pd.DataFrame(fold_results).to_csv(cfg_cv_path, index=False)
        print(f"  Saved: {cfg_cv_path}", flush=True)

    # Save combined CV results
    cv_df = pd.DataFrame(all_cv_results)
    cv_path = os.path.join(out_dir, "3class_mccv_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"\n  Saved CV results: {cv_path}", flush=True)

    # Select best config by mean val F1
    cv_summary = cv_df.groupby("config").agg(
        mean_f1=("val_f1", "mean"), std_f1=("val_f1", "std"),
        mean_acc=("val_acc", "mean"), std_acc=("val_acc", "std"),
    ).reset_index().sort_values("mean_f1", ascending=False)

    print(f"\n  CV SUMMARY (sorted by mean val F1):", flush=True)
    print(f"  {'Config':<25s} {'Mean F1':>10s} {'Std F1':>10s} {'Mean Acc':>10s} {'Std Acc':>10s}", flush=True)
    for _, row in cv_summary.iterrows():
        print(f"  {row['config']:<25s} {row['mean_f1']:>10.4f} {row['std_f1']:>10.4f} "
              f"{row['mean_acc']:>10.4f} {row['std_acc']:>10.4f}")

    best_config_name = cv_summary.iloc[0]["config"]
    best_cfg = [c for c in CONFIGS if c["name"] == best_config_name][0]
    print(f"\n  BEST CONFIG: {best_config_name}", flush=True)

    # ====================================================================
    # PHASE 2: Train final model on original train split
    # ====================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 2: Final model training (630 train, 210 val for early stopping)", flush=True)
    print(f"{'='*70}", flush=True)

    # Load train and val separately for final model
    train_graphs, train_labels, train_pids = load_graphs(GRAPH_DIR, splits["train"], gt_df, "cpu")
    val_graphs, val_labels, val_pids = load_graphs(GRAPH_DIR, splits["val"], gt_df, "cpu")
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}", flush=True)

    final_model = create_model(best_cfg, device)
    print(f"  Training {best_config_name}...", flush=True)

    best_state, best_val_f1 = train_with_patience(
        final_model, train_graphs, train_labels, val_graphs, val_labels,
        best_cfg["lr"], best_cfg["wd"], args.epochs, args.patience, device
    )
    final_model.load_state_dict(best_state)
    print(f"  Best val F1: {best_val_f1:.4f}", flush=True)

    # ====================================================================
    # PHASE 3: Evaluate on held-out test set + save
    # ====================================================================
    print(f"\n{'='*70}", flush=True)
    print(f"  PHASE 3: Test evaluation + save", flush=True)
    print(f"{'='*70}", flush=True)

    test_acc, test_f1, test_auc, test_preds, test_probs = evaluate(
        final_model, test_graphs, test_labels, device
    )
    print(f"  Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}", flush=True)
    print(f"  (GINConv baseline: Acc=0.6514)", flush=True)

    # Save model
    model_path = os.path.join(out_dir, f"3class_GENConv_{best_config_name}_final.pth")
    torch.save(final_model, model_path)
    print(f"  Saved model: {model_path}", flush=True)

    # Save predictions
    pred_rows = []
    for pid, y_true, y_pred, y_prob in zip(test_pids, test_labels, test_preds, test_probs):
        row = {"patient_id": pid, "y_true": y_true, "y_pred": int(y_pred)}
        for c in range(N_CLASSES):
            row[f"y_prob_{c}"] = float(y_prob[c])
        pred_rows.append(row)
    pred_df = pd.DataFrame(pred_rows)
    pred_path = os.path.join(RESULTS_DIR, "predictions", "BCNB_3class_CA_GENConv_final_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"  Saved predictions: {pred_path}", flush=True)

    # Save summary
    summary = {
        "best_config": best_config_name,
        "cv_mean_f1": cv_summary.iloc[0]["mean_f1"],
        "cv_std_f1": cv_summary.iloc[0]["std_f1"],
        "cv_mean_acc": cv_summary.iloc[0]["mean_acc"],
        "cv_std_acc": cv_summary.iloc[0]["std_acc"],
        "test_acc": test_acc, "test_f1": test_f1, "test_auc": test_auc,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "3class_final_summary.csv"), index=False)
    print(f"\n  DONE. Best config: {best_config_name}, Test Acc: {test_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
