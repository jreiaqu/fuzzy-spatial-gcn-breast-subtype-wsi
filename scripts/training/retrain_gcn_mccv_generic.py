"""
retrain_gcn_mccv_generic.py -- Retrain any task's GCN with GENConv + attention pooling

Follows the paper's protocol (04_Experimental_Setting.tex):
  Phase 1: 5-fold CV x N repeats on 840 patients (train+val) to select best config
  Phase 2: Train final model on 630 train, 210 val for early stopping
  Phase 3: Evaluate on 218 test, save model + predictions + provenance

Designed for overnight launch. Minimal HP grid (2 configs) to keep runs short.

Usage:
    pip install -r requirements.txt  # see repository root
    # 2-class
    python scripts/retrain_gcn_mccv_generic.py --task 2class --device cpu
    # 4-class
    python scripts/retrain_gcn_mccv_generic.py --task 4class --device cpu
    # Dry run
    python scripts/retrain_gcn_mccv_generic.py --task 2class --dry-run
"""

import sys, os, argparse, types, importlib, time, json, warnings

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ---------------------------------------------------------------------------
# Compat patches
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] CODE_DIR = f"{MOLSUB}/code"
# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)
# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"
SPLIT_DIR = f"{MOLSUB}/data/BCNB/patches_paths_class_perc"
GT_FILE = f"{MOLSUB}/data/BCNB/ground_truth/patient-clinical-data.xlsx"

# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------
TASKS = {
    "2class": {
        "graph_subdir": "graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
        "knn": 19,
        "n_classes": 2,
        "label_map": {
            "Luminal A": 0, "Luminal B": 0, "HER2(+)": 0, "HER2 enriched": 0,
            "Triple negative": 1, "TNBC": 1,
        },
        "class_names": {0: "Other", 1: "TNBC"},
        "configs": [
            {"name": "5L_attn_lr1e5",  "num_layers": 5, "lr": 1e-05, "wd": 0},
            {"name": "5L_attn_lr2e5",  "num_layers": 5, "lr": 2e-05, "wd": 0},
        ],
    },
    "4class": {
        "graph_subdir": "graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
        "knn": 19,  # standardize to k=19 (was k=25)
        "n_classes": 4,
        "label_map": {
            "Luminal A": 0, "Luminal B": 1,
            "HER2(+)": 2, "HER2 enriched": 2,
            "Triple negative": 3, "TNBC": 3,
        },
        "class_names": {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3: "TNBC"},
        "configs": [
            {"name": "5L_attn_lr2e5",  "num_layers": 5, "lr": 2e-05, "wd": 0},
            {"name": "5L_attn_lr1e5",  "num_layers": 5, "lr": 1e-05, "wd": 0},
        ],
    },
}

# All configs use GENConv + attention pooling (the standardization target)
POOLING = "attention"
GNN_TYPE = "GENConv"
HIDDEN_DIM = 128

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_patient_split():
    splits = {}
    for s in ["train", "val", "test"]:
        df = pd.read_csv(f"{SPLIT_DIR}/{s}_patches_class_perc_0_tp.csv")
        pids = df["patch_path"].str.extract(r"patches_512_fullWSIs_0/(\d+)/", expand=False)
        splits[s] = set(pids.dropna().astype(int).unique())
    return splits


def load_gt(label_map):
    gt = pd.read_excel(GT_FILE).rename(columns={"Patient ID": "patient_id", "Molecular subtype": "mol_subtype"})
    gt["label"] = gt["mol_subtype"].map(label_map)
    gt = gt.dropna(subset=["label"])
    gt["label"] = gt["label"].astype(int)
    return gt


def load_graphs(graph_dir, patient_ids, gt_df, device):
    graphs, labels, pids = [], [], []
    for pid in sorted(patient_ids):
        gpath = os.path.join(graph_dir, f"{pid}_graph.pt")
        if not os.path.exists(gpath): continue
        gt_row = gt_df[gt_df["patient_id"] == pid]
        if len(gt_row) == 0: continue
        label = gt_row["label"].values[0]
        graph = torch.load(gpath, map_location=device, weights_only=False)
        graphs.append(graph)
        labels.append(label)
        pids.append(pid)
    return graphs, labels, pids

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def create_model(n_classes, num_layers):
    from MIL_models import PatchGCN_MeanMax_LSelec
    return PatchGCN_MeanMax_LSelec(
        num_features=512, num_layers=num_layers, hidden_dim=HIDDEN_DIM,
        n_classes=n_classes, pooling=POOLING, gnn_layer_type=GNN_TYPE,
    )


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
    preds, probs, true = [], [], []
    with torch.no_grad():
        for g, l in zip(graphs, labels):
            g = g.to(device)
            Y_prob, Y_hat, _, _ = model(g)
            preds.append(Y_hat.item())
            probs.append(Y_prob.squeeze().cpu().numpy())
            true.append(l)
    y_true, y_pred, y_prob = np.array(true), np.array(preds), np.array(probs)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro") if y_prob.shape[1] > 2 else roc_auc_score(y_true, y_prob[:, 1])
    except ValueError:
        auc = 0.0
    return acc, f1, auc, y_pred, y_prob


def train_with_patience(model, train_g, train_l, val_g, val_l,
                        lr, wd, epochs, patience, device, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_f1, best_state, pat = 0, None, 0
    for ep in range(1, epochs + 1):
        _, _ = train_epoch(model, train_g, train_l, optimizer, device)
        _, val_f1, _, _, _ = evaluate(model, val_g, val_l, device)
        improved = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = deepcopy(model.state_dict())
            pat = 0
            improved = " *"
        else:
            pat += 1
        if ep <= 3 or ep % 10 == 0 or improved or pat >= patience:
            print(f"      {label}ep{ep:>3d} val_f1={val_f1:.3f} pat={pat}/{patience}{improved}", flush=True)
        if pat >= patience:
            print(f"      {label}early stop at epoch {ep}", flush=True)
            break
    return best_state, best_f1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASKS.keys()))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--config-override", type=str, default=None,
                        help="Override configs: 'name:layers:lr:wd' e.g. '5L_attn_lr5e6:5:5e-6:0'")
    args = parser.parse_args()

    task_cfg = TASKS[args.task]
    n_classes = task_cfg["n_classes"]
    knn = task_cfg["knn"]

    if args.config_override:
        parts = args.config_override.split(":")
        configs = [{"name": parts[0], "num_layers": int(parts[1]), "lr": float(parts[2]), "wd": float(parts[3])}]
    else:
        configs = task_cfg["configs"]
    device = torch.device(args.device)

    print(f"Task: {args.task} ({n_classes}-class, k={knn})", flush=True)
    print(f"Architecture: {GNN_TYPE} + {POOLING} pooling", flush=True)
    print(f"MC-CV: {args.cv_folds}-fold x {args.cv_repeats} repeats", flush=True)
    print(f"Configs: {[c['name'] for c in configs]}", flush=True)

    # Load data
    gt_df = load_gt(task_cfg["label_map"])
    splits = get_patient_split()
    graph_dir = f"{MOLSUB}/data/BCNB/results_graphs_november_23/{task_cfg['graph_subdir']}/graphs_k_{knn}"

    cv_pool_ids = splits["train"] | splits["val"]
    print(f"\nLoading CV pool ({len(cv_pool_ids)} patients)...", flush=True)
    cv_graphs, cv_labels, cv_pids = load_graphs(graph_dir, cv_pool_ids, gt_df, "cpu")
    print(f"  Loaded: {len(cv_graphs)} graphs", flush=True)

    print(f"Loading test set ({len(splits['test'])} patients)...", flush=True)
    test_graphs, test_labels, test_pids = load_graphs(graph_dir, splits["test"], gt_df, "cpu")
    print(f"  Loaded: {len(test_graphs)} graphs", flush=True)

    # Class distribution
    for name, labs in [("CV pool", cv_labels), ("Test", test_labels)]:
        counts = np.bincount(labs, minlength=n_classes)
        print(f"  {name}: {dict(zip(task_cfg['class_names'].values(), counts))}", flush=True)

    if args.dry_run:
        print(f"\nDry run. Would run {len(configs)} configs x {args.cv_folds} folds x {args.cv_repeats} repeats = {len(configs)*args.cv_folds*args.cv_repeats} runs", flush=True)
        return

    from MIL_models import PatchGCN_MeanMax_LSelec

    cv_labels_arr = np.array(cv_labels)
    out_dir = os.path.join(RESULTS, "retrained_models")
    os.makedirs(out_dir, exist_ok=True)

    # ====================================================================
    # PHASE 1: MC-CV
    # ====================================================================
    print(f"\n{'='*70}\n  PHASE 1: Monte Carlo CV\n{'='*70}", flush=True)

    all_cv = []
    for cfg in configs:
        print(f"\n  --- {cfg['name']} ---", flush=True)
        fold_results = []
        for rep in range(args.cv_repeats):
            skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=rep * 42)
            for fi, (tr_idx, va_idx) in enumerate(skf.split(cv_labels_arr, cv_labels_arr)):
                t0 = time.time()
                tr_g = [cv_graphs[i] for i in tr_idx]
                tr_l = [cv_labels[i] for i in tr_idx]
                va_g = [cv_graphs[i] for i in va_idx]
                va_l = [cv_labels[i] for i in va_idx]

                model = create_model(n_classes, cfg["num_layers"]).to(device)
                best_state, best_f1 = train_with_patience(
                    model, tr_g, tr_l, va_g, va_l,
                    cfg["lr"], cfg["wd"], args.epochs, args.patience, device,
                    label=f"R{rep+1}F{fi+1} "
                )
                model.load_state_dict(best_state)
                va_acc, va_f1, va_auc, _, _ = evaluate(model, va_g, va_l, device)
                elapsed = time.time() - t0

                fold_results.append({
                    "config": cfg["name"], "repeat": rep, "fold": fi,
                    "val_acc": va_acc, "val_f1": va_f1, "val_auc": va_auc, "elapsed_s": elapsed,
                })
                print(f"    R{rep+1}F{fi+1}: val_f1={va_f1:.3f} ({elapsed:.0f}s)", flush=True)
                del model, best_state

        df_cfg = pd.DataFrame(fold_results)
        mf1 = df_cfg["val_f1"].mean()
        sf1 = df_cfg["val_f1"].std()
        print(f"  >> {cfg['name']}: F1={mf1:.4f}+/-{sf1:.4f}", flush=True)

        # Save per-config results
        cfg_path = os.path.join(out_dir, f"{args.task}_mccv_{cfg['name']}.csv")
        df_cfg.to_csv(cfg_path, index=False)
        all_cv.extend(fold_results)

    # Select winner
    cv_df = pd.DataFrame(all_cv)
    cv_summary = cv_df.groupby("config").agg(
        mean_f1=("val_f1", "mean"), std_f1=("val_f1", "std"),
        mean_acc=("val_acc", "mean"), std_acc=("val_acc", "std"),
    ).reset_index().sort_values("mean_f1", ascending=False)

    print(f"\n  CV SUMMARY:", flush=True)
    for _, row in cv_summary.iterrows():
        print(f"    {row['config']}: F1={row['mean_f1']:.4f}+/-{row['std_f1']:.4f} Acc={row['mean_acc']:.4f}+/-{row['std_acc']:.4f}", flush=True)

    winner_name = cv_summary.iloc[0]["config"]
    winner_cfg = [c for c in configs if c["name"] == winner_name][0]
    print(f"\n  WINNER: {winner_name}", flush=True)

    # ====================================================================
    # PHASE 2: Final model
    # ====================================================================
    print(f"\n{'='*70}\n  PHASE 2: Final model training\n{'='*70}", flush=True)

    train_g, train_l, train_p = load_graphs(graph_dir, splits["train"], gt_df, "cpu")
    val_g, val_l, val_p = load_graphs(graph_dir, splits["val"], gt_df, "cpu")
    print(f"  Train: {len(train_g)}, Val: {len(val_g)}", flush=True)

    final_model = create_model(n_classes, winner_cfg["num_layers"]).to(device)
    best_state, best_f1 = train_with_patience(
        final_model, train_g, train_l, val_g, val_l,
        winner_cfg["lr"], winner_cfg["wd"], args.epochs, args.patience, device,
        label="FINAL "
    )
    final_model.load_state_dict(best_state)

    # ====================================================================
    # PHASE 3: Test + Save
    # ====================================================================
    print(f"\n{'='*70}\n  PHASE 3: Test + Save\n{'='*70}", flush=True)

    test_acc, test_f1, test_auc, test_preds, test_probs = evaluate(
        final_model, test_graphs, test_labels, device
    )
    print(f"  Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}", flush=True)

    # Save model
    model_name = f"{args.task}_GENConv_5L_attn_{winner_name}_final.pth"
    model_path = os.path.join(out_dir, model_name)
    torch.save(final_model, model_path)
    print(f"  Model: {model_path}", flush=True)

    # Save predictions (replaces existing Protocol 1 predictions)
    pred_rows = []
    for pid, yt, yp, yprob in zip(test_pids, test_labels, test_preds, test_probs):
        row = {"patient_id": pid, "y_true": yt, "y_pred": int(yp)}
        for c in range(n_classes):
            row[f"y_prob_{c}"] = float(yprob[c])
        pred_rows.append(row)
    pred_path = os.path.join(RESULTS, "predictions", f"BCNB_{args.task}_CA_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"  Predictions: {pred_path}", flush=True)

    # Save provenance
    provenance = {
        "task": args.task,
        "date": "2026-04-29",
        "description": f"{args.task} GCN retrained with GENConv + attention pooling for architectural consistency and interpretability",
        "methodology": f"MC-CV ({args.cv_folds}-fold x {args.cv_repeats} repeats) for config selection, final model on 630 train / 210 val / 218 test",
        "architecture": {
            "gnn_layer_type": GNN_TYPE, "pooling": POOLING,
            "num_layers": winner_cfg["num_layers"], "hidden_dim": HIDDEN_DIM,
            "lr": winner_cfg["lr"], "wd": winner_cfg["wd"],
            "knn": knn, "n_classes": n_classes,
        },
        "cv_winner": winner_name,
        "cv_results": {
            "mean_f1": float(cv_summary.iloc[0]["mean_f1"]),
            "std_f1": float(cv_summary.iloc[0]["std_f1"]),
            "mean_acc": float(cv_summary.iloc[0]["mean_acc"]),
            "std_acc": float(cv_summary.iloc[0]["std_acc"]),
        },
        "test_results": {
            "accuracy": float(test_acc), "f1": float(test_f1), "auc": float(test_auc),
            "n_patients": len(test_pids),
        },
        "files": {
            "model": model_path,
            "predictions": pred_path,
            "cv_results": os.path.join(out_dir, f"{args.task}_mccv_{winner_name}.csv"),
        },
        "reproduce": f"python scripts/retrain_gcn_mccv_generic.py --task {args.task} --device cpu",
    }
    prov_path = os.path.join(out_dir, f"{args.task}_retrain_provenance.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance: {prov_path}", flush=True)
    print(f"\n  DONE. {args.task}: {winner_name} -> Acc={test_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
