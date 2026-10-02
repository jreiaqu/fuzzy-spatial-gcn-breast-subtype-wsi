"""
retrain_gcn_mccv_generic.py -- Train the context-aware GCN on BCNB (Protocol 1)
with fuzzy edge weighting.

Protocol (same as the reference work):
  Phase 1: MC-CV (5 folds x 3 repeats) on the 840 train+val patients to select
           the configuration with the best mean weighted F1.
  Phase 2: train every configuration on the 630 train patients, with the 210
           val patients for early stopping.
  Phase 3: evaluate on the 218 test patients. Predictions and test metrics are
           saved for every configuration; the .pth only for the MC-CV winner.

--fuzzy-option selects the graph construction:
  1  graph rebuilt with the combined similarity (generate_fuzzy_graphs.py);
     edge weights precomputed in the .pt. Graphs read from
     data/BCNB/results_graphs_november_23_fuzzy/<--fuzzy-subdir>/<task>/k_19/
     edge mode: fuzzy_combined
  2  inherited spatial k-NN graph, weights computed in the forward pass.
     Graphs read from data/BCNB/results_graphs_november_23_morph/
     edge modes: spatial | morphological | spatial_fuzzy |
                 morphological_fuzzy | combined_fuzzy

The configurations evaluated for each edge mode are built by build_configs()
from the bandwidth tables below (retained-weight calibration). lr / wd are
fixed per task.

--mode direct skips Phase 1 and trains a single configuration (the only one
of the edge mode, or the one given with --config-override).

Outputs in results/option_<1|2>/...:
  {task}_mccv_{cfg}.csv                    Phase 1 folds, per configuration
  {task}_all_configs_test_results.json     test metrics of every configuration
  predictions/BCNB_{task}_{cfg}_CA_predictions.csv
  {task}_GENConv_5L_attn_{winner}_final.pth  {'state_dict', 'config'}
  {task}_retrain_provenance_{winner}.json

Usage:
    python scripts/training/retrain_gcn_mccv_generic.py --task 3class --edge-mode spatial_fuzzy --device cuda
    python scripts/training/retrain_gcn_mccv_generic.py --task 3class --fuzzy-option 1 \\
        --fuzzy-subdir sigmas_med_0.9 --device cuda
    python scripts/training/retrain_gcn_mccv_generic.py --task 3class --edge-mode combined_fuzzy --dry-run
    python scripts/training/retrain_gcn_mccv_generic.py --task 3class --mode direct \\
        --edge-mode combined_fuzzy --config-override my_cfg:5:2e-05:0:0.0714:0.2973:0.7 --device cuda
"""

import sys, os, argparse, types, importlib, time, json, warnings

# --- Repository path configuration (portable) ---
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
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
# graph_subdir: directory names produced by WSI2Graph (the "LAUMINALB" typo
# is part of the 4-class directory name). lr / wd / num_layers are the
# per-task values of the reference work and are shared by all edge modes.
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
        "num_layers": 5, "lr": 5e-06, "wd": 0,
    },
    "3class": {
        "graph_subdir": "graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
        "knn": 19,
        "n_classes": 3,
        "label_map": {
            "Luminal A": 0, "Luminal B": 0,
            "HER2(+)": 1, "HER2 enriched": 1,
            "Triple negative": 2, "TNBC": 2,
        },
        "class_names": {0: "Luminals", 1: "HER2(+)", 2: "TNBC"},
        "num_layers": 5, "lr": 2e-05, "wd": 0,
    },
    "4class": {
        "graph_subdir": "graphs_PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
        "knn": 19,
        "n_classes": 4,
        "label_map": {
            "Luminal A": 0, "Luminal B": 1,
            "HER2(+)": 2, "HER2 enriched": 2,
            "Triple negative": 3, "TNBC": 3,
        },
        "class_names": {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3: "TNBC"},
        "num_layers": 5, "lr": 1e-05, "wd": 0,
    },
}

# Option 2 bandwidths, keyed by the weight r kept by the median edge
# (exp(-median^2 / 2 sigma^2) = r; "med" means sigma = median, r ~ 0.607).
# Computed with calculate_sigma.py on the _morph graphs. The spatial distance
# does not depend on the task; the morphological one does (task-specific
# feature extractor).
SIGMA_SPATIAL = {"0.1": 0.0333, "0.3": 0.0460, "0.5": 0.0607, "med": 0.0714, "0.7": 0.0846, "0.9": 0.1556}
SIGMA_MORPHOLOGICAL = {
    "2class": {"0.1": 0.1579, "0.3": 0.2184, "0.5": 0.2878, "med": 0.3389, "0.7": 0.4012, "0.9": 0.7382},
    "3class": {"0.1": 0.1386, "0.3": 0.1916, "0.5": 0.2525, "med": 0.2973, "0.7": 0.3520, "0.9": 0.6477},
    "4class": {"0.1": 0.1363, "0.3": 0.1884, "0.5": 0.2484, "med": 0.2924, "0.7": 0.3462, "0.9": 0.6370},
}
ALPHAS = [0.1, 0.3, 0.5, 0.7, 0.9]

# combined_fuzzy is evaluated with two (r_spatial, r_morphological) pairs:
# the median calibration and the MC-CV winners of spatial_fuzzy and
# morphological_fuzzy for each task.
COMBINED_SIGMA_PAIRS = {
    "2class": {"med": ("med", "med"), "win": ("0.7", "0.1")},
    "3class": {"med": ("med", "med"), "win": ("0.3", "0.9")},
    "4class": {"med": ("med", "med"), "win": ("med", "0.3")},
}


def build_configs(task, edge_mode):
    """Configurations evaluated in Phase 1 for a task and edge mode."""
    base = {k: TASKS[task][k] for k in ("num_layers", "lr", "wd")}
    if edge_mode in ("spatial", "morphological", "fuzzy_combined"):
        return [{"name": edge_mode, **base}]
    if edge_mode == "spatial_fuzzy":
        return [{"name": f"spatial_fuzzy_r{r}", **base, "sigma_spatial": sig}
                for r, sig in SIGMA_SPATIAL.items()]
    if edge_mode == "morphological_fuzzy":
        return [{"name": f"morphological_fuzzy_r{r}", **base, "sigma_morphological": sig}
                for r, sig in SIGMA_MORPHOLOGICAL[task].items()]
    if edge_mode == "combined_fuzzy":
        configs = []
        for pair, (r_s, r_m) in COMBINED_SIGMA_PAIRS[task].items():
            for alpha in ALPHAS:
                configs.append({
                    "name": f"combined_fuzzy_{pair}_a{alpha}", **base,
                    "sigma_spatial": SIGMA_SPATIAL[r_s],
                    "sigma_morphological": SIGMA_MORPHOLOGICAL[task][r_m],
                    "alpha": alpha,
                })
        return configs
    raise ValueError(f"Unknown edge_mode: '{edge_mode}'")


# All configs use GENConv + attention pooling (the standardization target)
POOLING = "attention"
GNN_TYPE = "GENConv"
HIDDEN_DIM = 128

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_patient_split():
    """Official BCNB split: {"train" | "val" | "test": set of patient ids}."""
    splits = {}
    for s in ["train", "val", "test"]:
        df = pd.read_csv(f"{SPLIT_DIR}/{s}_patches_class_perc_0_tp.csv")
        pids = df["patch_path"].str.extract(r"patches_512_fullWSIs_0/(\d+)/", expand=False)
        splits[s] = set(pids.dropna().astype(int).unique())
    return splits


def load_gt(label_map):
    """BCNB ground truth with an integer "label" column mapped with label_map."""
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

EDGE_MODES_OPT1 = ["fuzzy_combined"]
EDGE_MODES_OPT2 = ["spatial", "morphological", "spatial_fuzzy", "morphological_fuzzy", "combined_fuzzy"]
EDGE_MODES = EDGE_MODES_OPT1 + EDGE_MODES_OPT2


def create_model(n_classes, num_layers, edge_mode="spatial",
                 sigma_spatial=0.5, sigma_morphological=0.5, alpha=0.5):
    from MIL_models import PatchGCN_MeanMax_LSelec
    return PatchGCN_MeanMax_LSelec(
        num_features=512, num_layers=num_layers, hidden_dim=HIDDEN_DIM,
        n_classes=n_classes, pooling=POOLING, gnn_layer_type=GNN_TYPE,
        edge_mode=edge_mode, sigma_spatial=sigma_spatial,
        sigma_morphological=sigma_morphological, alpha=alpha,
        use_edge_features=True,
    )


def train_epoch(model, graphs, labels, optimizer, device, n_classes, class_weights=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for idx in np.random.permutation(len(graphs)):
        graph = graphs[idx].to(device)
        label = torch.tensor([labels[idx]], dtype=torch.long, device=device)
        optimizer.zero_grad()
        _, Y_hat, logits, _ = model(graph)
        loss = F.cross_entropy(logits, label, weight=class_weights)
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
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            auc = roc_auc_score(y_true, y_prob if y_prob.ndim == 1 else y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except (ValueError, IndexError):
        auc = 0.0
    return acc, f1, auc, y_pred, y_prob


def train_with_patience(model, train_g, train_l, val_g, val_l,
                        lr, wd, epochs, patience, device, n_classes, label=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    counts = np.bincount(train_l, minlength=n_classes)
    class_weights = torch.tensor(np.where(counts > 0, 1.0 / counts, 0.0), dtype=torch.float, device=device)
    best_f1, best_state, pat = 0.0, deepcopy(model.state_dict()), 0
    for ep in range(1, epochs + 1):
        _, _ = train_epoch(model, train_g, train_l, optimizer, device, n_classes, class_weights)
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
    parser.add_argument("--mode", default="mccv", choices=["mccv", "direct"],
                        help="'mccv' (default): Phase 1 MC-CV, then Phase 2+3. "
                             "'direct': skip Phase 1 and train a single configuration "
                             "(the only one of the edge mode, or --config-override).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--fuzzy-option", type=int, default=2, choices=[1, 2],
                        help="1 = rebuilt fuzzy graphs (precomputed weights), "
                             "2 = inherited spatial k-NN graph with weights computed "
                             "in the forward pass (default)")
    parser.add_argument("--fuzzy-subdir", type=str, default=None,
                        help="Option 1 only (required): sigma combination inside "
                             "data/BCNB/results_graphs_november_23_fuzzy/, e.g. sigmas_med_0.9")
    parser.add_argument("--edge-mode", type=str, default=None,
                        help="Edge weighting mode. Option 1 default: fuzzy_combined. "
                             "Option 2 default: spatial. "
                             f"Option 1 choices: {EDGE_MODES_OPT1}. "
                             f"Option 2 choices: {EDGE_MODES_OPT2}.")
    parser.add_argument("--config-override", type=str, default=None,
                        help="Override configs: 'name:layers:lr:wd[:sigma_s[:sigma_m[:alpha]]]'")
    parser.add_argument("--no-save-model", action="store_true",
                        help="Skip saving the winner .pth to disk. Useful in large sweeps "
                             "(e.g. run_option1_sweep.py) where only metrics matter and "
                             "a .pth per run is not needed (about 8 MB each). "
                             "Re-run the winner combo without this flag to get the final .pth.")
    args = parser.parse_args()

    # Set and validate edge_mode per option
    valid_modes = EDGE_MODES_OPT1 if args.fuzzy_option == 1 else EDGE_MODES_OPT2
    if args.edge_mode is None:
        args.edge_mode = "fuzzy_combined" if args.fuzzy_option == 1 else "spatial"
    if args.edge_mode not in valid_modes:
        parser.error(f"--edge-mode '{args.edge_mode}' is not valid for --fuzzy-option {args.fuzzy_option}. "
                     f"Valid choices: {valid_modes}")
    if args.fuzzy_option == 1 and not args.fuzzy_subdir:
        parser.error("--fuzzy-option 1 requires --fuzzy-subdir (e.g. sigmas_med_0.9)")

    task_cfg = TASKS[args.task]
    n_classes = task_cfg["n_classes"]
    knn = task_cfg["knn"]

    if args.config_override:
        parts = args.config_override.split(":")
        configs = [{"name": parts[0], "num_layers": int(parts[1]), "lr": float(parts[2]), "wd": float(parts[3]),
                    "sigma_spatial":      float(parts[4]) if len(parts) > 4 else 0.5,
                    "sigma_morphological": float(parts[5]) if len(parts) > 5 else 0.5,
                    "alpha":               float(parts[6]) if len(parts) > 6 else 0.5}]
    else:
        configs = build_configs(args.task, args.edge_mode)

    # Direct mode: resolve the single config to use (no CV needed)
    if args.mode == "direct":
        if len(configs) > 1:
            parser.error(
                f"--mode direct needs a single configuration, but edge mode "
                f"'{args.edge_mode}' has {len(configs)}. Use --config-override."
            )
        direct_cfg = configs[0]

    device = torch.device(args.device)

    # graph_dir depends on fuzzy option
    if args.fuzzy_option == 1:
        graph_dir = os.path.join(
            MOLSUB, "data", "BCNB", "results_graphs_november_23_fuzzy",
            args.fuzzy_subdir, args.task, f"k_{knn}",
        )
    else:
        graph_dir = os.path.join(
            MOLSUB, "data", "BCNB", "results_graphs_november_23_morph",
            task_cfg["graph_subdir"], f"graphs_k_{knn}",
        )

    if not os.path.isdir(graph_dir):
        sys.exit(f"ERROR: graph_dir not found: {graph_dir}")

    # Option 1: the bandwidths are baked into the graphs; record the real values
    # (from generate_fuzzy_graphs.py's provenance) in the saved config.
    if args.fuzzy_option == 1:
        graph_cfg_path = os.path.join(graph_dir, "0_fuzzy_graph_config.json")
        if os.path.exists(graph_cfg_path):
            with open(graph_cfg_path) as f:
                graph_cfg = json.load(f)
            for c in configs:
                c["sigma_spatial"] = graph_cfg["sigma_spatial"]
                c["sigma_morphological"] = graph_cfg["sigma_morpho"]

    print(f"Task: {args.task} ({n_classes}-class, k={knn})", flush=True)
    print(f"Architecture: {GNN_TYPE} + {POOLING} pooling", flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Fuzzy option: {args.fuzzy_option}  |  Edge mode: {args.edge_mode}", flush=True)
    print(f"Graph dir: {graph_dir}", flush=True)
    if args.mode == "mccv":
        print(f"MC-CV: {args.cv_folds}-fold x {args.cv_repeats} repeats", flush=True)
        print(f"Configs: {[c['name'] for c in configs]}", flush=True)
    else:
        print(f"Direct config: {direct_cfg['name']}", flush=True)

    # Load data
    gt_df = load_gt(task_cfg["label_map"])
    splits = get_patient_split()

    print(f"Loading test set ({len(splits['test'])} patients)...", flush=True)
    test_graphs, test_labels, test_pids = load_graphs(graph_dir, splits["test"], gt_df, "cpu")
    print(f"  Loaded: {len(test_graphs)} graphs", flush=True)
    test_counts = np.bincount(test_labels, minlength=n_classes)
    print(f"  Test: {dict(zip(task_cfg['class_names'].values(), test_counts))}", flush=True)

    if args.mode == "mccv":
        cv_pool_ids = splits["train"] | splits["val"]
        print(f"\nLoading CV pool ({len(cv_pool_ids)} patients)...", flush=True)
        cv_graphs, cv_labels, _ = load_graphs(graph_dir, cv_pool_ids, gt_df, "cpu")
        print(f"  Loaded: {len(cv_graphs)} graphs", flush=True)
        cv_counts = np.bincount(cv_labels, minlength=n_classes)
        print(f"  CV pool: {dict(zip(task_cfg['class_names'].values(), cv_counts))}", flush=True)

    if args.dry_run:
        if args.mode == "mccv":
            print(f"\nDry run. Would run {len(configs)} configs x {args.cv_folds} folds x {args.cv_repeats} repeats = "
                  f"{len(configs)*args.cv_folds*args.cv_repeats} runs, then Phase 2+3 with the winner.", flush=True)
        else:
            print(f"\nDry run. Would skip MCCV and train final model directly with config "
                  f"'{direct_cfg['name']}' on train/val, then evaluate on test.", flush=True)
        return

    results_base = os.path.join(MOLSUB, "results")
    if args.fuzzy_option == 1:
        out_dir = os.path.join(results_base, "option_1", args.fuzzy_subdir, args.task)
    else:
        out_dir = os.path.join(results_base, f"option_{args.fuzzy_option}", args.edge_mode, args.task)
    os.makedirs(out_dir, exist_ok=True)

    # ====================================================================
    # PHASE 1: MC-CV  (skipped in --mode direct)
    # ====================================================================
    if args.mode == "mccv":
        print(f"\n{'='*70}\n  PHASE 1: Monte Carlo CV\n{'='*70}", flush=True)

        cv_labels_arr = np.array(cv_labels)
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

                    model = create_model(
                        n_classes, cfg["num_layers"], args.edge_mode,
                        cfg.get("sigma_spatial", 0.5), cfg.get("sigma_morphological", 0.5), cfg.get("alpha", 0.5),
                    ).to(device)
                    best_state, best_f1 = train_with_patience(
                        model, tr_g, tr_l, va_g, va_l,
                        cfg["lr"], cfg["wd"], args.epochs, args.patience, device, n_classes,
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

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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

    else:
        # Direct mode: no Phase 1
        winner_cfg = direct_cfg
        winner_name = direct_cfg["name"]
        cv_summary = None
        print(f"\n{'='*70}\n  PHASE 1: SKIPPED (--mode direct, using config '{winner_name}')\n{'='*70}", flush=True)

    # ====================================================================
    # PHASE 2 + 3: Final training + test evaluation for all configs
    # (pth and predictions CSV saved only for the winner)
    # ====================================================================
    print(f"\n{'='*70}\n  PHASE 2+3: Final training + test (all {len(configs)} configs)\n{'='*70}", flush=True)

    train_g, train_l, _ = load_graphs(graph_dir, splits["train"], gt_df, "cpu")
    val_g, val_l, _ = load_graphs(graph_dir, splits["val"], gt_df, "cpu")
    print(f"  Train: {len(train_g)}, Val: {len(val_g)}", flush=True)

    all_test_results = []
    winner_test_acc = winner_test_f1 = winner_test_auc = None
    model_path = pred_path = None

    for cfg in configs:
        is_winner = (cfg["name"] == winner_name)
        print(f"\n  --- {cfg['name']}{'  <<< WINNER' if is_winner else ''} ---", flush=True)

        model = create_model(
            n_classes, cfg["num_layers"], args.edge_mode,
            cfg.get("sigma_spatial", 0.5), cfg.get("sigma_morphological", 0.5), cfg.get("alpha", 0.5),
        ).to(device)
        best_state, _ = train_with_patience(
            model, train_g, train_l, val_g, val_l,
            cfg["lr"], cfg["wd"], args.epochs, args.patience, device, n_classes,
            label=f"{'WINNER' if is_winner else cfg['name']} ",
        )
        model.load_state_dict(best_state)

        test_acc, test_f1, test_auc, test_preds, test_probs = evaluate(
            model, test_graphs, test_labels, device
        )
        print(f"  Test Acc: {test_acc:.4f}  F1: {test_f1:.4f}  AUC: {test_auc:.4f}", flush=True)

        cfg_entry = {
            "config": cfg["name"],
            "is_winner": is_winner,
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc),
            "n_patients": len(test_pids),
        }
        if args.mode == "mccv":
            cv_row = cv_summary[cv_summary["config"] == cfg["name"]]
            if len(cv_row):
                cfg_entry["cv_mean_f1"] = float(cv_row.iloc[0]["mean_f1"])
                cfg_entry["cv_std_f1"] = float(cv_row.iloc[0]["std_f1"])
        all_test_results.append(cfg_entry)

        if is_winner:
            winner_test_acc, winner_test_f1, winner_test_auc = test_acc, test_f1, test_auc

            if not args.no_save_model:
                model_name = f"{args.task}_GENConv_{winner_cfg['num_layers']}L_attn_{winner_name}_final.pth"
                model_path = os.path.join(out_dir, model_name)
                torch.save({
                    "state_dict": model.state_dict(),
                    "config": {
                        "num_layers": winner_cfg["num_layers"],
                        "n_classes": n_classes,
                        "edge_mode": args.edge_mode,
                        # None = not used by this edge mode
                        "sigma_spatial": winner_cfg.get("sigma_spatial"),
                        "sigma_morphological": winner_cfg.get("sigma_morphological"),
                        "alpha": winner_cfg.get("alpha"),
                        **({"fuzzy_subdir": args.fuzzy_subdir} if args.fuzzy_option == 1 else {}),
                    },
                }, model_path)
                print(f"  Model: {model_path}", flush=True)
            else:
                print(f"  Model: skipped (--no-save-model)", flush=True)

        pred_rows = []
        for pid, yt, yp, yprob in zip(test_pids, test_labels, test_preds, test_probs):
            row = {"patient_id": pid, "y_true": yt, "y_pred": int(yp)}
            for c in range(n_classes):
                row[f"y_prob_{c}"] = float(yprob[c])
            pred_rows.append(row)
        cfg_pred_path = os.path.join(out_dir, "predictions", f"BCNB_{args.task}_{cfg['name']}_CA_predictions.csv")
        os.makedirs(os.path.dirname(cfg_pred_path), exist_ok=True)
        pd.DataFrame(pred_rows).to_csv(cfg_pred_path, index=False)
        print(f"  Predictions: {cfg_pred_path}", flush=True)
        if is_winner:
            pred_path = cfg_pred_path

        del model, best_state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save all-configs test results JSON (no overwrite: bump to (2), (3), ... if needed)
    base_path = os.path.join(out_dir, f"{args.task}_all_configs_test_results.json")
    all_cfg_path = base_path
    if os.path.exists(all_cfg_path):
        n = 2
        stem = base_path[:-5]  # strip .json
        while os.path.exists(f"{stem} ({n}).json"):
            n += 1
        all_cfg_path = f"{stem} ({n}).json"

    with open(all_cfg_path, "w") as f:
        json.dump({
            "task": args.task,
            "edge_mode": args.edge_mode,
            "winner": winner_name,
            "configs": all_test_results,
        }, f, indent=2)
    print(f"  All-config results: {all_cfg_path}", flush=True)

    # Save provenance for winner
    if args.mode == "mccv":
        methodology = (f"MC-CV ({args.cv_folds}-fold x {args.cv_repeats} repeats) for config selection, "
                       f"final model on 630 train / 210 val / 218 test")
        cv_provenance = {
            "cv_winner": winner_name,
            "cv_results": {
                "mean_f1": float(cv_summary.iloc[0]["mean_f1"]),
                "std_f1": float(cv_summary.iloc[0]["std_f1"]),
                "mean_acc": float(cv_summary.iloc[0]["mean_acc"]),
                "std_acc": float(cv_summary.iloc[0]["std_acc"]),
            },
            "cv_results_file": os.path.join(out_dir, f"{args.task}_mccv_{winner_name}.csv"),
        }
        reproduce_cmd = (
            f"python scripts/training/retrain_gcn_mccv_generic.py "
            f"--task {args.task} --fuzzy-option {args.fuzzy_option} "
            f"--edge-mode {args.edge_mode}"
            + (f" --fuzzy-subdir {args.fuzzy_subdir}" if args.fuzzy_option == 1 else "")
            + f" --device {args.device}"
        )
    else:
        methodology = "Direct training (no MCCV): final model on 630 train / 210 val / 218 test"
        cv_provenance = {
            "cv_winner": None,
            "cv_results": None,
            "note": "MCCV skipped (--mode direct)",
        }
        cfg_override_str = (
            f"{winner_cfg['name']}:{winner_cfg['num_layers']}:{winner_cfg['lr']}:{winner_cfg['wd']}"
            f":{winner_cfg.get('sigma_spatial', 0.5)}:{winner_cfg.get('sigma_morphological', 0.5)}"
            f":{winner_cfg.get('alpha', 0.5)}"
        )
        reproduce_cmd = (
            f"python scripts/training/retrain_gcn_mccv_generic.py "
            f"--task {args.task} --mode direct --fuzzy-option {args.fuzzy_option} "
            f"--edge-mode {args.edge_mode} --config-override {cfg_override_str}"
            + (f" --fuzzy-subdir {args.fuzzy_subdir}" if args.fuzzy_option == 1 else "")
            + f" --device {args.device}"
        )

    provenance = {
        "task": args.task,
        "date": time.strftime("%Y-%m-%d"),
        "mode": args.mode,
        "fuzzy_option": args.fuzzy_option,
        "graph_dir": graph_dir,
        "description": f"{args.task} GCN retrained with GENConv + attention pooling (fuzzy option {args.fuzzy_option})",
        "methodology": methodology,
        "architecture": {
            "gnn_layer_type": GNN_TYPE, "pooling": POOLING,
            "num_layers": winner_cfg["num_layers"], "hidden_dim": HIDDEN_DIM,
            "lr": winner_cfg["lr"], "wd": winner_cfg["wd"],
            "knn": knn, "n_classes": n_classes,
            "edge_mode": args.edge_mode,
            "sigma_spatial": winner_cfg.get("sigma_spatial"),
            "sigma_morphological": winner_cfg.get("sigma_morphological"),
            "alpha": winner_cfg.get("alpha"),
        },
        **cv_provenance,
        "test_results": {
            "accuracy": float(winner_test_acc), "f1": float(winner_test_f1), "auc": float(winner_test_auc),
            "n_patients": len(test_pids),
        },
        "files": {
            "model": model_path,
            "predictions": pred_path,
            "all_configs_test_results": all_cfg_path,
        },
        "reproduce": reproduce_cmd,
    }
    prov_path = os.path.join(out_dir, f"{args.task}_retrain_provenance_{winner_name}.json")
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"  Provenance: {prov_path}", flush=True)
    print(f"\n  DONE. {args.task}: {winner_name} -> Acc={winner_test_acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
