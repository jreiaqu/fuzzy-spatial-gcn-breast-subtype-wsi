"""
conch_classifier_predictions.py -- CMPB-D-25-07046 Revision, Protocol 3

Implements Protocol 3 (SBC/SBC CONCH foundation model): trains CONCH-based
aggregation + classifier from scratch using 3-fold CV with 2 repeats, then saves
per-patient predictions per fold.

Three architectures:
  - baseline: mean pooling + 2-layer classifier (512 -> 256 -> 128 -> n_classes)
  - NCA: feature transform + MILAttention pooling + linear classifier
  - CA: PatchGCN_MeanMax_LSelec with GENConv + DeepGCNLayer residuals +
        gated attention pooling (from MIL_models.py, same as Protocol 1)

Graph data: pre-extracted 512-dim CONCH features stored as PyTorch Geometric
Data objects in .pt files. Converted to dict format for model consumption.

Usage:
    pip install -r requirements.txt  # see repository root
    python conch_classifier_predictions.py --device cpu
    python conch_classifier_predictions.py --architectures baseline --tasks 2class
    python conch_classifier_predictions.py --dry-run
"""

import sys
import os
import re
import argparse
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB_ROOT = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] CODE_DIR = f"{MOLSUB_ROOT}/code"
SBC_CONCH = (
    f"{MOLSUB_ROOT}/data/SBC/results_graphs_january_25/graphs_CONCH"
)
SBC_GT = (
    f"{MOLSUB_ROOT}/data/SBC/ground_truth/CBDC_4_may2024_gt_extended.xlsx"
)
# [REPLACED by _paths.py] RESULTS_DIR = "/Users/kckj099/Documents/CMPB-Review/results"

# Add code dir for model class imports (conch_models.py, MIL_models.py)
# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Task definitions and label mappings
# ---------------------------------------------------------------------------
TASKS = {
    "2class": {
        "name": "OTHERvsTNBC",
        "labels": {
            "Luminal A": 0,
            "Luminal B": 0,
            "HER2(+)": 0,
            "TNBC": 1,
        },
        "class_names": {0: "Other", 1: "TNBC"},
    },
    "3class": {
        "name": "LUMINALSvsHER2vsTNBC",
        "labels": {
            "Luminal A": 0,
            "Luminal B": 0,
            "HER2(+)": 1,
            "TNBC": 2,
        },
        "class_names": {0: "Luminal", 1: "HER2(+)", 2: "TNBC"},
    },
    "4class": {
        "name": "LUMINALAvsLAUMINALBvsHER2vsTNBC",
        "labels": {
            "Luminal A": 0,
            "Luminal B": 1,
            "HER2(+)": 2,
            "TNBC": 3,
        },
        "class_names": {
            0: "Luminal A",
            1: "Luminal B",
            2: "HER2(+)",
            3: "TNBC",
        },
    },
}

# Map from task name to the CONCH graph directory name.
# The 2-class directory has a double PM_PM prefix.
TASK_GRAPH_DIRS = {
    "OTHERvsTNBC": "graphs_PM_PM_OTHERvsTNBC_BB_CONCH_dws_8",
    "LUMINALSvsHER2vsTNBC": "graphs_PM_LUMINALSvsHER2vsTNBC_BB_CONCH_dws_8",
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": (
        "graphs_PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_CONCH_dws_8"
    ),
}

# ---------------------------------------------------------------------------
# Per-architecture, per-task hyperparameters (recovered from MLflow logs)
# See docs/CONCH_CA_GCN_Experiment_Log.md for full provenance.
# Format: (lr, weight_decay)
# ---------------------------------------------------------------------------
CONCH_HP_LOOKUP = {
    "baseline": {
        "2class": (0.001, 0.001),   # CONCH_BugFixed_Fair_Comparison
        "3class": (0.001, 0.001),   # CONCH_BugFixed_Fair_Comparison
        "4class": (0.001, 0.001),   # CONCH_BugFixed_Fair_Comparison
    },
    "NCA": {
        "2class": (0.0001, 0.001),  # CONCH_BugFixed_Fair_Comparison
        "3class": (0.001, 0.0001),  # CONCH_BugFixed_Fair_Comparison
        "4class": (0.001, 0.001),   # CONCH_BugFixed_Fair_Comparison
    },
    "CA": {
        "2class": (5e-6, 1e-5),     # CONCH_BugFixed_Extended_Final
        "3class": (5e-5, 0.0001),   # CONCH_BugFixed_Extended_Final
        "4class": (5e-5, 0.0001),   # CONCH_BugFixed_Extended_Final
    },
}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------


def load_sbc_gt():
    """Load SBC ground truth, filter excluded, return DataFrame.

    Columns used:
      - SUS_number: patient ID
      - Molsub_surr_4clf: molecular subtype label
      - Molsub_surr_7clf: used only for exclusion filter
    """
    df = pd.read_excel(SBC_GT)
    # Filter out excluded patients (per Molsub_surr_7clf, matching original code)
    df = df[df["Molsub_surr_7clf"] != "Excluded"].copy()
    return df


def extract_patient_id(filename):
    """Extract SUS patient ID from CONCH graph filename.
    e.g. 'SUS001-2021-09-24_13.39.24_graph.pt' -> 'SUS001'
    """
    match = re.search(r"(SUS\d+)", filename)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Graph data loading
# ---------------------------------------------------------------------------


def find_conch_graph_dir(task_name, knn):
    """Find the CONCH graph directory for a given task and knn value."""
    dir_name = TASK_GRAPH_DIRS.get(task_name)
    if dir_name is None:
        return None
    graph_dir = os.path.join(SBC_CONCH, dir_name, f"graphs_k_{knn}")
    if os.path.isdir(graph_dir):
        return graph_dir
    return None


def load_graph_data(task_key, knn, device="cpu"):
    """Load all CONCH graphs and labels for a given task.

    Returns:
        graph_data: list of dicts with keys 'graph', 'label', 'patient_id'
        n_classes: number of classes for this task
    """
    task_cfg = TASKS[task_key]
    task_name = task_cfg["name"]
    label_map = task_cfg["labels"]
    n_classes = len(task_cfg["class_names"])

    graph_dir = find_conch_graph_dir(task_name, knn)
    if graph_dir is None:
        raise FileNotFoundError(
            f"Graph directory not found for task={task_name}, knn={knn}"
        )

    gt_df = load_sbc_gt()

    # Build patient_id -> label lookup
    pid_to_label = {}
    for _, row in gt_df.iterrows():
        pid = row["SUS_number"]
        molsub = row["Molsub_surr_4clf"]
        encoded = label_map.get(molsub)
        if encoded is not None:
            pid_to_label[pid] = encoded

    # Load graph files
    graph_files = sorted(
        [f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")]
    )
    graph_data = []
    skipped = 0

    for gfile in graph_files:
        pid = extract_patient_id(gfile)
        if pid is None or pid not in pid_to_label:
            skipped += 1
            continue

        graph_path = os.path.join(graph_dir, gfile)
        graph_obj = torch.load(graph_path, map_location=device, weights_only=False)

        # Convert PyTorch Geometric Data object to dict
        graph_dict = {
            "x": graph_obj.x,
            "edge_index": graph_obj.edge_index,
        }

        graph_data.append(
            {
                "graph": graph_dict,
                "label": pid_to_label[pid],
                "patient_id": pid,
            }
        )

    print(
        f"  Loaded {len(graph_data)} graphs for {task_key} "
        f"(skipped {skipped} without matching GT)"
    )
    return graph_data, n_classes


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------


class _PatchGCNWrapper(nn.Module):
    """Wraps PatchGCN_MeanMax_LSelec to match the (logits, features) return
    interface expected by our training loop and evaluation functions.

    PatchGCN_MeanMax_LSelec.forward() returns (Y_prob, Y_hat, logits, h)
    where logits is [1, n_classes]. We return (logits_squeezed, h) where
    logits_squeezed is [n_classes] (1D).
    """

    def __init__(self, **kwargs):
        super().__init__()
        from MIL_models import PatchGCN_MeanMax_LSelec

        self.model = PatchGCN_MeanMax_LSelec(**kwargs)

    def forward(self, graph):
        Y_prob, Y_hat, logits, h = self.model(graph)
        return logits.squeeze(0), h


def create_model(architecture, n_classes, input_dim=512):
    """Create a CONCH model instance.

    Args:
        architecture: 'baseline', 'NCA', or 'CA'
        n_classes: number of output classes
        input_dim: feature dimension (512 for CONCH)

    Returns:
        nn.Module instance
    """
    from conch_models import CONCHBaselineModel, CONCHAttentionModel

    arch_lower = architecture.lower()
    if arch_lower == "baseline":
        return CONCHBaselineModel(input_dim=input_dim, n_classes=n_classes)
    elif arch_lower == "nca":
        return CONCHAttentionModel(
            input_dim=input_dim, hidden_dim=128, n_classes=n_classes
        )
    elif arch_lower == "ca":
        # PatchGCN_MeanMax_LSelec: the same GCN architecture used in
        # Protocol 1 (BCNB within-domain) and the paper's CONCH+GCN results.
        # Uses DeepGCNLayer residuals, layer concatenation, and gated
        # attention pooling. Source: conch_experiments_sophisticated.py.
        return _PatchGCNWrapper(
            num_features=input_dim,
            hidden_dim=128,
            n_classes=n_classes,
            num_layers=4,
            gnn_layer_type="GENConv",
            pooling="attention",
            include_edge_features=False,
            dropout=0.25,
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


# ---------------------------------------------------------------------------
# Training loop (no MLflow dependency)
# ---------------------------------------------------------------------------


def compute_class_weights(labels):
    """Compute inverse-frequency class weights for imbalanced classes."""
    labels_np = np.array(labels)
    classes = np.unique(labels_np)
    n_samples = len(labels_np)
    weights = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    for c in classes:
        count = np.sum(labels_np == c)
        weights[c] = n_samples / (len(classes) * count)
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model,
    train_graphs,
    train_labels,
    val_graphs,
    val_labels,
    epochs=150,
    lr=5e-5,
    weight_decay=1e-4,
    device="cpu",
    patience=20,
    val_frequency=10,
):
    """Train a CONCH model with early stopping. No MLflow dependency.

    Processes one graph at a time (not batched), matching the original
    training loop from conch_training.py.

    Args:
        model: nn.Module to train
        train_graphs: list of graph dicts with keys 'x', 'edge_index'
        train_labels: list or tensor of integer labels
        val_graphs: list of graph dicts
        val_labels: list or tensor of integer labels
        epochs: max training epochs
        lr: learning rate
        weight_decay: optimizer weight decay
        device: 'cpu' or 'cuda'
        patience: early stopping patience (number of validation checks)
        val_frequency: validate every N epochs

    Returns:
        model: best model (state loaded)
        history: dict with train_loss, val_loss, val_accuracy, val_f1 lists
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Weighted cross-entropy for class imbalance
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_labels_t = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_labels_t = torch.tensor(val_labels, dtype=torch.long).to(device)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    best_epoch = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    for epoch in range(epochs):
        # -- Training phase --
        model.train()
        epoch_loss = 0.0

        # Shuffle training order each epoch
        perm = np.random.permutation(len(train_graphs))
        for idx in perm:
            graph_dict = train_graphs[idx]
            label = train_labels_t[idx]

            graph_on_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in graph_dict.items()
            }

            optimizer.zero_grad()
            logits, _ = model(graph_on_device)
            loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_graphs)
        history["train_loss"].append(avg_train_loss)

        # -- Progress logging every 25 epochs --
        if (epoch + 1) % 25 == 0 or epoch == 0:
            latest_val_acc = history["val_accuracy"][-1] if history["val_accuracy"] else 0.0
            latest_val_f1 = history["val_f1"][-1] if history["val_f1"] else 0.0
            print(
                f"      Epoch {epoch + 1:>3}/{epochs}  "
                f"train_loss={avg_train_loss:.4f}  "
                f"val_acc={latest_val_acc:.4f}  "
                f"val_f1={latest_val_f1:.4f}  "
                f"patience={patience_counter}/{patience}",
                flush=True,
            )

        # -- Validation phase (every val_frequency epochs) --
        if (epoch + 1) % val_frequency == 0:
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_true = val_labels_t.cpu().numpy()

            with torch.no_grad():
                for i, graph_dict in enumerate(val_graphs):
                    label = val_labels_t[i]
                    graph_on_device = {
                        k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in graph_dict.items()
                    }
                    logits, _ = model(graph_on_device)
                    loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))
                    val_loss += loss.item()
                    val_preds.append(torch.argmax(logits).cpu().item())

            avg_val_loss = val_loss / len(val_graphs)
            val_acc = accuracy_score(val_true, val_preds)
            val_f1 = f1_score(val_true, val_preds, average="weighted", zero_division=0)

            history["val_loss"].append(avg_val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_f1"].append(val_f1)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = deepcopy(model.state_dict())
                best_epoch = epoch
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"      Early stop at epoch {epoch + 1}, "
                    f"best epoch {best_epoch + 1}",
                    flush=True,
                )
                break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, history


# ---------------------------------------------------------------------------
# Evaluation (per-sample predictions + probabilities)
# ---------------------------------------------------------------------------


def evaluate_model(model, graphs, labels, device="cpu"):
    """Evaluate model and return per-sample predictions and probabilities.

    Args:
        model: trained nn.Module
        graphs: list of graph dicts
        labels: list of integer labels
        device: 'cpu' or 'cuda'

    Returns:
        predictions: np.array of predicted class indices
        probabilities: np.array of shape [N, n_classes]
        metrics: dict with accuracy, f1, precision, recall, auc
    """
    model.eval()
    model = model.to(device)
    labels_np = np.array(labels)

    predictions = []
    probabilities = []

    with torch.no_grad():
        for graph_dict in graphs:
            graph_on_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in graph_dict.items()
            }
            logits, _ = model(graph_on_device)
            pred = torch.argmax(logits).cpu().item()
            prob = F.softmax(logits, dim=0).cpu().numpy()
            predictions.append(pred)
            probabilities.append(prob)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    metrics = {
        "accuracy": accuracy_score(labels_np, predictions),
        "f1": f1_score(
            labels_np, predictions, average="weighted", zero_division=0
        ),
        "precision": precision_score(
            labels_np, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            labels_np, predictions, average="weighted", zero_division=0
        ),
    }

    # AUC calculation
    n_classes = len(np.unique(labels_np))
    try:
        if n_classes == 2:
            metrics["auc"] = roc_auc_score(labels_np, probabilities[:, 1])
        elif n_classes > 2:
            unique_labels = np.unique(labels_np)
            if probabilities.shape[1] > len(unique_labels):
                prob_subset = probabilities[:, unique_labels]
                metrics["auc"] = roc_auc_score(
                    labels_np, prob_subset, multi_class="ovr", average="macro"
                )
            else:
                metrics["auc"] = roc_auc_score(
                    labels_np,
                    probabilities,
                    multi_class="ovr",
                    average="macro",
                )
        else:
            metrics["auc"] = float("nan")
    except Exception as e:
        print(f"    AUC calculation failed: {e}")
        metrics["auc"] = float("nan")

    return predictions, probabilities, metrics


# ---------------------------------------------------------------------------
# CV splits: 3-fold with 2 repeats
# ---------------------------------------------------------------------------


def create_cv_splits(labels, n_folds=3, n_repeats=2, random_state=42):
    """Create stratified k-fold CV splits with repeats.

    For each fold, uses the two non-test folds as train+val, splitting
    val as approximately 20% of the train portion (for early stopping).

    Args:
        labels: np.array of integer labels
        n_folds: number of CV folds (default 3)
        n_repeats: number of repeats (default 2)
        random_state: base random seed

    Returns:
        list of dicts with keys: repeat, fold, train, val, test
        (each value is a list of integer indices)
    """
    all_splits = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state + rep,
        )

        for fold_idx, (train_val_indices, test_indices) in enumerate(
            skf.split(np.zeros(len(labels)), labels)
        ):
            # Split train_val into train and val (approx 80/20)
            train_val_labels = labels[train_val_indices]
            inner_skf = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=random_state + rep + fold_idx,
            )
            # Take the first split: 4/5 train, 1/5 val
            for inner_train_idx, inner_val_idx in inner_skf.split(
                np.zeros(len(train_val_labels)), train_val_labels
            ):
                train_indices = train_val_indices[inner_train_idx]
                val_indices = train_val_indices[inner_val_idx]
                break  # Only take first inner split

            all_splits.append(
                {
                    "repeat": rep,
                    "fold": fold_idx,
                    "train": train_indices.tolist(),
                    "val": val_indices.tolist(),
                    "test": test_indices.tolist(),
                }
            )

    return all_splits


# ---------------------------------------------------------------------------
# Run one fold: train + evaluate + collect predictions
# ---------------------------------------------------------------------------


def run_single_fold(
    graph_data,
    split_info,
    architecture,
    n_classes,
    epochs,
    lr,
    weight_decay,
    device,
    model_save_path=None,
):
    """Train and evaluate one fold, returning per-patient predictions.

    Args:
        graph_data: list of dicts with 'graph', 'label', 'patient_id'
        split_info: dict with 'repeat', 'fold', 'train', 'val', 'test' indices
        architecture: 'baseline', 'NCA', or 'CA'
        n_classes: number of classes
        epochs, lr, weight_decay: training hyperparameters
        device: 'cpu' or 'cuda'
        model_save_path: if set, save best model state_dict to this path

    Returns:
        fold_preds: list of dicts (one per test patient) with keys:
            patient_id, repeat, fold, y_true, y_pred, y_prob_0, ...
        fold_metrics: dict with accuracy, f1, precision, recall, auc
    """
    rep = split_info["repeat"]
    fold = split_info["fold"]

    train_graphs = [graph_data[i]["graph"] for i in split_info["train"]]
    train_labels = [graph_data[i]["label"] for i in split_info["train"]]

    val_graphs = [graph_data[i]["graph"] for i in split_info["val"]]
    val_labels = [graph_data[i]["label"] for i in split_info["val"]]

    test_graphs = [graph_data[i]["graph"] for i in split_info["test"]]
    test_labels = [graph_data[i]["label"] for i in split_info["test"]]
    test_pids = [graph_data[i]["patient_id"] for i in split_info["test"]]

    # Create fresh model
    model = create_model(architecture, n_classes)

    # Train
    model, history = train_model(
        model,
        train_graphs,
        train_labels,
        val_graphs,
        val_labels,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
    )

    # Evaluate on test fold
    predictions, probabilities, metrics = evaluate_model(
        model, test_graphs, test_labels, device=device
    )

    # Build per-patient prediction rows
    fold_preds = []
    for i, pid in enumerate(test_pids):
        row = {
            "patient_id": pid,
            "repeat": rep,
            "fold": fold,
            "y_true": test_labels[i],
            "y_pred": int(predictions[i]),
        }
        for c in range(n_classes):
            row[f"y_prob_{c}"] = float(probabilities[i, c])
        fold_preds.append(row)

    # Save best model weights
    if model_save_path is not None:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"      Saved model: {model_save_path}", flush=True)

    # Clean up
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return fold_preds, metrics


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiments(args):
    """Run all requested task x architecture combinations with CV."""
    # Resolve which tasks and architectures to run
    if args.tasks == "all":
        task_keys = list(TASKS.keys())
    else:
        task_keys = [args.tasks]

    if args.architectures == "all":
        architectures = ["baseline", "NCA", "CA"]
    else:
        architectures = [args.architectures]

    # Ensure output directories exist
    pred_dir = os.path.join(RESULTS_DIR, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_metrics_rows = []

    for task_key in task_keys:
        task_cfg = TASKS[task_key]
        task_name = task_cfg["name"]
        n_classes = len(task_cfg["class_names"])

        print(f"\n{'=' * 70}")
        print(f"Task: {task_key} ({task_name}), {n_classes} classes")
        print(f"{'=' * 70}")

        # Load graph data
        t0 = time.time()
        graph_data, n_classes = load_graph_data(
            task_key, knn=args.knn, device=args.device
        )
        print(f"  Graph loading took {time.time() - t0:.1f}s")

        if len(graph_data) == 0:
            print("  WARNING: no graph data loaded, skipping task")
            continue

        # Create CV splits
        labels = np.array([d["label"] for d in graph_data])
        cv_splits = create_cv_splits(
            labels,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            random_state=42,
        )

        print(
            f"  CV: {args.n_folds} folds x {args.n_repeats} repeats "
            f"= {len(cv_splits)} total runs"
        )
        print(f"  Label distribution: {np.bincount(labels).tolist()}")

        for architecture in architectures:
            # Look up per-task, per-architecture HPs from MLflow recovery
            hp = CONCH_HP_LOOKUP.get(architecture, {}).get(task_key)
            if hp is not None:
                fold_lr, fold_wd = hp
            else:
                fold_lr, fold_wd = args.lr, args.weight_decay
            print(f"\n  --- Architecture: {architecture} (lr={fold_lr}, wd={fold_wd}) ---")

            all_fold_preds = []
            fold_metrics_list = []

            for split_info in cv_splits:
                rep = split_info["repeat"]
                fold = split_info["fold"]

                print(
                    f"    Repeat {rep + 1}/{args.n_repeats}, "
                    f"Fold {fold + 1}/{args.n_folds} "
                    f"(train={len(split_info['train'])}, "
                    f"val={len(split_info['val'])}, "
                    f"test={len(split_info['test'])})"
                )

                # Model save path
                model_dir = os.path.join(RESULTS_DIR, "conch_models")
                model_save_path = os.path.join(
                    model_dir,
                    f"CONCH_{task_key}_{architecture}_R{rep+1}_F{fold+1}.pth",
                )

                t1 = time.time()
                fold_preds, fold_metrics = run_single_fold(
                    graph_data,
                    split_info,
                    architecture,
                    n_classes,
                    epochs=args.epochs,
                    lr=fold_lr,
                    weight_decay=fold_wd,
                    device=args.device,
                    model_save_path=model_save_path,
                )
                elapsed = time.time() - t1

                print(
                    f"      Acc={fold_metrics['accuracy']:.4f}  "
                    f"F1={fold_metrics['f1']:.4f}  "
                    f"AUC={fold_metrics.get('auc', float('nan')):.4f}  "
                    f"({elapsed:.1f}s)"
                )

                all_fold_preds.extend(fold_preds)
                fold_metrics_list.append(
                    {
                        "task": task_key,
                        "architecture": architecture,
                        "repeat": rep,
                        "fold": fold,
                        **fold_metrics,
                    }
                )

            # Save per-patient predictions CSV
            preds_df = pd.DataFrame(all_fold_preds)
            pred_path = os.path.join(
                pred_dir,
                f"SBC_{task_key}_{architecture}_conch_predictions.csv",
            )
            preds_df.to_csv(pred_path, index=False)
            print(f"\n  Saved predictions: {pred_path} ({len(preds_df)} rows)")

            # Aggregate metrics across folds
            metrics_df = pd.DataFrame(fold_metrics_list)
            mean_metrics = metrics_df[
                ["accuracy", "f1", "precision", "recall", "auc"]
            ].mean()
            std_metrics = metrics_df[
                ["accuracy", "f1", "precision", "recall", "auc"]
            ].std()

            print(f"  Mean Accuracy: {mean_metrics['accuracy']:.4f} +/- {std_metrics['accuracy']:.4f}")
            print(f"  Mean F1:       {mean_metrics['f1']:.4f} +/- {std_metrics['f1']:.4f}")
            print(f"  Mean AUC:      {mean_metrics['auc']:.4f} +/- {std_metrics['auc']:.4f}")

            all_metrics_rows.extend(fold_metrics_list)

    # Save aggregate metrics CSV
    if all_metrics_rows:
        metrics_path = os.path.join(pred_dir, "SBC_conch_metrics.csv")
        metrics_all_df = pd.DataFrame(all_metrics_rows)
        metrics_all_df.to_csv(metrics_path, index=False)
        print(f"\nSaved aggregate metrics: {metrics_path}")


# ---------------------------------------------------------------------------
# Dry run: check paths only
# ---------------------------------------------------------------------------


def dry_run_check(args):
    """Verify all paths and data files exist without loading anything heavy."""
    print("=== DRY RUN: Checking paths and configs ===\n")
    ok = True

    # Check ground truth
    exists = os.path.exists(SBC_GT)
    status = "OK" if exists else "MISSING"
    if not exists:
        ok = False
    print(f"  [{status}] SBC GT: {SBC_GT}")

    # Check CONCH graph base directory
    exists = os.path.exists(SBC_CONCH)
    status = "OK" if exists else "MISSING"
    if not exists:
        ok = False
    print(f"  [{status}] CONCH graphs base: {SBC_CONCH}")

    # Check per-task graph directories
    if args.tasks == "all":
        task_keys = list(TASKS.keys())
    else:
        task_keys = [args.tasks]

    print()
    for task_key in task_keys:
        task_name = TASKS[task_key]["name"]
        graph_dir = find_conch_graph_dir(task_name, args.knn)
        if graph_dir:
            nfiles = len(
                [f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")]
            )
            print(f"  [OK] {task_key} ({task_name}) k={args.knn}: {nfiles} graphs in {graph_dir}")
        else:
            print(f"  [MISSING] {task_key} ({task_name}) k={args.knn}")
            ok = False

    # Check CODE_DIR imports
    print()
    for module_name in ["conch_models", "MIL_models"]:
        module_path = os.path.join(CODE_DIR, f"{module_name}.py")
        exists = os.path.exists(module_path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {module_name}: {module_path}")

    # Check output directory
    print()
    pred_dir = os.path.join(RESULTS_DIR, "predictions")
    exists = os.path.exists(pred_dir)
    status = "OK" if exists else "WILL CREATE"
    print(f"  [{status}] Output dir: {pred_dir}")

    # Environment info
    print()
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  Device: {args.device}")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")

    print()
    print(f"Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Protocol 3: Train CONCH aggregation+classifier from scratch "
            "with 3-fold CV, 2 repeats. Saves per-patient predictions."
        )
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for training (default: cpu)",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        default="all",
        choices=["baseline", "NCA", "CA", "all"],
        help="Which architecture(s) to run (default: all)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        choices=["2class", "3class", "4class", "all"],
        help="Which classification task(s) to run (default: all)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=3,
        help="Number of CV folds (default: 3)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=2,
        help="Number of CV repeats (default: 2)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Max training epochs (default: 150)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, paper optimal)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (default: 1e-4, paper optimal)",
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=19,
        help="KNN for graph construction (default: 19)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check paths, don't train",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CMPB Revision - Protocol 3: CONCH Classifier Predictions")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {args.device}")
    print(f"Architectures: {args.architectures}")
    print(f"Tasks: {args.tasks}")
    print(f"CV: {args.n_folds} folds x {args.n_repeats} repeats")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, WD: {args.weight_decay}")
    print(f"KNN: {args.knn}")
    print()

    if args.dry_run:
        dry_run_check(args)
        return

    run_experiments(args)
    print("\nDone.")


if __name__ == "__main__":
    main()
