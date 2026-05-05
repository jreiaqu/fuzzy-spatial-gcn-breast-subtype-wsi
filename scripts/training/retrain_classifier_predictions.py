"""
retrain_classifier_predictions.py -- CMPB-D-25-07046 Revision, Protocol 2

Implements constrained transfer learning (Protocol 2 from the paper):
  1. Loads BCNB-trained NCA (VGG16 + attention MIL) and CA (VGG16 + GCN) models
  2. Freezes everything except the classifier layer
  3. Extracts aggregated WSI-level feature vectors from SBC/SBC graphs
  4. Retrains ONLY the final classifier on SBC/SBC using 5-fold CV x 3 repeats
  5. Saves per-patient predictions for each test fold

This is a standalone re-implementation of the original retrain_ca_nca_classifiers.py,
stripped of MLflow dependencies and producing CSV-based per-patient predictions
compatible with the downstream analysis notebooks in this review project.

Usage:
    pip install -r requirements.txt  # see repository root
    python retrain_classifier_predictions.py --device cpu
    python retrain_classifier_predictions.py --models ca --n-repeats 1  # quick test
    python retrain_classifier_predictions.py --dry-run                  # check paths only
"""

import sys
import os
import argparse
import types
import importlib
import warnings
import re
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
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle as sklearn_shuffle

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

# ---------------------------------------------------------------------------
# PyTorch + torch_geometric compatibility shim (from generate_predictions.py)
# ---------------------------------------------------------------------------

def _apply_compat_patches():
    """Apply all compatibility patches needed for loading pickled GCN models."""
    patches = []

    # Patch 1: _rebuild_parameter_v2 (needed for torch.load of pickled models)
    if not hasattr(torch._utils, '_rebuild_parameter_v2'):
        if hasattr(torch._utils, '_rebuild_parameter_with_state'):
            torch._utils._rebuild_parameter_v2 = torch._utils._rebuild_parameter_with_state
            patches.append("_rebuild_parameter_v2")

    # Patch 2: _lazy_load_hook and other missing Module attrs
    _orig_getattr = nn.Module.__getattr__
    def _patched_getattr(self, name):
        if name in ('_lazy_load_hook', 'decomposed_layers', 'explain'):
            return None
        return _orig_getattr(self, name)
    nn.Module.__getattr__ = _patched_getattr
    patches.append("nn.Module.__getattr__")

    # Patch 3: torch_geometric Inspector module path redirect
    import torch_geometric
    import torch_geometric.nn.conv
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

    # Patch 4: Inspector.implements safe fallback
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
# Path constants
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB_ROOT    = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] CODE_DIR       = f"{MOLSUB_ROOT}/code"
# [REPLACED by _paths.py] GCN_WEIGHTS    = f"{MOLSUB_ROOT}/data/gcn_pretrained_models"
NCA_WEIGHTS    = f"{MOLSUB_ROOT}/data/feature_extractors"
SBC_GRAPHS = f"{MOLSUB_ROOT}/data/SBC/results_graphs_january_25"
SBC_GT     = f"{MOLSUB_ROOT}/data/SBC/ground_truth/CBDC_4_may2024_gt_extended.xlsx"
# [REPLACED by _paths.py] RESULTS_DIR    = "/Users/kckj099/Documents/CMPB-Review/results"

# Add code dir for model class imports
# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Model and task definitions (same as generate_predictions.py)
# ---------------------------------------------------------------------------

GCN_MODELS = {
    "2class": {
        # Retrained with GENConv/attention (Session 4, 2026-05-01). MC-CV selected.
        "filename": "2class_GENConv_5L_attn_5L_attn_lr1e5_wd_final.pth",
        "weights_dir": os.path.join(WEIGHTS_DIR, "retrained"),
        "task": "OTHERvsTNBC",
        "n_classes": 2,
        "knn": 19,
        "pooling": "attention",
        "gnn_layer_type": "GENConv",
        "num_layers": 5,
    },
    "3class": {
        # Retrained with GENConv (Session 3b, 2026-04-29). MC-CV selected.
        # See results/retrained_models/3class_retrain_provenance.json
        "filename": "3class_GENConv_5L_attn_lr2e5_final.pth",
        "weights_dir": os.path.join(WEIGHTS_DIR, "retrained"),
        "task": "LUMINALSvsHER2vsTNBC",
        "n_classes": 3,
        "knn": 19,
        "pooling": "attention",
        "gnn_layer_type": "GENConv",
        "num_layers": 5,
    },
    "4class": {
        # Retrained with GENConv/attention (Session 4, 2026-05-01). MC-CV selected.
        "filename": "4class_GENConv_5L_attn_5L_attn_lr1e5_final.pth",
        "weights_dir": os.path.join(WEIGHTS_DIR, "retrained"),
        "task": "LUMINALAvsLAUMINALBvsHER2vsTNBC",
        "n_classes": 4,
        "knn": 19,
        "pooling": "attention",
        "gnn_layer_type": "GENConv",
        "num_layers": 5,
    },
}

NCA_MODELS = {
    "2class": {
        "filename": "PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10x_N_100_Anetwork_weights_best_f1.pth",
        "task": "OTHERvsTNBC",
        "n_classes": 2,
    },
    "3class": {
        "filename": "PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10network_weights_best_f1.pth",
        "task": "LUMINALSvsHER2vsTNBC",
        "n_classes": 3,
    },
    "4class": {
        "filename": "PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_Tnetwork_weights_best_f1.pth",
        "task": "LUMINALAvsLAUMINALBvsHER2vsTNBC",
        "n_classes": 4,
    },
}

# Task label mappings (ground truth text -> integer label)
TASK_LABEL_MAPS = {
    "OTHERvsTNBC": {"Other": 0, "TNBC": 1},
    "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
}

# 4-class task name has a typo (LAUMINALB) in model filenames vs graph dirs (LUMINALB)
_TASK_NAME_ALIASES = {
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": "LUMINALAvsLUMINALBvsHER2vsTNBC",
}

# Reverse map: class index -> class name (for CSV output)
TASK_CLASS_NAMES = {
    "OTHERvsTNBC": {0: "Other", 1: "TNBC"},
    "LUMINALSvsHER2vsTNBC": {0: "Luminal", 1: "HER2(+)", 2: "TNBC"},
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3: "TNBC"},
}


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_sbc_gt():
    """Load SBC ground truth, filter excluded, map to task labels."""
    df = pd.read_excel(SBC_GT)
    df = df.rename(columns={"SUS_number": "patient_id", "Molsub_surr_4clf": "mol_subtype"})

    # Filter where Molsub_surr_7clf != 'Excluded' (as in original script)
    df = df[df["Molsub_surr_7clf"] != "Excluded"].copy()

    # Also exclude rows where mol_subtype itself is 'Excluded' or missing
    df = df[df["mol_subtype"].notna()].copy()
    df = df[df["mol_subtype"] != "Excluded"].copy()

    # Map to task-specific labels
    df["label_4class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3
    })
    df["label_3class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 0, "HER2(+)": 1, "TNBC": 2
    })
    df["label_2class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 0, "HER2(+)": 0, "TNBC": 1
    })
    return df


# ---------------------------------------------------------------------------
# Patient ID extraction
# ---------------------------------------------------------------------------

def extract_sbc_patient_id(filename):
    """Extract patient ID (SUS###) from SBC graph filename."""
    match = re.search(r'(SUS\d+)', filename)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Graph directory resolution (from generate_predictions.py)
# ---------------------------------------------------------------------------

def find_graph_dir(base_path, task_name, knn):
    """Find the graph directory for a given task and k-value.

    Returns the path to the graphs_k_{knn} subdirectory, or None.
    Handles task name typo aliases (LAUMINALB vs LUMINALB).
    """
    if not os.path.exists(base_path):
        return None

    names_to_try = [task_name]
    if task_name in _TASK_NAME_ALIASES:
        names_to_try.append(_TASK_NAME_ALIASES[task_name])

    for dirname in os.listdir(base_path):
        dirpath = os.path.join(base_path, dirname)
        if not os.path.isdir(dirpath):
            continue
        for name in names_to_try:
            if name in dirname:
                k_dir = os.path.join(dirpath, f"graphs_k_{knn}")
                if os.path.exists(k_dir):
                    return k_dir
    return None


# ---------------------------------------------------------------------------
# Model loading (from generate_predictions.py)
# ---------------------------------------------------------------------------

def load_nca_model(model_path, device):
    """Load an NCA model (pickled object, loads directly without issues)."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"NCA model not found: {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def load_gcn_model(model_path, config, device):
    """Load a GCN model using state_dict reconstruction.

    The pickled model is loaded to extract weights, then a fresh
    PatchGCN_MeanMax_LSelec is instantiated and weights transferred.
    This avoids torch_geometric Inspector incompatibility on forward pass.
    """
    from MIL_models import PatchGCN_MeanMax_LSelec

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"GCN model not found: {model_path}")

    # Step 1: Load pickle to extract state_dict only
    old_model = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = old_model.state_dict()
    del old_model

    # Step 2: Create fresh model with known constructor params
    new_model = PatchGCN_MeanMax_LSelec(
        num_features=512,                     # VGG16 features
        num_layers=config["num_layers"],
        hidden_dim=128,
        n_classes=config["n_classes"],
        pooling=config["pooling"],
        gnn_layer_type=config["gnn_layer_type"],
    )

    # Step 3: Transfer weights
    load_result = new_model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        print(f"  WARNING: state_dict mismatch. Missing: {load_result.missing_keys}, "
              f"Unexpected: {load_result.unexpected_keys}")

    new_model = new_model.to(device)
    new_model.eval()
    return new_model


# ---------------------------------------------------------------------------
# Feature extraction: extract aggregated WSI-level features from all graphs
# ---------------------------------------------------------------------------

def extract_features_nca(model, graph_dir, gt_df, n_classes, device):
    """Extract aggregated feature vectors for all SBC graphs using frozen NCA model.

    For NCA: embedding = model.milAggregation(graph['x'])
    The milAggregation with attention internally pools all patch features into
    a single 512-dim WSI-level embedding.

    Returns:
        features: np.ndarray of shape (N_patients, feat_dim)
        labels: np.ndarray of shape (N_patients,) with integer class labels
        patient_ids: list of str with patient IDs in same order
    """
    label_col = f"label_{n_classes}class"
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])

    all_features = []
    all_labels = []
    all_pids = []

    model.eval()
    with torch.no_grad():
        for gfile in graph_files:
            pid = extract_sbc_patient_id(gfile)
            if pid is None:
                continue

            # Look up ground truth
            gt_row = gt_df[gt_df["patient_id"] == pid]
            if len(gt_row) == 0:
                continue
            label = gt_row[label_col].values[0]
            if pd.isna(label):
                continue
            label = int(label)

            # Load graph and extract patch features
            graph_path = os.path.join(graph_dir, gfile)
            graph = torch.load(graph_path, map_location=device, weights_only=False)
            graph_features = graph["x"].to(device)

            # NCA aggregation: milAggregation pools patches -> single WSI embedding
            embedding = model.milAggregation(graph_features)

            all_features.append(embedding.cpu().numpy().flatten())
            all_labels.append(label)
            all_pids.append(pid)

    features = np.stack(all_features)
    labels = np.array(all_labels)
    return features, labels, all_pids


def extract_features_ca(model, graph_dir, gt_df, n_classes, device):
    """Extract aggregated feature vectors for all SBC graphs using frozen CA (GCN) model.

    For CA: Y_prob, Y_hat, logits, h = model(graph)
    The 4th return value (h) is the pooled graph-level embedding after
    GCN layers + path_phi + attention/mean/max pooling + path_rho.

    Returns:
        features: np.ndarray of shape (N_patients, feat_dim)
        labels: np.ndarray of shape (N_patients,) with integer class labels
        patient_ids: list of str with patient IDs in same order
    """
    label_col = f"label_{n_classes}class"
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])

    all_features = []
    all_labels = []
    all_pids = []

    model.eval()
    with torch.no_grad():
        for gfile in graph_files:
            pid = extract_sbc_patient_id(gfile)
            if pid is None:
                continue

            gt_row = gt_df[gt_df["patient_id"] == pid]
            if len(gt_row) == 0:
                continue
            label = gt_row[label_col].values[0]
            if pd.isna(label):
                continue
            label = int(label)

            graph_path = os.path.join(graph_dir, gfile)
            graph = torch.load(graph_path, map_location=device, weights_only=False)
            graph = graph.to(device)

            # CA forward: returns (Y_prob, Y_hat, logits, h)
            # h is the pooled graph embedding after path_rho
            _, _, _, h = model(graph)

            all_features.append(h.cpu().numpy().flatten())
            all_labels.append(label)
            all_pids.append(pid)

    features = np.stack(all_features)
    labels = np.array(all_labels)
    return features, labels, all_pids


# ---------------------------------------------------------------------------
# Weighted cross-entropy loss (from original retrain script)
# ---------------------------------------------------------------------------

def weighted_cross_entropy(y_pred, y_true, class_weights=None):
    """Compute cross-entropy loss with optional per-sample class weighting.

    Args:
        y_pred: (batch, n_classes) raw logits
        y_true: (batch,) integer class labels
        class_weights: (n_classes,) tensor of class weights, or None
    """
    loss = F.cross_entropy(y_pred, y_true, reduction='none')
    if class_weights is not None:
        weight_per_sample = class_weights[y_true]
        loss = loss * weight_per_sample
    return loss.mean()


# ---------------------------------------------------------------------------
# Monte Carlo CV: retrain classifier only
# ---------------------------------------------------------------------------

def monte_carlo_cv(X, y, patient_ids, classifier, task_name, n_classes,
                   n_folds=5, n_repeats=3, batch_size=128, epochs=200,
                   lr=0.0001, weight_decay=None, device_str="cpu",
                   verbose=True):
    """Retrain the classifier layer using stratified k-fold CV with repeats.

    The approach follows the original retrain_ca_nca_classifiers.py:
      - For each repeat, shuffle X/y with a new random seed
      - For each fold, deepcopy the original classifier, retrain from scratch
      - Train with Adam optimizer, weighted cross-entropy, for `epochs` epochs
      - Record per-patient predictions on the test fold

    Args:
        X: np.ndarray (N, feat_dim), aggregated WSI-level features
        y: np.ndarray (N,), integer class labels
        patient_ids: list of str, patient IDs aligned with X/y
        classifier: nn.Module, the original pretrained classifier layer to deepcopy
        task_name: str, task identifier
        n_classes: int
        n_folds: int, number of CV folds (default 5)
        n_repeats: int, number of Monte Carlo repeats (default 3)
        batch_size: int (default 128)
        epochs: int, training epochs per fold (default 200)
        lr: float, learning rate (default 0.0001)
        device_str: str, 'cpu' or 'cuda'

    Returns:
        predictions_df: DataFrame with columns
            [patient_id, repeat, fold, y_true, y_pred, y_prob_0, y_prob_1, ...]
        metrics_df: DataFrame with columns
            [repeat, fold, accuracy, f1, precision, recall, auc]
    """
    device = torch.device(device_str)
    patient_ids = np.array(patient_ids)

    all_predictions = []
    all_metrics = []

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    for repeat in range(n_repeats):
        if verbose:
            print(f"  Repeat {repeat + 1}/{n_repeats}")

        # Reshuffle data with repeat-specific seed (same as original script)
        X_shuffled, y_shuffled, pids_shuffled = sklearn_shuffle(
            X, y, patient_ids, random_state=repeat
        )

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_shuffled, y_shuffled)):
            X_train = torch.tensor(X_shuffled[train_idx], dtype=torch.float32).to(device)
            y_train = torch.tensor(y_shuffled[train_idx], dtype=torch.long).to(device)
            X_test = torch.tensor(X_shuffled[test_idx], dtype=torch.float32).to(device)
            y_test = torch.tensor(y_shuffled[test_idx], dtype=torch.long).to(device)
            pids_test = pids_shuffled[test_idx]

            # Compute balanced class weights for this fold's training set
            classes_in_fold = np.unique(y_shuffled[train_idx])
            weights = compute_class_weight(
                'balanced', classes=classes_in_fold, y=y_shuffled[train_idx]
            )
            fold_class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

            # Reinitialize classifier from original pretrained weights
            fold_classifier = deepcopy(classifier).to(device)
            fold_classifier.train()

            opt_kwargs = {"lr": lr}
            if weight_decay is not None and weight_decay > 0:
                opt_kwargs["weight_decay"] = weight_decay
            optimizer = torch.optim.Adam(fold_classifier.parameters(), **opt_kwargs)

            # Train for the specified number of epochs
            for epoch in range(epochs):
                fold_classifier.train()
                for i in range(0, len(X_train), batch_size):
                    X_batch = X_train[i:i + batch_size]
                    y_batch = y_train[i:i + batch_size]

                    optimizer.zero_grad()
                    logits = fold_classifier(X_batch)
                    loss = weighted_cross_entropy(logits, y_batch, class_weights=fold_class_weights)
                    loss.backward()
                    optimizer.step()

            # Evaluate on test fold
            fold_classifier.eval()
            with torch.no_grad():
                test_logits = fold_classifier(X_test)
                test_probs = F.softmax(test_logits, dim=1).cpu().numpy()
                y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
                y_true = y_test.cpu().numpy()

            # Compute fold metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)

            # AUC: binary vs multiclass handling
            try:
                if n_classes == 2:
                    auc = roc_auc_score(y_true, test_probs[:, 1])
                else:
                    auc = roc_auc_score(y_true, test_probs, multi_class='ovr')
            except ValueError:
                # Can happen if a class is missing from the test fold
                auc = float("nan")

            all_metrics.append({
                "repeat": repeat,
                "fold": fold,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "auc": auc,
            })

            # Record per-patient predictions
            for j in range(len(y_true)):
                row = {
                    "patient_id": pids_test[j],
                    "repeat": repeat,
                    "fold": fold,
                    "y_true": int(y_true[j]),
                    "y_pred": int(y_pred[j]),
                }
                for c in range(n_classes):
                    row[f"y_prob_{c}"] = float(test_probs[j, c])
                all_predictions.append(row)

            if verbose:
                print(f"    Fold {fold + 1}/{n_folds}: "
                      f"acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

            # Cleanup
            del fold_classifier, optimizer
            if device.type == "cuda":
                torch.cuda.empty_cache()

    predictions_df = pd.DataFrame(all_predictions)
    metrics_df = pd.DataFrame(all_metrics)
    return predictions_df, metrics_df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_transfer_learning(model_type, args):
    """Run Protocol 2 (constrained transfer) for CA or NCA models.

    Steps:
      1. Load SBC ground truth
      2. For each task (2/3/4 class):
         a. Load BCNB-pretrained model
         b. Extract aggregated features from all SBC graphs (frozen model)
         c. Retrain classifier via stratified k-fold CV with repeats
         d. Save per-patient predictions CSV
    """
    device = torch.device(args.device)
    gt_df = load_sbc_gt()
    print(f"SBC non-excluded patients in GT: {len(gt_df)}")

    models_config = GCN_MODELS if model_type == "ca" else NCA_MODELS
    weights_dir = GCN_WEIGHTS if model_type == "ca" else NCA_WEIGHTS
    model_label = "CA" if model_type == "ca" else "NCA"

    task_results = []

    for task_key, config in models_config.items():
        task_name = config["task"]
        n_classes = config["n_classes"]
        knn = config.get("knn", 19)  # NCA graphs use knn=19 by default

        print(f"\n--- SBC Transfer {model_label} {task_key} "
              f"(task={task_name}, k={knn}) ---")

        # Step 1: Load model (use per-task weights_dir if specified, else default)
        task_weights_dir = config.get("weights_dir", weights_dir)
        model_path = os.path.join(task_weights_dir, config["filename"])
        try:
            if model_type == "ca":
                model = load_gcn_model(model_path, config, device)
            else:
                model = load_nca_model(model_path, device)
            print(f"  Model loaded: {type(model).__name__}")
        except Exception as e:
            print(f"  ERROR loading model: {e}")
            continue

        # Step 2: Find graph directory
        graph_dir = find_graph_dir(SBC_GRAPHS, task_name, knn)
        if graph_dir is None:
            print(f"  ERROR: graph dir not found for {task_name} k={knn}")
            continue
        print(f"  Graphs: {graph_dir}")

        # Step 3: Extract aggregated features from ALL graphs (once)
        print(f"  Extracting aggregated features...")
        if model_type == "ca":
            features, labels, pids = extract_features_ca(
                model, graph_dir, gt_df, n_classes, device
            )
        else:
            features, labels, pids = extract_features_nca(
                model, graph_dir, gt_df, n_classes, device
            )

        print(f"  Extracted features: {features.shape[0]} patients, "
              f"{features.shape[1]}-dim features")
        print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

        # Step 4: Get the pretrained classifier layer for deepcopy during CV
        classifier = model.classifier
        print(f"  Classifier: {classifier}")

        # Step 5: Run Monte Carlo CV (retrain classifier only)
        if args.sweep:
            # Sweep mode: try multiple HP combos, keep the best
            import itertools
            combos = list(itertools.product(SWEEP_LRS, SWEEP_WDS, SWEEP_BATCH_SIZES))
            print(f"  SWEEP MODE: testing {len(combos)} HP combos "
                  f"({args.n_folds}-fold x {args.n_repeats} repeats, {args.epochs} epochs)")

            best_acc = -1.0
            best_predictions_df = None
            best_metrics_df = None
            best_hp = None

            for combo_idx, (lr, wd, bs) in enumerate(combos):
                wd_str = f"{wd}" if wd is not None else "None"
                print(f"    [{combo_idx+1}/{len(combos)}] lr={lr}, wd={wd_str}, bs={bs}", end="")

                preds_df, mets_df = monte_carlo_cv(
                    X=features, y=labels, patient_ids=pids,
                    classifier=classifier, task_name=task_name,
                    n_classes=n_classes, n_folds=args.n_folds,
                    n_repeats=args.n_repeats, batch_size=bs,
                    epochs=args.epochs, lr=lr, weight_decay=wd,
                    device_str=args.device, verbose=False,
                )

                mean_acc = mets_df["accuracy"].mean()
                print(f" -> acc={mean_acc:.4f}")

                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_predictions_df = preds_df
                    best_metrics_df = mets_df
                    best_hp = {"lr": lr, "wd": wd, "bs": bs}

            predictions_df = best_predictions_df
            metrics_df = best_metrics_df
            print(f"  BEST: lr={best_hp['lr']}, wd={best_hp['wd']}, "
                  f"bs={best_hp['bs']} -> acc={best_acc:.4f}")
        else:
            # Single config mode
            print(f"  Starting {args.n_folds}-fold CV x {args.n_repeats} repeats "
                  f"({args.epochs} epochs, lr={args.lr}, batch={args.batch_size})...")

            predictions_df, metrics_df = monte_carlo_cv(
                X=features, y=labels, patient_ids=pids,
                classifier=classifier, task_name=task_name,
                n_classes=n_classes, n_folds=args.n_folds,
                n_repeats=args.n_repeats, batch_size=args.batch_size,
                epochs=args.epochs, lr=args.lr, weight_decay=None,
                device_str=args.device,
            )

        # Step 6: Save per-patient predictions CSV
        pred_dir = os.path.join(RESULTS_DIR, "predictions")
        os.makedirs(pred_dir, exist_ok=True)

        pred_path = os.path.join(
            pred_dir,
            f"SBC_{task_key}_{model_label}_transfer_predictions.csv"
        )
        predictions_df.to_csv(pred_path, index=False)
        print(f"  Saved predictions: {pred_path} "
              f"({len(predictions_df)} rows, "
              f"{predictions_df['patient_id'].nunique()} unique patients)")

        # Summary metrics for this task
        avg_metrics = metrics_df.mean(numeric_only=True)
        std_metrics = metrics_df.std(numeric_only=True)
        print(f"  Mean accuracy: {avg_metrics['accuracy']:.4f} "
              f"(+/- {std_metrics['accuracy']:.4f})")
        print(f"  Mean F1:       {avg_metrics['f1']:.4f} "
              f"(+/- {std_metrics['f1']:.4f})")
        print(f"  Mean AUC:      {avg_metrics['auc']:.4f} "
              f"(+/- {std_metrics['auc']:.4f})")

        # Add task/model context to metrics
        metrics_df["task"] = task_name
        metrics_df["task_key"] = task_key
        metrics_df["model_type"] = model_label
        metrics_df["n_classes"] = n_classes
        task_results.append(metrics_df)

        # Cleanup
        del model, features, labels
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Step 7: Save aggregate metrics CSV
    if task_results:
        all_metrics_df = pd.concat(task_results, ignore_index=True)
        metrics_path = os.path.join(
            RESULTS_DIR, "predictions", "SBC_transfer_metrics.csv"
        )

        # If file exists and we're only running one model type, merge with existing
        if os.path.exists(metrics_path):
            existing = pd.read_csv(metrics_path)
            # Remove rows for the current model_type to avoid duplicates
            existing = existing[existing["model_type"] != model_label]
            all_metrics_df = pd.concat([existing, all_metrics_df], ignore_index=True)

        all_metrics_df.to_csv(metrics_path, index=False)
        print(f"\nSaved aggregate metrics: {metrics_path}")


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run_check():
    """Verify all paths and data files exist without loading models."""
    print("=== DRY RUN: Checking paths and configs ===\n")
    ok = True

    # Check model weights
    for label, models in [("GCN (CA)", GCN_MODELS), ("NCA", NCA_MODELS)]:
        weights_dir = GCN_WEIGHTS if label.startswith("GCN") else NCA_WEIGHTS
        for key, cfg in models.items():
            task_wd = cfg.get("weights_dir", weights_dir)
            path = os.path.join(task_wd, cfg["filename"])
            exists = os.path.exists(path)
            status = "OK" if exists else "MISSING"
            if not exists:
                ok = False
            print(f"  [{status}] {label} {key}: {path}")

    print()

    # Check ground truth
    exists = os.path.exists(SBC_GT)
    status = "OK" if exists else "MISSING"
    if not exists:
        ok = False
    print(f"  [{status}] SBC GT: {SBC_GT}")

    if exists:
        gt_df = load_sbc_gt()
        print(f"  SBC non-excluded patients: {len(gt_df)}")

    print()

    # Check graph directories for each task/model combo
    for model_label, models in [("CA", GCN_MODELS), ("NCA", NCA_MODELS)]:
        for key, cfg in models.items():
            task = cfg["task"]
            knn = cfg.get("knn", 19)
            gdir = find_graph_dir(SBC_GRAPHS, task, knn)
            if gdir:
                nfiles = len([f for f in os.listdir(gdir) if f.endswith("_graph.pt")])
                print(f"  [OK] {model_label} {key} SBC graphs "
                      f"(task={task}, k={knn}): {nfiles} files in {gdir}")
            else:
                print(f"  [MISSING] {model_label} {key} SBC graphs "
                      f"(task={task}, k={knn})")
                ok = False

    print()

    # Check output dirs
    pred_dir = os.path.join(RESULTS_DIR, "predictions")
    exists = os.path.exists(pred_dir)
    status = "OK" if exists else "WILL CREATE"
    print(f"  [{status}] Output dir: {pred_dir}")

    print()
    print(f"  PyTorch version: {torch.__version__}")
    if _applied_patches:
        print(f"  [OK] Compatibility patches applied: {', '.join(_applied_patches)}")
    else:
        print(f"  [OK] No compatibility patches needed")

    print()
    print(f"Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Protocol 2: Retrain classifier on SBC with 5-fold CV"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for computation (default: cpu)"
    )
    parser.add_argument(
        "--models", type=str, default="both",
        choices=["ca", "nca", "both"],
        help="Which model types to run (default: both)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--n-repeats", type=int, default=3,
        help="Number of Monte Carlo CV repeats (default: 3)"
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Training epochs per fold (default: 200)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001,
        help="Learning rate for classifier retraining (default: 0.0001)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for classifier training (default: 128)"
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep hyperparameters (LR x WD). Picks the best combo per task "
             "based on mean accuracy, then saves those predictions."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check paths and configs, don't load models"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Hyperparameter sweep grid (matches original retrain_ca_nca_classifiers.py)
# ---------------------------------------------------------------------------
SWEEP_LRS = [0.01, 0.001, 0.0001, 0.00001]
SWEEP_WDS = [0.001, 0.0001, 0.00001, None]
SWEEP_BATCH_SIZES = [128]  # original also tried 256, but 128 is fine


def main():
    args = parse_args()

    print("=" * 70)
    print("CMPB Revision - Protocol 2: Constrained Transfer Learning")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"CV: {args.n_folds}-fold x {args.n_repeats} repeats")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print()

    if args.dry_run:
        dry_run_check()
        return

    if _applied_patches:
        print(f"Compatibility patches: {', '.join(_applied_patches)}")
        print()

    model_types = ["ca", "nca"] if args.models == "both" else [args.models]

    for model_type in model_types:
        print(f"\n{'=' * 70}")
        print(f"Running {model_type.upper()} transfer learning on SBC")
        print(f"{'=' * 70}")

        run_transfer_learning(model_type, args)

    print("\nDone.")


if __name__ == "__main__":
    main()
