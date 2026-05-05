"""
generate_predictions.py  --  CMPB-D-25-07046 Revision, Session 1 (S1.1)

Loads trained GCN (context-aware) and NCA (non-context-aware) models,
runs inference on test graphs for BCNB and SBC datasets, saves
per-patient prediction CSVs and intermediate data for attention analysis.

Feeds downstream tasks:
  - R2.5: statistical tests (fold-level accuracy comparison)
  - R2.7: clinical metrics (sensitivity, specificity, COC curve)
  - R1.1: attention scatter plots (node embeddings + spatial coords)

Usage:
    pip install -r requirements.txt  # see repository root
    python generate_predictions.py --device cpu
    python generate_predictions.py --device cuda  # if GPU available
    python generate_predictions.py --models ca     # CA only
    python generate_predictions.py --models nca    # NCA only
    python generate_predictions.py --datasets bcnb # BCNB only

GCN Model Loading Strategy:
    GCN models were saved as full pickled objects (torch.save(model, path)) with
    an older PyTorch + torch_geometric version. Direct loading and forward pass
    fails with modern versions due to Inspector class incompatibility. The fix
    uses state_dict reconstruction: load the pickle to extract weights, create
    a fresh model instance from the class definition, and load the state_dict.
    NCA models load directly without issues.
"""

import sys
import os
import argparse
import types
import importlib
import warnings
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# ---------------------------------------------------------------------------
# PyTorch + torch_geometric compatibility shim
# ---------------------------------------------------------------------------
# GCN models require these patches to be deserialized from pickle.
# NCA models do not need patches (they load cleanly).
# After deserialization, GCN models are reconstructed fresh (state_dict only)
# to avoid torch_geometric Inspector incompatibility during forward pass.

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
# [REPLACED by _paths.py] BCNB_GRAPHS    = f"{MOLSUB_ROOT}/data/BCNB/results_graphs_november_23"
SBC_GRAPHS = f"{MOLSUB_ROOT}/data/SBC/results_graphs_january_25"
SBC_CONCH  = f"{SBC_GRAPHS}/graphs_CONCH"
# [REPLACED by _paths.py] BCNB_GT        = f"{MOLSUB_ROOT}/data/BCNB/ground_truth/patient-clinical-data.xlsx"
SBC_GT     = f"{MOLSUB_ROOT}/data/SBC/ground_truth/CBDC_4_may2024_gt_extended.xlsx"
# [REPLACED by _paths.py] BCNB_SPLITS    = f"{MOLSUB_ROOT}/data/BCNB/patches_paths_class_perc"
SBC_FOLDS  = f"{MOLSUB_ROOT}/data/SBC/new_CV_folds_SBC_DB"
# [REPLACED by _paths.py] RESULTS_DIR    = "/Users/kckj099/Documents/CMPB-Review/results"
# [REPLACED by _paths.py] VENV_PATH      = f"{MOLSUB_ROOT}/molsub_venv"

# Add code dir for model class imports (code/MIL_utils.py has nmslib commented out)
# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)

# ---------------------------------------------------------------------------
# Model and task definitions
# ---------------------------------------------------------------------------

# GCN (CA) model filenames -> parsed config
GCN_MODELS = {
    "2class": {
        "filename": "[24_11_2023]_GCN_Final_BCNB_OTHERvsTNBC_GT_GENConv_GL_5_KNN_19_EA_spatial_EF_False_GP_mean_DO_True_LR_1e-05.pth",
        "task": "OTHERvsTNBC",
        "n_classes": 2,
        "knn": 19,
        "pooling": "mean",
        "gnn_layer_type": "GENConv",
        "num_layers": 5,
    },
    "3class": {
        # Retrained with GENConv (Session 3b, 2026-04-29). MC-CV selected.
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
        "filename": "[24_11_2023]_GCN_Final_BCNB_LUMINALAvsLUMINALBvsHER2vsTNBC_GT_GENConv_GL_4_KNN_25_GP_max_LR_2e-05_Optim_adam_OWD_1e-05_CVFold_0.pth",
        "task": "LUMINALAvsLAUMINALBvsHER2vsTNBC",  # Note: typo in original (LAUMINALB)
        "n_classes": 4,
        "knn": 25,
        "pooling": "max",
        "gnn_layer_type": "GENConv",
        "num_layers": 4,
    },
}

# NCA (VGG16 + attention) model filenames
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
        "task": "LUMINALAvsLAUMINALBvsHER2vsTNBC",  # Same typo as GCN
        "n_classes": 4,
    },
}

# Task label mappings (ground truth text -> integer label)
TASK_LABEL_MAPS = {
    "OTHERvsTNBC": {"Other": 0, "TNBC": 1},
    "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
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

def load_bcnb_gt():
    """Load BCNB ground truth and map molecular subtypes to task labels.

    BCNB GT columns: Patient ID (int), Molecular subtype (str)
    Molecular subtype values: 'Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative'
    """
    df = pd.read_excel(BCNB_GT)
    # Rename for consistency
    df = df.rename(columns={"Patient ID": "patient_id", "Molecular subtype": "mol_subtype"})
    df["patient_id"] = df["patient_id"].astype(str)

    # Map to task-specific labels
    # 4-class: direct mapping
    df["label_4class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "Triple negative": 3
    })
    # 3-class: merge Luminal A + Luminal B -> Luminal
    df["label_3class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 0, "HER2(+)": 1, "Triple negative": 2
    })
    # 2-class: Other (non-TNBC) vs TNBC
    df["label_2class"] = df["mol_subtype"].map({
        "Luminal A": 0, "Luminal B": 0, "HER2(+)": 0, "Triple negative": 1
    })
    return df


def load_sbc_gt():
    """Load SBC ground truth and map molecular subtypes to task labels.

    SBC GT columns: SUS_number (str), Molsub_surr_4clf (str), Molsub_surr_7clf (str)
    Molsub_surr_4clf values: 'Luminal A', 'Luminal B', 'HER2(+)', 'TNBC', 'Excluded'
    """
    df = pd.read_excel(SBC_GT)
    df = df.rename(columns={"SUS_number": "patient_id", "Molsub_surr_4clf": "mol_subtype"})

    # Filter out excluded patients
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
# Patient ID extraction from graph filenames
# ---------------------------------------------------------------------------

def extract_bcnb_patient_id(filename):
    """Extract patient ID from BCNB graph filename.
    Format: '{patient_id}_graph.pt' e.g. '1000_graph.pt' -> '1000'
    """
    return filename.replace("_graph.pt", "")


def extract_sbc_patient_id(filename):
    """Extract patient ID from SBC graph filename.
    Format: 'SUS{NNN}-{datetime}_graph.pt' e.g. 'SUS001-2021-09-24_13.39.24_graph.pt' -> 'SUS001'
    """
    return filename.split("-")[0]


# ---------------------------------------------------------------------------
# BCNB test set patient IDs
# ---------------------------------------------------------------------------

def get_bcnb_test_patient_ids():
    """Get the list of BCNB test set patient IDs from the patches_paths_class_perc split."""
    test_csv = os.path.join(BCNB_SPLITS, "test_patches_class_perc_0_tp.csv")
    df = pd.read_csv(test_csv)
    # Extract patient IDs from patch paths
    # Path format varies; the patient ID is the folder name containing the patches
    patient_ids = df["patch_path"].apply(lambda x: x.split("/")[-2]).unique()
    # Clean up IDs: remove any non-numeric prefix
    clean_ids = set()
    for pid in patient_ids:
        # The folder names in BCNB are numeric patient IDs
        clean_ids.add(pid)
    return clean_ids


# ---------------------------------------------------------------------------
# Graph directory resolution
# ---------------------------------------------------------------------------

# 4-class task name has a typo in model filenames (LAUMINALB) vs graph dirs
# (LUMINALB). This map handles the mismatch.
_TASK_NAME_ALIASES = {
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": "LUMINALAvsLUMINALBvsHER2vsTNBC",
}


def find_graph_dir(base_path, task_name, knn, feature_type="vgg16"):
    """Find the graph directory for a given task and k-value.

    Returns the path to the graphs_k_{knn} subdirectory, or None.
    Handles task name typo aliases (e.g. LAUMINALB vs LUMINALB).
    """
    if not os.path.exists(base_path):
        return None

    # Try original name and alias
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


def find_conch_graph_dir(task_name, knn):
    """Find CONCH graph directory for a task. Handles naming quirks."""
    if not os.path.exists(SBC_CONCH):
        return None

    names_to_try = [task_name]
    if task_name in _TASK_NAME_ALIASES:
        names_to_try.append(_TASK_NAME_ALIASES[task_name])

    for dirname in os.listdir(SBC_CONCH):
        dirpath = os.path.join(SBC_CONCH, dirname)
        if not os.path.isdir(dirpath):
            continue
        for name in names_to_try:
            if name in dirname:
                k_dir = os.path.join(dirpath, f"graphs_k_{knn}")
                if os.path.exists(k_dir):
                    return k_dir
    return None


# ---------------------------------------------------------------------------
# Model loading
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

    Args:
        model_path: Path to the .pth file
        config: Dict with keys: num_layers, n_classes, pooling, gnn_layer_type
        device: torch.device
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
# Inference: CA (GCN) models
# ---------------------------------------------------------------------------

def run_ca_inference(model, graph_dir, gt_df, task_name, n_classes,
                     dataset_name, patient_id_extractor, device,
                     test_ids=None):
    """Run context-aware (GCN) inference on all graphs in a directory.

    Args:
        model: Loaded PatchGCN_MeanMax_LSelec model
        graph_dir: Path to graphs_k_{knn} directory
        gt_df: Ground truth DataFrame with patient_id and label columns
        task_name: Task identifier for label column selection
        n_classes: Number of classes
        dataset_name: 'BCNB' or 'SBC'
        patient_id_extractor: Function to extract patient ID from filename
        device: torch device
        test_ids: Optional set of patient IDs to restrict to (for BCNB test set)

    Returns:
        predictions_df: DataFrame with patient_id, y_true, y_pred, y_prob_*
        node_data: Dict of patient_id -> {embeddings, centroids, logits} for attention analysis
    """
    # Select label column based on n_classes
    label_col = f"label_{n_classes}class"

    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])
    results = []
    node_data = {}

    for gfile in graph_files:
        pid = patient_id_extractor(gfile)

        # Filter to test set if specified
        if test_ids is not None and pid not in test_ids:
            continue

        # Look up ground truth
        gt_row = gt_df[gt_df["patient_id"] == pid]
        if len(gt_row) == 0:
            continue
        label = gt_row[label_col].values[0]
        if pd.isna(label):
            continue
        label = int(label)

        # Load graph
        graph_path = os.path.join(graph_dir, gfile)
        graph = torch.load(graph_path, map_location=device, weights_only=False)
        graph = graph.to(device)

        with torch.no_grad():
            # PatchGCN_MeanMax_LSelec.forward() returns: Y_prob, Y_hat, logits, h
            Y_prob, Y_hat, logits, h = model(graph)

        y_pred = Y_hat.squeeze().cpu().item()
        y_prob = Y_prob.squeeze().cpu().numpy()

        row = {
            "patient_id": pid,
            "y_true": label,
            "y_pred": y_pred,
        }
        for c in range(n_classes):
            row[f"y_prob_{c}"] = float(y_prob[c]) if c < len(y_prob) else 0.0
        results.append(row)

        # Save intermediate node data for potential attention analysis later
        # h_path after path_phi: needed for manual attention extraction
        # Also save centroids for spatial scatter plots
        graph_x = graph["x"]
        centroids = graph["centroid"].cpu().numpy() if "centroid" in graph.keys() else None

        # Extract node embeddings after GCN layers + phi transform
        # Replicate forward pass up to path_phi to get node-level embeddings
        with torch.no_grad():
            x = model.fc(graph["x"])
            x_ = x
            if model.edge_agg == "spatial":
                edge_index = graph["edge_index"]
            else:
                edge_index = graph.get("edge_latent", graph["edge_index"])
            edge_attr = graph.get("edge_features", None) if model.include_edge_features else None

            x = model.layers[0].conv(x_, edge_index, edge_attr)
            x_ = torch.cat([x_, x], axis=1)
            for layer in model.layers[1:]:
                x = layer(x, edge_index, edge_attr)
                x_ = torch.cat([x_, x], axis=1)

            h_path = model.path_phi(x_)

        node_data[pid] = {
            "node_embeddings": h_path.cpu().numpy(),  # [N_nodes, hidden_dim*num_layers]
            "centroids": centroids,                     # [N_nodes, 2]
            "y_true": label,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

    predictions_df = pd.DataFrame(results)
    return predictions_df, node_data


# ---------------------------------------------------------------------------
# Inference: NCA (VGG16 + attention) models
# ---------------------------------------------------------------------------

def run_nca_inference(model, graph_dir, gt_df, task_name, n_classes,
                      dataset_name, patient_id_extractor, device,
                      test_ids=None):
    """Run non-context-aware (VGG16 + attention MIL) inference.

    NCA models use: model.milAggregation(graph.x) -> model.classifier(embedding)
    The milAggregation with attention returns (embedding, attention_weights).

    Returns:
        predictions_df: DataFrame with patient_id, y_true, y_pred, y_prob_*
        attention_data: Dict of patient_id -> {attention_weights, centroids}
    """
    label_col = f"label_{n_classes}class"

    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])
    results = []
    attention_data = {}

    for gfile in graph_files:
        pid = patient_id_extractor(gfile)

        if test_ids is not None and pid not in test_ids:
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
        graph_features = graph["x"].to(device)
        centroids = graph["centroid"].cpu().numpy() if "centroid" in graph.keys() else None

        with torch.no_grad():
            # NCA inference path (from evaluate_BCNB_gcns.py)
            # milAggregation with attention returns (embedding, attention_weights)
            mil_agg = model.milAggregation
            if mil_agg.aggregation == "attention":
                embedding, attention_weights = mil_agg.attention_pooling(graph_features)
                attn_w = attention_weights.cpu().numpy()
            else:
                embedding = mil_agg(graph_features)
                attn_w = None

            logits = model.classifier(embedding)
            Y_prob = F.softmax(logits, dim=0)
            Y_hat = torch.argmax(logits).item()

        y_prob = Y_prob.cpu().numpy()

        row = {
            "patient_id": pid,
            "y_true": label,
            "y_pred": Y_hat,
        }
        for c in range(n_classes):
            row[f"y_prob_{c}"] = float(y_prob[c]) if c < len(y_prob) else 0.0
        results.append(row)

        attention_data[pid] = {
            "attention_weights": attn_w,  # [N_nodes, 1] or None
            "centroids": centroids,        # [N_nodes, 2]
            "y_true": label,
            "y_pred": Y_hat,
            "y_prob": y_prob,
        }

    predictions_df = pd.DataFrame(results)
    return predictions_df, attention_data


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_bcnb_inference(model_type, device):
    """Run inference on BCNB test set for all tasks.

    BCNB uses a fixed train/val/test split. We only run inference on the test set.
    """
    gt_df = load_bcnb_gt()
    test_ids = get_bcnb_test_patient_ids()
    print(f"BCNB test set: {len(test_ids)} patients")

    models_config = GCN_MODELS if model_type == "ca" else NCA_MODELS
    weights_dir = GCN_WEIGHTS if model_type == "ca" else NCA_WEIGHTS
    model_label = "CA" if model_type == "ca" else "NCA"

    for task_key, config in models_config.items():
        task_name = config["task"]
        n_classes = config["n_classes"]
        knn = config.get("knn", 19)  # NCA defaults to 19

        print(f"\n--- BCNB {model_label} {task_key} (task={task_name}, k={knn}) ---")

        # Load model (use per-task weights_dir if specified, else default)
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

        # Find graph directory
        graph_dir = find_graph_dir(BCNB_GRAPHS, task_name, knn)
        if graph_dir is None:
            print(f"  ERROR: graph dir not found for {task_name} k={knn}")
            continue
        print(f"  Graphs: {graph_dir}")

        # Run inference
        if model_type == "ca":
            preds_df, extra_data = run_ca_inference(
                model, graph_dir, gt_df, task_name, n_classes,
                "BCNB", extract_bcnb_patient_id, device,
                test_ids=test_ids
            )
        else:
            preds_df, extra_data = run_nca_inference(
                model, graph_dir, gt_df, task_name, n_classes,
                "BCNB", extract_bcnb_patient_id, device,
                test_ids=test_ids
            )

        if len(preds_df) == 0:
            print("  WARNING: no predictions generated")
            continue

        # Save predictions CSV
        pred_path = os.path.join(RESULTS_DIR, "predictions", f"BCNB_{task_key}_{model_label}_predictions.csv")
        preds_df.to_csv(pred_path, index=False)
        print(f"  Saved: {pred_path} ({len(preds_df)} patients)")

        # Save extra data (node embeddings or attention weights)
        extra_path = os.path.join(RESULTS_DIR, "attention", f"BCNB_{task_key}_{model_label}_node_data.pkl")
        with open(extra_path, "wb") as f:
            pickle.dump(extra_data, f)
        print(f"  Saved: {extra_path}")

        # Quick summary
        acc = (preds_df["y_true"] == preds_df["y_pred"]).mean()
        print(f"  Accuracy: {acc:.4f} ({len(preds_df)} samples)")

        # Free GPU memory
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_sbc_inference(model_type, device, feature_type="vgg16"):
    """Run inference on SBC dataset.

    SBC uses 5-fold CV. For VGG16 models trained on BCNB, we run on ALL
    non-excluded SBC patients (cross-domain transfer). For CONCH models,
    we use the predefined fold structure.
    """
    gt_df = load_sbc_gt()
    print(f"SBC non-excluded patients: {len(gt_df)}")

    models_config = GCN_MODELS if model_type == "ca" else NCA_MODELS
    weights_dir = GCN_WEIGHTS if model_type == "ca" else NCA_WEIGHTS
    model_label = "CA" if model_type == "ca" else "NCA"

    graphs_base = SBC_GRAPHS

    for task_key, config in models_config.items():
        task_name = config["task"]
        n_classes = config["n_classes"]
        knn = config.get("knn", 19)

        print(f"\n--- SBC {model_label} {task_key} (task={task_name}, k={knn}) ---")

        # Load model (same BCNB-trained model, cross-domain)
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

        # Find graph directory
        graph_dir = find_graph_dir(graphs_base, task_name, knn)
        if graph_dir is None:
            print(f"  ERROR: graph dir not found for {task_name} k={knn}")
            continue
        print(f"  Graphs: {graph_dir}")

        # Run inference on all non-excluded patients
        if model_type == "ca":
            preds_df, extra_data = run_ca_inference(
                model, graph_dir, gt_df, task_name, n_classes,
                "SBC", extract_sbc_patient_id, device,
                test_ids=None  # All non-excluded patients
            )
        else:
            preds_df, extra_data = run_nca_inference(
                model, graph_dir, gt_df, task_name, n_classes,
                "SBC", extract_sbc_patient_id, device,
                test_ids=None
            )

        if len(preds_df) == 0:
            print("  WARNING: no predictions generated")
            continue

        # Save predictions CSV
        pred_path = os.path.join(RESULTS_DIR, "predictions", f"SBC_{task_key}_{model_label}_predictions.csv")
        preds_df.to_csv(pred_path, index=False)
        print(f"  Saved: {pred_path} ({len(preds_df)} patients)")

        # Save extra data
        extra_path = os.path.join(RESULTS_DIR, "attention", f"SBC_{task_key}_{model_label}_node_data.pkl")
        with open(extra_path, "wb") as f:
            pickle.dump(extra_data, f)
        print(f"  Saved: {extra_path}")

        # Quick summary
        acc = (preds_df["y_true"] == preds_df["y_pred"]).mean()
        print(f"  Accuracy: {acc:.4f} ({len(preds_df)} samples)")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate per-patient predictions for CMPB revision"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for inference (default: cpu)"
    )
    parser.add_argument(
        "--models", type=str, default="both",
        choices=["ca", "nca", "both"],
        help="Which model types to run (default: both)"
    )
    parser.add_argument(
        "--datasets", type=str, default="both",
        choices=["bcnb", "sbc", "both"],
        help="Which datasets to run (default: both)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check paths and configs, don't load models"
    )
    return parser.parse_args()


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
    for label, path in [("BCNB GT", BCNB_GT), ("SBC GT", SBC_GT)]:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {label}: {path}")

    print()

    # Check graph directories
    for ds_label, base, tasks_knn in [
        ("BCNB VGG16", BCNB_GRAPHS, [(t, c["knn"]) for c in GCN_MODELS.values() for t in [c["task"]]]),
        ("SBC VGG16", SBC_GRAPHS, [(t, c["knn"]) for c in GCN_MODELS.values() for t in [c["task"]]]),
    ]:
        for task, knn in tasks_knn:
            gdir = find_graph_dir(base, task, knn)
            if gdir:
                nfiles = len([f for f in os.listdir(gdir) if f.endswith("_graph.pt")])
                print(f"  [OK] {ds_label} {task} k={knn}: {nfiles} graphs")
            else:
                print(f"  [MISSING] {ds_label} {task} k={knn}")
                ok = False

    print()

    # Check splits
    for label, path in [("BCNB splits", BCNB_SPLITS), ("SBC folds", SBC_FOLDS)]:
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] {label}: {path}")

    print()

    # Check output dirs
    for subdir in ["predictions", "attention", "topology", "clinical", "interpretability"]:
        path = os.path.join(RESULTS_DIR, subdir)
        exists = os.path.exists(path)
        status = "OK" if exists else "MISSING"
        if not exists:
            ok = False
        print(f"  [{status}] Output dir: {path}")

    print()
    # PyTorch version check
    print(f"  PyTorch version: {torch.__version__}")
    if _applied_patches:
        print(f"  [OK] Compatibility patches applied: {', '.join(_applied_patches)}")
    else:
        print("  [OK] No compatibility patches needed")

    print()
    print(f"Overall: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    return ok


def main():
    args = parse_args()

    print("=" * 70)
    print("CMPB Revision - Generate Predictions (S1.1)")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {args.device}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print()

    if args.dry_run:
        dry_run_check()
        return

    # Show compatibility info
    if _applied_patches:
        print(f"Compatibility patches: {', '.join(_applied_patches)}")
        print()

    device = torch.device(args.device)

    # Ensure output dirs exist
    for subdir in ["predictions", "attention"]:
        os.makedirs(os.path.join(RESULTS_DIR, subdir), exist_ok=True)

    # Run inference
    model_types = ["ca", "nca"] if args.models == "both" else [args.models]
    datasets = ["bcnb", "sbc"] if args.datasets == "both" else [args.datasets]

    for model_type in model_types:
        for dataset in datasets:
            print(f"\n{'=' * 70}")
            print(f"Running {model_type.upper()} on {dataset.upper()}")
            print(f"{'=' * 70}")

            if dataset == "bcnb":
                run_bcnb_inference(model_type, device)
            elif dataset == "sbc":
                run_sbc_inference(model_type, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
