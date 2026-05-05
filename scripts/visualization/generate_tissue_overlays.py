"""
generate_tissue_overlays.py  --  CMPB-D-25-07046 Revision (R1.1)

Generates tissue overlay visualizations with a modular panel/layer system.
Each panel is independently selectable via --panels. Overlay layers
(annotations, edges, nodes) can be toggled independently.

Available panels:
  hne            Raw H&E image only
  graph          H&E + graph structure (nodes + KNN edges)
  nca_attention  H&E + NCA attention weight heatmap
  ca_gradient    H&E + CA gradient importance heatmap (input-level)
  tissue_seg     H&E + tissue segmentation composition overlay

Overlay layers (drawn on panels as configured):
  --annotations / --no-annotations  Pathologist polygons (default: on)
  --edges / --no-edges              KNN edges (default: on for graph, off elsewhere)
  --nodes / --no-nodes              Node center dots (default: on)

Usage:
    pip install -r requirements.txt  # see repository root
    python generate_tissue_overlays.py --task 2class --patients 6 984 304
    python generate_tissue_overlays.py --panels graph nca_attention ca_gradient tissue_seg
    python generate_tissue_overlays.py --panels hne graph --no-annotations --dpi 200
"""

import sys
import os
import argparse
import json
import pickle
import types
import importlib
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

Image.MAX_IMAGE_PIXELS = None  # BCNB images are ~285MP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap

# ---------------------------------------------------------------------------
# PyTorch compatibility patches (same as generate_predictions.py)
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

# Add CODE_DIR for model imports
# [REPLACED by _paths.py] CODE_DIR = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}"
# [REPLACED by _paths.py] MOLSUB_CODE = "/Users/kckj099/Documents/Programming/molsub_article/code"
# [REPLACED by _paths.py] sys.path.insert(0, MOLSUB_CODE)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB_ROOT    = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] BCNB_GRAPHS    = f"{MOLSUB_ROOT}/data/BCNB/results_graphs_november_23"
# [REPLACED by _paths.py] BCNB_IMAGES    = "/Users/kckj099/Documents/CMPB-Review/data/BCNB/images"
# [REPLACED by _paths.py] BCNB_ANNOTATIONS = "/Users/kckj099/Documents/CMPB-Review/data/BCNB/annotations"
# [REPLACED by _paths.py] RESULTS_DIR    = "/Users/kckj099/Documents/CMPB-Review/results"
PRED_DIR       = f"{RESULTS_DIR}/predictions"
ATTN_DIR       = f"{RESULTS_DIR}/attention"
OUT_DIR        = f"{RESULTS_DIR}/interpretability"

TISSUE_COMP_DIR = f"{MOLSUB_ROOT}/data/BCNB/patches_paths_class_perc"

PATCH_SIZE = 512  # pixels in the JPEG coordinate system

# Valid panel types
VALID_PANELS = {"hne", "graph", "nca_attention", "ca_attention", "ca_gradient", "tissue_seg"}
DEFAULT_PANELS = ["graph", "nca_attention", "ca_attention"]

# Task -> graph directory suffix
TASK_GRAPH_DIRS = {
    "2class": "graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
    "3class": "graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
    "4class": "graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
}

# Task name alias for 4-class directory lookup
_TASK_NAME_ALIASES = {
    "4class": "graphs_PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0",
}

TASK_CLASS_NAMES = {
    "2class": {0: "Other", 1: "TNBC"},
    "3class": {0: "Luminal", 1: "HER2(+)", 2: "TNBC"},
    "4class": {0: "Luminal A", 1: "Luminal B", 2: "HER2(+)", 3: "TNBC"},
}

# GCN model configs (for gradient-based and attention-based importance)
# NOTE: 3-class model was retrained (Session 3b, 2026-04-29) from GINConv to GENConv
# with attention pooling, selected via MC-CV (5-fold x 3 repeats). The retrained model
# has REAL attention weights (like NCA), enabling direct attention extraction in addition
# to gradient/GNNExplainer methods. See results/retrained_models/3class_retrain_provenance.json.
# [REPLACED by _paths.py] GCN_WEIGHTS_ORIGINAL = f"{MOLSUB_ROOT}/data/gcn_pretrained_models"
# [REPLACED by _paths.py] GCN_WEIGHTS_RETRAINED = "/Users/kckj099/Documents/CMPB-Review/results/retrained_models"
GCN_MODELS = {
    "2class": {
        "weights_dir": GCN_WEIGHTS_ORIGINAL,
        "filename": "[24_11_2023]_GCN_Final_BCNB_OTHERvsTNBC_GT_GENConv_GL_5_KNN_19_EA_spatial_EF_False_GP_mean_DO_True_LR_1e-05.pth",
        "n_classes": 2, "pooling": "mean", "gnn_layer_type": "GENConv", "num_layers": 5,
        "has_attention": False,
    },
    "3class": {
        "weights_dir": GCN_WEIGHTS_RETRAINED,
        "filename": "3class_GENConv_5L_attn_lr2e5_final.pth",
        "n_classes": 3, "pooling": "attention", "gnn_layer_type": "GENConv", "num_layers": 5,
        "has_attention": True,  # Retrained model uses gated attention pooling
    },
    "4class": {
        "weights_dir": GCN_WEIGHTS_ORIGINAL,
        "filename": "[24_11_2023]_GCN_Final_BCNB_LUMINALAvsLUMINALBvsHER2vsTNBC_GT_GENConv_GL_4_KNN_25_GP_max_LR_2e-05_Optim_adam_OWD_1e-05_CVFold_0.pth",
        "n_classes": 4, "pooling": "max", "gnn_layer_type": "GENConv", "num_layers": 4,
        "has_attention": False,
    },
}

# Tissue segmentation color scheme (all 5 classes from TSM)
TISSUE_COLORS = {
    0: (0.85, 0.85, 0.85),    # Other/Background: light gray
    1: (1.0, 0.0, 0.0),       # Tumor: red
    2: (0.0, 0.0, 1.0),       # Stroma: blue
    3: (0.0, 0.8, 0.0),       # Inflammation/TILs: green
    4: (0.2, 0.2, 0.2),       # Necrosis: dark gray
    "mixed": (0.7, 0.5, 0.9), # Mixed (no dominant >30%): light purple
}

TISSUE_LABELS = {
    0: "Other/Background",
    1: "Tumor",
    2: "Stroma",
    3: "Inflammation",
    4: "Necrosis",
    "mixed": "Mixed",
}


# ---------------------------------------------------------------------------
# GCN model loading
# ---------------------------------------------------------------------------

def load_gcn_model(task_key):
    """Load a GCN model using state_dict reconstruction.

    For the retrained 3-class model (saved as full model via torch.save),
    loads directly. For original models (saved as pickled objects),
    extracts state_dict and reconstructs.
    """
    from MIL_models import PatchGCN_MeanMax_LSelec
    config = GCN_MODELS[task_key]
    weights_dir = config.get("weights_dir", GCN_WEIGHTS_ORIGINAL)
    model_path = os.path.join(weights_dir, config["filename"])

    old_model = torch.load(model_path, map_location="cpu", weights_only=False)

    # Handle both formats: full model object or state_dict
    if isinstance(old_model, dict):
        state_dict = old_model
    elif hasattr(old_model, "state_dict"):
        state_dict = old_model.state_dict()
        del old_model
    else:
        raise ValueError(f"Unexpected model format: {type(old_model)}")

    new_model = PatchGCN_MeanMax_LSelec(
        num_features=512, num_layers=config["num_layers"], hidden_dim=128,
        n_classes=config["n_classes"], pooling=config["pooling"],
        gnn_layer_type=config["gnn_layer_type"],
    )
    new_model.load_state_dict(state_dict, strict=True)
    new_model.eval()

    if config.get("has_attention"):
        print(f"  NOTE: {task_key} CA model uses attention pooling. "
              f"Direct attention extraction available via path_attention_head hook.")

    return new_model


def compute_gradient_importance(model, graph, predicted_class):
    """Compute per-node gradient importance for a CA (GCN) model.

    Computes the gradient of the predicted class logit w.r.t. the INPUT
    node features (before GCN layers). The L2 norm of each node's gradient
    measures how sensitive the prediction is to that node's features.

    This works even with mean pooling (where post-GCN gradients are uniform)
    because the gradient flows back through the GCN layers, which create
    node-dependent non-linear transformations based on graph structure.

    Analogous to input-level GradCAM: captures which tissue patches the
    GCN's spatial message-passing deems most relevant for classification.

    Returns:
        importance: np.ndarray of shape (N_nodes,), normalized to [0, 1]
    """
    model.eval()

    # Make input features require grad
    x_input = graph["x"].clone().detach().requires_grad_(True)
    edge_index = graph["edge_index"]
    edge_attr = graph.get("edge_features", None) if model.include_edge_features else None

    # Forward through fc
    x = model.fc(x_input)
    x_ = x.clone()

    # Forward through GCN layers
    x = model.layers[0].conv(x_, edge_index, edge_attr)
    x_ = torch.cat([x_, x], axis=1)
    for layer in model.layers[1:]:
        x = layer(x, edge_index, edge_attr)
        x_ = torch.cat([x_, x], axis=1)

    h_path = model.path_phi(x_)

    # Global pooling
    if model.pooling == "mean":
        pooled = torch.mean(h_path, dim=0, keepdim=True)
    elif model.pooling == "max":
        pooled = torch.max(h_path, dim=0, keepdim=True)[0]
    else:
        pooled = torch.mean(h_path, dim=0, keepdim=True)

    # Classifier
    logits = model.path_rho(pooled)

    # Backpropagate to input features
    logits[0, predicted_class].backward()

    # Per-node importance = L2 norm of input gradient
    grad = x_input.grad  # [N_nodes, 512]
    if grad is None:
        return None
    importance = grad.norm(dim=1).detach().numpy()

    # Normalize to [0, 1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance


# ---------------------------------------------------------------------------
# GNNExplainer (PyG native, optimization-based, saturation-proof)
# ---------------------------------------------------------------------------

class PatchGCNExplainerWrapper(torch.nn.Module):
    """Wrapper to make PatchGCN compatible with PyG Explainer API.

    PyG's Explainer expects model(x, edge_index, **kwargs) -> logits.
    PatchGCN expects a graph Data object and returns (Y_prob, Y_hat, logits, h).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        from torch_geometric.data import Data
        graph = Data(x=x, edge_index=edge_index)
        # Add edge features if the model expects them
        if hasattr(self.model, 'include_edge_features') and self.model.include_edge_features:
            if 'edge_features' in kwargs:
                graph.edge_features = kwargs['edge_features']
        Y_prob, Y_hat, logits, h = self.model(graph)
        # logits shape varies: could be [n_classes] or [1, n_classes]
        # Explainer expects [1, n_classes] (batch dim = 1)
        if logits.dim() == 1:
            return logits.unsqueeze(0)
        elif logits.dim() == 2 and logits.shape[0] == 1:
            return logits
        else:
            return logits.view(1, -1)


_gnnexplainer_instance = None  # Cache to avoid re-creating per patient


def compute_gnnexplainer_importance(model, graph, predicted_class, epochs=200, lr=0.01):
    """Compute per-node importance using GNNExplainer (PyG native).

    GNNExplainer learns soft masks on nodes by optimizing mutual information
    between the prediction and masked input. This is optimization-based
    (not gradient-based) so it does NOT suffer from gradient saturation
    for high-confidence predictions.

    Args:
        model: loaded PatchGCN model
        graph: PyG Data object with x, edge_index
        predicted_class: int, the class to explain
        epochs: optimization steps (default 200)
        lr: learning rate for mask optimization

    Returns:
        importance: np.ndarray (N_nodes,), normalized to [0, 1]
    """
    global _gnnexplainer_instance

    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

    if _gnnexplainer_instance is None:
        wrapped = PatchGCNExplainerWrapper(model)
        _gnnexplainer_instance = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=epochs, lr=lr),
            explanation_type="model",
            model_config=ModelConfig(
                mode="multiclass_classification",
                task_level="graph",
                return_type="raw",
            ),
            node_mask_type="object",
            edge_mask_type=None,
        )

    # explanation_type="model" uses the model's own prediction as target,
    # so we don't pass an explicit target
    explanation = _gnnexplainer_instance(
        x=graph["x"],
        edge_index=graph["edge_index"],
    )

    node_mask = explanation.node_mask
    if node_mask is None:
        return None

    importance = node_mask.detach().cpu().numpy().flatten()
    # Normalize to [0, 1]
    importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
    return importance


# ---------------------------------------------------------------------------
# Graph directory resolution
# ---------------------------------------------------------------------------

def find_graph_dir(task_key, knn=19):
    """Find the graph directory for a task."""
    base = BCNB_GRAPHS
    target = TASK_GRAPH_DIRS.get(task_key)
    if target:
        d = os.path.join(base, target, f"graphs_k_{knn}")
        if os.path.exists(d):
            return d
    # Try alias
    alias = _TASK_NAME_ALIASES.get(task_key)
    if alias:
        for dirname in os.listdir(base):
            if alias in dirname:
                d = os.path.join(base, dirname, f"graphs_k_{knn}")
                if os.path.exists(d):
                    return d
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(task_key):
    """Load CA and NCA predictions for a task."""
    ca_path = os.path.join(PRED_DIR, f"BCNB_{task_key}_CA_predictions.csv")
    nca_path = os.path.join(PRED_DIR, f"BCNB_{task_key}_NCA_predictions.csv")
    ca_df = pd.read_csv(ca_path)
    nca_df = pd.read_csv(nca_path)
    ca_df["patient_id"] = ca_df["patient_id"].astype(int).astype(str)
    nca_df["patient_id"] = nca_df["patient_id"].astype(int).astype(str)
    return ca_df, nca_df


def load_node_data(task_key):
    """Load attention/embedding pkl files for CA and NCA."""
    ca_path = os.path.join(ATTN_DIR, f"BCNB_{task_key}_CA_node_data.pkl")
    nca_path = os.path.join(ATTN_DIR, f"BCNB_{task_key}_NCA_node_data.pkl")
    with open(ca_path, "rb") as f:
        ca_data = pickle.load(f)
    with open(nca_path, "rb") as f:
        nca_data = pickle.load(f)
    return ca_data, nca_data


def load_tissue_composition():
    """Load and merge tissue segmentation composition from all three split CSVs.

    Returns:
        dict keyed by (patient_id_str, row, col) -> {
            'class_perc_0': float, ..., 'class_perc_4': float,
            'dominant_tissue': int or 'mixed'
        }
    """
    frames = []
    for split in ("train", "val", "test"):
        csv_path = os.path.join(TISSUE_COMP_DIR, f"{split}_patches_class_perc_0_tp.csv")
        if os.path.exists(csv_path):
            frames.append(pd.read_csv(csv_path))
        else:
            print(f"  WARNING: tissue composition CSV not found: {csv_path}")

    if not frames:
        print("  ERROR: no tissue composition CSVs found")
        return {}

    df = pd.concat(frames, ignore_index=True)
    print(f"  Loaded tissue composition: {len(df)} patches from {len(frames)} splits")

    tissue_map = {}
    for _, row in df.iterrows():
        path = row["patch_path"]
        # Parse patient_id, row_idx, col_idx from filename
        # Format: .../patient_id/patient_id_row_col.jpg
        # Handle both Windows (backslash) and Unix (forward slash) paths
        filename = path.replace("\\", "/").split("/")[-1]
        parts = filename.replace(".jpg", "").split("_")
        if len(parts) < 3:
            continue
        patient_id = parts[0]
        try:
            row_idx = int(parts[1])
            col_idx = int(parts[2])
        except ValueError:
            continue

        # Determine dominant tissue class (including background/other as class 0)
        percs = {
            0: row["class_perc_0"],  # other/background
            1: row["class_perc_1"],  # tumor
            2: row["class_perc_2"],  # stroma
            3: row["class_perc_3"],  # inflammation
            4: row["class_perc_4"],  # necrosis
        }
        max_class = max(percs, key=percs.get)
        dominant = max_class if percs[max_class] > 0.3 else "mixed"

        tissue_map[(patient_id, row_idx, col_idx)] = {
            "class_perc_0": row["class_perc_0"],
            "class_perc_1": row["class_perc_1"],
            "class_perc_2": row["class_perc_2"],
            "class_perc_3": row["class_perc_3"],
            "class_perc_4": row["class_perc_4"],
            "dominant_tissue": dominant,
        }

    return tissue_map


def load_patient_data(pid, task_key, graph_dir, ca_data, nca_data,
                      gcn_model=None, ca_pred_class=None, ca_method="gradient"):
    """Load all data for a single patient.

    Args:
        gcn_model: loaded GCN model for gradient importance (if None, falls back to embedding norm)
        ca_pred_class: predicted class index for gradient computation
    """
    # Image
    img_path = os.path.join(BCNB_IMAGES, f"{pid}.jpg")
    if not os.path.exists(img_path):
        return None
    img = Image.open(img_path)

    # Graph
    graph_path = os.path.join(graph_dir, f"{pid}_graph.pt")
    if not os.path.exists(graph_path):
        return None
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    # Annotations
    ann_path = os.path.join(BCNB_ANNOTATIONS, f"{pid}.json")
    annotations = {"positive": [], "negative": []}
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            annotations = json.load(f)

    # NCA attention weights (from pre-saved pkl)
    nca_attn = None
    if pid in nca_data and nca_data[pid]["attention_weights"] is not None:
        nca_attn = nca_data[pid]["attention_weights"].flatten()
        nca_attn = (nca_attn - nca_attn.min()) / (nca_attn.max() - nca_attn.min() + 1e-8)

    # CA importance: gradient-based if model available, else fall back to embedding norm
    ca_importance = None
    ca_importance_label = "CA Embedding Norm"

    if gcn_model is not None and ca_pred_class is not None:
        try:
            if ca_method == "gnnexplainer":
                importance = compute_gnnexplainer_importance(gcn_model, graph, ca_pred_class)
                if importance is not None:
                    ca_importance = importance
                    ca_importance_label = "CA GNNExplainer"
            else:
                importance = compute_gradient_importance(gcn_model, graph, ca_pred_class)
                if importance is not None:
                    ca_importance = importance
                    ca_importance_label = "CA Gradient Importance"
        except Exception as e:
            print(f"    CA importance computation failed for {pid}: {e}")

    # Fall back to embedding norm if gradient failed
    if ca_importance is None and pid in ca_data and ca_data[pid]["node_embeddings"] is not None:
        emb = ca_data[pid]["node_embeddings"]
        ca_importance = np.linalg.norm(emb, axis=1)
        ca_importance = (ca_importance - ca_importance.min()) / (ca_importance.max() - ca_importance.min() + 1e-8)
        ca_importance_label = "CA Embedding Norm (fallback)"

    # CA direct attention (from attention-pooling models, pre-extracted pkl)
    ca_direct_attn = None
    if pid in ca_data and "attention_weights" in ca_data[pid]:
        ca_direct_attn = ca_data[pid]["attention_weights"].flatten()
        ca_direct_attn = (ca_direct_attn - ca_direct_attn.min()) / (ca_direct_attn.max() - ca_direct_attn.min() + 1e-8)

    return {
        "image": np.array(img),
        "centroids": graph["centroid"].numpy(),
        "edge_index": graph["edge_index"].numpy(),
        "annotations": annotations,
        "nca_attention": nca_attn,
        "ca_attention": ca_direct_attn,
        "ca_importance": ca_importance,
        "ca_importance_label": ca_importance_label,
    }


# ---------------------------------------------------------------------------
# Overlay layer helpers (shared across panels)
# ---------------------------------------------------------------------------

def draw_annotations(ax, annotations):
    """Draw pathologist annotation polygons on an axes."""
    for ann_region in annotations.get("positive", []):
        verts = np.array(ann_region["vertices"])
        polygon = plt.Polygon(verts, fill=False, edgecolor="lime",
                              linewidth=2, linestyle="--")
        ax.add_patch(polygon)


def draw_edges(ax, centroids, edge_index, color="cyan", alpha=0.15, linewidth=0.3):
    """Draw KNN edges between node centroids."""
    for i in range(edge_index.shape[1]):
        src, tgt = edge_index[0, i], edge_index[1, i]
        sx = centroids[src, 1] * PATCH_SIZE + PATCH_SIZE // 2
        sy = centroids[src, 0] * PATCH_SIZE + PATCH_SIZE // 2
        tx = centroids[tgt, 1] * PATCH_SIZE + PATCH_SIZE // 2
        ty = centroids[tgt, 0] * PATCH_SIZE + PATCH_SIZE // 2
        ax.plot([sx, tx], [sy, ty], color=color, alpha=alpha, linewidth=linewidth)


def draw_nodes(ax, centroids, values=None, cmap_name=None, norm=None):
    """Draw node center dots on an axes.

    If values/cmap_name provided, color nodes by value. Otherwise cyan dots.
    """
    node_x = centroids[:, 1] * PATCH_SIZE + PATCH_SIZE // 2
    node_y = centroids[:, 0] * PATCH_SIZE + PATCH_SIZE // 2

    if values is not None and cmap_name is not None:
        if cmap_name == "custom_alpha":
            cmap_obj = LinearSegmentedColormap.from_list(
                "red_alpha", [(0.2, 0.2, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)], N=256
            )
        else:
            cmap_obj = plt.get_cmap(cmap_name)
        ax.scatter(node_x, node_y, c=values, cmap=cmap_obj, s=8, alpha=0.8,
                   edgecolors="none", norm=norm)
    else:
        ax.scatter(node_x, node_y, c="cyan", s=8, alpha=0.7, edgecolors="none")


# ---------------------------------------------------------------------------
# Panel rendering functions
# ---------------------------------------------------------------------------

def draw_hne_panel(ax, img_arr, patient_data, title, layer_cfg):
    """Panel: Raw H&E image only."""
    ax.imshow(img_arr)

    if layer_cfg["annotations"]:
        draw_annotations(ax, patient_data["annotations"])
    if layer_cfg["nodes"]:
        draw_nodes(ax, patient_data["centroids"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def draw_graph_panel(ax, img_arr, patient_data, title, layer_cfg):
    """Panel: H&E + graph structure (nodes as dots, KNN edges as lines)."""
    ax.imshow(img_arr)

    # Edges are ON by default for the graph panel (unless user passed --no-edges)
    if layer_cfg["edges"]:
        draw_edges(ax, patient_data["centroids"], patient_data["edge_index"],
                   color="cyan", alpha=0.15, linewidth=0.3)

    if layer_cfg["nodes"]:
        draw_nodes(ax, patient_data["centroids"])

    if layer_cfg["annotations"]:
        draw_annotations(ax, patient_data["annotations"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def _draw_heatmap_panel(ax, img_arr, patient_data, values, title, layer_cfg,
                        cmap_name, colorbar_label, alpha_mode="fixed"):
    """Shared logic for heatmap-style panels (NCA attention, CA gradient)."""
    ax.imshow(img_arr)

    if values is not None:
        norm = Normalize(vmin=0, vmax=1)

        if cmap_name == "custom_alpha":
            base_cmap = LinearSegmentedColormap.from_list(
                "red_alpha", [(0.2, 0.2, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)], N=256
            )
        else:
            base_cmap = plt.get_cmap(cmap_name)

        for i, (row, col) in enumerate(patient_data["centroids"]):
            px, py = int(col * PATCH_SIZE), int(row * PATCH_SIZE)
            color = base_cmap(norm(values[i]))

            if alpha_mode == "scaled":
                patch_alpha = 0.1 + 0.6 * values[i]
            else:
                patch_alpha = 0.5

            rect = mpatches.Rectangle(
                (px, py), PATCH_SIZE, PATCH_SIZE,
                alpha=patch_alpha, facecolor=color, edgecolor="none"
            )
            ax.add_patch(rect)

        # Inset colorbar (skip for individual panels to keep dimensions uniform)
        if not layer_cfg.get("no_panel_labels", False):
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            sm = cm.ScalarMappable(cmap=base_cmap, norm=norm)
            sm.set_array([])
            cax = inset_axes(ax, width="3%", height="40%", loc="upper right",
                             borderpad=1.0)
            cbar = plt.colorbar(sm, cax=cax, label=colorbar_label)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(colorbar_label, fontsize=9)

        # Edges are OFF by default for heatmap panels (unless user passed --edges)
        if layer_cfg["edges"]:
            draw_edges(ax, patient_data["centroids"], patient_data["edge_index"],
                       color="white", alpha=0.15, linewidth=0.3)

        if layer_cfg["nodes"]:
            draw_nodes(ax, patient_data["centroids"], values=values,
                       cmap_name=cmap_name, norm=norm)
    else:
        # No values available; draw plain
        if layer_cfg["edges"]:
            draw_edges(ax, patient_data["centroids"], patient_data["edge_index"])
        if layer_cfg["nodes"]:
            draw_nodes(ax, patient_data["centroids"])

    if layer_cfg["annotations"]:
        draw_annotations(ax, patient_data["annotations"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


def draw_nca_panel(ax, img_arr, patient_data, title, layer_cfg,
                   cmap_name="coolwarm", alpha_mode="fixed"):
    """Panel: H&E + NCA attention weight heatmap."""
    _draw_heatmap_panel(
        ax, img_arr, patient_data,
        values=patient_data["nca_attention"],
        title=title, layer_cfg=layer_cfg,
        cmap_name=cmap_name, colorbar_label="Attention Weight",
        alpha_mode=alpha_mode,
    )


def draw_ca_panel(ax, img_arr, patient_data, title, layer_cfg,
                  cmap_name="coolwarm", alpha_mode="fixed"):
    """Panel: H&E + CA gradient importance heatmap (input-level gradients)."""
    ca_label = patient_data.get("ca_importance_label", "CA Importance")
    _draw_heatmap_panel(
        ax, img_arr, patient_data,
        values=patient_data["ca_importance"],
        title=title, layer_cfg=layer_cfg,
        cmap_name=cmap_name, colorbar_label=ca_label,
        alpha_mode=alpha_mode,
    )


def draw_ca_attention_panel(ax, img_arr, patient_data, title, layer_cfg,
                            cmap_name="coolwarm", alpha_mode="fixed"):
    """Panel: H&E + CA direct attention weights from gated attention pooling.

    Only available for models with attention pooling (all tasks after Session 4
    standardization). Uses pre-extracted attention from
    results/attention/BCNB_{task}_CA_attention_direct.pkl.
    """
    _draw_heatmap_panel(
        ax, img_arr, patient_data,
        values=patient_data["ca_attention"],
        title=title, layer_cfg=layer_cfg,
        cmap_name=cmap_name, colorbar_label="CA Attention Weight",
        alpha_mode=alpha_mode,
    )


def _get_tissue_percs(patch_info):
    """Extract all 5 tissue percentages (including background) from patch info."""
    return {
        0: patch_info["class_perc_0"],  # other/background
        1: patch_info["class_perc_1"],  # tumor
        2: patch_info["class_perc_2"],  # stroma
        3: patch_info["class_perc_3"],  # inflammation
        4: patch_info["class_perc_4"],  # necrosis
    }


def _tissue_color_dominant(patch_info):
    """Mode 'dominant': single color for the dominant class (>30% threshold).
    Falls back to 'mixed' if no class exceeds 30%."""
    percs = _get_tissue_percs(patch_info)
    max_class = max(percs, key=percs.get)
    if percs[max_class] > 0.30:
        return TISSUE_COLORS[max_class], "none", 0
    return TISSUE_COLORS["mixed"], "none", 0


def _tissue_color_blend(patch_info):
    """Mode 'blend': proportional color mixing weighted by tissue percentages.
    Each tissue class contributes its color proportionally."""
    percs = _get_tissue_percs(patch_info)
    total = sum(percs.values())
    if total < 1e-6:
        return (0.5, 0.5, 0.5), "none", 0  # gray for empty

    r, g, b = 0.0, 0.0, 0.0
    for cls_id, pct in percs.items():
        weight = pct / total
        tc = TISSUE_COLORS[cls_id]
        r += weight * tc[0]
        g += weight * tc[1]
        b += weight * tc[2]
    return (min(r, 1.0), min(g, 1.0), min(b, 1.0)), "none", 0


def _tissue_color_border(patch_info):
    """Mode 'border': fill with dominant class color, border with second class.
    If dominant >50%, solid fill + second-type border.
    If dominant 30-50%, lighter fill + second-type border.
    If dominant <30%, gray fill."""
    percs = _get_tissue_percs(patch_info)
    sorted_classes = sorted(percs, key=percs.get, reverse=True)
    top_class = sorted_classes[0]
    second_class = sorted_classes[1]
    top_pct = percs[top_class]

    if top_pct > 0.30:
        fill_color = TISSUE_COLORS[top_class]
        border_color = TISSUE_COLORS[second_class]
        border_width = 3 if percs[second_class] > 0.20 else 1.5
        return fill_color, border_color, border_width
    return TISSUE_COLORS["mixed"], "none", 0


def _tissue_draw_stacked(ax, px, py, patch_info):
    """Mode 'stacked': subdivide the patch into vertical strips proportional
    to each tissue class percentage. Like a horizontal stacked bar chart
    inside each patch square."""
    percs = _get_tissue_percs(patch_info)
    total = sum(percs.values())
    if total < 1e-6:
        rect = mpatches.Rectangle((px, py), PATCH_SIZE, PATCH_SIZE,
                                   alpha=0.4, facecolor=(0.5, 0.5, 0.5), edgecolor="none")
        ax.add_patch(rect)
        return

    # Sort by percentage descending so largest class is leftmost
    sorted_classes = sorted(percs.items(), key=lambda x: -x[1])
    x_offset = 0
    for cls_id, pct in sorted_classes:
        if pct < 0.01:
            continue  # skip negligible classes
        width = (pct / total) * PATCH_SIZE
        color = TISSUE_COLORS[cls_id]
        rect = mpatches.Rectangle(
            (px + x_offset, py), width, PATCH_SIZE,
            alpha=0.4, facecolor=color, edgecolor="none"
        )
        ax.add_patch(rect)
        x_offset += width


TISSUE_MODE_FUNCS = {
    "dominant": _tissue_color_dominant,
    "blend": _tissue_color_blend,
    "border": _tissue_color_border,
    # "stacked" is handled separately in draw_tissue_seg_panel (draws multiple rects)
}


def draw_tissue_seg_panel(ax, img_arr, patient_data, title, layer_cfg,
                          tissue_map=None, patient_id=None, tissue_mode="dominant"):
    """Panel: H&E + tissue segmentation composition overlay.

    tissue_mode:
      'dominant': single color per dominant class (30% threshold)
      'blend': proportional color mixing by tissue percentages
      'border': dominant fill + second-type border color
    """
    ax.imshow(img_arr)

    color_func = TISSUE_MODE_FUNCS.get(tissue_mode, _tissue_color_dominant)

    if tissue_map is not None and patient_id is not None:
        matched = 0
        unmatched = 0
        for row, col in patient_data["centroids"]:
            row_i, col_i = int(row), int(col)
            key = (patient_id, row_i, col_i)
            patch_info = tissue_map.get(key)

            if patch_info is None:
                unmatched += 1
                continue

            matched += 1
            px, py = int(col * PATCH_SIZE), int(row * PATCH_SIZE)

            if tissue_mode == "stacked":
                _tissue_draw_stacked(ax, px, py, patch_info)
            else:
                fill_color, border_color, border_width = color_func(patch_info)
                rect = mpatches.Rectangle(
                    (px, py), PATCH_SIZE, PATCH_SIZE,
                    alpha=0.4, facecolor=fill_color,
                    edgecolor=border_color if border_color != "none" else "none",
                    linewidth=border_width,
                )
                ax.add_patch(rect)

        if unmatched > 0:
            print(f"    tissue_seg: {matched} matched, {unmatched} unmatched patches")

        # Legend (all 5 tissue classes)
        legend_patches = []
        for cls_id in (0, 1, 2, 3, 4):
            legend_patches.append(
                mpatches.Patch(color=TISSUE_COLORS[cls_id],
                               label=TISSUE_LABELS[cls_id])
            )
        if tissue_mode == "dominant":
            legend_patches.append(
                mpatches.Patch(color=TISSUE_COLORS["mixed"],
                               label="Mixed (<30%)")
            )
        elif tissue_mode == "border":
            legend_patches.append(
                mpatches.Patch(facecolor=(0.8, 0.8, 0.8),
                               edgecolor=(0.0, 0.0, 1.0), linewidth=2,
                               label="Border = 2nd tissue")
            )
        ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
                  framealpha=0.7, edgecolor="gray")

    # Edges (off by default for tissue seg, same as heatmap panels)
    if layer_cfg["edges"]:
        draw_edges(ax, patient_data["centroids"], patient_data["edge_index"],
                   color="white", alpha=0.15, linewidth=0.3)

    if layer_cfg["nodes"]:
        draw_nodes(ax, patient_data["centroids"])

    if layer_cfg["annotations"]:
        draw_annotations(ax, patient_data["annotations"])

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def build_panel_titles(pid, panels, task_key, ca_pred_row, nca_pred_row, patient_data):
    """Build title strings for each panel."""
    class_names = TASK_CLASS_NAMES[task_key]
    true_label = class_names[int(ca_pred_row.y_true)]
    ca_pred_label = class_names[int(ca_pred_row.y_pred)]
    nca_pred_label = class_names[int(nca_pred_row.y_pred)]
    ca_correct = "correct" if ca_pred_row.y_true == ca_pred_row.y_pred else "WRONG"
    nca_correct = "correct" if nca_pred_row.y_true == nca_pred_row.y_pred else "WRONG"
    ca_label = patient_data.get("ca_importance_label", "CA Importance")

    titles = {}
    for panel in panels:
        if panel == "hne":
            titles[panel] = f"H&E (Patient {pid})\nTrue: {true_label}"
        elif panel == "graph":
            titles[panel] = f"H&E + Graph (Patient {pid})\nTrue: {true_label}"
        elif panel == "nca_attention":
            titles[panel] = f"NCA Attention\nPred: {nca_pred_label} ({nca_correct})"
        elif panel == "ca_attention":
            titles[panel] = f"CA Attention\nPred: {ca_pred_label} ({ca_correct})"
        elif panel == "ca_gradient":
            titles[panel] = f"{ca_label}\nPred: {ca_pred_label} ({ca_correct})"
        elif panel == "tissue_seg":
            titles[panel] = f"Tissue Segmentation\nTrue: {true_label}"
    return titles


def generate_patient_overlay(pid, task_key, patient_data, ca_pred_row, nca_pred_row,
                              panels, layer_cfg, cmap_name="coolwarm",
                              alpha_mode="fixed", dpi=150, tissue_map=None,
                              tissue_mode="dominant"):
    """Generate a multi-panel overlay figure for one patient.

    Number of panels and figure width scale with the panels list.
    """
    n_panels = len(panels)
    img_arr = patient_data["image"]

    # Compute biopsy bounding box from centroids BEFORE creating figure,
    # so figure dimensions adapt to the biopsy's aspect ratio.
    centroids = patient_data["centroids"]
    rows_c, cols_c = centroids[:, 0], centroids[:, 1]
    pad = 2  # grid units of padding around the biopsy
    y_min = max(0, (rows_c.min() - pad) * PATCH_SIZE)
    y_max = min(img_arr.shape[0], (rows_c.max() + pad + 1) * PATCH_SIZE)
    x_min = max(0, (cols_c.min() - pad) * PATCH_SIZE)
    x_max = min(img_arr.shape[1], (cols_c.max() + pad + 1) * PATCH_SIZE)

    biopsy_w = x_max - x_min
    biopsy_h = y_max - y_min
    aspect = biopsy_w / biopsy_h if biopsy_h > 0 else 1.0

    # Each panel matches the biopsy aspect ratio; fixed panel height of 10 inches
    panel_height = 10
    panel_width = panel_height * aspect
    fig_width = panel_width * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, panel_height))

    # Handle single-panel case (axes is not a list)
    if n_panels == 1:
        axes = [axes]

    titles = build_panel_titles(pid, panels, task_key, ca_pred_row, nca_pred_row, patient_data)

    # Panel letter labels for manuscript figures
    panel_letters = [chr(ord('a') + i) for i in range(n_panels)]

    # Build per-panel layer config:
    # edges default ON for graph, OFF for everything else
    for i, panel in enumerate(panels):
        ax = axes[i]
        title = titles[panel]

        # Per-panel edge config: if user explicitly set edges, use that;
        # otherwise default ON for graph, OFF for others
        panel_layer_cfg = dict(layer_cfg)
        if not layer_cfg.get("edges_explicit"):
            panel_layer_cfg["edges"] = (panel == "graph")

        if panel == "hne":
            draw_hne_panel(ax, img_arr, patient_data, title, panel_layer_cfg)
        elif panel == "graph":
            draw_graph_panel(ax, img_arr, patient_data, title, panel_layer_cfg)
        elif panel == "nca_attention":
            draw_nca_panel(ax, img_arr, patient_data, title, panel_layer_cfg,
                           cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "ca_attention":
            draw_ca_attention_panel(ax, img_arr, patient_data, title, panel_layer_cfg,
                                   cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "ca_gradient":
            draw_ca_panel(ax, img_arr, patient_data, title, panel_layer_cfg,
                          cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "tissue_seg":
            draw_tissue_seg_panel(ax, img_arr, patient_data, title, panel_layer_cfg,
                                  tissue_map=tissue_map, patient_id=pid,
                                  tissue_mode=tissue_mode)

        # Panel letter labels (can be disabled with --no-panel-labels)
        if not layer_cfg.get("no_panel_labels", False):
            ax.text(0.02, 0.98, f"({panel_letters[i]})", transform=ax.transAxes,
                    fontsize=16, fontweight="bold", va="top", ha="left",
                    color="white", bbox=dict(boxstyle="round,pad=0.2",
                    facecolor="black", alpha=0.7))

    # Crop all panels to the biopsy bounding box (computed before figure creation)
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # inverted y for image coordinates

    fig.subplots_adjust(wspace=0.08, left=0.01, right=0.99,
                        top=0.92, bottom=0.01)

    # Save
    class_names = TASK_CLASS_NAMES[task_key]
    true_label = class_names[int(ca_pred_row.y_true)]
    ca_correct = "correct" if ca_pred_row.y_true == ca_pred_row.y_pred else "WRONG"
    tag = f"{true_label.replace(' ', '_')}_{ca_correct}"
    tissue_suffix = f"_{tissue_mode}" if tissue_mode != "dominant" else ""
    save_png = os.path.join(OUT_DIR, f"tissue_overlay_{task_key}_{pid}_{tag}{tissue_suffix}.png")
    plt.savefig(save_png, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Saved: {save_png}")

    # Individual panel mode: save each panel as a separate plain image
    if layer_cfg.get("individual_panels", False):
        _save_individual_panels(
            pid, task_key, patient_data, ca_pred_row, nca_pred_row,
            panels, layer_cfg, cmap_name, alpha_mode, dpi, tissue_map,
            tissue_mode, tag, tissue_suffix,
            x_min, x_max, y_min, y_max
        )

    return save_png


def _save_individual_panels(pid, task_key, patient_data, ca_pred_row, nca_pred_row,
                             panels, layer_cfg, cmap_name, alpha_mode, dpi,
                             tissue_map, tissue_mode, tag, tissue_suffix,
                             x_min, x_max, y_min, y_max):
    """Save each panel as its own standalone image (no title, no labels, no embedded colorbar)."""
    img_arr = patient_data["image"]

    # Determine figure size from biopsy bbox
    biopsy_w = x_max - x_min
    biopsy_h = y_max - y_min
    aspect = biopsy_w / biopsy_h if biopsy_h > 0 else 1.0
    panel_height = 8
    panel_width = panel_height * aspect

    # Subdirectory for individual panels
    panel_dir = os.path.join(OUT_DIR, f"panels_{task_key}_{pid}")
    os.makedirs(panel_dir, exist_ok=True)

    # Suppress titles, labels, colorbars for individual panels
    plain_layer_cfg = dict(layer_cfg)
    plain_layer_cfg["no_panel_labels"] = True

    for i, panel in enumerate(panels):
        fig_single, ax = plt.subplots(1, 1, figsize=(panel_width, panel_height))

        # Per-panel edge config
        panel_layer_cfg = dict(plain_layer_cfg)
        if not plain_layer_cfg.get("edges_explicit"):
            panel_layer_cfg["edges"] = (panel == "graph")

        # Draw panel WITHOUT title (pass empty string)
        if panel == "hne":
            draw_hne_panel(ax, img_arr, patient_data, "", panel_layer_cfg)
        elif panel == "graph":
            draw_graph_panel(ax, img_arr, patient_data, "", panel_layer_cfg)
        elif panel == "nca_attention":
            draw_nca_panel(ax, img_arr, patient_data, "", panel_layer_cfg,
                           cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "ca_attention":
            draw_ca_attention_panel(ax, img_arr, patient_data, "", panel_layer_cfg,
                                   cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "ca_gradient":
            draw_ca_panel(ax, img_arr, patient_data, "", panel_layer_cfg,
                          cmap_name=cmap_name, alpha_mode=alpha_mode)
        elif panel == "tissue_seg":
            draw_tissue_seg_panel(ax, img_arr, patient_data, "", panel_layer_cfg,
                                  tissue_map=tissue_map, patient_id=pid,
                                  tissue_mode=tissue_mode)

        # Crop to biopsy bounding box
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)
        ax.axis("off")

        # Fixed margins: no bbox_inches="tight" to ensure all panels are
        # identical dimensions regardless of colorbar/legend presence
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        panel_path = os.path.join(panel_dir, f"{panel}{tissue_suffix}.png")
        fig_single.savefig(panel_path, dpi=dpi, pad_inches=0)
        plt.close(fig_single)

    print(f"  Saved individual panels: {panel_dir}/")


# ---------------------------------------------------------------------------
# Patient selection
# ---------------------------------------------------------------------------

def select_representative_patients(ca_df, nca_df, nca_data, task_key, max_patients=6):
    """Select representative patients for visualization.

    Strategy: pick one correctly classified and one misclassified per
    clinically important subtype, prioritizing patients with annotations
    and images available.
    """
    class_names = TASK_CLASS_NAMES[task_key]
    selected = []

    for y_true in sorted(class_names.keys()):
        label = class_names[y_true]

        # Correctly classified by CA
        correct = ca_df[(ca_df.y_true == y_true) & (ca_df.y_pred == y_true)]
        for _, row in correct.iterrows():
            pid = row.patient_id
            if (os.path.exists(os.path.join(BCNB_IMAGES, f"{pid}.jpg")) and
                pid in nca_data):
                selected.append({"pid": pid, "type": f"{label}_correct"})
                break

        # Misclassified by CA
        wrong = ca_df[(ca_df.y_true == y_true) & (ca_df.y_pred != y_true)]
        for _, row in wrong.iterrows():
            pid = row.patient_id
            if (os.path.exists(os.path.join(BCNB_IMAGES, f"{pid}.jpg")) and
                pid in nca_data):
                selected.append({"pid": pid, "type": f"{label}_wrong"})
                break

    return selected[:max_patients]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate tissue overlay visualizations for BCNB patients"
    )
    parser.add_argument(
        "--task", type=str, default="2class",
        choices=["2class", "3class", "4class"],
        help="Classification task (default: 2class)"
    )
    parser.add_argument(
        "--patients", nargs="+", type=str, default=None,
        help="Specific patient IDs to visualize (default: auto-select)"
    )
    parser.add_argument(
        "--max-patients", type=int, default=6,
        help="Max patients to auto-select (default: 6)"
    )
    parser.add_argument(
        "--panels", nargs="+", type=str, default=None,
        choices=sorted(VALID_PANELS),
        help="Panel types to render (default: graph nca_attention ca_attention)"
    )
    parser.add_argument(
        "--cmap", type=str, default="coolwarm",
        choices=["coolwarm", "RdYlBu_r", "jet", "hot", "custom_alpha"],
        help="Colormap for heatmaps (default: coolwarm)"
    )
    parser.add_argument(
        "--alpha-mode", type=str, default="fixed",
        choices=["fixed", "scaled"],
        help="Alpha mode: 'fixed' (uniform 0.5) or 'scaled' (proportional to value)"
    )
    parser.add_argument(
        "--knn", type=int, default=19,
        help="KNN value for graph directory (default: 19)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for output PNG (default: 150)"
    )

    # Overlay layer toggles
    parser.add_argument(
        "--annotations", dest="annotations", action="store_true", default=True,
        help="Draw pathologist annotation polygons (default: on)"
    )
    parser.add_argument(
        "--no-annotations", dest="annotations", action="store_false",
        help="Suppress pathologist annotation polygons"
    )
    parser.add_argument(
        "--edges", dest="edges", action="store_true", default=None,
        help="Draw KNN edges on all panels"
    )
    parser.add_argument(
        "--no-edges", dest="edges", action="store_false",
        help="Suppress KNN edges on all panels"
    )
    parser.add_argument(
        "--nodes", dest="nodes", action="store_true", default=True,
        help="Draw node center dots (default: on)"
    )
    parser.add_argument(
        "--no-nodes", dest="nodes", action="store_false",
        help="Suppress node center dots"
    )
    parser.add_argument(
        "--ca-method", type=str, default="gradient",
        choices=["gradient", "gnnexplainer"],
        help="CA importance method: 'gradient' (input-level gradient, fast but "
             "saturates for high-confidence predictions) or 'gnnexplainer' "
             "(optimization-based, saturation-proof, slower). Default: gradient"
    )
    parser.add_argument(
        "--tissue-mode", type=str, default="dominant",
        choices=["dominant", "blend", "border", "stacked"],
        help="Tissue seg coloring: 'dominant' (single color, 30%% threshold), "
             "'blend' (proportional color mixing), "
             "'border' (dominant fill + second-type border), "
             "'stacked' (vertical strips proportional to class %%). Default: dominant"
    )
    parser.add_argument(
        "--no-panel-labels", dest="no_panel_labels", action="store_true", default=False,
        help="Suppress (a), (b), (c), (d) panel labels in upper-left corner"
    )
    parser.add_argument(
        "--individual-panels", dest="individual_panels", action="store_true", default=False,
        help="Save each panel as a separate plain image (no title, no labels) "
             "in a subdirectory panels_{task}_{pid}/. For LaTeX subfigure composition."
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    panels = args.panels if args.panels else DEFAULT_PANELS

    # Determine if user explicitly set edges
    edges_explicit = args.edges is not None
    edges_value = args.edges if edges_explicit else True  # default on for graph panel

    layer_cfg = {
        "annotations": args.annotations,
        "edges": edges_value,
        "edges_explicit": edges_explicit,
        "nodes": args.nodes,
        "no_panel_labels": args.no_panel_labels,
        "individual_panels": args.individual_panels,
    }

    print("=" * 70)
    print("CMPB Revision - Tissue Overlay Visualizations (R1.1)")
    print("=" * 70)
    print(f"Task: {args.task}")
    print(f"Panels: {' '.join(panels)}")
    print(f"Colormap: {args.cmap}")
    print(f"Alpha mode: {args.alpha_mode}")
    print(f"DPI: {args.dpi}")
    print(f"Layers: annotations={args.annotations}, "
          f"edges={'explicit=' + str(edges_value) if edges_explicit else 'per-panel default'}, "
          f"nodes={args.nodes}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    graph_dir = find_graph_dir(args.task, args.knn)
    if graph_dir is None:
        print(f"ERROR: graph directory not found for {args.task} k={args.knn}")
        return
    print(f"Graph dir: {graph_dir}")

    ca_df, nca_df = load_predictions(args.task)
    ca_data, nca_data = load_node_data(args.task)
    print(f"Loaded predictions: {len(ca_df)} CA, {len(nca_df)} NCA")
    print(f"Loaded node data: {len(ca_data)} CA, {len(nca_data)} NCA")

    # Load CA direct attention if ca_attention panel is requested
    if "ca_attention" in panels:
        ca_attn_pkl = os.path.join(ATTN_DIR, f"BCNB_{args.task}_CA_attention_direct.pkl")
        if os.path.exists(ca_attn_pkl):
            import pickle
            with open(ca_attn_pkl, "rb") as f:
                ca_attn_data = pickle.load(f)
            # Merge into ca_data (keyed by int pid in attn pkl, str pid in ca_data)
            for pid_int, attn_entry in ca_attn_data.items():
                pid_str = str(pid_int)
                if pid_str not in ca_data:
                    ca_data[pid_str] = {}
                ca_data[pid_str]["attention_weights"] = attn_entry["attention_weights"]
            print(f"  Loaded CA direct attention: {len(ca_attn_data)} patients from {ca_attn_pkl}")
        else:
            print(f"  WARNING: CA attention pkl not found: {ca_attn_pkl}")
            print(f"  Run: python scripts/extract_ca_attention_bcnb.py --tasks {args.task}")

    # Load GCN model only if ca_gradient panel is requested
    gcn_model = None
    if "ca_gradient" in panels:
        print("Loading GCN model for gradient importance...")
        try:
            gcn_model = load_gcn_model(args.task)
            print(f"  GCN model loaded: {type(gcn_model).__name__}")
        except Exception as e:
            print(f"  WARNING: Could not load GCN model ({e}). Falling back to embedding norm.")

    # Load tissue composition only if tissue_seg panel is requested
    tissue_map = None
    if "tissue_seg" in panels:
        print("Loading tissue composition data...")
        tissue_map = load_tissue_composition()

    # Select patients
    if args.patients:
        patients = [{"pid": p, "type": "manual"} for p in args.patients]
    else:
        patients = select_representative_patients(
            ca_df, nca_df, nca_data, args.task, args.max_patients
        )

    print(f"\nPatients to visualize: {len(patients)}")
    for p in patients:
        print(f"  {p['pid']} ({p['type']})")
    print()

    # Generate overlays
    for p_info in patients:
        pid = p_info["pid"]
        print(f"Processing patient {pid} ({p_info['type']})...")

        # Get predicted class for gradient computation
        ca_row_lookup = ca_df[ca_df.patient_id == pid]
        ca_pred_class = int(ca_row_lookup.iloc[0].y_pred) if len(ca_row_lookup) > 0 else None

        patient_data = load_patient_data(
            pid, args.task, graph_dir, ca_data, nca_data,
            gcn_model=gcn_model, ca_pred_class=ca_pred_class,
            ca_method=args.ca_method,
        )
        if patient_data is None:
            print(f"  SKIP: missing data for patient {pid}")
            continue

        ca_row = ca_df[ca_df.patient_id == pid]
        nca_row = nca_df[nca_df.patient_id == pid]
        if len(ca_row) == 0 or len(nca_row) == 0:
            print(f"  SKIP: no predictions for patient {pid}")
            continue

        generate_patient_overlay(
            pid=pid,
            task_key=args.task,
            patient_data=patient_data,
            ca_pred_row=ca_row.iloc[0],
            nca_pred_row=nca_row.iloc[0],
            panels=panels,
            layer_cfg=layer_cfg,
            cmap_name=args.cmap,
            alpha_mode=args.alpha_mode,
            dpi=args.dpi,
            tissue_map=tissue_map,
            tissue_mode=args.tissue_mode,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
