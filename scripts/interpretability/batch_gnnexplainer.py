"""
batch_gnnexplainer.py -- Run GNNExplainer on all 218 BCNB test patients

Saves per-patient per-node importance scores + TSM tissue composition
for quantitative tissue-type correlation analysis (R1.1 validation).

Output: results/interpretability/gnnexplainer_{task}_importance.pkl
  Dict[patient_id] -> {importance: [N], tissue_comp: [N,5], centroid: [N,2],
                       y_true: int, y_pred: int, n_nodes: int}

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/batch_gnnexplainer.py --task 2class --device cpu
"""

import sys, os, argparse, pickle, time, types, importlib
import numpy as np
import torch
import torch.nn as nn

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402


# Compat patches
def _apply_compat_patches():
    patches = []
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
    return patches

_apply_compat_patches()

# [REPLACED by _paths.py] MOLSUB_ROOT = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] CODE_DIR = f"{MOLSUB_ROOT}/code"
# [REPLACED by _paths.py] sys.path.insert(0, CODE_DIR)

# [REPLACED by _paths.py] RESULTS_DIR = "/Users/kckj099/Documents/CMPB-Review/results"
SPLIT_DIR = f"{MOLSUB_ROOT}/data/BCNB/patches_paths_class_perc"
GT_FILE = f"{MOLSUB_ROOT}/data/BCNB/ground_truth/patient-clinical-data.xlsx"

import pandas as pd

GCN_MODELS = {
    "2class": {
        "weights_dir": f"{MOLSUB_ROOT}/data/gcn_pretrained_models",
        "filename": "[24_11_2023]_GCN_Final_BCNB_OTHERvsTNBC_GT_GENConv_GL_5_KNN_19_EA_spatial_EF_False_GP_mean_DO_True_LR_1e-05.pth",
        "n_classes": 2, "pooling": "mean", "gnn_layer_type": "GENConv", "num_layers": 5,
        "graph_subdir": "graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
    },
    "3class": {
        "weights_dir": f"{RESULTS_DIR}/retrained_models",
        "filename": "3class_GENConv_5L_attn_lr2e5_final.pth",
        "n_classes": 3, "pooling": "attention", "gnn_layer_type": "GENConv", "num_layers": 5,
        "graph_subdir": "graphs_PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
    },
    "4class": {
        "weights_dir": f"{MOLSUB_ROOT}/data/gcn_pretrained_models",
        "filename": "[24_11_2023]_GCN_Final_BCNB_LUMINALAvsLUMINALBvsHER2vsTNBC_GT_GENConv_GL_4_KNN_25_GP_max_LR_2e-05_Optim_adam_OWD_1e-05_CVFold_0.pth",
        "n_classes": 4, "pooling": "max", "gnn_layer_type": "GENConv", "num_layers": 4,
        "graph_subdir": "graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x",
    },
}

LABEL_MAPS = {
    "2class": {"Luminal A": 0, "Luminal B": 0, "HER2(+)": 0, "HER2 enriched": 0, "Triple negative": 1, "TNBC": 1},
    "3class": {"Luminal A": 0, "Luminal B": 0, "HER2(+)": 1, "HER2 enriched": 1, "Triple negative": 2, "TNBC": 2},
    "4class": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "HER2 enriched": 2, "Triple negative": 3, "TNBC": 3},
}


def get_test_ids():
    df = pd.read_csv(f"{SPLIT_DIR}/test_patches_class_perc_0_tp.csv")
    pids = df["patch_path"].str.extract(r"patches_512_fullWSIs_0/(\d+)/", expand=False)
    return set(pids.dropna().astype(int).unique())


def load_gt(task):
    gt = pd.read_excel(GT_FILE)
    gt = gt.rename(columns={"Patient ID": "patient_id", "Molecular subtype": "mol_subtype"})
    gt["label"] = gt["mol_subtype"].map(LABEL_MAPS[task])
    gt = gt.dropna(subset=["label"])
    gt["label"] = gt["label"].astype(int)
    return gt


def load_model(task, device):
    from MIL_models import PatchGCN_MeanMax_LSelec
    cfg = GCN_MODELS[task]
    path = os.path.join(cfg["weights_dir"], cfg["filename"])
    old = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(old, "state_dict"):
        sd = old.state_dict()
        del old
    else:
        sd = old
    model = PatchGCN_MeanMax_LSelec(
        num_features=512, num_layers=cfg["num_layers"], hidden_dim=128,
        n_classes=cfg["n_classes"], pooling=cfg["pooling"],
        gnn_layer_type=cfg["gnn_layer_type"],
    )
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model


def run_gnnexplainer(model, graph, predicted_class, epochs=200, lr=0.01):
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    from generate_tissue_overlays import PatchGCNExplainerWrapper

    wrapper = PatchGCNExplainerWrapper(model)
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs, lr=lr),
        explanation_type="model",
        node_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )

    x = graph["x"]
    edge_index = graph["edge_index"]
    explanation = explainer(x=x, edge_index=edge_index, target=torch.tensor([predicted_class]))
    node_mask = explanation.node_mask.squeeze()
    if node_mask.dim() > 1:
        node_mask = node_mask.mean(dim=1)
    nmin, nmax = node_mask.min(), node_mask.max()
    if nmax > nmin:
        node_mask = (node_mask - nmin) / (nmax - nmin)
    return node_mask.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="2class", choices=["2class", "3class", "4class"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gnnexplainer-epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg = GCN_MODELS[args.task]
    knn = 25 if args.task == "4class" else 19

    print(f"Task: {args.task}, Device: {device}", flush=True)

    # Load model
    print("Loading model...", flush=True)
    model = load_model(args.task, device)

    # Load test patient IDs and ground truth
    test_ids = get_test_ids()
    gt = load_gt(args.task)
    print(f"Test patients: {len(test_ids)}", flush=True)

    # Graph directory
    graph_base = f"{MOLSUB_ROOT}/data/BCNB/results_graphs_november_23/{cfg['graph_subdir']}/graphs_k_{knn}"
    print(f"Graphs: {graph_base}", flush=True)

    results = {}
    t0 = time.time()

    for i, pid in enumerate(sorted(test_ids)):
        gpath = os.path.join(graph_base, f"{pid}_graph.pt")
        if not os.path.exists(gpath):
            continue
        gt_row = gt[gt["patient_id"] == pid]
        if len(gt_row) == 0:
            continue
        label = gt_row["label"].values[0]

        graph = torch.load(gpath, map_location=device, weights_only=False)

        # Get prediction
        with torch.no_grad():
            Y_prob, Y_hat, _, _ = model(graph)
        pred = Y_hat.item()

        # Run GNNExplainer
        importance = run_gnnexplainer(model, graph, pred, epochs=args.gnnexplainer_epochs)

        # Get tissue composition (from graph node features' TSM percentages if available)
        centroid = graph["centroid"].cpu().numpy() if hasattr(graph, "centroid") and graph["centroid"] is not None else None

        results[pid] = {
            "importance": importance,
            "y_true": label,
            "y_pred": pred,
            "y_prob": Y_prob.squeeze().cpu().numpy(),
            "n_nodes": graph["x"].shape[0],
            "centroid": centroid,
        }

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(test_ids) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(test_ids)}] pid={pid} nodes={graph['x'].shape[0]} "
                  f"pred={pred} true={label} ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    # Save
    out_dir = os.path.join(RESULTS_DIR, "interpretability")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"gnnexplainer_{args.task}_importance.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.time() - t0
    print(f"\nDone. {len(results)} patients, {elapsed:.0f}s total.", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
