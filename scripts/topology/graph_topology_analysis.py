"""
graph_topology_analysis_v2.py  --  CMPB-D-25-07046 Revision, Session 2

Advanced graph topology analysis comparing BCNB vs SBC datasets.
Extends v1 with additional structural metrics, statistical tests, and
publication-quality visualizations.

Feeds downstream tasks:
  - R2.4: graph topology analysis explaining cross-domain performance drops
  - Supplements tissue micro-environment differences between BCNB and SBC

Metrics computed per graph (NEW items marked with *):
  --- Structural (from v1) ---
  - Node count, Edge count, Mean degree, Graph diameter, Connected components
  --- Structural (new) ---
  * Graph density
  * Average clustering coefficient
  * Degree assortativity
  * Betweenness centrality (mean, std)
  * Closeness centrality (mean)
  * Spectral gap (Fiedler value / algebraic connectivity)
  --- Feature-level (from v1) ---
  - Feature mean, std, L2 norm
  - Edge distance mean, std
  - Spatial extent

Statistical comparisons (all new):
  * Mann-Whitney U test per metric (with Bonferroni correction)
  * Cohen's d effect size per metric
  * Kolmogorov-Smirnov test on degree distributions
  * Wasserstein distance on feature distributions

Visualizations (upgraded):
  * Violin plots with overlaid strip plots (replaces boxplots)
  * KDE overlay plots for continuous distributions
  * Statistical summary heatmap (effect sizes)
  * Summary table with p-values and effect sizes

Usage:
    pip install -r requirements.txt  # see repository root
    python graph_topology_analysis_v2.py
    python graph_topology_analysis_v2.py --tasks 2class 3class
    python graph_topology_analysis_v2.py --k-values 19
"""

import sys
import os
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import networkx as nx
import matplotlib

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from scipy import stats as sp_stats
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] MOLSUB_ROOT    = "/Users/kckj099/Documents/Programming/molsub_article"
# [REPLACED by _paths.py] BCNB_GRAPHS    = f"{MOLSUB_ROOT}/data/BCNB/results_graphs_november_23"
SBC_GRAPHS = f"{MOLSUB_ROOT}/data/SBC/results_graphs_january_25"
SBC_CONCH  = f"{SBC_GRAPHS}/graphs_CONCH"
# [REPLACED by _paths.py] RESULTS_DIR    = "/Users/kckj099/Documents/CMPB-Review/results"

# 4-class task name has a typo in some directories (LAUMINALB vs LUMINALB).
# BCNB uses the correct spelling, SBC uses the typo. Try both.
_TASK_NAME_ALIASES = {
    "LUMINALAvsLAUMINALBvsHER2vsTNBC": "LUMINALAvsLUMINALBvsHER2vsTNBC",
}

# Task definitions
TASKS = {
    "2class": {
        "name": "OTHERvsTNBC",
        "display": "2-class (Other vs TNBC)",
    },
    "3class": {
        "name": "LUMINALSvsHER2vsTNBC",
        "display": "3-class (Lum vs HER2 vs TNBC)",
    },
    "4class": {
        "name": "LUMINALAvsLAUMINALBvsHER2vsTNBC",
        "display": "4-class (LumA vs LumB vs HER2 vs TNBC)",
    },
}


# ---------------------------------------------------------------------------
# Graph directory resolution (unchanged from v1)
# ---------------------------------------------------------------------------

def find_graph_dirs(base_path, task_name, k_values):
    """Find all matching graph directories for a task and k-values.

    Handles the 4-class task name typo: BCNB uses LUMINALB (correct),
    SBC uses LAUMINALB (typo). We try both spellings via _TASK_NAME_ALIASES.
    """
    dirs = {}
    if not os.path.exists(base_path):
        return dirs

    names_to_try = [task_name]
    if task_name in _TASK_NAME_ALIASES:
        names_to_try.append(_TASK_NAME_ALIASES[task_name])

    for dirname in os.listdir(base_path):
        dirpath = os.path.join(base_path, dirname)
        if not os.path.isdir(dirpath):
            continue
        for name in names_to_try:
            if name in dirname:
                for k in k_values:
                    k_dir = os.path.join(dirpath, f"graphs_k_{k}")
                    if os.path.exists(k_dir):
                        dirs[k] = k_dir
    return dirs


def find_conch_graph_dirs(task_name, k_values):
    """Find CONCH graph directories."""
    dirs = {}
    if not os.path.exists(SBC_CONCH):
        return dirs

    names_to_try = [task_name]
    if task_name in _TASK_NAME_ALIASES:
        names_to_try.append(_TASK_NAME_ALIASES[task_name])

    for dirname in os.listdir(SBC_CONCH):
        dirpath = os.path.join(SBC_CONCH, dirname)
        if not os.path.isdir(dirpath):
            continue
        for name in names_to_try:
            if name in dirname:
                for k in k_values:
                    k_dir = os.path.join(dirpath, f"graphs_k_{k}")
                    if os.path.exists(k_dir):
                        dirs[k] = k_dir
    return dirs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pyg_to_nx(edge_index, num_nodes):
    """Convert PyG edge_index tensor to an undirected NetworkX graph."""
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    edges = edge_index.t().numpy()
    G.add_edges_from(edges.tolist())
    return G


def compute_spectral_gap(G):
    """Compute the Fiedler value (algebraic connectivity) of the largest
    connected component. Uses sparse eigenvalue decomposition for efficiency.

    Returns NaN if the graph has fewer than 3 nodes in its largest CC.
    """
    if G.number_of_nodes() < 3:
        return np.nan

    # Use largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    if G.number_of_nodes() < 3:
        return np.nan

    try:
        L = nx.laplacian_matrix(G).astype(float)
        # We need the second smallest eigenvalue; compute 2 smallest
        eigenvalues = eigsh(L, k=2, which="SM", return_eigenvectors=False)
        eigenvalues.sort()
        return float(eigenvalues[1])
    except Exception:
        # Fallback: dense computation for small graphs
        try:
            L_dense = nx.laplacian_matrix(G).toarray().astype(float)
            evals = np.linalg.eigvalsh(L_dense)
            return float(evals[1])
        except Exception:
            return np.nan


# ---------------------------------------------------------------------------
# Topology computation (v2 - extended)
# ---------------------------------------------------------------------------

def compute_topology_stats(graph_path):
    """Compute extended topology statistics for a single graph.

    Returns stats dict or None if loading fails.
    """
    try:
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    except Exception:
        return None

    x = graph["x"]                   # [N_nodes, feature_dim]
    edge_index = graph["edge_index"] # [2, N_edges]
    num_nodes = x.shape[0]
    feature_dim = x.shape[1]
    num_edges = edge_index.shape[1]

    # --- Build NetworkX graph (undirected) ---
    G = pyg_to_nx(edge_index, num_nodes)

    # ===== V1 metrics =====
    mean_degree = num_edges / num_nodes if num_nodes > 0 else 0

    # Feature statistics
    feat_mean = x.mean().item()
    feat_std = x.std().item()
    feat_l2_norm_mean = torch.norm(x, dim=1).mean().item()

    # Connected components
    components = list(nx.connected_components(G))
    n_components = len(components)

    # Diameter (largest CC)
    largest_cc = max(components, key=len) if components else set()
    if len(largest_cc) > 1:
        G_lcc = G.subgraph(largest_cc)
        # Sample-based approximation for large graphs
        if len(largest_cc) > 500:
            sample_nodes = np.random.choice(
                list(largest_cc), size=min(50, len(largest_cc)), replace=False
            )
            diameter = 0
            for src in sample_nodes:
                lengths = nx.single_source_shortest_path_length(G_lcc, src)
                diameter = max(diameter, max(lengths.values()))
        else:
            diameter = nx.diameter(G_lcc)
    else:
        diameter = 0

    # Spatial extent
    if "centroid" in graph.keys():
        centroids = graph["centroid"]
        spatial_extent = torch.norm(
            centroids.max(dim=0).values - centroids.min(dim=0).values
        ).item()
    else:
        spatial_extent = np.nan

    # Edge feature statistics
    if "edge_features" in graph.keys():
        edge_feats = graph["edge_features"]
        edge_dist_mean = edge_feats.mean().item()
        edge_dist_std = edge_feats.std().item()
    else:
        edge_dist_mean = np.nan
        edge_dist_std = np.nan

    # ===== V2 new metrics =====

    # Graph density
    density = nx.density(G)

    # Average clustering coefficient
    avg_clustering = nx.average_clustering(G)

    # Degree assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G)
    except nx.NetworkXError:
        assortativity = np.nan

    # Betweenness centrality (mean + std over nodes)
    bc = nx.betweenness_centrality(G)
    bc_vals = np.array(list(bc.values()))
    bc_mean = bc_vals.mean()
    bc_std = bc_vals.std()

    # Closeness centrality (mean, on largest CC only for meaningful values)
    if len(largest_cc) > 1:
        cc = nx.closeness_centrality(G_lcc)
        cc_vals = np.array(list(cc.values()))
        closeness_mean = cc_vals.mean()
    else:
        closeness_mean = np.nan

    # Spectral gap (Fiedler value)
    spectral_gap = compute_spectral_gap(G)

    # Degree distribution stats (for per-graph summary)
    degrees = np.array([d for _, d in G.degree()])
    degree_std = degrees.std()
    degree_max = degrees.max() if len(degrees) > 0 else 0
    degree_skew = float(sp_stats.skew(degrees)) if len(degrees) > 2 else np.nan

    return {
        # V1 metrics
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "feature_dim": feature_dim,
        "mean_degree": mean_degree,
        "diameter": diameter,
        "n_components": n_components,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "feat_l2_norm_mean": feat_l2_norm_mean,
        "spatial_extent": spatial_extent,
        "edge_dist_mean": edge_dist_mean,
        "edge_dist_std": edge_dist_std,
        # V2 structural metrics
        "density": density,
        "avg_clustering": avg_clustering,
        "assortativity": assortativity,
        "betweenness_mean": bc_mean,
        "betweenness_std": bc_std,
        "closeness_mean": closeness_mean,
        "spectral_gap": spectral_gap,
        "degree_std": degree_std,
        "degree_max": int(degree_max),
        "degree_skew": degree_skew,
    }


# ---------------------------------------------------------------------------
# Dataset processing
# ---------------------------------------------------------------------------

def process_dataset(graph_dir, dataset_label, task_label, k_value, feature_type,
                    max_graphs=None):
    """Process all graphs in a directory and return stats DataFrame."""
    graph_files = sorted([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])

    if max_graphs is not None:
        graph_files = graph_files[:max_graphs]

    all_stats = []
    n_failed = 0

    for i, gfile in enumerate(graph_files):
        if (i + 1) % 50 == 0:
            print(f"    Processing {i + 1}/{len(graph_files)}...")

        s = compute_topology_stats(os.path.join(graph_dir, gfile))
        if s is None:
            n_failed += 1
            continue

        s["filename"] = gfile
        s["dataset"] = dataset_label
        s["task"] = task_label
        s["k_value"] = k_value
        s["feature_type"] = feature_type
        all_stats.append(s)

    if n_failed > 0:
        print(f"    WARNING: {n_failed} graphs failed to load")

    return pd.DataFrame(all_stats)


# ---------------------------------------------------------------------------
# Statistical comparisons
# ---------------------------------------------------------------------------

def cohens_d(group1, group2):
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def effect_size_label(d):
    """Interpret Cohen's d magnitude."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def run_statistical_comparisons(stats_df, dataset_a="BCNB", dataset_b="SBC",
                                feature_type="VGG16", k_value=19):
    """Run Mann-Whitney U, KS tests, and effect sizes for all topology metrics.

    Returns a DataFrame with one row per metric.
    """
    df = stats_df[
        (stats_df["feature_type"] == feature_type) &
        (stats_df["k_value"] == k_value)
    ]
    grp_a = df[df["dataset"] == dataset_a]
    grp_b = df[df["dataset"] == dataset_b]

    if len(grp_a) == 0 or len(grp_b) == 0:
        print(f"  No data for {dataset_a} vs {dataset_b} ({feature_type}, k={k_value})")
        return pd.DataFrame()

    metrics = [
        ("num_nodes", "Node count"),
        ("num_edges", "Edge count"),
        ("mean_degree", "Mean degree"),
        ("diameter", "Diameter"),
        ("n_components", "Connected components"),
        ("density", "Graph density"),
        ("avg_clustering", "Clustering coefficient"),
        ("assortativity", "Assortativity"),
        ("betweenness_mean", "Betweenness centrality (mean)"),
        ("closeness_mean", "Closeness centrality (mean)"),
        ("spectral_gap", "Spectral gap (Fiedler)"),
        ("degree_std", "Degree std"),
        ("degree_max", "Degree max"),
        ("degree_skew", "Degree skewness"),
        ("feat_mean", "Feature mean"),
        ("feat_std", "Feature std"),
        ("feat_l2_norm_mean", "Feature L2 norm"),
        ("spatial_extent", "Spatial extent"),
        ("edge_dist_mean", "Edge distance mean"),
        ("edge_dist_std", "Edge distance std"),
    ]

    results = []
    for col, label in metrics:
        a = grp_a[col].dropna()
        b = grp_b[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue

        # Mann-Whitney U
        u_stat, mw_p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")

        # Kolmogorov-Smirnov
        ks_stat, ks_p = sp_stats.ks_2samp(a, b)

        # Wasserstein distance
        w_dist = sp_stats.wasserstein_distance(a, b)

        # Effect size
        d = cohens_d(a.values, b.values)

        results.append({
            "metric": label,
            "column": col,
            f"{dataset_a}_mean": a.mean(),
            f"{dataset_a}_std": a.std(),
            f"{dataset_a}_median": a.median(),
            f"{dataset_b}_mean": b.mean(),
            f"{dataset_b}_std": b.std(),
            f"{dataset_b}_median": b.median(),
            "mann_whitney_U": u_stat,
            "mann_whitney_p": mw_p,
            "ks_statistic": ks_stat,
            "ks_p": ks_p,
            "wasserstein_dist": w_dist,
            "cohens_d": d,
            "effect_size": effect_size_label(d),
            "n_a": len(a),
            "n_b": len(b),
        })

    result_df = pd.DataFrame(results)

    # Bonferroni correction
    if len(result_df) > 0:
        n_tests = len(result_df)
        result_df["mw_p_bonferroni"] = np.minimum(
            result_df["mann_whitney_p"] * n_tests, 1.0
        )
        result_df["ks_p_bonferroni"] = np.minimum(
            result_df["ks_p"] * n_tests, 1.0
        )
        result_df["mw_significant"] = result_df["mw_p_bonferroni"] < 0.05
        result_df["ks_significant"] = result_df["ks_p_bonferroni"] < 0.05

    return result_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def create_violin_plots(stats_df, output_path, k_value=19):
    """Create violin plots comparing BCNB vs SBC (VGG16) for all
    structural and feature metrics."""
    df = stats_df[
        (stats_df["k_value"] == k_value) &
        (stats_df["feature_type"] == "VGG16")
    ].copy()

    if len(df) == 0:
        print(f"  No VGG16 data for k={k_value}")
        return

    metrics = [
        ("num_nodes", "Node Count"),
        ("mean_degree", "Mean Degree"),
        ("diameter", "Diameter"),
        ("density", "Graph Density"),
        ("avg_clustering", "Clustering Coeff."),
        ("assortativity", "Assortativity"),
        ("betweenness_mean", "Betweenness (mean)"),
        ("closeness_mean", "Closeness (mean)"),
        ("spectral_gap", "Spectral Gap"),
        ("feat_l2_norm_mean", "Feature L2 Norm"),
        ("spatial_extent", "Spatial Extent"),
        ("degree_skew", "Degree Skewness"),
    ]

    n_metrics = len(metrics)
    ncols = 4
    nrows = (n_metrics + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    palette = {"BCNB": "#4C72B0", "SBC": "#DD8452"}

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        plot_df = df[["dataset", metric]].dropna()

        if len(plot_df) == 0:
            ax.set_visible(False)
            continue

        sns.violinplot(
            data=plot_df, x="dataset", y=metric, ax=ax,
            palette=palette, inner="quartile", linewidth=1.0, cut=0,
        )
        # Overlay strip for individual points (subsampled)
        if len(plot_df) > 200:
            sample_df = plot_df.groupby("dataset", group_keys=False).apply(
                lambda g: g.sample(min(80, len(g)), random_state=42)
            )
        else:
            sample_df = plot_df
        sns.stripplot(
            data=sample_df, x="dataset", y=metric, ax=ax,
            color="0.25", alpha=0.25, size=1.5, jitter=True,
        )

        # Add Mann-Whitney p-value annotation
        a = plot_df[plot_df["dataset"] == "BCNB"][metric]
        b = plot_df[plot_df["dataset"] == "SBC"][metric]
        if len(a) > 1 and len(b) > 1:
            _, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
            d = cohens_d(a.values, b.values)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.set_title(f"{title}\n{sig}  d={d:.2f}", fontsize=10, fontweight="bold")
        else:
            ax.set_title(title, fontsize=10, fontweight="bold")

        ax.set_xlabel("")
        ax.set_ylabel("")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f"Graph Topology: BCNB vs SBC (k={k_value}, VGG16)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved violin plots: {output_path}")


def create_kde_overlay_plots(stats_df, output_path, k_value=19):
    """Create KDE overlay plots for key metrics (BCNB vs SBC)."""
    df = stats_df[
        (stats_df["k_value"] == k_value) &
        (stats_df["feature_type"] == "VGG16")
    ].copy()

    if len(df) == 0:
        return

    metrics = [
        ("num_nodes", "Node Count"),
        ("mean_degree", "Mean Degree"),
        ("avg_clustering", "Clustering Coefficient"),
        ("spectral_gap", "Spectral Gap (Fiedler)"),
        ("betweenness_mean", "Betweenness Centrality"),
        ("feat_l2_norm_mean", "Feature L2 Norm"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    palette = {"BCNB": "#4C72B0", "SBC": "#DD8452"}

    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        for ds, color in palette.items():
            vals = df[df["dataset"] == ds][metric].dropna()
            if len(vals) > 2:
                sns.kdeplot(vals, ax=ax, color=color, label=ds, fill=True, alpha=0.3)

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.legend(fontsize=9)

    plt.suptitle(
        f"Distribution Overlays: BCNB vs SBC (k={k_value}, VGG16)",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved KDE overlays: {output_path}")


def create_effect_size_heatmap(stat_results_df, output_path):
    """Create a heatmap of Cohen's d effect sizes for all metrics."""
    if len(stat_results_df) == 0:
        return

    plot_df = stat_results_df[["metric", "cohens_d", "mw_significant"]].copy()
    plot_df = plot_df.set_index("metric")

    fig, ax = plt.subplots(figsize=(6, max(4, len(plot_df) * 0.45)))

    # Color by effect size, annotate with significance
    values = plot_df["cohens_d"].values.reshape(-1, 1)
    im = ax.imshow(values, cmap="RdBu_r", aspect="auto",
                   vmin=-max(2, abs(values).max()),
                   vmax=max(2, abs(values).max()))

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df.index, fontsize=9)
    ax.set_xticks([0])
    ax.set_xticklabels(["Cohen's d"], fontsize=10)

    # Annotate cells
    for i, (d_val, sig) in enumerate(
        zip(plot_df["cohens_d"], plot_df["mw_significant"])
    ):
        marker = "*" if sig else ""
        ax.text(0, i, f"{d_val:.2f}{marker}", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if abs(d_val) > 1.0 else "black")

    plt.colorbar(im, ax=ax, label="Cohen's d (BCNB - SBC)")
    ax.set_title("Effect Sizes (* = significant after Bonferroni)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved effect size heatmap: {output_path}")


def create_feature_distribution_plots(stats_df, output_path, k_value=19):
    """Create feature distribution comparison between VGG16 and CONCH."""
    df = stats_df[stats_df["k_value"] == k_value].copy()
    if len(df) == 0:
        return

    feature_types = df["feature_type"].unique()
    if len(feature_types) < 2:
        print("  Skipping feature comparison: only one feature type available")
        return

    metrics = [
        ("feat_mean", "Feature Mean"),
        ("feat_std", "Feature Std"),
        ("feat_l2_norm_mean", "Mean Feature L2 Norm"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        plot_df = df[["dataset", "feature_type", metric]].dropna()
        plot_df["group"] = plot_df["dataset"] + "\n" + plot_df["feature_type"]

        if len(plot_df) > 0:
            sns.violinplot(
                data=plot_df, x="group", y=metric, ax=ax,
                palette="Set3", inner="quartile", cut=0,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle(
        f"Feature Distribution: VGG16 vs CONCH (k={k_value})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved feature comparison: {output_path}")


# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_statistical_summary(stat_results_df, dataset_a="BCNB", dataset_b="SBC"):
    """Print a formatted summary of statistical comparisons."""
    if len(stat_results_df) == 0:
        return

    print(f"\n{'='*80}")
    print(f"  Statistical Comparison: {dataset_a} vs {dataset_b}")
    print(f"{'='*80}")
    print(f"  {'Metric':<30} {'d':>7} {'Effect':>10} {'MW p':>10} {'Sig':>5}")
    print(f"  {'-'*30} {'-'*7} {'-'*10} {'-'*10} {'-'*5}")

    for _, row in stat_results_df.iterrows():
        sig = "YES" if row["mw_significant"] else "no"
        p_str = f"{row['mw_p_bonferroni']:.2e}" if row["mw_p_bonferroni"] < 0.01 else f"{row['mw_p_bonferroni']:.4f}"
        print(
            f"  {row['metric']:<30} {row['cohens_d']:>7.2f} "
            f"{row['effect_size']:>10} {p_str:>10} {sig:>5}"
        )

    n_sig = stat_results_df["mw_significant"].sum()
    n_total = len(stat_results_df)
    print(f"\n  {n_sig}/{n_total} metrics significantly different after Bonferroni correction")

    # Highlight largest effect sizes
    top = stat_results_df.reindex(
        stat_results_df["cohens_d"].abs().sort_values(ascending=False).index
    ).head(5)
    print(f"\n  Top 5 largest effect sizes:")
    for _, row in top.iterrows():
        direction = f"{dataset_a} > {dataset_b}" if row["cohens_d"] > 0 else f"{dataset_b} > {dataset_a}"
        print(f"    {row['metric']}: d={row['cohens_d']:.2f} ({direction})")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Advanced graph topology analysis for CMPB revision (v2)"
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["2class"],
        choices=["2class", "3class", "4class"],
        help="Which classification tasks to analyze (default: 2class)",
    )
    parser.add_argument(
        "--k-values", nargs="+", type=int, default=[19],
        help="KNN values to analyze (default: 19)",
    )
    parser.add_argument(
        "--max-graphs", type=int, default=None,
        help="Max graphs per dataset (for testing). None = all",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only check paths, don't process graphs",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CMPB Revision - Advanced Graph Topology Analysis v2")
    print("=" * 70)
    print(f"Tasks: {args.tasks}")
    print(f"K-values: {args.k_values}")
    if args.max_graphs:
        print(f"Max graphs per dataset: {args.max_graphs}")
    print()

    out_dir = os.path.join(RESULTS_DIR, "topology_v2")
    os.makedirs(out_dir, exist_ok=True)

    all_stats = []

    for task_key in args.tasks:
        task_cfg = TASKS[task_key]
        task_name = task_cfg["name"]
        task_display = task_cfg["display"]

        print(f"\n--- Task: {task_display} ---")

        for k in args.k_values:
            # BCNB VGG16
            bcnb_dirs = find_graph_dirs(BCNB_GRAPHS, task_name, [k])
            if k in bcnb_dirs:
                graph_dir = bcnb_dirs[k]
                n_graphs = len([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])
                print(f"\n  BCNB VGG16 k={k}: {n_graphs} graphs in {graph_dir}")
                if not args.dry_run:
                    df = process_dataset(
                        graph_dir, "BCNB", task_key, k, "VGG16",
                        max_graphs=args.max_graphs,
                    )
                    all_stats.append(df)
                    print(f"    Processed: {len(df)} graphs")
            else:
                print(f"  BCNB VGG16 k={k}: NOT FOUND")

            # SBC VGG16
            sbc_dirs = find_graph_dirs(SBC_GRAPHS, task_name, [k])
            if k in sbc_dirs:
                graph_dir = sbc_dirs[k]
                n_graphs = len([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])
                print(f"\n  SBC VGG16 k={k}: {n_graphs} graphs in {graph_dir}")
                if not args.dry_run:
                    df = process_dataset(
                        graph_dir, "SBC", task_key, k, "VGG16",
                        max_graphs=args.max_graphs,
                    )
                    all_stats.append(df)
                    print(f"    Processed: {len(df)} graphs")
            else:
                print(f"  SBC VGG16 k={k}: NOT FOUND")

            # SBC CONCH
            conch_dirs = find_conch_graph_dirs(task_name, [k])
            if k in conch_dirs:
                graph_dir = conch_dirs[k]
                n_graphs = len([f for f in os.listdir(graph_dir) if f.endswith("_graph.pt")])
                print(f"\n  SBC CONCH k={k}: {n_graphs} graphs in {graph_dir}")
                if not args.dry_run:
                    df = process_dataset(
                        graph_dir, "SBC", task_key, k, "CONCH",
                        max_graphs=args.max_graphs,
                    )
                    all_stats.append(df)
                    print(f"    Processed: {len(df)} graphs")
            else:
                print(f"  SBC CONCH k={k}: NOT FOUND")

    if args.dry_run:
        print("\nDry run complete. No graphs processed.")
        return

    if not all_stats:
        print("\nNo graphs processed. Check paths.")
        return

    # Combine all stats
    combined_df = pd.concat(all_stats, ignore_index=True)

    # Save raw stats
    raw_path = os.path.join(out_dir, "topology_stats_v2.csv")
    combined_df.to_csv(raw_path, index=False, float_format="%.6f")
    print(f"\nSaved raw stats: {raw_path} ({len(combined_df)} total graphs)")

    # ---- Statistical comparisons and visualizations (per task) ----
    for task_key in args.tasks:
        task_df = combined_df[combined_df["task"] == task_key]
        task_label = TASKS[task_key]["display"]

        for k in args.k_values:
            suffix = f"{task_key}_k{k}"

            # Statistical comparison (BCNB vs SBC, VGG16 only)
            stat_df = run_statistical_comparisons(
                task_df, "BCNB", "SBC", "VGG16", k
            )
            if len(stat_df) > 0:
                stat_path = os.path.join(out_dir, f"statistical_comparison_{suffix}.csv")
                stat_df.to_csv(stat_path, index=False, float_format="%.6f")
                print(f"\n  Saved statistical comparison ({task_label}): {stat_path}")
                print_statistical_summary(stat_df)

                # Effect size heatmap
                heatmap_path = os.path.join(out_dir, f"effect_size_heatmap_{suffix}.pdf")
                create_effect_size_heatmap(stat_df, heatmap_path)

            # Violin plots (VGG16 only)
            vgg16_df = task_df[task_df["feature_type"] == "VGG16"]
            if len(vgg16_df) > 0:
                violin_path = os.path.join(out_dir, f"violin_plots_{suffix}.pdf")
                create_violin_plots(vgg16_df, violin_path, k_value=k)

                kde_path = os.path.join(out_dir, f"kde_overlays_{suffix}.pdf")
                create_kde_overlay_plots(vgg16_df, kde_path, k_value=k)

            # Feature comparison (VGG16 vs CONCH, SBC only)
            sbc_df = task_df[task_df["dataset"] == "SBC"]
            if len(sbc_df["feature_type"].unique()) > 1:
                feat_path = os.path.join(out_dir, f"feature_comparison_{suffix}.pdf")
                create_feature_distribution_plots(sbc_df, feat_path, k_value=k)

    print("\nDone. All outputs in:", out_dir)


if __name__ == "__main__":
    main()
