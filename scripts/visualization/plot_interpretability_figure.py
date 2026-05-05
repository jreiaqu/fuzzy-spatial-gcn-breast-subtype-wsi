"""
plot_interpretability_figure.py -- Generate manuscript-ready interpretability figure

Creates a grouped bar chart showing:
  - Overall biopsy tissue composition (baseline)
  - NCA most-attended patches tissue composition (50% attention mass)
  - CA most-attended patches tissue composition (50% attention mass)
  across all 3 classification tasks on the BCNB test set (217 patients).

Uses the same tissue color palette as the overlay visualizations for consistency:
  Tumor=red, Stroma=blue, Inflammation=green, Necrosis=dark gray.

Usage:
    pip install -r requirements.txt  # see repository root
    python scripts/plot_interpretability_figure.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# [REPLACED by _paths.py] RESULTS = "/Users/kckj099/Documents/CMPB-Review/results"
OUT_DIR = os.path.join(RESULTS, "interpretability")

# Consistent palette with generate_tissue_overlays.py
TISSUE_COLORS = {
    "other": (0.85, 0.85, 0.85),        # light gray
    "tumor": (1.0, 0.0, 0.0),           # red
    "stroma": (0.0, 0.0, 1.0),          # blue
    "inflammation": (0.0, 0.8, 0.0),    # green
    "necrosis": (0.2, 0.2, 0.2),        # dark gray
}
TISSUE_ORDER = ["tumor", "inflammation", "stroma", "necrosis", "other"]
TISSUE_LABELS = ["Tumor", "Inflammation", "Stroma", "Necrosis", "Other"]

TASK_LABELS = {
    "2class": "Binary\n(Other vs TNBC)",
    "3class": "Ternary\n(Luminals vs HER2+ vs TNBC)",
    "4class": "Quaternary\n(L-A vs L-B vs HER2+ vs TNBC)",
}


def load_data():
    """Load the attention tissue composition CSV."""
    path = os.path.join(OUT_DIR, "attention_tissue_composition_corrected.csv")
    return pd.read_csv(path)


def compute_summary(df):
    """Compute per-task summary: overall, NCA 50%, CA 50% tissue composition."""
    summary = {}
    for task in ["2class", "3class", "4class"]:
        dt = df[df["task"] == task]
        n = len(dt)

        overall = [dt[f"overall_{tn}"].mean() for tn in TISSUE_ORDER]
        nca_50 = [dt[f"nca_50pct_{tn}"].mean() for tn in TISSUE_ORDER]
        ca_50 = [dt[f"ca_50pct_{tn}"].mean() for tn in TISSUE_ORDER]

        # Concentration info for annotation
        nca_n = dt["nca_50pct_n_patches"].mean()
        nca_pct = dt["nca_50pct_pct_graph"].mean()
        ca_n = dt["ca_50pct_n_patches"].mean()
        ca_pct = dt["ca_50pct_pct_graph"].mean()

        summary[task] = {
            "overall": np.array(overall) * 100,
            "nca": np.array(nca_50) * 100,
            "ca": np.array(ca_50) * 100,
            "nca_info": f"n={nca_n:.0f} ({nca_pct:.0f}%)",
            "ca_info": f"n={ca_n:.0f} ({ca_pct:.0f}%)",
            "n_patients": n,
        }
    return summary


def plot_grouped_bars(summary):
    """
    Create a grouped bar chart: 3 task groups, each with 3 bars (Overall, NCA, CA),
    stacked by tissue type.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    tasks = ["2class", "3class", "4class"]
    bar_labels = ["Overall\nbiopsy", "NCA\n(top patches)", "CA\n(top patches)"]
    bar_width = 0.6
    x = np.arange(3)  # 3 bars per task

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        s = summary[task]

        # Stack bars: bottom-up for each tissue type
        for bar_idx, (data, label) in enumerate(
            [(s["overall"], bar_labels[0]),
             (s["nca"], bar_labels[1]),
             (s["ca"], bar_labels[2])]
        ):
            bottom = 0
            for ti, tn in enumerate(TISSUE_ORDER):
                color = TISSUE_COLORS[tn]
                height = data[ti]
                ax.bar(bar_idx, height, bar_width, bottom=bottom, color=color,
                       edgecolor="white", linewidth=0.5)

                # Label percentage inside bar if tall enough
                if height > 4:
                    ax.text(bar_idx, bottom + height / 2, f"{height:.1f}%",
                            ha="center", va="center", fontsize=7,
                            fontweight="bold",
                            color="white" if tn in ["stroma", "necrosis"] else "black")
                bottom += height

        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_title(TASK_LABELS[task], fontsize=11, fontweight="bold", pad=10)
        ax.set_ylim(0, 105)

        if ax_idx == 0:
            ax.set_ylabel("Tissue composition (%)", fontsize=10)

        # Add concentration info below x-axis
        ax.text(1, -18, f"NCA: {s['nca_info']}", ha="center", fontsize=7,
                color="gray", transform=ax.transData)
        ax.text(2, -18, f"CA: {s['ca_info']}", ha="center", fontsize=7,
                color="gray", transform=ax.transData)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legend (horizontal, below title, above plots)
    legend_patches = [
        mpatches.Patch(color=TISSUE_COLORS[tn], label=tl)
        for tn, tl in zip(TISSUE_ORDER, TISSUE_LABELS)
    ]
    fig.legend(handles=legend_patches, loc="upper center",
               bbox_to_anchor=(0.5, 0.99), ncol=len(TISSUE_ORDER), fontsize=9,
               frameon=True, edgecolor="gray", fancybox=True)

    fig.suptitle(
        "Tissue composition of most-attended patches (50% attention mass)\n"
        "BCNB test set, 217 patients, all tasks with GENConv + gated attention pooling",
        fontsize=12, fontweight="bold", y=1.10
    )

    plt.tight_layout()
    return fig


def plot_enrichment_bars(summary):
    """
    Enrichment chart: for each tissue type, show NCA and CA enrichment (pp)
    across tasks. Cleaner layout: one subplot per task, paired NCA/CA bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    tasks = ["2class", "3class", "4class"]
    # Only show diagnostically relevant tissues (skip "other")
    enrich_tissues = ["tumor", "inflammation", "stroma", "necrosis"]
    enrich_labels = ["Tumor", "Inflammation", "Stroma", "Necrosis"]
    n_tissues = len(enrich_tissues)
    x = np.arange(n_tissues)
    bar_width = 0.35

    for ax_idx, task in enumerate(tasks):
        ax = axes[ax_idx]
        s = summary[task]

        nca_enrich = np.array([s["nca"][TISSUE_ORDER.index(tn)] - s["overall"][TISSUE_ORDER.index(tn)]
                               for tn in enrich_tissues])
        ca_enrich = np.array([s["ca"][TISSUE_ORDER.index(tn)] - s["overall"][TISSUE_ORDER.index(tn)]
                              for tn in enrich_tissues])

        bars_nca = ax.bar(x - bar_width / 2, nca_enrich, bar_width, label="NCA",
                          color=[TISSUE_COLORS[tn] for tn in enrich_tissues],
                          edgecolor="black", linewidth=0.8)
        bars_ca = ax.bar(x + bar_width / 2, ca_enrich, bar_width, label="CA",
                         color=[tuple(min(1, c + 0.35) for c in TISSUE_COLORS[tn]) for tn in enrich_tissues],
                         edgecolor="black", linewidth=0.8, hatch="//")

        # Value labels
        for bar in list(bars_nca) + list(bars_ca):
            h = bar.get_height()
            if abs(h) > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + (0.3 if h > 0 else -0.8),
                        f"{h:+.1f}", ha="center", va="bottom" if h > 0 else "top",
                        fontsize=7, fontweight="bold")

        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(enrich_labels, fontsize=9)
        ax.set_title(TASK_LABELS[task], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if ax_idx == 0:
            ax.set_ylabel("Enrichment vs biopsy average (pp)", fontsize=10)

        # Concentration annotation
        ax.text(0.98, 0.02,
                f"NCA: {s['nca_info']}\nCA: {s['ca_info']}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color="gray",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Legend: solid = NCA, hatched = CA
    legend_elements = [
        mpatches.Patch(facecolor="gray", edgecolor="black", label="NCA (top patches)"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", hatch="//", label="CA (top patches)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=10, frameon=True)

    fig.suptitle(
        "Tissue enrichment in most-attended patches (50% attention mass) vs overall biopsy\n"
        "BCNB test set, 217 patients",
        fontsize=12, fontweight="bold", y=1.08
    )

    plt.tight_layout()
    return fig


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()
    summary = compute_summary(df)

    # Figure 1: Stacked composition bars
    fig1 = plot_grouped_bars(summary)
    path1 = os.path.join(OUT_DIR, "manuscript_tissue_composition_figure.pdf")
    fig1.savefig(path1, dpi=300, bbox_inches="tight")
    fig1.savefig(path1.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path1}")
    print(f"Saved: {path1.replace('.pdf', '.png')}")

    # Figure 2: Enrichment bars
    fig2 = plot_enrichment_bars(summary)
    path2 = os.path.join(OUT_DIR, "manuscript_tissue_enrichment_figure.pdf")
    fig2.savefig(path2, dpi=300, bbox_inches="tight")
    fig2.savefig(path2.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {path2}")
    print(f"Saved: {path2.replace('.pdf', '.png')}")

    # Print summary for reference
    print("\nSummary:")
    for task in ["2class", "3class", "4class"]:
        s = summary[task]
        print(f"\n  {task} (n={s['n_patients']}):")
        print(f"    Overall: {', '.join(f'{tn}={v:.1f}%' for tn, v in zip(TISSUE_ORDER, s['overall']))}")
        print(f"    NCA ({s['nca_info']}): {', '.join(f'{tn}={v:.1f}%' for tn, v in zip(TISSUE_ORDER, s['nca']))}")
        print(f"    CA  ({s['ca_info']}): {', '.join(f'{tn}={v:.1f}%' for tn, v in zip(TISSUE_ORDER, s['ca']))}")


if __name__ == "__main__":
    main()
