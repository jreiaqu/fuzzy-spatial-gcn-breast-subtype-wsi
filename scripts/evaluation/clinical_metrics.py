"""
clinical_metrics.py -- CMPB-D-25-07046 Revision, Session 2 (S2.1 / R2.7)

Computes clinical metrics and COC (Confidence Operating Characteristic)
curves from per-patient prediction CSVs across all three experimental
protocols.

Metrics computed per model/task:
  - Per-class: sensitivity (recall), specificity, PPV, NPV
  - Overall: accuracy, weighted F1, macro AUC
  - COC curve: accuracy vs fraction delegated to expert
  - AUC-OC: area under the COC curve
  - ECE: expected calibration error (15 bins)

COC curve reference:
  Salustiano et al., "Expert load matters: operating networks at high
  accuracy and low manual effort," NeurIPS 2023.
  https://arxiv.org/abs/2308.05035

Usage:
    pip install -r requirements.txt  # see repository root
    python clinical_metrics.py
    python clinical_metrics.py --protocols 1 2       # specific protocols
    python clinical_metrics.py --tasks 2class        # specific task
"""

import os
import sys
import argparse
import math

import numpy as np
import pandas as pd
import matplotlib

# --- Repository path configuration (portable) ---
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
from _paths import *  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# [REPLACED by _paths.py] RESULTS_DIR = "/Users/kckj099/Documents/CMPB-Review/results"
PRED_DIR = os.path.join(RESULTS_DIR, "predictions")
OUT_DIR = os.path.join(RESULTS_DIR, "clinical")

# ---------------------------------------------------------------------------
# Task definitions (must match prediction scripts)
# ---------------------------------------------------------------------------
TASKS = {
    "2class": {
        "class_names": ["Other", "TNBC"],
        "n_classes": 2,
    },
    "3class": {
        "class_names": ["Luminals", "HER2(+)", "TNBC"],
        "n_classes": 3,
    },
    "4class": {
        "class_names": ["Luminal A", "Luminal B", "HER2(+)", "TNBC"],
        "n_classes": 4,
    },
}

# ---------------------------------------------------------------------------
# Prediction file discovery
# ---------------------------------------------------------------------------

# Protocol 1: BCNB within-domain (single test set, no folds)
# Format: patient_id, y_true, y_pred, y_prob_0, ...
PROTO1_FILES = {
    ("2class", "CA"):  "BCNB_2class_CA_predictions.csv",
    ("2class", "NCA"): "BCNB_2class_NCA_predictions.csv",
    ("3class", "CA"):  "BCNB_3class_CA_predictions.csv",
    ("3class", "NCA"): "BCNB_3class_NCA_predictions.csv",
    ("4class", "CA"):  "BCNB_4class_CA_predictions.csv",
    ("4class", "NCA"): "BCNB_4class_NCA_predictions.csv",
}

# Protocol 2: SBC VGG16 transfer (5-fold x 3 repeats)
# Format: patient_id, repeat, fold, y_true, y_pred, y_prob_0, ...
PROTO2_FILES = {
    ("2class", "CA"):  "SBC_2class_CA_transfer_predictions.csv",
    ("2class", "NCA"): "SBC_2class_NCA_transfer_predictions.csv",
    ("3class", "CA"):  "SBC_3class_CA_transfer_predictions.csv",
    ("3class", "NCA"): "SBC_3class_NCA_transfer_predictions.csv",
    ("4class", "CA"):  "SBC_4class_CA_transfer_predictions.csv",
    ("4class", "NCA"): "SBC_4class_NCA_transfer_predictions.csv",
}

# Protocol 3: SBC CONCH (3-fold x 2 repeats)
# Format: patient_id, repeat, fold, y_true, y_pred, y_prob_0, ...
PROTO3_FILES = {
    ("2class", "baseline"): "SBC_2class_baseline_conch_predictions.csv",
    ("2class", "NCA"):      "SBC_2class_NCA_conch_predictions.csv",
    ("2class", "CA"):       "SBC_2class_CA_conch_predictions.csv",
    ("3class", "baseline"): "SBC_3class_baseline_conch_predictions.csv",
    ("3class", "NCA"):      "SBC_3class_NCA_conch_predictions.csv",
    ("3class", "CA"):       "SBC_3class_CA_conch_predictions.csv",
    ("4class", "baseline"): "SBC_4class_baseline_conch_predictions.csv",
    ("4class", "NCA"):      "SBC_4class_NCA_conch_predictions.csv",
    ("4class", "CA"):       "SBC_4class_CA_conch_predictions.csv",
}

PROTOCOL_MAP = {
    1: ("Protocol 1: BCNB within-domain", PROTO1_FILES),
    2: ("Protocol 2: SBC VGG16 transfer", PROTO2_FILES),
    3: ("Protocol 3: SBC CONCH", PROTO3_FILES),
}


# ---------------------------------------------------------------------------
# Clinical metrics computation
# ---------------------------------------------------------------------------

def compute_per_class_metrics(y_true, y_pred, class_names):
    """Compute sensitivity, specificity, PPV, NPV per class.

    Uses one-vs-rest binarization for each class.

    Returns list of dicts, one per class.
    """
    n_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    results = []

    for c in range(n_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        results.append({
            "class": class_names[c],
            "class_idx": c,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "support": int(tp + fn),
        })

    return results


def compute_coc_curve(y_true, y_pred, probabilities):
    """Compute COC (Confidence Operating Characteristic) curve.

    Based on Salustiano et al., NeurIPS 2023.

    Args:
        y_true: array of true labels
        y_pred: array of predicted labels
        probabilities: array of shape [N, n_classes] (softmax outputs)

    Returns:
        fractions_delegated: array of fraction of samples delegated to expert
        accuracies: array of accuracy on model-retained samples
        aucoc: area under the COC curve
    """
    confidences = np.max(probabilities, axis=1)
    thresholds = np.sort(confidences)
    n_samples = len(y_true)

    fractions_delegated = np.zeros(len(thresholds))
    accuracies = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        retained = confidences >= t
        n_retained = retained.sum()

        if n_retained > 0:
            correct = (y_true[retained] == y_pred[retained]).sum()
            accuracies[i] = correct / n_retained
        else:
            accuracies[i] = 1.0  # no samples retained = perfect accuracy trivially

        fractions_delegated[i] = (confidences < t).sum() / n_samples

    # Compute AUC-OC using trapezoidal rule
    from sklearn.metrics import auc
    aucoc = auc(fractions_delegated, accuracies)

    return fractions_delegated, accuracies, aucoc


def compute_coc_per_class(y_true, y_pred, probabilities, class_idx):
    """Compute COC curve for patients where the model PREDICTS a specific class.

    Answers: "When the model says 'HER2+', how often is it right, and at
    what confidence threshold should we trust it?"

    Only considers patients whose predicted class == class_idx.
    Confidence = the probability assigned to that predicted class.

    Returns:
        fractions_delegated, accuracies, aucoc (same format as compute_coc_curve)
        Returns None, None, None if fewer than 5 patients predicted as this class.
    """
    mask = y_pred == class_idx
    if mask.sum() < 5:
        return None, None, None

    y_true_sub = y_true[mask]
    y_pred_sub = y_pred[mask]
    confidences = probabilities[mask, class_idx]

    thresholds = np.sort(confidences)
    n_samples = len(y_true_sub)

    fractions_delegated = np.zeros(len(thresholds))
    accuracies = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        retained = confidences >= t
        n_retained = retained.sum()
        if n_retained > 0:
            correct = (y_true_sub[retained] == y_pred_sub[retained]).sum()
            accuracies[i] = correct / n_retained
        else:
            accuracies[i] = 1.0
        fractions_delegated[i] = (confidences < t).sum() / n_samples

    from sklearn.metrics import auc
    aucoc = auc(fractions_delegated, accuracies)
    return fractions_delegated, accuracies, aucoc


def compute_coc_one_vs_rest(y_true, probabilities, class_idx):
    """Compute COC curve for a one-vs-rest binary problem.

    Binarizes the problem: is this patient class_idx or not?
    Confidence = probability assigned to class_idx.
    Prediction = 1 if prob[class_idx] > 0.5, else 0.

    Answers: "If we use this model as a TNBC screener, how does the
    accuracy-delegation trade-off look?"

    Returns:
        fractions_delegated, accuracies, aucoc
        Returns None, None, None if fewer than 5 positive samples.
    """
    y_binary = (y_true == class_idx).astype(int)
    if y_binary.sum() < 5:
        return None, None, None

    prob_class = probabilities[:, class_idx]
    y_pred_binary = (prob_class >= 0.5).astype(int)

    # Use prob_class as confidence for positive predictions,
    # and (1 - prob_class) as confidence for negative predictions
    confidences = np.where(y_pred_binary == 1, prob_class, 1.0 - prob_class)

    thresholds = np.sort(confidences)
    n_samples = len(y_true)

    fractions_delegated = np.zeros(len(thresholds))
    accuracies = np.zeros(len(thresholds))

    for i, t in enumerate(thresholds):
        retained = confidences >= t
        n_retained = retained.sum()
        if n_retained > 0:
            correct = (y_binary[retained] == y_pred_binary[retained]).sum()
            accuracies[i] = correct / n_retained
        else:
            accuracies[i] = 1.0
        fractions_delegated[i] = (confidences < t).sum() / n_samples

    from sklearn.metrics import auc
    aucoc = auc(fractions_delegated, accuracies)
    return fractions_delegated, accuracies, aucoc


def compute_ece(y_true, y_pred, confidences, n_bins=15):
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bl, bu in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bl) & (confidences <= bu)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            acc_in_bin = (y_true[in_bin] == y_pred[in_bin]).mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
    return ece


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(filepath):
    """Load a prediction CSV and return arrays.

    For multi-fold files (Protocols 2/3), returns one row per patient using
    majority vote across folds/repeats, with mean probabilities.
    For single-test files (Protocol 1), returns as-is.
    """
    df = pd.read_csv(filepath)
    prob_cols = [c for c in df.columns if c.startswith("y_prob_")]
    n_classes = len(prob_cols)

    if "fold" in df.columns:
        # Multi-fold: aggregate per patient
        # Majority vote for y_pred, mean for probabilities
        agg = df.groupby("patient_id").agg(
            y_true=("y_true", "first"),
            y_pred=("y_pred", lambda x: x.mode().iloc[0]),
            **{pc: (pc, "mean") for pc in prob_cols},
        ).reset_index()
    else:
        agg = df

    y_true = agg["y_true"].astype(int).values
    y_pred = agg["y_pred"].astype(int).values
    probs = agg[prob_cols].values
    n_patients = len(agg)

    return y_true, y_pred, probs, n_patients


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_protocol(protocol_num, task_keys, out_dir):
    """Run clinical metrics analysis for one protocol.

    Returns list of summary rows and list of COC curve data.
    """
    proto_name, file_map = PROTOCOL_MAP[protocol_num]
    print(f"\n{'='*70}")
    print(f"  {proto_name}")
    print(f"{'='*70}")

    all_summary_rows = []
    all_per_class_rows = []
    coc_curves = {}

    for task_key in task_keys:
        task_cfg = TASKS[task_key]
        class_names = task_cfg["class_names"]
        n_classes = task_cfg["n_classes"]

        for (tk, model), filename in file_map.items():
            if tk != task_key:
                continue

            filepath = os.path.join(PRED_DIR, filename)
            if not os.path.exists(filepath):
                print(f"  SKIP {task_key}/{model}: {filename} not found")
                continue

            y_true, y_pred, probs, n_patients = load_predictions(filepath)

            # Overall metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            try:
                if n_classes == 2:
                    auc_val = roc_auc_score(y_true, probs[:, 1])
                else:
                    auc_val = roc_auc_score(
                        y_true, probs, multi_class="ovr", average="macro"
                    )
            except ValueError:
                auc_val = np.nan

            # Confidence and calibration
            confidences = np.max(probs, axis=1)
            ece = compute_ece(y_true, y_pred, confidences)

            # COC curve (overall: max confidence across all classes)
            frac_del, coc_acc, aucoc = compute_coc_curve(y_true, y_pred, probs)
            coc_curves[(task_key, model, "overall")] = (frac_del, coc_acc, aucoc)

            # COC per-class: for patients where model predicts class C
            for c_idx, c_name in enumerate(class_names):
                fd, ca, aoc = compute_coc_per_class(y_true, y_pred, probs, c_idx)
                if fd is not None:
                    coc_curves[(task_key, model, f"perclass_{c_name}")] = (fd, ca, aoc)

            # COC one-vs-rest: binary screening for each subtype
            for c_idx, c_name in enumerate(class_names):
                fd, ca, aoc = compute_coc_one_vs_rest(y_true, probs, c_idx)
                if fd is not None:
                    coc_curves[(task_key, model, f"ovr_{c_name}")] = (fd, ca, aoc)

            # Per-class metrics
            per_class = compute_per_class_metrics(y_true, y_pred, class_names)
            for pc in per_class:
                pc["protocol"] = protocol_num
                pc["task"] = task_key
                pc["model"] = model
                pc["n_patients"] = n_patients
            all_per_class_rows.extend(per_class)

            # Summary row
            summary = {
                "protocol": protocol_num,
                "task": task_key,
                "model": model,
                "n_patients": n_patients,
                "accuracy": acc,
                "f1_weighted": f1,
                "auc_macro": auc_val,
                "aucoc": aucoc,
                "ece": ece,
                "mean_confidence": confidences.mean(),
            }

            # Add per-class AUC-OC to summary
            for c_idx, c_name in enumerate(class_names):
                key_pc = (task_key, model, f"perclass_{c_name}")
                key_ovr = (task_key, model, f"ovr_{c_name}")
                if key_pc in coc_curves:
                    summary[f"aucoc_perclass_{c_name}"] = coc_curves[key_pc][2]
                if key_ovr in coc_curves:
                    summary[f"aucoc_ovr_{c_name}"] = coc_curves[key_ovr][2]

            all_summary_rows.append(summary)

            # Print
            print(f"\n  {task_key} / {model} (n={n_patients})")
            print(f"    Accuracy: {acc:.3f}  F1: {f1:.3f}  AUC: {auc_val:.3f}")
            print(f"    AUC-OC (overall): {aucoc:.3f}  ECE: {ece:.3f}  Mean conf: {confidences.mean():.3f}")
            for c_idx, c_name in enumerate(class_names):
                ovr_key = (task_key, model, f"ovr_{c_name}")
                pc_key = (task_key, model, f"perclass_{c_name}")
                ovr_str = f"{coc_curves[ovr_key][2]:.3f}" if ovr_key in coc_curves else "n/a"
                pc_str = f"{coc_curves[pc_key][2]:.3f}" if pc_key in coc_curves else "n/a"
                print(f"    AUC-OC {c_name:>12s}: one-vs-rest={ovr_str}  per-class={pc_str}")
            print(f"    Per-class clinical metrics:")
            for pc in per_class:
                print(
                    f"      {pc['class']:>12s}: Sens={pc['sensitivity']:.3f}  "
                    f"Spec={pc['specificity']:.3f}  PPV={pc['PPV']:.3f}  "
                    f"NPV={pc['NPV']:.3f}  (n={pc['support']})"
                )

    return all_summary_rows, all_per_class_rows, coc_curves


def create_coc_plots(all_coc_data, protocol_num, task_keys, out_dir):
    """Create all three types of COC curve plots for one protocol.

    1. Overall COC: one plot per task, comparing models (CA vs NCA vs baseline)
    2. Per-class COC: one plot per task, showing per-predicted-class curves for each model
    3. One-vs-rest COC: one plot per subtype of interest, comparing models as binary screeners
    """
    proto_name = PROTOCOL_MAP[protocol_num][0]
    model_palette = {"CA": "#E24A33", "NCA": "#348ABD", "baseline": "#988ED5"}

    tasks_in_data = sorted(set(
        tk for (tk, _, coc_type) in all_coc_data.keys() if coc_type == "overall"
    ))

    # --- 1. Overall COC (one plot per task, comparing models) ---
    for task_key in tasks_in_data:
        fig, ax = plt.subplots(figsize=(8, 6))
        for (tk, model, ctype), (frac_del, coc_acc, aucoc) in sorted(all_coc_data.items()):
            if tk != task_key or ctype != "overall":
                continue
            color = model_palette.get(model, "#777777")
            ax.plot(frac_del, coc_acc, label=f"{model} (AUC-OC={aucoc:.3f})",
                    color=color, linewidth=2)

        ax.set_xlabel("Fraction delegated to expert", fontsize=12)
        ax.set_ylabel("Accuracy on retained samples", fontsize=12)
        task_display = ", ".join(TASKS[task_key]["class_names"])
        ax.set_title(f"Overall COC: {task_display}\n{proto_name}",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=10, loc="lower right")
        ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.02)
        ax.grid(True, alpha=0.3)
        outpath = os.path.join(out_dir, f"coc_overall_proto{protocol_num}_{task_key}.pdf")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved overall COC: {outpath}")

    # --- 2. Per-class COC (per task: subplots for each class, lines for each model) ---
    for task_key in tasks_in_data:
        class_names = TASKS[task_key]["class_names"]
        n_classes = len(class_names)
        fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5))
        if n_classes == 1:
            axes = [axes]

        for c_idx, c_name in enumerate(class_names):
            ax = axes[c_idx]
            has_data = False
            for (tk, model, ctype), (fd, ca, aoc) in sorted(all_coc_data.items()):
                if tk != task_key or ctype != f"perclass_{c_name}":
                    continue
                color = model_palette.get(model, "#777777")
                ax.plot(fd, ca, label=f"{model} ({aoc:.3f})",
                        color=color, linewidth=2)
                has_data = True

            ax.set_title(f"Predicted: {c_name}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Frac. delegated", fontsize=10)
            if c_idx == 0:
                ax.set_ylabel("Accuracy on retained", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0.3, 1.02)
            ax.grid(True, alpha=0.3)
            if has_data:
                ax.legend(fontsize=9, loc="lower right")

        plt.suptitle(f"Per-class COC: {proto_name}", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        outpath = os.path.join(out_dir, f"coc_perclass_proto{protocol_num}_{task_key}.pdf")
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  Saved per-class COC: {outpath}")

    # --- 3. One-vs-rest COC (per clinically important subtype, comparing models) ---
    # Focus on TNBC and HER2+ (the clinically actionable subtypes)
    subtypes_of_interest = ["TNBC", "HER2(+)"]
    for subtype in subtypes_of_interest:
        for task_key in tasks_in_data:
            class_names = TASKS[task_key]["class_names"]
            if subtype not in class_names:
                continue

            fig, ax = plt.subplots(figsize=(8, 6))
            has_data = False
            for (tk, model, ctype), (fd, ca, aoc) in sorted(all_coc_data.items()):
                if tk != task_key or ctype != f"ovr_{subtype}":
                    continue
                color = model_palette.get(model, "#777777")
                ax.plot(fd, ca, label=f"{model} (AUC-OC={aoc:.3f})",
                        color=color, linewidth=2)
                has_data = True

            if not has_data:
                plt.close()
                continue

            ax.set_xlabel("Fraction delegated to expert", fontsize=12)
            ax.set_ylabel(f"Accuracy ({subtype} vs rest)", fontsize=12)
            ax.set_title(
                f"One-vs-Rest COC: {subtype} screening\n{proto_name} ({task_key})",
                fontsize=12, fontweight="bold",
            )
            ax.legend(fontsize=10, loc="lower right")
            ax.set_xlim(0, 1); ax.set_ylim(0.4, 1.02)
            ax.grid(True, alpha=0.3)
            safe_subtype = subtype.replace("(", "").replace(")", "").replace("+", "plus")
            outpath = os.path.join(
                out_dir, f"coc_ovr_{safe_subtype}_proto{protocol_num}_{task_key}.pdf"
            )
            plt.tight_layout()
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"  Saved one-vs-rest COC ({subtype}): {outpath}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clinical metrics and COC curves for CMPB revision"
    )
    parser.add_argument(
        "--protocols", nargs="+", type=int, default=[1, 2, 3],
        choices=[1, 2, 3],
        help="Which protocols to analyze (default: all)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["2class", "3class", "4class"],
        choices=["2class", "3class", "4class"],
        help="Which tasks to analyze (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("CMPB Revision - Clinical Metrics + COC Curves (S2.1 / R2.7)")
    print("=" * 70)
    print(f"Protocols: {args.protocols}")
    print(f"Tasks: {args.tasks}")

    os.makedirs(OUT_DIR, exist_ok=True)

    all_summary = []
    all_per_class = []

    for proto in args.protocols:
        summary, per_class, coc_data = analyze_protocol(
            proto, args.tasks, OUT_DIR
        )
        all_summary.extend(summary)
        all_per_class.extend(per_class)

        # COC plots (all three types)
        if coc_data:
            create_coc_plots(coc_data, proto, args.tasks, OUT_DIR)

    # Save summary tables
    if all_summary:
        summary_df = pd.DataFrame(all_summary)
        summary_path = os.path.join(OUT_DIR, "clinical_metrics_summary.csv")
        summary_df.to_csv(summary_path, index=False, float_format="%.4f")
        print(f"\nSaved summary: {summary_path}")

        # Print formatted table
        print(f"\n{'='*90}")
        print(f"  {'Proto':>5} {'Task':>7} {'Model':>10} {'N':>5} "
              f"{'Acc':>6} {'F1':>6} {'AUC':>6} {'AUC-OC':>7} {'ECE':>6}")
        print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*5} "
              f"{'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*6}")
        for _, r in summary_df.iterrows():
            print(
                f"  {int(r['protocol']):>5} {r['task']:>7} {r['model']:>10} "
                f"{int(r['n_patients']):>5} {r['accuracy']:>6.3f} "
                f"{r['f1_weighted']:>6.3f} {r['auc_macro']:>6.3f} "
                f"{r['aucoc']:>7.3f} {r['ece']:>6.3f}"
            )

    if all_per_class:
        per_class_df = pd.DataFrame(all_per_class)
        per_class_path = os.path.join(OUT_DIR, "clinical_metrics_per_class.csv")
        per_class_df.to_csv(per_class_path, index=False, float_format="%.4f")
        print(f"\nSaved per-class metrics: {per_class_path}")

    print("\nDone. All outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()
