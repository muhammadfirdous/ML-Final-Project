"""Metric computation — classification and regression.

All functions return plain dicts so they can be JSON-serialised straight
into ``results/<task>_<dataset>_<model>.json``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: list[str],
) -> dict[str, Any]:
    """Accuracy, Precision/Recall/F1 (macro), AUC (one-vs-rest), confusion matrix.

    Also computes a per-class ROC for the rendered chart by averaging the
    one-vs-rest curves into a single ``fpr/tpr`` pair (macro-style).
    """
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    n_classes = len(classes)
    # One-vs-rest AUC (works for binary and multiclass).
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc = float("nan")

    # Macro-averaged ROC for the dashboard chart (single curve per model).
    fpr_grid = np.linspace(0.0, 1.0, 31)
    tpr_acc = np.zeros_like(fpr_grid)
    actual_classes = 0
    for c in range(n_classes):
        if (y_true == c).sum() == 0:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true == c, y_proba[:, c])
            tpr_interp = np.interp(fpr_grid, fpr, tpr)
            tpr_acc += tpr_interp
            actual_classes += 1
        except ValueError:
            continue
    if actual_classes > 0:
        tpr_acc /= actual_classes

    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).tolist()

    return {
        "accuracy": int(round(acc * 100)),  # dashboard expects 0-100 scale
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4) if not np.isnan(auc) else None,
        "confusion": cm,
        "roc": {
            "fpr": [round(float(v), 4) for v in fpr_grid],
            "tpr": [round(float(v), 4) for v in tpr_acc],
            "auc": round(auc, 4) if not np.isnan(auc) else None,
        },
    }


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """MAE, RMSE, MAPE, R² (rendered as ``accuracy`` in 0–100 scale)."""
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # MAPE blows up when targets cross zero (e.g. winter °F). Floor the
    # denominator at 5% of the target std and clip per-sample at 100% so a
    # handful of near-zero ground-truth rows can't dominate the mean.
    floor = max(1.0, float(np.std(y_true)) * 0.05)
    denom = np.where(np.abs(y_true) < floor, floor, np.abs(y_true))
    sample_mape = np.minimum(np.abs(y_true - y_pred) / denom, 1.0)
    mape = float(np.mean(sample_mape) * 100)
    r2 = float(r2_score(y_true, y_pred))
    accuracy_pct = max(0, min(100, int(round(r2 * 100))))

    return {
        "accuracy": accuracy_pct,
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 2),
        "r2": round(r2, 4),
    }


def format_training_time(seconds: float) -> str:
    """e.g. 372.4 → '6m 12s'."""
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m {secs:02d}s"


def format_param_count(n: int) -> str:
    """e.g. 142983 → '142,983', 3214567 → '3.21M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    return f"{n:,}"
