"""Plot generation — saves PNGs to plots/<task>_<dataset>_<model>_<plot>.png.

Matplotlib is heavy; import lazily inside each function so the Flask import
path stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from config import PLOTS_DIR


def plot_paths(task: str, dataset_id: str, model: str) -> dict[str, Path]:
    base = f"{task}_{dataset_id}_{model}"
    return {
        "loss": PLOTS_DIR / f"{base}_loss.png",
        "acc": PLOTS_DIR / f"{base}_acc.png",
        "roc": PLOTS_DIR / f"{base}_roc.png",
        "confusion": PLOTS_DIR / f"{base}_confusion.png",
        "predVsActual": PLOTS_DIR / f"{base}_pred_vs_actual.png",
    }


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    return plt


def plot_loss_curve(curve: list[dict], out: Path, classify: bool, model_name: str) -> None:
    plt = _setup_matplotlib()
    epochs = [pt["epoch"] for pt in curve]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(epochs, [pt["trainLoss"] for pt in curve], label="Train", color="#22d3ee", lw=2)
    ax.plot(epochs, [pt["valLoss"] for pt in curve], label="Validation", color="#22d3ee",
            lw=1.5, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy" if classify else "MSE")
    ax.set_title(f"{model_name} — loss curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_accuracy_curve(curve: list[dict], out: Path, model_name: str) -> None:
    plt = _setup_matplotlib()
    epochs = [pt["epoch"] for pt in curve]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    ax.plot(epochs, [pt["trainAcc"] for pt in curve], label="Train", color="#34d399", lw=2)
    ax.plot(epochs, [pt["valAcc"] for pt in curve], label="Validation", color="#34d399",
            lw=1.5, ls="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_name} — accuracy curve")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_confusion_matrix(matrix, classes: list[str], out: Path, model_name: str) -> None:
    plt = _setup_matplotlib()
    import numpy as np
    cm = np.asarray(matrix, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_name} — confusion matrix")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_roc_curve(roc: dict[str, Any], out: Path, model_name: str) -> None:
    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    auc_val = roc.get("auc")
    if auc_val is None:
        auc_label = "n/a"
    else:
        auc_label = f"{auc_val:.3f}"
    ax.plot(roc["fpr"], roc["tpr"], color="#22d3ee", lw=2,
            label=f"{model_name} (AUC = {auc_label})")
    ax.plot([0, 1], [0, 1], color="#94a3b8", lw=1, ls="--", label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — ROC curve")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_pred_vs_actual(actual: Sequence[float], predicted: Sequence[float],
                         out: Path, model_name: str) -> None:
    plt = _setup_matplotlib()
    days = list(range(1, len(actual) + 1))
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(days, actual, label="Actual", color="#94a3b8", lw=1.5)
    ax.plot(days, predicted, label="Predicted", color="#22d3ee", lw=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Value")
    ax.set_title(f"{model_name} — predicted vs actual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def render_all(task: str, dataset_id: str, model: str, result: dict[str, Any]) -> dict[str, str]:
    """Render every plot the dashboard might want; returns map of name → relative URL."""
    paths = plot_paths(task, dataset_id, model)
    classify = bool(result.get("classify"))
    out: dict[str, str] = {}
    if result.get("curve"):
        plot_loss_curve(result["curve"], paths["loss"], classify, model)
        out["loss"] = f"/static/plots/{paths['loss'].name}"
        plot_accuracy_curve(result["curve"], paths["acc"], model)
        out["acc"] = f"/static/plots/{paths['acc'].name}"
    if classify and result.get("confusion") and result.get("classes"):
        plot_confusion_matrix(result["confusion"], result["classes"], paths["confusion"], model)
        out["confusion"] = f"/static/plots/{paths['confusion'].name}"
    if classify and result.get("roc"):
        plot_roc_curve(result["roc"], paths["roc"], model)
        out["roc"] = f"/static/plots/{paths['roc'].name}"
    if not classify and result.get("predVsActual"):
        actual = [pt["actual"] for pt in result["predVsActual"]]
        predicted = [pt["predicted"] for pt in result["predVsActual"]]
        plot_pred_vs_actual(actual, predicted, paths["predVsActual"], model)
        out["predVsActual"] = f"/static/plots/{paths['predVsActual'].name}"
    return out
