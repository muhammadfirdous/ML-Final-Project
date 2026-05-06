"""Generic training loop with EarlyStopping + result caching.

Output schema (saved to ``results/<task>_<dataset>_<model>.json``) matches
what /api/metrics returns and is compatible with the dashboard's existing
``metrics[algo]``, ``curves[algo]``, ``roc.merged[*][algo]``,
``confusion[algo]``, ``predVsActual[algo]`` shapes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    HYPERPARAMS,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    seed_tensorflow,
)


def cache_paths(task: str, dataset_id: str, model: str) -> tuple[Path, Path]:
    base = f"{task}_{dataset_id}_{model}"
    return SAVED_MODELS_DIR / f"{base}.h5", RESULTS_DIR / f"{base}.json"


def is_cached(task: str, dataset_id: str, model: str) -> bool:
    _, results_path = cache_paths(task, dataset_id, model)
    return results_path.exists()


# ── Public entrypoint ──────────────────────────────────────────────────────
def run_training(
    task: str,
    dataset_id: str,
    model: str,
    hyperparams: dict[str, Any] | None = None,
    force_retrain: bool = False,
    progress_cb=None,
) -> dict[str, Any]:
    """Train (or load from cache) and return the per-model results dict."""
    model_path, results_path = cache_paths(task, dataset_id, model)
    if results_path.exists() and not force_retrain:
        return json.loads(results_path.read_text(encoding="utf-8"))

    seed_tensorflow()
    # Lazy imports — TF is heavy.
    from tensorflow.keras.callbacks import EarlyStopping

    from src import data_loader, evaluate, models, preprocessing, visualize

    hp_base = HYPERPARAMS[task].copy()
    hp_base.update(hyperparams or {})

    print(f"\n[{task}/{dataset_id}/{model}] loading data…")
    raw = data_loader.load(task, dataset_id)

    print(f"[{task}/{dataset_id}/{model}] preprocessing…")
    data = preprocessing.preprocess(task, dataset_id, raw, hp_base)

    print(f"[{task}/{dataset_id}/{model}] building model…")
    builder = models.BUILDERS[model]
    keras_model = builder(
        input_shape=data["input_shape"],
        output_dim=data["output_dim"],
        hp=hp_base,
        classify=data["classify"],
    )
    keras_model.summary(print_fn=lambda s: print(f"   {s}"))
    n_params = int(keras_model.count_params())

    print(f"[{task}/{dataset_id}/{model}] fitting up to {hp_base['epochs']} epochs…")
    es = EarlyStopping(
        monitor="val_loss",
        patience=hp_base["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    t0 = time.time()
    history = keras_model.fit(
        data["X_train"], data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=hp_base["epochs"],
        batch_size=hp_base["batch_size"],
        callbacks=[es],
        verbose=2,
    )
    train_seconds = time.time() - t0
    print(f"[{task}/{dataset_id}/{model}] training done in {train_seconds:.1f}s")

    # Build per-epoch curve in dashboard shape.
    h = history.history
    epochs = list(range(1, len(h["loss"]) + 1))
    curve = []
    for i, ep in enumerate(epochs):
        row = {
            "epoch": ep,
            "trainLoss": round(float(h["loss"][i]), 4),
            "valLoss": round(float(h["val_loss"][i]), 4),
        }
        if data["classify"]:
            row["trainAcc"] = round(float(h.get("accuracy", [0.0] * len(epochs))[i]), 4)
            row["valAcc"] = round(float(h.get("val_accuracy", [0.0] * len(epochs))[i]), 4)
        else:
            # Fill with 0 so the dashboard keeps a uniform shape.
            row["trainAcc"] = 0.0
            row["valAcc"] = 0.0
        curve.append(row)

    # Evaluate on val.
    print(f"[{task}/{dataset_id}/{model}] evaluating…")
    if data["classify"]:
        proba = keras_model.predict(data["X_val"], verbose=0)
        y_pred = np.argmax(proba, axis=1)
        eval_metrics = evaluate.evaluate_classification(
            data["y_val"], y_pred, proba, data["classes"]
        )
        # Pred-vs-actual not applicable for pure classification.
        pred_vs_actual = None
    else:
        y_pred_scaled = keras_model.predict(data["X_val"], verbose=0).flatten()
        # Inverse-transform predictions back to original units.
        scaler = data["scaler"]
        target_idx = data["target_idx"]
        if scaler is not None and target_idx is not None:
            n_features = scaler.n_features_in_
            tmp_pred = np.zeros((len(y_pred_scaled), n_features))
            tmp_pred[:, target_idx] = y_pred_scaled
            y_pred = scaler.inverse_transform(tmp_pred)[:, target_idx]
            tmp_true = np.zeros_like(tmp_pred)
            tmp_true[:, target_idx] = data["y_val"]
            y_true = scaler.inverse_transform(tmp_true)[:, target_idx]
        else:
            y_pred = y_pred_scaled
            y_true = data["y_val"]
        eval_metrics = evaluate.evaluate_regression(y_true, y_pred)
        # Show first 60 predictions side-by-side with actuals (matches mock shape).
        n_show = min(60, len(y_true))
        pred_vs_actual = [
            {"day": i + 1, "actual": round(float(y_true[i]), 2),
             "predicted": round(float(y_pred[i]), 2)}
            for i in range(n_show)
        ]

    # Build dashboard-shape ``metrics`` block.
    if data["classify"]:
        metrics_block = {
            "accuracy": eval_metrics["accuracy"],
            "precision": eval_metrics["precision"],
            "recall": eval_metrics["recall"],
            "f1": eval_metrics["f1"],
            "auc": eval_metrics["auc"],
            "params": evaluate.format_param_count(n_params),
            "trainingTime": evaluate.format_training_time(train_seconds),
        }
    else:
        metrics_block = {
            "accuracy": eval_metrics["accuracy"],   # R² as 0–100
            "mae": eval_metrics["mae"],
            "rmse": eval_metrics["rmse"],
            "mape": eval_metrics["mape"],
            "params": evaluate.format_param_count(n_params),
            "trainingTime": evaluate.format_training_time(train_seconds),
        }

    result: dict[str, Any] = {
        "task": task,
        "datasetId": dataset_id,
        "model": model,
        "metrics": metrics_block,
        "curve": curve,
        "classify": bool(data["classify"]),
    }
    if data["classify"]:
        result["roc"] = eval_metrics["roc"]
        result["confusion"] = eval_metrics["confusion"]
        result["classes"] = data["classes"]
    else:
        result["predVsActual"] = pred_vs_actual

    # Persist model + JSON.
    print(f"[{task}/{dataset_id}/{model}] saving model + result JSON…")
    try:
        keras_model.save(model_path)
    except Exception as exc:  # pragma: no cover — disk full, etc.
        print(f"   warning: could not save .h5: {exc}")
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Render plots (best-effort; don't fail training if plotting fails).
    try:
        visualize.render_all(task, dataset_id, model, result)
    except Exception as exc:
        print(f"   warning: plot rendering failed: {exc}")

    return result


# ── CLI: train one combination, or all of them ────────────────────────────
def _parse_combos(arg: str) -> list[tuple[str, str, str]]:
    """Accept "task/dataset/model" or "all" or "task/dataset/all"."""
    from config import TASKS

    if arg == "all":
        out = []
        for task, cfg in TASKS.items():
            for ds in cfg["datasets"]:
                for algo in cfg["algos"]:
                    out.append((task, ds, algo))
        return out
    parts = arg.split("/")
    if len(parts) == 3:
        task, ds, algo = parts
        if algo == "all":
            return [(task, ds, a) for a in TASKS[task]["algos"]]
        return [(task, ds, algo)]
    raise SystemExit(f"Unrecognised arg: {arg}")


def _cli():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.train <task/dataset/model | task/dataset/all | all>")
        return
    combos = _parse_combos(sys.argv[1])
    force = "--force" in sys.argv
    overall_t0 = time.time()
    for task, ds, algo in combos:
        try:
            r = run_training(task, ds, algo, force_retrain=force)
            print(f"=> {task}/{ds}/{algo}: {r['metrics']}")
        except Exception as exc:
            print(f"ERROR on {task}/{ds}/{algo}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\nTotal wall time: {time.time() - overall_t0:.1f}s")


if __name__ == "__main__":
    _cli()
