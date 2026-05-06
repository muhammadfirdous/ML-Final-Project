"""Flask backend serving the ML dashboard.

Phase 2: every endpoint returns shape-correct mock data so the frontend
can be wired and tested before any TF training runs. Once a per-model
results JSON exists in results/, the dispatch helpers prefer it over the
mock — so as Phase 3+ trains real models, the same endpoints transparently
upgrade to real numbers.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_from_directory
from flask_cors import CORS

from config import (
    ALLOWED_ORIGINS,
    HYPERPARAMS,
    PLOTS_DIR,
    RESULTS_DIR,
    TASKS,
)
from src import mock_data
from src.train import cache_paths, is_cached
from verify_data import check_all


app = Flask(__name__)
CORS(app, origins=ALLOWED_ORIGINS)


# ── Helpers ────────────────────────────────────────────────────────────────
def _read_results_json(task: str, dataset_id: str, model: str) -> dict[str, Any] | None:
    _, results_path = cache_paths(task, dataset_id, model)
    if not results_path.exists():
        return None
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_dataset_slice(task: str, dataset_id: str) -> dict[str, Any]:
    """Prefer real cached results; fall back to mock. Always returns shape-correct."""
    sl = mock_data.get_dataset_slice(task, dataset_id)
    if sl is None:
        abort(404, description=f"Unknown task/dataset: {task}/{dataset_id}")

    # If any per-model JSON exists, splice it into the slice so the dashboard
    # shows real numbers as soon as one model is trained, even if others lag.
    for algo in sl["algos"]:
        cached = _read_results_json(task, dataset_id, algo)
        if not cached:
            continue
        sl["metrics"][algo] = cached.get("metrics", sl["metrics"][algo])
        sl["curves"][algo] = cached.get("curve", sl["curves"][algo])
        if "roc" in cached and "roc" in sl:
            # Per-model ROC re-merged into the dataset-level structure.
            roc_idx = next(
                (
                    i
                    for i, pt in enumerate(sl["roc"]["merged"])
                    if abs(pt["fpr"] - cached["roc"]["fpr"][0]) < 1e-9
                ),
                None,
            )
            if roc_idx is not None:
                for i, fpr in enumerate(cached["roc"]["fpr"]):
                    if i < len(sl["roc"]["merged"]):
                        sl["roc"]["merged"][i][algo] = cached["roc"]["tpr"][i]
                sl["roc"]["aucMap"][algo] = cached["roc"]["auc"]
        if "confusion" in cached and "confusion" in sl:
            sl["confusion"][algo] = cached["confusion"]
        if "predVsActual" in cached and "predVsActual" in sl:
            sl["predVsActual"][algo] = cached["predVsActual"]

    # Recompute bestId if any real metrics were spliced in.
    if "bestId" in sl:
        if sl.get("classify"):
            sl["bestId"] = max(sl["algos"], key=lambda a: sl["metrics"][a].get("accuracy", 0))
        else:
            sl["bestId"] = min(sl["algos"], key=lambda a: sl["metrics"][a].get("rmse", 1e9))

    # The dashboard uses ``metrics[a].best`` for classification tasks (Speech,
    # Sentiment) and ``bestId`` for time-series. Stamp ``best`` on the winner
    # for classification regardless of whether mock or real metrics are in use.
    if sl.get("classify"):
        winner = max(sl["algos"], key=lambda a: sl["metrics"][a].get("accuracy", 0))
        for a in sl["algos"]:
            sl["metrics"][a].pop("best", None)
        sl["metrics"][winner]["best"] = True
    return sl


# ── Health / verification ──────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "phase": 2, "message": "Mock data + frontend wiring."})


@app.get("/api/verify")
def verify():
    checks = check_all()
    return jsonify(
        {
            "ok": all(c.ok for c in checks),
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail} for c in checks],
        }
    )


def _recompute_summary(payload: dict[str, Any]) -> tuple[list[dict], list[dict]]:
    """Derive ``summaryRows`` + ``heatmapData`` from whichever per-model metrics
    were spliced into the dashboard payload.

    summaryRows: one row per dataset with the best model + headline metric.
    heatmapData: row per dataset, columns ANN/RNN/LSTM/GRU showing primary metric
    (accuracy for classification, R² accuracy* for regression). ``None`` if
    a model wasn't trained on that dataset.
    """
    tasks = payload["tasks"]
    rows: list[dict] = []
    heat: list[dict] = []

    # Speech / Sentiment — single embedded dataset each.
    for task_id, dataset_label, dataset_id in [
        ("speech", "TESS", "tess"),
        ("sentiment", "Twitter", "twitter"),
    ]:
        t = tasks[task_id]
        m = t["metrics"]
        best = max(m, key=lambda a: m[a]["accuracy"])
        rows.append({
            "task": "Speech Emotion" if task_id == "speech" else "Sentiment",
            "taskId": task_id, "dataset": dataset_label,
            "best": best, "metric": "Accuracy",
            "score": f"{m[best]['accuracy']}%", "classify": True,
        })
        heat.append({
            "dataset": f"{dataset_label} ({task_id.title()})",
            "ANN": None,
            "RNN": m.get("RNN", {}).get("accuracy"),
            "LSTM": m.get("LSTM", {}).get("accuracy"),
            "GRU": m.get("GRU", {}).get("accuracy"),
            "metric": "Acc",
        })

    # Time-series datasets.
    for ds in tasks["timeseries"]["datasets"]:
        m = ds["metrics"]
        if ds.get("classify"):
            best = max(m, key=lambda a: m[a]["accuracy"])
            rows.append({
                "task": "Time Series", "taskId": "timeseries",
                "dataset": ds["name"], "best": best, "metric": "Accuracy",
                "score": f"{m[best]['accuracy']}%", "classify": True,
            })
        else:
            best = min(m, key=lambda a: m[a].get("rmse", 1e9))
            rows.append({
                "task": "Time Series", "taskId": "timeseries",
                "dataset": ds["name"], "best": best, "metric": "RMSE",
                "score": f"{m[best]['rmse']:.2f}", "classify": False,
            })
        heat.append({
            "dataset": ds["name"],
            "ANN": m.get("ANN", {}).get("accuracy"),
            "RNN": m.get("RNN", {}).get("accuracy"),
            "LSTM": m.get("LSTM", {}).get("accuracy"),
            "GRU": m.get("GRU", {}).get("accuracy"),
            "metric": "Acc" if ds.get("classify") else "Acc*",
        })

    return rows, heat


# ── Bulk read for dashboard first-paint ────────────────────────────────────
@app.get("/api/dashboard")
def dashboard():
    payload = mock_data.build_dashboard()
    # If real cached results exist, splice them into each task slice.
    for task_id, task_obj in payload["tasks"].items():
        if task_id == "timeseries":
            for ds in task_obj["datasets"]:
                merged = _resolve_dataset_slice("timeseries", ds["id"])
                ds.update(
                    {
                        "metrics": merged["metrics"],
                        "curves": merged["curves"],
                        "predVsActual": merged.get("predVsActual", ds.get("predVsActual")),
                        "bestId": merged.get("bestId", ds.get("bestId")),
                    }
                )
        else:
            dataset_id = "tess" if task_id == "speech" else "twitter"
            merged = _resolve_dataset_slice(task_id, dataset_id)
            task_obj["metrics"] = merged["metrics"]
            task_obj["curves"] = merged["curves"]
            if "roc" in merged:
                task_obj["roc"] = merged["roc"]
            if "confusion" in merged:
                task_obj["confusion"] = merged["confusion"]

    # Recompute the cross-task tables from the (now real) per-task metrics.
    payload["summaryRows"], payload["heatmapData"] = _recompute_summary(payload)
    return jsonify(payload)


# ── Dataset metadata ───────────────────────────────────────────────────────
@app.get("/api/datasets")
def datasets():
    out = []
    for task_id, cfg in TASKS.items():
        for ds_id in cfg["datasets"]:
            sl = mock_data.get_dataset_slice(task_id, ds_id)
            if sl is None:
                continue
            out.append(
                {
                    "task": task_id,
                    "datasetId": ds_id,
                    "name": sl["name"],
                    "source": sl["source"],
                    "target": sl["target"],
                    "classify": bool(sl.get("classify", task_id != "timeseries")),
                    "stats": sl["stats"],
                    "classDist": sl.get("classDist"),
                }
            )
    return jsonify(out)


@app.get("/api/dataset/<task>/<dataset_id>/summary")
def dataset_summary(task: str, dataset_id: str):
    sl = mock_data.get_dataset_slice(task, dataset_id)
    if sl is None:
        abort(404, description=f"Unknown task/dataset: {task}/{dataset_id}")
    return jsonify(
        {
            "task": task,
            "datasetId": dataset_id,
            "name": sl["name"],
            "source": sl["source"],
            "type": sl["type"],
            "target": sl["target"],
            "classify": bool(sl.get("classify", task != "timeseries")),
            "stats": sl["stats"],
            "classDist": sl.get("classDist"),
            "pipeline": sl["pipeline"],
            "classes": sl.get("classes"),
        }
    )


# ── Per-model + per-dataset results ────────────────────────────────────────
@app.get("/api/metrics/<task>/<dataset_id>/<model>")
def metrics(task: str, dataset_id: str, model: str):
    cached = _read_results_json(task, dataset_id, model)
    if cached is not None:
        return jsonify(cached)
    out = mock_data.get_model_metrics(task, dataset_id, model)
    if out is None:
        abort(404, description=f"Unknown task/dataset/model: {task}/{dataset_id}/{model}")
    return jsonify(out)


@app.get("/api/compare/<task>/<dataset_id>")
def compare(task: str, dataset_id: str):
    return jsonify(_resolve_dataset_slice(task, dataset_id))


# ── Plots ──────────────────────────────────────────────────────────────────
@app.get("/api/plots/<task>/<dataset_id>/<model>")
def plot_index(task: str, dataset_id: str, model: str):
    base = f"{task}_{dataset_id}_{model}"
    plots = {
        "loss": f"/static/plots/{base}_loss.png",
        "acc": f"/static/plots/{base}_acc.png",
        "roc": f"/static/plots/{base}_roc.png",
        "confusion": f"/static/plots/{base}_confusion.png",
        "predVsActual": f"/static/plots/{base}_pred_vs_actual.png",
    }
    available = {k: v for k, v in plots.items() if (PLOTS_DIR / Path(v).name).exists()}
    return jsonify({"task": task, "datasetId": dataset_id, "model": model, "plots": available})


@app.get("/static/plots/<path:filename>")
def serve_plot(filename: str):
    return send_from_directory(PLOTS_DIR, filename)


# ── Train (Phase 2 stub) ───────────────────────────────────────────────────
@app.post("/api/train")
def train():
    body = request.get_json(silent=True) or {}
    task = body.get("task")
    dataset_id = body.get("datasetId") or body.get("dataset")
    model = body.get("model")
    force = bool(body.get("force_retrain", False))
    overrides = body.get("hyperparams", {}) or {}

    if not (task and dataset_id and model):
        abort(400, description="Required fields: task, datasetId, model.")

    if task not in TASKS or model not in mock_data.get_dataset_slice(task, dataset_id)["algos"]:
        abort(400, description=f"Unsupported combination: {task}/{dataset_id}/{model}")

    # Phase 2: training pipeline not implemented yet. Report cache status so the
    # frontend can wire up the train button without crashing.
    cached = is_cached(task, dataset_id, model)
    base_hp = HYPERPARAMS.get(task, {}).copy()
    base_hp.update(overrides)
    return jsonify(
        {
            "status": "cached" if (cached and not force) else "not_implemented",
            "runId": str(uuid.uuid4()),
            "task": task,
            "datasetId": dataset_id,
            "model": model,
            "hyperparams": base_hp,
            "message": (
                "Cached result available." if cached and not force
                else "Phase 2 stub — real training arrives in Phase 3+."
            ),
        }
    ), (200 if cached and not force else 501)


@app.post("/api/predict")
def predict():
    """Single-sample inference for a previously-trained model.

    Three input shapes:
      - sentiment   → JSON body ``{task, model, text}``
      - speech      → multipart upload, file field ``audio``,
                       form fields ``task=speech`` and ``model=...``
      - timeseries  → JSON body ``{task, datasetId, model, window}`` OR
                       ``{task, datasetId, model, sampleId}`` (uses cached sample)
    """
    from src import inference

    # Speech needs multipart; everything else is JSON.
    if request.content_type and request.content_type.startswith("multipart/"):
        task = (request.form.get("task") or "").strip()
        model = (request.form.get("model") or "").strip()
        f = request.files.get("audio")
        if task != "speech" or not model or f is None:
            abort(400, description="Speech requests need task=speech, model, and audio file.")
        try:
            return jsonify(inference.predict_speech(model, f.read()))
        except FileNotFoundError as exc:
            abort(404, description=str(exc))
        except Exception as exc:
            abort(400, description=f"Inference failed: {exc}")

    body = request.get_json(silent=True) or {}
    task = (body.get("task") or "").strip()
    model = (body.get("model") or "").strip()
    if not task or not model:
        abort(400, description="Required fields: task, model.")

    try:
        if task == "sentiment":
            text = body.get("text") or body.get("input") or ""
            if not text.strip():
                abort(400, description="Empty text input.")
            return jsonify(inference.predict_sentiment(model, text))

        if task == "timeseries":
            dataset_id = body.get("datasetId") or body.get("dataset")
            if not dataset_id:
                abort(400, description="timeseries predictions need datasetId.")
            window = body.get("window")
            sample_id = body.get("sampleId")
            if window is None and sample_id is not None:
                # Look up the cached window by id.
                samples = inference.list_ts_samples(dataset_id)
                match = next((s for s in samples if str(s.get("id")) == str(sample_id)), None)
                if match is None:
                    abort(404, description=f"Unknown sampleId for {dataset_id}: {sample_id!r}")
                result = inference.predict_timeseries(dataset_id, model, match["window"], sample_id)
                # Merge actual + label so the frontend can show side-by-side.
                result["sample"] = {k: match.get(k) for k in ("label", "actual", "actualClass", "date")}
                return jsonify(result)
            if window is None:
                abort(400, description="timeseries predictions need either window[][] or sampleId.")
            return jsonify(inference.predict_timeseries(dataset_id, model, window, sample_id))

        if task == "speech":
            abort(400, description="Speech requires multipart/form-data (audio file).")
    except FileNotFoundError as exc:
        abort(404, description=str(exc))
    except ValueError as exc:
        abort(400, description=str(exc))
    abort(400, description=f"Unknown task: {task!r}")


@app.get("/api/predict/samples/<dataset_id>")
def predict_samples(dataset_id: str):
    """List the pre-built test-set windows the frontend can offer for the
    'try it yourself' picker on time-series datasets."""
    from src import inference
    try:
        samples = inference.list_ts_samples(dataset_id)
    except ValueError as exc:
        abort(400, description=str(exc))
    # Trim out raw windows (large) — frontend doesn't need them, just labels + ids.
    return jsonify([
        {k: v for k, v in s.items() if k != "window"}
        for s in samples
    ])


# ── Error handlers ─────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(err):
    return jsonify({"error": "not_found", "message": str(getattr(err, "description", err))}), 404


@app.errorhandler(400)
def bad_request(err):
    return jsonify({"error": "bad_request", "message": str(getattr(err, "description", err))}), 400


@app.errorhandler(501)
def not_implemented(err):
    return jsonify({"error": "not_implemented", "message": str(getattr(err, "description", err))}), 501


if __name__ == "__main__":
    import os
    print(f"\nResults dir: {RESULTS_DIR}")
    print(f"Plots dir:   {PLOTS_DIR}")
    print("CORS allowed:", ", ".join(ALLOWED_ORIGINS))
    print()
    # Default debug off — the reloader follows TensorFlow's lazy imports and
    # restarts itself in a loop. Set FLASK_DEBUG=1 to opt in.
    debug = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=5050, debug=debug, use_reloader=False)
