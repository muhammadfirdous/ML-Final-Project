"""Inference layer for the /api/predict endpoint.

A ``Predictor`` loads a trained .h5 once and the per-task preprocessing
artifacts (Tokenizer for sentiment, MFCC mean/std for speech, MinMaxScaler
for time-series) once, then exposes ``.predict(input)`` for single samples.

Artifacts are cached to ``saved_models/<task>_<dataset>.preproc.pkl`` after
their first build so subsequent process restarts are fast.
"""

from __future__ import annotations

import io
import pickle
import re
from typing import Any

import numpy as np

from config import HYPERPARAMS, SAVED_MODELS_DIR


# ── Artifact cache (per-task/dataset, separate from per-model H5) ──────────
_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}


def _artifacts_path(task: str, dataset_id: str):
    return SAVED_MODELS_DIR / f"{task}_{dataset_id}.preproc.pkl"


def get_artifacts(task: str, dataset_id: str) -> dict[str, Any]:
    key = f"{task}/{dataset_id}"
    if key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[key]

    path = _artifacts_path(task, dataset_id)
    if path.exists():
        with open(path, "rb") as f:
            artifacts = pickle.load(f)
        _ARTIFACT_CACHE[key] = artifacts
        return artifacts

    # Build for the first time; pickle for next process.
    from src.preprocessing import build_artifacts
    hp = HYPERPARAMS[task]
    artifacts = build_artifacts(task, dataset_id, hp)
    try:
        with open(path, "wb") as f:
            pickle.dump(artifacts, f)
    except Exception as exc:  # pragma: no cover
        print(f"   warning: failed to pickle artifacts to {path}: {exc}")
    _ARTIFACT_CACHE[key] = artifacts
    return artifacts


# ── Model cache ────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, Any] = {}


def get_model(task: str, dataset_id: str, model_name: str):
    key = f"{task}/{dataset_id}/{model_name}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from tensorflow.keras.models import load_model

    h5 = SAVED_MODELS_DIR / f"{task}_{dataset_id}_{model_name}.h5"
    if not h5.exists():
        raise FileNotFoundError(
            f"No trained model at {h5}. Train it first with "
            f"`python -m src.train {task}/{dataset_id}/{model_name}`."
        )
    m = load_model(h5, compile=False)
    _MODEL_CACHE[key] = m
    return m


# ── Per-task predictors ────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_tweet_runtime(text: str, stopwords: set[str]) -> str:
    s = (text or "").lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return " ".join(t for t in s.split() if t not in stopwords)


def predict_sentiment(model_name: str, text: str) -> dict[str, Any]:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    from src.preprocessing import _load_stopwords

    art = get_artifacts("sentiment", "twitter")
    model = get_model("sentiment", "twitter", model_name)

    cleaned = _clean_tweet_runtime(text, _load_stopwords())
    seq = art["tokenizer"].texts_to_sequences([cleaned])
    X = pad_sequences(seq, maxlen=art["max_seq_len"], padding="post", truncating="post")
    proba = model.predict(X, verbose=0)[0]
    classes = art["classes"]
    idx = int(np.argmax(proba))

    return {
        "task": "sentiment",
        "datasetId": "twitter",
        "model": model_name,
        "input": text,
        "cleaned": cleaned,
        "predictedClass": classes[idx],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }


def predict_speech(model_name: str, wav_bytes: bytes) -> dict[str, Any]:
    import librosa  # noqa: WPS433

    art = get_artifacts("speech", "tess")
    model = get_model("speech", "tess", model_name)

    # Read from bytes via soundfile (librosa accepts a file-like object).
    y, _sr = librosa.load(io.BytesIO(wav_bytes), sr=art["sample_rate"], mono=True)
    m = librosa.feature.mfcc(y=y, sr=art["sample_rate"], n_mfcc=art["n_mfcc"]).T
    if m.shape[0] >= art["max_frames"]:
        m = m[: art["max_frames"]]
    else:
        pad = np.zeros((art["max_frames"] - m.shape[0], art["n_mfcc"]), dtype=np.float32)
        m = np.concatenate([m, pad], axis=0)
    # mfcc_mean / std are saved with keepdims=(N, time) → shape (1, 1, n_mfcc).
    # Squeeze to (1, n_mfcc) so the broadcast against a (n_frames, n_mfcc)
    # single clip produces (n_frames, n_mfcc) without inventing a leading axis.
    mu = np.asarray(art["mfcc_mean"]).reshape(1, -1)
    sigma = np.asarray(art["mfcc_std"]).reshape(1, -1)
    m = ((m.astype(np.float32) - mu) / sigma).astype(np.float32)
    X = m[None, ...]

    proba = model.predict(X, verbose=0)[0]
    classes = art["classes"]
    idx = int(np.argmax(proba))

    return {
        "task": "speech",
        "datasetId": "tess",
        "model": model_name,
        "predictedClass": classes[idx],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "audioDurationSec": round(float(len(y) / art["sample_rate"]), 2),
    }


def predict_timeseries(
    dataset_id: str,
    model_name: str,
    window: list[list[float]],
    sample_id: str | None = None,
) -> dict[str, Any]:
    art = get_artifacts("timeseries", dataset_id)
    model = get_model("timeseries", dataset_id, model_name)

    win = np.asarray(window, dtype=np.float32)
    if win.ndim != 2 or win.shape[0] != art["window_size"]:
        raise ValueError(
            f"Expected window of shape ({art['window_size']}, n_features); "
            f"got {tuple(win.shape)}"
        )
    X = win[None, ...]

    if art["classify"]:
        proba = model.predict(X, verbose=0)[0]
        classes = art["classes"]
        idx = int(np.argmax(proba))
        return {
            "task": "timeseries",
            "datasetId": dataset_id,
            "model": model_name,
            "predictedClass": classes[idx],
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "sampleId": sample_id,
        }

    pred_scaled = float(model.predict(X, verbose=0)[0, 0])
    # Inverse-transform back to real units.
    target_idx = art["target_idx"]
    n_features = art["scaler"].n_features_in_
    tmp = np.zeros((1, n_features), dtype=np.float32)
    tmp[0, target_idx] = pred_scaled
    pred = float(art["scaler"].inverse_transform(tmp)[0, target_idx])

    return {
        "task": "timeseries",
        "datasetId": dataset_id,
        "model": model_name,
        "predictedValue": round(pred, 2),
        "targetCol": art["target_col"],
        "sampleId": sample_id,
    }


def list_ts_samples(dataset_id: str) -> list[dict[str, Any]]:
    """Return pre-computed sample windows + actuals for the frontend's
    'try it yourself' picker on time-series datasets."""
    art = get_artifacts("timeseries", dataset_id)
    return art.get("sample_windows", [])
