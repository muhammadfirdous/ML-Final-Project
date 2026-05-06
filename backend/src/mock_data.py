"""Deterministic mock data matching the exact shapes from data.jsx / mockData.js.

The Flask routes serve from this module until real training writes results to
``backend/results/<task>_<dataset>_<model>.json``. Once a results file exists
on disk, the routes prefer that over the mock.

The intent: a fully-shaped /api/dashboard payload from day one so the
frontend can be wired and verified before any TF training runs.
"""

from __future__ import annotations

import math
from typing import Any


# ── Deterministic PRNG (matches the JS makeRand) ───────────────────────────
def _make_rand(seed: int):
    s = [seed]

    def _next() -> float:
        s[0] = (s[0] * 9301 + 49297) % 233280
        return s[0] / 233280

    return _next


# ── Curve / matrix / ROC generators (ports of data.jsx generators) ─────────
def _gen_training(epochs: int, final_loss: float, final_acc: float, seed: int) -> list[dict]:
    r = _make_rand(seed)
    start_loss = max(0.6, final_loss + 0.7)
    start_acc = max(0.2, final_acc - 0.4)
    out: list[dict] = []
    for i in range(epochs):
        t = i / (epochs - 1) if epochs > 1 else 0
        decay = (1 - t) ** 2.0
        train_loss = final_loss + (start_loss - final_loss) * decay + (r() - 0.5) * 0.012
        val_loss = (
            final_loss * 1.12
            + (start_loss * 1.05 - final_loss * 1.12) * decay
            + (r() - 0.5) * 0.025
        )
        train_acc = start_acc + (final_acc - start_acc) * (1 - decay) + (r() - 0.5) * 0.006
        val_acc = (
            start_acc * 0.95
            + (final_acc * 0.97 - start_acc * 0.95) * (1 - decay)
            + (r() - 0.5) * 0.018
        )
        out.append(
            {
                "epoch": i + 1,
                "trainLoss": round(max(0.0, train_loss), 4),
                "valLoss": round(max(0.0, val_loss), 4),
                "trainAcc": round(min(0.99, max(0.0, train_acc)), 4),
                "valAcc": round(min(0.99, max(0.0, val_acc)), 4),
            }
        )
    return out


def _gen_roc(auc: float, seed: int, steps: int = 30) -> list[dict]:
    r = _make_rand(seed)
    power = 1 / max(0.05, auc * 2 - 0.7)
    pts: list[dict] = []
    for i in range(steps + 1):
        fpr = i / steps
        tpr = min(1.0, (fpr ** (1 / power)) * (auc * 1.05) + (r() - 0.5) * 0.012)
        pts.append({"fpr": round(fpr, 3), "tpr": round(max(0.0, min(1.0, tpr)), 3)})
    pts[0] = {"fpr": 0, "tpr": 0}
    pts[-1] = {"fpr": 1, "tpr": 1}
    return pts


def _build_roc(auc_map: dict[str, float]) -> dict[str, Any]:
    series = {
        algo: _gen_roc(auc, 13 + i * 7) for i, (algo, auc) in enumerate(auc_map.items())
    }
    merged: list[dict] = []
    for i in range(31):
        row: dict[str, Any] = {"fpr": i / 30}
        for algo in auc_map:
            row[algo] = series[algo][i]["tpr"]
        merged.append(row)
    return {"merged": merged, "aucMap": auc_map}


def _gen_pred_vs_actual(n: int, rmse: float, seed: int) -> list[dict]:
    r = _make_rand(seed)
    out: list[dict] = []
    base = 18.0
    for i in range(n):
        seasonal = math.sin(i / n * math.pi * 4) * 6 + math.sin(i / n * math.pi * 8) * 2
        drift = i / n * 4 - 2
        noise = (r() - 0.5) * 1.2
        actual = base + seasonal + drift + noise
        pred_err = (r() - 0.5) * rmse * 1.6
        pred = actual + pred_err
        out.append({"day": i + 1, "actual": round(actual, 2), "predicted": round(pred, 2)})
    return out


def _gen_confusion(classes: list[str], accuracy: float, seed: int) -> list[list[int]]:
    r = _make_rand(seed)
    k = len(classes)
    total = 80
    m = [[0] * k for _ in range(k)]
    for i in range(k):
        correct = round(total * accuracy + (r() - 0.5) * 4)
        m[i][i] = max(40, min(total - 4, correct))
        remaining = total - m[i][i]
        for j in range(k):
            if j == i:
                continue
            share = remaining / (k - 1)
            v = max(0, round(share + (r() - 0.5) * share))
            m[i][j] = v
        # Renormalise so each row sums to total.
        row_sum = sum(m[i])
        diff = total - row_sum
        for j in range(k - 1, -1, -1):
            if j != i and m[i][j] + diff >= 0:
                m[i][j] += diff
                break
    return m


# ── TASK 1: Speech (TESS) ──────────────────────────────────────────────────
EMOTION_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def _task_speech() -> dict[str, Any]:
    metrics = {
        "RNN": {
            "accuracy": 78, "precision": 0.77, "recall": 0.76, "f1": 0.76, "auc": 0.84,
            "params": "58,247", "trainingTime": "6m 12s",
        },
        "LSTM": {
            "accuracy": 88, "precision": 0.89, "recall": 0.87, "f1": 0.88, "auc": 0.94,
            "params": "142,983", "trainingTime": "11m 04s", "best": True,
        },
        "GRU": {
            "accuracy": 86, "precision": 0.86, "recall": 0.85, "f1": 0.85, "auc": 0.92,
            "params": "108,601", "trainingTime": "8m 47s",
        },
    }
    return {
        "id": "speech",
        "num": "01",
        "kicker": "Task 1",
        "title": "Speech Emotion Classification",
        "subtitle": "Recognising 7 emotional states from short Toronto Emotional Speech Set (TESS) recordings.",
        "iconName": "AudioWaveform",
        "dataset": {
            "name": "Toronto Emotional Speech Set (TESS)",
            "source": "kaggle.com/ejlok1/toronto-emotional-speech-set-tess",
            "type": "Audio (.wav)",
            "target": "7 emotion classes",
            "stats": [
                {"label": "Audio clips", "value": "2,800", "icon": "Database"},
                {"label": "Emotion classes", "value": "7", "icon": "Layers"},
                {"label": "Sample rate", "value": "24 kHz", "icon": "Activity"},
                {"label": "Avg clip length", "value": "2.1 s", "icon": "Clock"},
                {"label": "Speakers", "value": "2", "icon": "Users"},
                {"label": "MFCC coefficients", "value": "40", "icon": "BarChart3", "mono": True},
            ],
            "classDist": [
                {"name": c, "value": 400 + (i * 7) % 30}
                for i, c in enumerate(EMOTION_CLASSES)
            ],
        },
        "pipeline": [
            {"id": "load", "title": "Load Audio", "icon": "FolderOpen",
             "desc": "Read .wav files with librosa, preserve original sample rates, attach emotion labels from filename."},
            {"id": "resample", "title": "Resample", "icon": "Repeat",
             "desc": "Resample every clip to a uniform 24 kHz so feature extraction is comparable across speakers."},
            {"id": "mfcc", "title": "MFCC Extract", "icon": "AudioWaveform",
             "desc": "40 Mel-frequency cepstral coefficients per frame — captures timbre, the cue most correlated with emotion."},
            {"id": "pad", "title": "Pad / Truncate", "icon": "Maximize2",
             "desc": "Pad shorter clips and truncate longer ones to 174 frames so the model sees fixed-length sequences."},
            {"id": "split", "title": "Train / Test", "icon": "SplitSquareHorizontal",
             "desc": "Stratified 80/20 split by emotion class — keeps the class balance identical between train and test."},
        ],
        "algos": ["RNN", "LSTM", "GRU"],
        "metrics": metrics,
        "arch": {
            "RNN": [
                {"name": "Input", "shape": "(174, 40)", "type": "input"},
                {"name": "SimpleRNN", "shape": "128", "type": "rnn", "act": "tanh"},
                {"name": "Dropout", "shape": "0.3", "type": "reg"},
                {"name": "Dense", "shape": "7", "type": "dense", "act": "softmax"},
            ],
            "LSTM": [
                {"name": "Input", "shape": "(174, 40)", "type": "input"},
                {"name": "LSTM", "shape": "128", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dropout", "shape": "0.4", "type": "reg"},
                {"name": "LSTM", "shape": "64", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dense", "shape": "32", "type": "dense", "act": "ReLU"},
                {"name": "Dense", "shape": "7", "type": "dense", "act": "softmax"},
            ],
            "GRU": [
                {"name": "Input", "shape": "(174, 40)", "type": "input"},
                {"name": "GRU", "shape": "96", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dropout", "shape": "0.3", "type": "reg"},
                {"name": "GRU", "shape": "48", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dense", "shape": "7", "type": "dense", "act": "softmax"},
            ],
        },
        "archMeta": {
            "RNN": {"activation": "tanh + softmax",
                    "regularization": "Dropout (0.3), Early Stopping (p=10)"},
            "LSTM": {"activation": "tanh, sigmoid, ReLU, softmax",
                     "regularization": "Dropout (0.4), L2 (1e-4), Early Stopping (p=12)"},
            "GRU": {"activation": "tanh, sigmoid, softmax",
                    "regularization": "Dropout (0.3), Early Stopping (p=10)"},
        },
        "curves": {
            "RNN": _gen_training(40, 0.55, 0.78, 11),
            "LSTM": _gen_training(40, 0.32, 0.88, 23),
            "GRU": _gen_training(40, 0.38, 0.86, 41),
        },
        "roc": _build_roc({"RNN": 0.84, "LSTM": 0.94, "GRU": 0.92}),
        "confusion": {
            "RNN": _gen_confusion(EMOTION_CLASSES, 0.78, 11),
            "LSTM": _gen_confusion(EMOTION_CLASSES, 0.88, 23),
            "GRU": _gen_confusion(EMOTION_CLASSES, 0.86, 41),
        },
        "classes": EMOTION_CLASSES,
    }


# ── TASK 2: Sentiment (Twitter) ────────────────────────────────────────────
SENTIMENT_CLASSES = ["Positive", "Negative", "Neutral", "Irrelevant"]


def _task_sentiment() -> dict[str, Any]:
    return {
        "id": "sentiment",
        "num": "02",
        "kicker": "Task 2",
        "title": "Sentiment Analysis",
        "subtitle": "Classifying tweets into Positive, Negative, Neutral, or Irrelevant given a target entity.",
        "iconName": "MessageSquare",
        "dataset": {
            "name": "Twitter Entity Sentiment Analysis",
            "source": "kaggle.com/jp797498e/twitter-entity-sentiment-analysis",
            "type": "Text",
            "target": "4 sentiment classes",
            "stats": [
                {"label": "Total tweets", "value": "74,682", "icon": "Database"},
                {"label": "Sentiment classes", "value": "4", "icon": "Layers"},
                {"label": "Unique entities", "value": "32", "icon": "Tag"},
                {"label": "Avg tweet length", "value": "18 tok", "icon": "Type"},
                {"label": "Vocab size", "value": "24,500", "icon": "Hash"},
                {"label": "Embedding dim", "value": "100", "icon": "BarChart3", "mono": True},
            ],
            "classDist": [
                {"name": "Positive", "value": 20832},
                {"name": "Negative", "value": 22542},
                {"name": "Neutral", "value": 18318},
                {"name": "Irrelevant", "value": 12990},
            ],
        },
        "pipeline": [
            {"id": "lower", "title": "Lowercase", "icon": "CaseLower",
             "desc": 'Force every character to lowercase so "Apple" and "apple" share a single embedding.'},
            {"id": "tokenize", "title": "Tokenize", "icon": "Type",
             "desc": "Whitespace + punctuation tokenizer; URLs, mentions, and hashtags collapsed to special tokens."},
            {"id": "stop", "title": "Stopword Filter", "icon": "Filter",
             "desc": 'Drop common English stopwords ("the", "is", "and") that contribute no sentiment signal.'},
            {"id": "embed", "title": "Embedding", "icon": "Hash",
             "desc": "100-dim trainable embedding initialised on the training vocabulary — 24,500 unique tokens."},
            {"id": "pad", "title": "Pad Sequences", "icon": "Maximize2",
             "desc": "Pad/truncate every tokenised tweet to length 50 — 95th percentile of the training distribution."},
        ],
        "algos": ["RNN", "LSTM", "GRU"],
        "metrics": {
            "RNN": {"accuracy": 72, "precision": 0.71, "recall": 0.70, "f1": 0.70, "auc": 0.78,
                    "params": "3.21M", "trainingTime": "4m 50s"},
            "LSTM": {"accuracy": 84, "precision": 0.85, "recall": 0.83, "f1": 0.84, "auc": 0.91,
                     "params": "3.42M", "trainingTime": "9m 18s", "best": True},
            "GRU": {"accuracy": 82, "precision": 0.83, "recall": 0.81, "f1": 0.82, "auc": 0.89,
                    "params": "3.38M", "trainingTime": "7m 22s"},
        },
        "arch": {
            "RNN": [
                {"name": "Input", "shape": "(50,)", "type": "input"},
                {"name": "Embedding", "shape": "24500x100", "type": "dense"},
                {"name": "SimpleRNN", "shape": "64", "type": "rnn", "act": "tanh"},
                {"name": "Dropout", "shape": "0.3", "type": "reg"},
                {"name": "Dense", "shape": "4", "type": "dense", "act": "softmax"},
            ],
            "LSTM": [
                {"name": "Input", "shape": "(50,)", "type": "input"},
                {"name": "Embedding", "shape": "24500x100", "type": "dense"},
                {"name": "LSTM", "shape": "128", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dropout", "shape": "0.4", "type": "reg"},
                {"name": "Dense", "shape": "32", "type": "dense", "act": "ReLU"},
                {"name": "Dense", "shape": "4", "type": "dense", "act": "softmax"},
            ],
            "GRU": [
                {"name": "Input", "shape": "(50,)", "type": "input"},
                {"name": "Embedding", "shape": "24500x100", "type": "dense"},
                {"name": "GRU", "shape": "128", "type": "rnn", "act": "tanh / sigmoid"},
                {"name": "Dropout", "shape": "0.3", "type": "reg"},
                {"name": "Dense", "shape": "4", "type": "dense", "act": "softmax"},
            ],
        },
        "archMeta": {
            "RNN": {"activation": "tanh + softmax",
                    "regularization": "Dropout (0.3), Early Stopping (p=8)"},
            "LSTM": {"activation": "tanh, sigmoid, ReLU, softmax",
                     "regularization": "Dropout (0.4), Early Stopping (p=10)"},
            "GRU": {"activation": "tanh, sigmoid, softmax",
                    "regularization": "Dropout (0.3), Early Stopping (p=10)"},
        },
        "curves": {
            "RNN": _gen_training(30, 0.62, 0.72, 51),
            "LSTM": _gen_training(30, 0.38, 0.84, 53),
            "GRU": _gen_training(30, 0.42, 0.82, 55),
        },
        "roc": _build_roc({"RNN": 0.78, "LSTM": 0.91, "GRU": 0.89}),
        "classes": SENTIMENT_CLASSES,
    }


# ── TASK 3: Time Series (3 datasets) ───────────────────────────────────────
TS_PIPELINE = [
    {"id": "raw", "title": "Raw Data", "icon": "Database",
     "desc": "Daily observations across one or more climate variables loaded from CSV."},
    {"id": "clean", "title": "Cleaning", "icon": "Sparkles",
     "desc": "Outliers detected via IQR, duplicate timestamps removed, anomalous readings clipped to 4σ."},
    {"id": "norm", "title": "Normalization", "icon": "Sliders",
     "desc": "MinMax scaling per feature → every channel lives in [0, 1] so magnitude bias dies."},
    {"id": "window", "title": "Windowing", "icon": "LayoutGrid",
     "desc": "Sliding window of 30 days predicts day 31. Stride = 1. Sequence length is uniform across models."},
    {"id": "split", "title": "Train / Test", "icon": "SplitSquareHorizontal",
     "desc": "Chronological 80/20 split — no leakage from future into past."},
]
TS_ALGOS = ["ANN", "RNN", "LSTM", "GRU"]
TS_ARCH = {
    "ANN": [
        {"name": "Input", "shape": "(150,)", "type": "input"},
        {"name": "Dense", "shape": "128", "type": "dense", "act": "ReLU"},
        {"name": "Dropout", "shape": "0.2", "type": "reg"},
        {"name": "Dense", "shape": "64", "type": "dense", "act": "ReLU"},
        {"name": "Dense", "shape": "1", "type": "dense", "act": "Linear"},
    ],
    "RNN": [
        {"name": "Input", "shape": "(30, F)", "type": "input"},
        {"name": "SimpleRNN", "shape": "64", "type": "rnn", "act": "tanh"},
        {"name": "Dropout", "shape": "0.3", "type": "reg"},
        {"name": "Dense", "shape": "1", "type": "dense", "act": "Linear"},
    ],
    "LSTM": [
        {"name": "Input", "shape": "(30, F)", "type": "input"},
        {"name": "LSTM", "shape": "128", "type": "rnn", "act": "tanh / sigmoid"},
        {"name": "Dropout", "shape": "0.3", "type": "reg"},
        {"name": "LSTM", "shape": "64", "type": "rnn", "act": "tanh / sigmoid"},
        {"name": "Dense", "shape": "1", "type": "dense", "act": "Linear"},
    ],
    "GRU": [
        {"name": "Input", "shape": "(30, F)", "type": "input"},
        {"name": "GRU", "shape": "96", "type": "rnn", "act": "tanh / sigmoid"},
        {"name": "Dropout", "shape": "0.3", "type": "reg"},
        {"name": "GRU", "shape": "48", "type": "rnn", "act": "tanh / sigmoid"},
        {"name": "Dense", "shape": "1", "type": "dense", "act": "Linear"},
    ],
}
TS_ARCH_META = {
    "ANN":  {"activation": "ReLU + Linear",
             "regularization": "Dropout (0.2), Early Stopping (p=8)"},
    "RNN":  {"activation": "tanh + Linear",
             "regularization": "Dropout (0.3), Early Stopping (p=10)"},
    "LSTM": {"activation": "tanh, sigmoid, Linear",
             "regularization": "Dropout (0.3), L2 (1e-4), ES (p=12)"},
    "GRU":  {"activation": "tanh, sigmoid, Linear",
             "regularization": "Dropout (0.3), Early Stopping (p=10)"},
}


def _build_ts_dataset(spec: dict[str, Any]) -> dict[str, Any]:
    classify = bool(spec.get("classify"))
    metrics = spec["metrics"]
    pred_seed = spec["predSeed"]

    curves: dict[str, list[dict]] = {}
    for a in TS_ALGOS:
        if classify:
            final_loss = 1 - (metrics[a]["accuracy"] / 100) + 0.05
        else:
            final_loss = min(0.6, metrics[a]["rmse"] / 10)
        final_acc = (
            metrics[a]["accuracy"] / 100 if classify else 0.5 + (1 - final_loss) * 0.5
        )
        curves[a] = _gen_training(40, final_loss, final_acc, pred_seed + ord(a[0]))

    pred_vs_actual = {
        a: _gen_pred_vs_actual(60, metrics[a].get("rmse", 2.5) or 2.5, pred_seed + ord(a[0]) * 3)
        for a in TS_ALGOS
    }

    if classify:
        best_id = max(TS_ALGOS, key=lambda a: metrics[a]["accuracy"])
    else:
        best_id = min(TS_ALGOS, key=lambda a: metrics[a]["rmse"])

    return {
        "id": spec["id"],
        "name": spec["name"],
        "source": spec["source"],
        "type": spec["type"],
        "target": spec["target"],
        "stats": spec["stats"],
        "classify": classify,
        "metrics": metrics,
        "bestId": best_id,
        "curves": curves,
        "predVsActual": pred_vs_actual,
        "arch": TS_ARCH,
        "archMeta": TS_ARCH_META,
        "pipeline": TS_PIPELINE,
        "algos": TS_ALGOS,
    }


_TS_SPECS = [
    {
        "id": "corgis",
        "name": "CORGIS Weather",
        "source": "corgis-edu.github.io/corgis/csv/weather",
        "type": "Tabular time series",
        "target": "Mean temperature (°C)",
        "stats": [
            {"label": "Records", "value": "16,743", "icon": "Database"},
            {"label": "Features", "value": "14", "icon": "Layers"},
            {"label": "Stations", "value": "49", "icon": "MapPin"},
            {"label": "Time span", "value": "2016", "icon": "Calendar"},
            {"label": "Frequency", "value": "Daily", "icon": "Clock"},
            {"label": "Target", "value": "Mean Temp", "icon": "Target", "mono": True},
        ],
        "metrics": {
            "ANN":  {"accuracy": 78, "mae": 2.94, "mape": 6.2, "rmse": 3.8, "params": "52,481",  "trainingTime": "2m 44s"},
            "RNN":  {"accuracy": 82, "mae": 2.43, "mape": 5.1, "rmse": 3.2, "params": "64,193",  "trainingTime": "5m 18s"},
            "LSTM": {"accuracy": 88, "mae": 1.92, "mape": 4.0, "rmse": 2.6, "params": "128,917", "trainingTime": "8m 32s"},
            "GRU":  {"accuracy": 87, "mae": 2.05, "mape": 4.2, "rmse": 2.7, "params": "96,801",  "trainingTime": "6m 41s"},
        },
        "predSeed": 71,
    },
    {
        "id": "climate",
        "name": "Daily Climate",
        "source": "kaggle.com/sumanthvrao/daily-climate-time-series-data",
        "type": "Daily climate measurements",
        "target": "Mean temperature (°C)",
        "stats": [
            {"label": "Records", "value": "1,576", "icon": "Database"},
            {"label": "Features", "value": "5", "icon": "Layers"},
            {"label": "Time range", "value": "2013–2017", "icon": "Calendar"},
            {"label": "Frequency", "value": "Daily", "icon": "Clock"},
            {"label": "Missing", "value": "0", "icon": "CheckCircle2"},
            {"label": "Target", "value": "meantemp", "icon": "Target", "mono": True},
        ],
        "metrics": {
            "ANN":  {"accuracy": 81, "mae": 2.21, "mape": 7.8, "rmse": 2.9, "params": "48,193",  "trainingTime": "2m 14s"},
            "RNN":  {"accuracy": 85, "mae": 1.81, "mape": 6.4, "rmse": 2.4, "params": "62,401",  "trainingTime": "4m 32s"},
            "LSTM": {"accuracy": 92, "mae": 1.42, "mape": 4.9, "rmse": 1.9, "params": "124,673", "trainingTime": "7m 48s"},
            "GRU":  {"accuracy": 90, "mae": 1.51, "mape": 5.2, "rmse": 2.0, "params": "94,209",  "trainingTime": "5m 56s"},
        },
        "predSeed": 41,
    },
    {
        "id": "seattle",
        "name": "Seattle Weather",
        "source": "kaggle.com/petalme/seattle-weather-prediction-dataset",
        "type": "Daily weather records",
        "target": "Weather category (5 classes)",
        "classify": True,
        "stats": [
            {"label": "Records", "value": "1,461", "icon": "Database"},
            {"label": "Features", "value": "5", "icon": "Layers"},
            {"label": "Classes", "value": "5", "icon": "Tag"},
            {"label": "Time range", "value": "2012–2015", "icon": "Calendar"},
            {"label": "Frequency", "value": "Daily", "icon": "Clock"},
            {"label": "Target", "value": "weather", "icon": "Target", "mono": True},
        ],
        "metrics": {
            "ANN":  {"accuracy": 74, "precision": 0.73, "recall": 0.71, "f1": 0.71, "rmse": 0, "params": "46,721",  "trainingTime": "2m 02s"},
            "RNN":  {"accuracy": 78, "precision": 0.77, "recall": 0.76, "f1": 0.76, "rmse": 0, "params": "60,201",  "trainingTime": "4m 11s"},
            "LSTM": {"accuracy": 85, "precision": 0.86, "recall": 0.84, "f1": 0.85, "rmse": 0, "params": "120,481", "trainingTime": "7m 24s"},
            "GRU":  {"accuracy": 83, "precision": 0.84, "recall": 0.82, "f1": 0.83, "rmse": 0, "params": "92,018",  "trainingTime": "5m 41s"},
        },
        "predSeed": 89,
    },
]


def _task_timeseries() -> dict[str, Any]:
    return {
        "id": "timeseries",
        "num": "03",
        "kicker": "Task 3",
        "title": "Time-Series Forecasting",
        "subtitle": "Three weather datasets, the full ANN/RNN/LSTM/GRU lineup, head-to-head on each.",
        "iconName": "LineChart",
        "datasets": [_build_ts_dataset(s) for s in _TS_SPECS],
    }


# ── Cross-task tables ──────────────────────────────────────────────────────
SUMMARY_ROWS = [
    {"task": "Speech Emotion", "taskId": "speech", "dataset": "TESS",
     "best": "LSTM", "metric": "Accuracy", "score": "88%", "classify": True},
    {"task": "Sentiment", "taskId": "sentiment", "dataset": "Twitter",
     "best": "LSTM", "metric": "Accuracy", "score": "84%", "classify": True},
    {"task": "Time Series", "taskId": "timeseries", "dataset": "CORGIS Weather",
     "best": "LSTM", "metric": "RMSE", "score": "2.6", "classify": False},
    {"task": "Time Series", "taskId": "timeseries", "dataset": "Daily Climate",
     "best": "LSTM", "metric": "RMSE", "score": "1.9", "classify": False},
    {"task": "Time Series", "taskId": "timeseries", "dataset": "Seattle Weather",
     "best": "LSTM", "metric": "Accuracy", "score": "85%", "classify": True},
]

HEATMAP_DATA = [
    {"dataset": "TESS (Speech)",       "ANN": None, "RNN": 78, "LSTM": 88, "GRU": 86, "metric": "Acc"},
    {"dataset": "Twitter (Sentiment)", "ANN": None, "RNN": 72, "LSTM": 84, "GRU": 82, "metric": "Acc"},
    {"dataset": "CORGIS Weather",      "ANN": 78,   "RNN": 82, "LSTM": 88, "GRU": 87, "metric": "Acc*"},
    {"dataset": "Daily Climate",       "ANN": 81,   "RNN": 85, "LSTM": 92, "GRU": 90, "metric": "Acc*"},
    {"dataset": "Seattle Weather",     "ANN": 74,   "RNN": 78, "LSTM": 85, "GRU": 83, "metric": "Acc"},
]

TEAM = [
    {
        "name": "Muhammad Firdous",
        "role": "Sole author — full pipeline",
        "initials": "MF",
        "contribution": (
            "Built the full pipeline end-to-end: TESS audio preprocessing & MFCC extraction, "
            "Twitter text tokenisation & embedding training, three time-series datasets, all 18 "
            "model trainings (ANN/RNN/LSTM/GRU), and the comparative analysis writeup."
        ),
        "worked": [
            "TESS · RNN/LSTM/GRU",
            "Twitter · RNN/LSTM/GRU",
            "CORGIS · ANN/RNN/LSTM/GRU",
            "Daily Climate · ANN/RNN/LSTM/GRU",
            "Seattle · ANN/RNN/LSTM/GRU",
        ],
    },
]

PROJECT_MAP = [
    {"task": "Speech",     "icon": "AudioWaveform", "datasets": ["TESS"],
     "algos": ["RNN", "LSTM", "GRU"]},
    {"task": "Sentiment",  "icon": "MessageSquare", "datasets": ["Twitter Entity"],
     "algos": ["RNN", "LSTM", "GRU"]},
    {"task": "Time Series", "icon": "LineChart",
     "datasets": ["CORGIS", "Daily Climate", "Seattle Weather"],
     "algos": ["ANN", "RNN", "LSTM", "GRU"]},
]

HERO_STATS = [
    {"value": 3,  "suffix": "", "label": "Task categories"},
    {"value": 5,  "suffix": "", "label": "Datasets"},
    {"value": 4,  "suffix": "", "label": "Algorithm families"},
    {"value": 18, "suffix": "", "label": "Models trained"},
]


# ── Public API ─────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "ANN": "#fbbf24", "RNN": "#60a5fa", "LSTM": "#22d3ee", "GRU": "#c084fc",
}
ALGO_FULL = {
    "ANN": "Artificial Neural Network",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short-Term Memory",
    "GRU": "Gated Recurrent Unit",
}


def build_dashboard() -> dict[str, Any]:
    """Bundle endpoint payload — every shape data.jsx exports, in one object."""
    return {
        "algoColors": ALGO_COLORS,
        "algoFull": ALGO_FULL,
        "heroStats": HERO_STATS,
        "projectMap": PROJECT_MAP,
        "team": TEAM,
        "summaryRows": SUMMARY_ROWS,
        "heatmapData": HEATMAP_DATA,
        "tasks": {
            "speech": _task_speech(),
            "sentiment": _task_sentiment(),
            "timeseries": _task_timeseries(),
        },
    }


def get_task(task_id: str) -> dict[str, Any] | None:
    if task_id == "speech":
        return _task_speech()
    if task_id == "sentiment":
        return _task_sentiment()
    if task_id == "timeseries":
        return _task_timeseries()
    return None


def get_dataset_slice(task_id: str, dataset_id: str) -> dict[str, Any] | None:
    """Return the per-task/per-dataset slice the dashboard renders."""
    task = get_task(task_id)
    if task is None:
        return None
    if task_id == "timeseries":
        for ds in task["datasets"]:
            if ds["id"] == dataset_id:
                return ds
        return None
    # speech and sentiment have a single dataset embedded directly in the task.
    expected = {"speech": "tess", "sentiment": "twitter"}.get(task_id)
    if dataset_id != expected:
        return None
    out: dict[str, Any] = {
        "id": dataset_id,
        "name": task["dataset"]["name"],
        "source": task["dataset"]["source"],
        "type": task["dataset"]["type"],
        "target": task["dataset"]["target"],
        "stats": task["dataset"]["stats"],
        "classDist": task["dataset"]["classDist"],
        "classify": True,
        "metrics": task["metrics"],
        "bestId": next(a for a, m in task["metrics"].items() if m.get("best")),
        "curves": task["curves"],
        "roc": task["roc"],
        "classes": task["classes"],
        "arch": task["arch"],
        "archMeta": task["archMeta"],
        "pipeline": task["pipeline"],
        "algos": task["algos"],
    }
    if "confusion" in task:
        out["confusion"] = task["confusion"]
    return out


def get_model_metrics(task_id: str, dataset_id: str, model: str) -> dict[str, Any] | None:
    """Per-model slice: metrics + curves + (classification: roc/confusion) +
    (regression-style TS: predVsActual)."""
    sl = get_dataset_slice(task_id, dataset_id)
    if sl is None or model not in sl["metrics"]:
        return None
    out: dict[str, Any] = {
        "task": task_id,
        "datasetId": dataset_id,
        "model": model,
        "metrics": sl["metrics"][model],
        "curve": sl["curves"][model],
        "classify": bool(sl.get("classify", True)),
    }
    if "roc" in sl:
        out["roc"] = {
            "fpr": [pt["fpr"] for pt in sl["roc"]["merged"]],
            "tpr": [pt[model] for pt in sl["roc"]["merged"]],
            "auc": sl["roc"]["aucMap"][model],
        }
    if "confusion" in sl:
        out["confusion"] = sl["confusion"][model]
    if "predVsActual" in sl:
        out["predVsActual"] = sl["predVsActual"][model]
    if "classes" in sl:
        out["classes"] = sl["classes"]
    return out
