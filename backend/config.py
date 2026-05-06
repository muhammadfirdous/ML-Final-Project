"""Centralised paths, seeds, and hyperparameter defaults.

Every module that needs a path or a default reads it from here so that
verify_data.py, the Flask routes, and the training scripts all agree.
"""

import os
import random
from pathlib import Path

import numpy as np

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def seed_tensorflow() -> None:
    """Seed TF only when it's actually imported — keeps cold imports cheap."""
    import tensorflow as tf  # noqa: WPS433 (deliberate lazy import)

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


# ── Paths ───────────────────────────────────────────────────────────────────
BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
TESS_DIR = DATA_DIR / "tess"
TWITTER_DIR = DATA_DIR / "twitter"
WEATHER_DIR = DATA_DIR / "weather"

SAVED_MODELS_DIR = BACKEND_DIR / "saved_models"
RESULTS_DIR = BACKEND_DIR / "results"
PLOTS_DIR = BACKEND_DIR / "plots"

for _d in (SAVED_MODELS_DIR, RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ── Expected dataset files (used by verify_data.py) ─────────────────────────
EXPECTED_FILES = {
    "twitter": [
        TWITTER_DIR / "twitter_training.csv",
        TWITTER_DIR / "twitter_validation.csv",
    ],
    "weather": [
        WEATHER_DIR / "corgis_weather.csv",
        WEATHER_DIR / "DailyDelhiClimateTrain.csv",
        WEATHER_DIR / "DailyDelhiClimateTest.csv",
        WEATHER_DIR / "seattle-weather.csv",
    ],
}
# TESS folder names vary by mirror — some use "ps" shorthand, others use
# "Pleasant_surprise" / "pleasant_surprised". Map every variant to the canonical
# dashboard label.
EMOTION_LABEL_MAP = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "ps": "Surprise",
    "pleasant_surprise": "Surprise",
    "pleasant_surprised": "Surprise",
}
# Canonical emotion keys used by verify_data.py to confirm at least one folder
# per emotion is present.
TESS_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


# ── Algorithm metadata (single source of truth for the dashboard) ──────────
ALGO_COLORS = {
    "ANN": "#fbbf24",
    "RNN": "#60a5fa",
    "LSTM": "#22d3ee",
    "GRU": "#c084fc",
}
ALGO_FULL = {
    "ANN": "Artificial Neural Network",
    "RNN": "Recurrent Neural Network",
    "LSTM": "Long Short-Term Memory",
    "GRU": "Gated Recurrent Unit",
}


# ── Per-task hyperparameter defaults ───────────────────────────────────────
# Frontend doesn't override these yet (see Phase 1 contract). /api/train accepts
# overrides; missing fields fall back to whatever's here.
HYPERPARAMS = {
    "speech": {
        "epochs": 40,
        "batch_size": 32,
        # 174-frame MFCCs need higher LR + temporal pooling to converge.
        # Without these, GRU stays at chance for 40 epochs.
        "learning_rate": 5e-3,
        "dropout": 0.3,
        "hidden_units": {"RNN": 128, "LSTM": [128, 64], "GRU": 128},
        "pool_over_time": True,
        "max_frames": 174,
        "n_mfcc": 40,
        "sample_rate": 24000,
        "early_stopping_patience": 5,
    },
    "sentiment": {
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "dropout": 0.3,
        "hidden_units": {"RNN": 64, "LSTM": 128, "GRU": 128},
        "vocab_size": 24500,
        "embedding_dim": 100,
        "max_seq_len": 50,
        "early_stopping_patience": 5,
    },
    "timeseries": {
        "epochs": 40,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "dropout": 0.3,
        "hidden_units": {
            "ANN": [128, 64],
            "RNN": 64,
            "LSTM": [128, 64],
            "GRU": [96, 48],
        },
        "window_size": 30,
        "early_stopping_patience": 5,
    },
}


# ── Task / dataset registry ─────────────────────────────────────────────────
# Keep aligned with what data.jsx exports. Used to enumerate trainings.
TASKS = {
    "speech": {
        "datasets": ["tess"],
        "algos": ["RNN", "LSTM", "GRU"],
        "classify": True,
    },
    "sentiment": {
        "datasets": ["twitter"],
        "algos": ["RNN", "LSTM", "GRU"],
        "classify": True,
    },
    "timeseries": {
        "datasets": ["corgis", "climate", "seattle"],
        "algos": ["ANN", "RNN", "LSTM", "GRU"],
        # Per-dataset classify flag; matches data.jsx
        "per_dataset_classify": {"corgis": False, "climate": False, "seattle": True},
    },
}


# ── CORS / server ───────────────────────────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
