"""Task-specific preprocessing.

Each function takes raw output from ``data_loader`` and returns a dict of
arrays + metadata keyed in a way ``train.run_training`` understands.

Common return shape:
    {
        "X_train": np.ndarray,
        "y_train": np.ndarray,
        "X_val":   np.ndarray,
        "y_val":   np.ndarray,
        "input_shape": tuple,
        "output_dim":  int,
        "classify":    bool,
        "classes":     list[str] | None,
        "scaler":      sklearn.preprocessing.MinMaxScaler | None,  # for inverse_transform
        "target_idx":  int | None,                                 # column index of y in the scaler
        "test_dates":  np.ndarray | None,                          # for pred-vs-actual plots
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


# ── Common windowing helper ────────────────────────────────────────────────
def _make_windows(values: np.ndarray, target_idx: int, window: int):
    """Sliding window: predict step t+1 from steps [t-window+1 .. t]."""
    X, y = [], []
    for i in range(len(values) - window):
        X.append(values[i : i + window])
        y.append(values[i + window, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# ── TESS speech ─────────────────────────────────────────────────────────────
def prep_tess(wav_paths: list[Path], labels: list[str], hp: dict[str, Any]) -> dict[str, Any]:
    """librosa load → resample → 40 MFCC → pad/truncate to max_frames → label encode."""
    import librosa  # heavy import, lazy

    sr_target = hp["sample_rate"]
    n_mfcc = hp["n_mfcc"]
    max_frames = hp["max_frames"]

    feats: list[np.ndarray] = []
    print(f"[TESS] extracting MFCCs from {len(wav_paths)} wav files…")
    for i, p in enumerate(wav_paths, 1):
        y, sr = librosa.load(p, sr=sr_target, mono=True)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (frames, n_mfcc)
        if m.shape[0] >= max_frames:
            m = m[:max_frames]
        else:
            pad = np.zeros((max_frames - m.shape[0], n_mfcc), dtype=np.float32)
            m = np.concatenate([m, pad], axis=0)
        feats.append(m.astype(np.float32))
        if i % 200 == 0:
            print(f"[TESS]   {i}/{len(wav_paths)} done")
    X = np.stack(feats)  # (N, max_frames, n_mfcc)
    # Z-score per MFCC coefficient using ALL frames — raw MFCCs span ~[-700, 200]
    # which saturates RNN gating (especially GRU). Without this, GRU stays at
    # chance for the entire training run.
    mu = X.mean(axis=(0, 1), keepdims=True)
    sigma = X.std(axis=(0, 1), keepdims=True) + 1e-6
    X = ((X - mu) / sigma).astype(np.float32)

    classes = sorted(set(labels))
    le = LabelEncoder()
    le.fit(classes)
    y = le.transform(labels)
    n_classes = len(classes)

    # Stratified 80/20 split.
    rng = np.random.default_rng(42)
    idxs_per_class = {c: np.where(y == le.transform([c])[0])[0] for c in classes}
    train_idx, val_idx = [], []
    for c, idxs in idxs_per_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        cut = int(0.8 * len(idxs))
        train_idx.extend(idxs[:cut])
        val_idx.extend(idxs[cut:])
    train_idx = np.array(train_idx)
    val_idx = np.array(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val":   X[val_idx],
        "y_val":   y[val_idx],
        "input_shape": (max_frames, n_mfcc),
        "output_dim": n_classes,
        "classify": True,
        "classes": classes,
        "scaler": None,
        "target_idx": None,
        "test_dates": None,
    }


# ── Twitter sentiment ───────────────────────────────────────────────────────
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS_CACHE: set[str] | None = None


def _load_stopwords() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        try:
            from nltk.corpus import stopwords
            _STOPWORDS_CACHE = set(stopwords.words("english"))
        except LookupError:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            _STOPWORDS_CACHE = set(stopwords.words("english"))
    return _STOPWORDS_CACHE


def _clean_tweet(s: str, stopwords: set[str]) -> str:
    s = s.lower()
    s = _URL_RE.sub(" ", s)
    s = _MENTION_RE.sub(" ", s)
    s = _NON_ALNUM_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return " ".join(t for t in s.split() if t not in stopwords)


def prep_twitter(train_df: pd.DataFrame, val_df: pd.DataFrame, hp: dict[str, Any]) -> dict[str, Any]:
    """lowercase → strip URLs/mentions/punct → stopwords → Tokenizer → pad."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.preprocessing.text import Tokenizer

    stops = _load_stopwords()
    train_texts = [_clean_tweet(t, stops) for t in train_df["text"].astype(str)]
    val_texts = [_clean_tweet(t, stops) for t in val_df["text"].astype(str)]

    classes = ["Positive", "Negative", "Neutral", "Irrelevant"]
    le = LabelEncoder().fit(classes)
    y_train = le.transform(train_df["sentiment"].astype(str))
    y_val = le.transform(val_df["sentiment"].astype(str))

    vocab_size = hp["vocab_size"]
    max_len = hp["max_seq_len"]
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_texts), maxlen=max_len, padding="post", truncating="post"
    )
    X_val = pad_sequences(
        tokenizer.texts_to_sequences(val_texts), maxlen=max_len, padding="post", truncating="post"
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val":   X_val,
        "y_val":   y_val,
        "input_shape": (max_len,),
        "output_dim": len(classes),
        "classify": True,
        "classes": classes,
        "scaler": None,
        "target_idx": None,
        "test_dates": None,
        "vocab_size": vocab_size,
    }


# ── Weather time-series — regression ────────────────────────────────────────
def _split_and_scale(values: np.ndarray, train_frac: float = 0.8):
    """Chronological split + fit MinMax on train only, transform both."""
    cut = int(len(values) * train_frac)
    train_raw, val_raw = values[:cut], values[cut:]
    scaler = MinMaxScaler().fit(train_raw)
    return scaler.transform(train_raw), scaler.transform(val_raw), scaler


def prep_weather_regression(
    df: pd.DataFrame,
    hp: dict[str, Any],
    target_col: str,
    feature_cols: list[str] | None = None,
    group_col: str | None = None,
    test_dates_col: str | None = "date",
) -> dict[str, Any]:
    """Null-fill → MinMax scale → 30-step window → next-step target → temporal split.

    If ``group_col`` is given (e.g. ``station`` for CORGIS), each group is
    windowed and split independently, then concatenated. The resulting
    samples carry no leakage across the chronological boundary within each
    group.
    """
    window = hp["window_size"]
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {"date", group_col} and df[c].dtype != "O"]
    cols = list(feature_cols)
    if target_col not in cols:
        cols.insert(0, target_col)
    target_idx = cols.index(target_col)

    df = df.copy()
    df[cols] = df[cols].ffill().bfill()
    # Any remaining NaNs (entirely empty columns) → 0.
    df[cols] = df[cols].fillna(0.0)

    if group_col is None:
        groups = [(None, df)]
    else:
        groups = list(df.groupby(group_col, sort=True))

    # Fit scaler on the train portion of each group concatenated → no val leakage.
    train_chunks: list[np.ndarray] = []
    for _, sub in groups:
        vals = sub[cols].values.astype(np.float32)
        if len(vals) < window + 2:
            continue
        cut = int(len(vals) * 0.8)
        train_chunks.append(vals[:cut])
    fit_scaler = MinMaxScaler().fit(np.concatenate(train_chunks, axis=0))

    X_train_parts, y_train_parts, X_val_parts, y_val_parts, val_dates_parts = [], [], [], [], []

    for _, sub in groups:
        vals = sub[cols].values.astype(np.float32)
        if len(vals) < window + 2:
            continue
        scaled = fit_scaler.transform(vals)
        cut = int(len(scaled) * 0.8)
        train_arr = scaled[:cut]
        val_arr = scaled[cut - window :]   # include last `window` of train so first val target is predictable
        if len(train_arr) > window and len(val_arr) > window:
            Xt, yt = _make_windows(train_arr, target_idx, window)
            Xv, yv = _make_windows(val_arr, target_idx, window)
            X_train_parts.append(Xt)
            y_train_parts.append(yt)
            X_val_parts.append(Xv)
            y_val_parts.append(yv)
            if test_dates_col and test_dates_col in sub.columns:
                dates = sub[test_dates_col].values[cut:]
                val_dates_parts.append(dates[: len(yv)])

    X_train = np.concatenate(X_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    X_val = np.concatenate(X_val_parts, axis=0)
    y_val = np.concatenate(y_val_parts, axis=0)
    test_dates = np.concatenate(val_dates_parts) if val_dates_parts else None

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "input_shape": (window, len(cols)),
        "output_dim": 1,
        "classify": False,
        "classes": None,
        "scaler": fit_scaler,
        "target_idx": target_idx,
        "test_dates": test_dates,
    }


def prep_weather_classification(
    df: pd.DataFrame,
    hp: dict[str, Any],
    target_col: str,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Window weather features → predict day t+1's category."""
    window = hp["window_size"]
    if feature_cols is None:
        feature_cols = [
            c for c in df.columns
            if c not in {"date", target_col} and df[c].dtype != "O"
        ]
    df = df.copy()
    df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)

    classes = sorted(df[target_col].dropna().unique().tolist())
    le = LabelEncoder().fit(classes)
    y_all = le.transform(df[target_col].astype(str))
    feats = df[feature_cols].values.astype(np.float32)

    cut = int(len(df) * 0.8)
    scaler = MinMaxScaler().fit(feats[:cut])
    feats_scaled = scaler.transform(feats)

    X_all, y_target = [], []
    for i in range(len(feats_scaled) - window):
        X_all.append(feats_scaled[i : i + window])
        y_target.append(y_all[i + window])
    X_all = np.asarray(X_all, dtype=np.float32)
    y_target = np.asarray(y_target, dtype=np.int64)

    cut_w = max(0, cut - window)  # account for the window offset
    X_train, X_val = X_all[:cut_w], X_all[cut_w:]
    y_train, y_val = y_target[:cut_w], y_target[cut_w:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "input_shape": (window, len(feature_cols)),
        "output_dim": len(classes),
        "classify": True,
        "classes": classes,
        "scaler": scaler,
        "target_idx": None,
        "test_dates": None,
    }


# ── Dispatch helper ────────────────────────────────────────────────────────
def preprocess(task: str, dataset_id: str, raw, hp: dict[str, Any]) -> dict[str, Any]:
    if task == "speech" and dataset_id == "tess":
        wav_paths, labels = raw
        return prep_tess(wav_paths, labels, hp)
    if task == "sentiment" and dataset_id == "twitter":
        train_df, val_df = raw
        return prep_twitter(train_df, val_df, hp)
    if task == "timeseries":
        if dataset_id == "corgis":
            df = raw
            return prep_weather_regression(
                df, hp, target_col="avg_temp",
                feature_cols=["precipitation", "avg_temp", "max_temp", "min_temp",
                              "wind_direction", "wind_speed"],
                group_col="station",
            )
        if dataset_id == "climate":
            train_df, test_df = raw
            full = pd.concat([train_df, test_df], ignore_index=True)
            return prep_weather_regression(
                full, hp, target_col="meantemp",
                feature_cols=["meantemp", "humidity", "wind_speed", "meanpressure"],
                group_col=None,
            )
        if dataset_id == "seattle":
            df = raw
            return prep_weather_classification(
                df, hp, target_col="weather",
                feature_cols=["precipitation", "temp_max", "temp_min", "wind"],
            )
    raise ValueError(f"Unknown (task, dataset): ({task!r}, {dataset_id!r})")
