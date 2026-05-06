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
TRAIN_SPEAKER = "OAF"   # held in for training
VAL_SPEAKER = "YAF"     # held out for evaluation


def prep_tess(
    wav_paths: list[Path],
    labels: list[str],
    hp: dict[str, Any],
    speakers: list[str] | None = None,
) -> dict[str, Any]:
    """librosa load → 40 MFCC → pad/truncate → speaker-independent split.

    TESS contains only two actresses (OAF, YAF) reading the same 200 sentences,
    so a stratified random split lets the model memorise voice patterns and
    drift to ~100% — that's not real emotion recognition. We hold out one
    speaker entirely (YAF) and train on the other (OAF). MFCC z-score stats
    are also computed from the train portion only so the val data is genuinely
    unseen.
    """
    import librosa  # heavy import, lazy

    sr_target = hp["sample_rate"]
    n_mfcc = hp["n_mfcc"]
    max_frames = hp["max_frames"]

    if speakers is None:
        # Backwards compat — assume all training speaker if no info.
        speakers = [TRAIN_SPEAKER] * len(wav_paths)

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
        if i % 400 == 0:
            print(f"[TESS]   {i}/{len(wav_paths)} done")
    X = np.stack(feats)  # (N, max_frames, n_mfcc)

    classes = sorted(set(labels))
    le = LabelEncoder().fit(classes)
    y = le.transform(labels)

    speakers_arr = np.array(speakers)
    train_mask = speakers_arr == TRAIN_SPEAKER
    val_mask = speakers_arr == VAL_SPEAKER
    if train_mask.sum() == 0 or val_mask.sum() == 0:
        raise RuntimeError(
            f"Speaker-independent split needs both {TRAIN_SPEAKER} and {VAL_SPEAKER} "
            f"folders; got speakers={set(speakers)}"
        )

    # Z-score per coefficient using ONLY the training speaker so val is genuinely
    # out-of-distribution. Without normalisation, GRU's sigmoid gates saturate.
    X_train_raw = X[train_mask]
    mu = X_train_raw.mean(axis=(0, 1), keepdims=True)
    sigma = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-6
    X = ((X - mu) / sigma).astype(np.float32)

    return {
        "X_train": X[train_mask],
        "y_train": y[train_mask],
        "X_val":   X[val_mask],
        "y_val":   y[val_mask],
        "input_shape": (max_frames, n_mfcc),
        "output_dim": len(classes),
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

    # LabelEncoder sorts alphabetically; align ``classes`` to its real order
    # so downstream argmax → class-name lookup is correct.
    le = LabelEncoder().fit(["Positive", "Negative", "Neutral", "Irrelevant"])
    classes = list(le.classes_)  # → ['Irrelevant', 'Negative', 'Neutral', 'Positive']
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


# ── Artifact builders (for inference; pickleable subset of preprocess output) ──
def build_artifacts_speech(hp: dict[str, Any]) -> dict[str, Any]:
    """Compute (mfcc_mean, mfcc_std, classes) from the TRAIN speaker (OAF)
    only. Inference uses these same stats for any user audio.
    Slow (~3-5 min on CPU) but runs once — pickled and reused."""
    from src.data_loader import load_tess
    import librosa  # noqa: WPS433

    wav_paths, labels, speakers = load_tess()
    n_mfcc = hp["n_mfcc"]
    max_frames = hp["max_frames"]
    sr_target = hp["sample_rate"]

    train_feats: list[np.ndarray] = []
    print(f"[artifacts/tess] extracting MFCCs from {len(wav_paths)} clips "
          f"(stats computed from {TRAIN_SPEAKER} only)…")
    for i, (p, sp) in enumerate(zip(wav_paths, speakers), 1):
        if sp != TRAIN_SPEAKER:
            continue
        y, sr = librosa.load(p, sr=sr_target, mono=True)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
        if m.shape[0] >= max_frames:
            m = m[:max_frames]
        else:
            pad = np.zeros((max_frames - m.shape[0], n_mfcc), dtype=np.float32)
            m = np.concatenate([m, pad], axis=0)
        train_feats.append(m.astype(np.float32))
        if i % 400 == 0:
            print(f"[artifacts/tess]   {i}/{len(wav_paths)}")
    X_train = np.stack(train_feats)
    mu = X_train.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    sigma = (X_train.std(axis=(0, 1), keepdims=True) + 1e-6).astype(np.float32)

    return {
        "mfcc_mean": mu,
        "mfcc_std": sigma,
        "classes": sorted(set(labels)),
        "max_frames": max_frames,
        "n_mfcc": n_mfcc,
        "sample_rate": sr_target,
        "train_speaker": TRAIN_SPEAKER,
        "val_speaker": VAL_SPEAKER,
    }


def build_artifacts_sentiment(hp: dict[str, Any]) -> dict[str, Any]:
    from tensorflow.keras.preprocessing.text import Tokenizer

    from src.data_loader import load_twitter

    train_df, _ = load_twitter()
    stops = _load_stopwords()
    train_texts = [_clean_tweet(t, stops) for t in train_df["text"].astype(str)]

    tokenizer = Tokenizer(num_words=hp["vocab_size"], oov_token="<oov>")
    tokenizer.fit_on_texts(train_texts)

    le = LabelEncoder().fit(["Positive", "Negative", "Neutral", "Irrelevant"])
    return {
        "tokenizer": tokenizer,
        "classes": list(le.classes_),
        "max_seq_len": hp["max_seq_len"],
        "vocab_size": hp["vocab_size"],
    }


def build_artifacts_timeseries(dataset_id: str, hp: dict[str, Any]) -> dict[str, Any]:
    """For each TS dataset return the fitted scaler + feature config so a single
    user-supplied window (or one drawn from the test set) can be predicted
    without re-running the full preprocessing pipeline."""
    from src.data_loader import load_corgis, load_daily_climate, load_seattle

    window = hp["window_size"]

    if dataset_id == "corgis":
        df = load_corgis()
        target_col = "avg_temp"
        feature_cols = ["precipitation", "avg_temp", "max_temp", "min_temp",
                        "wind_direction", "wind_speed"]
        cols = list(feature_cols)
        if target_col not in cols:
            cols.insert(0, target_col)
        target_idx = cols.index(target_col)
        df = df.copy()
        df[cols] = df[cols].ffill().bfill().fillna(0.0)
        # Train portion of every station → fit scaler.
        train_chunks: list[np.ndarray] = []
        for _, sub in df.groupby("station", sort=True):
            vals = sub[cols].values.astype(np.float32)
            if len(vals) < window + 2:
                continue
            cut = int(len(vals) * 0.8)
            train_chunks.append(vals[:cut])
        scaler = MinMaxScaler().fit(np.concatenate(train_chunks, axis=0))
        # Pre-compute val windows so the frontend can offer "pick a sample".
        sample_windows = _ts_collect_windows(df, "station", cols, scaler, window, target_idx, n_samples=8)
        return {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "all_cols": cols,
            "target_idx": target_idx,
            "target_col": target_col,
            "window_size": window,
            "classify": False,
            "classes": None,
            "sample_windows": sample_windows,
        }

    if dataset_id == "climate":
        train_df, test_df = load_daily_climate()
        full = pd.concat([train_df, test_df], ignore_index=True).copy()
        target_col = "meantemp"
        feature_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
        cols = list(feature_cols)
        if target_col not in cols:
            cols.insert(0, target_col)
        target_idx = cols.index(target_col)
        full[cols] = full[cols].ffill().bfill().fillna(0.0)
        cut = int(len(full) * 0.8)
        scaler = MinMaxScaler().fit(full[cols].values[:cut].astype(np.float32))
        sample_windows = _ts_collect_windows(full, None, cols, scaler, window, target_idx, n_samples=8)
        return {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "all_cols": cols,
            "target_idx": target_idx,
            "target_col": target_col,
            "window_size": window,
            "classify": False,
            "classes": None,
            "sample_windows": sample_windows,
        }

    if dataset_id == "seattle":
        df = load_seattle().copy()
        feature_cols = ["precipitation", "temp_max", "temp_min", "wind"]
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0.0)
        classes = sorted(df["weather"].dropna().unique().tolist())
        cut = int(len(df) * 0.8)
        scaler = MinMaxScaler().fit(df[feature_cols].values[:cut].astype(np.float32))
        # Pre-pad sample windows from the val portion.
        feats = scaler.transform(df[feature_cols].values.astype(np.float32))
        labels_all = LabelEncoder().fit(classes).transform(df["weather"].astype(str))
        cut_w = max(0, cut - window)
        val_windows = []
        for i, idx in enumerate(range(cut_w, len(feats) - window)):
            if i >= 8:
                break
            val_windows.append({
                "id": int(idx),
                "label": "Day-of-test #{}".format(i + 1),
                "window": feats[idx : idx + window].tolist(),
                "actualClass": classes[int(labels_all[idx + window])],
                "date": str(df["date"].iloc[idx + window].date()),
            })
        return {
            "scaler": scaler,
            "feature_cols": feature_cols,
            "all_cols": feature_cols,
            "target_idx": None,
            "target_col": "weather",
            "window_size": window,
            "classify": True,
            "classes": classes,
            "sample_windows": val_windows,
        }

    raise ValueError(f"Unknown timeseries dataset_id: {dataset_id!r}")


def _ts_collect_windows(df, group_col, cols, scaler, window, target_idx, n_samples=8):
    """Collect (id, label, window, actual) tuples from the val portion of the
    dataset so the frontend can offer 'pick a sample'.

    With ``group_col`` (e.g. CORGIS by station): one sample per station, up to
    n_samples. Without grouping (Climate, Seattle classification): multiple
    samples spaced across the val window so the picker isn't a single option.
    """
    out: list[dict[str, Any]] = []
    if group_col is None:
        # Single time series — sample at evenly-spaced offsets through the val
        # portion. The first val target sits at ``cut``; subsequent windows
        # advance by ``stride`` steps.
        vals = df[cols].values.astype(np.float32)
        if len(vals) < window + 2:
            return out
        cut = int(len(vals) * 0.8)
        scaled = scaler.transform(vals)
        n_val = len(scaled) - cut
        if n_val < 1:
            return out
        # Pick up to n_samples evenly-spaced offsets in [0, n_val-1].
        step = max(1, n_val // n_samples)
        offsets = [i * step for i in range(n_samples) if i * step < n_val]
        for sample_idx, off in enumerate(offsets):
            target_pos = cut + off
            start = target_pos - window
            if start < 0 or target_pos >= len(scaled):
                continue
            win = scaled[start : target_pos]
            if len(win) != window:
                continue
            actual = float(vals[target_pos, target_idx])
            date_str = ""
            if "date" in df.columns:
                d = df["date"].iloc[target_pos]
                if hasattr(d, "strftime"):
                    date_str = d.strftime("%Y-%m-%d")
            label = f"Test set — {date_str}" if date_str else f"Test set — sample {sample_idx + 1}"
            out.append({
                "id": f"sample-{sample_idx}",
                "label": label,
                "window": win.tolist(),
                "actual": round(actual, 2),
            })
        return out

    # Grouped (per-station) — one window per station.
    groups = list(df.groupby(group_col, sort=True))
    sample_idx = 0
    for gid, sub in groups:
        vals = sub[cols].values.astype(np.float32)
        if len(vals) < window + 2:
            continue
        cut = int(len(vals) * 0.8)
        if cut + 1 >= len(vals):
            continue
        scaled = scaler.transform(vals)
        win = scaled[cut - window : cut]
        if len(win) != window:
            continue
        actual_unscaled = float(vals[cut, target_idx])
        label = f"{gid} — {sub['date'].iloc[cut].strftime('%Y-%m-%d')}" \
            if "date" in sub.columns and hasattr(sub['date'].iloc[cut], 'strftime') \
            else f"{gid} — sample {sample_idx + 1}"
        out.append({
            "id": str(gid),
            "label": label,
            "window": win.tolist(),
            "actual": round(actual_unscaled, 2),
        })
        sample_idx += 1
        if sample_idx >= n_samples:
            break
    return out


def build_artifacts(task: str, dataset_id: str, hp: dict[str, Any]) -> dict[str, Any]:
    if task == "speech":
        return build_artifacts_speech(hp)
    if task == "sentiment":
        return build_artifacts_sentiment(hp)
    if task == "timeseries":
        return build_artifacts_timeseries(dataset_id, hp)
    raise ValueError(f"Unknown task: {task!r}")


# ── Dispatch helper ────────────────────────────────────────────────────────
def preprocess(task: str, dataset_id: str, raw, hp: dict[str, Any]) -> dict[str, Any]:
    if task == "speech" and dataset_id == "tess":
        if len(raw) == 3:
            wav_paths, labels, speakers = raw
        else:  # backwards compat
            wav_paths, labels = raw
            speakers = None
        return prep_tess(wav_paths, labels, hp, speakers=speakers)
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
