"""Dataset loaders.

One function per dataset; each returns raw arrays/dataframes ready for
``preprocessing``. Loaders do NOT scale, tokenise, or window — that lives
in preprocessing.py so caching and reuse stay clean.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    EMOTION_LABEL_MAP,
    TESS_DIR,
    TWITTER_DIR,
    WEATHER_DIR,
)


# ── TESS (audio) ────────────────────────────────────────────────────────────
def load_tess() -> tuple[list[Path], list[str], list[str]]:
    """Scan every ACTRESS_<emotion>/ folder and return
    (wav_paths, emotion_labels, speaker_labels).

    Emotion labels are the dashboard-facing strings (Angry/Disgust/…/Surprise),
    derived from the folder suffix via EMOTION_LABEL_MAP. Speaker is the
    actress prefix in upper-case (e.g. "OAF" or "YAF") — used downstream to
    enforce a speaker-independent train/test split so the model can't cheat
    by memorising voice+sentence patterns.
    """
    if not TESS_DIR.exists():
        raise FileNotFoundError(f"TESS directory missing: {TESS_DIR}")

    wav_paths: list[Path] = []
    labels: list[str] = []
    speakers: list[str] = []
    for sub in sorted(TESS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        parts = sub.name.lower().split("_", 1)
        if len(parts) != 2:
            continue
        speaker = parts[0].upper()
        emotion_key = parts[1]
        if emotion_key not in EMOTION_LABEL_MAP:
            continue
        label = EMOTION_LABEL_MAP[emotion_key]
        for wav in sorted(sub.glob("*.wav")):
            wav_paths.append(wav)
            labels.append(label)
            speakers.append(speaker)
    if not wav_paths:
        raise RuntimeError(f"No .wav files found under {TESS_DIR}")
    return wav_paths, labels, speakers


# ── Twitter (text) ──────────────────────────────────────────────────────────
_TWITTER_COLS = ["id", "entity", "sentiment", "text"]


def load_twitter() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) from twitter_training.csv + twitter_validation.csv.

    The Kaggle CSVs ship without headers; we attach explicit column names so
    downstream code is clear.
    """
    train_path = TWITTER_DIR / "twitter_training.csv"
    val_path = TWITTER_DIR / "twitter_validation.csv"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected {train_path} and {val_path}. See backend/README.md."
        )
    train_df = pd.read_csv(train_path, header=None, names=_TWITTER_COLS)
    val_df = pd.read_csv(val_path, header=None, names=_TWITTER_COLS)
    # Drop rows with NaN text — there are a few in the original dataset.
    train_df = train_df.dropna(subset=["text"]).reset_index(drop=True)
    val_df = val_df.dropna(subset=["text"]).reset_index(drop=True)
    return train_df, val_df


# ── Weather (3 datasets) ────────────────────────────────────────────────────
def load_corgis() -> pd.DataFrame:
    """Load CORGIS Weather (16,743 rows × 14 cols, 307 stations × ~53 weeks).

    Returns multi-station weekly observations with cleaned column names. The
    raw CSV has dotted column names (``Data.Temperature.Avg Temp``) which
    pandas tolerates but downstream code shouldn't assume. Columns:
    ``date, station, precipitation, avg_temp, max_temp, min_temp,
    wind_direction, wind_speed``.

    Preprocessing windows each station independently and concatenates.
    """
    path = WEATHER_DIR / "corgis_weather.csv"
    if not path.exists():
        raise FileNotFoundError(f"CORGIS missing: {path}. See backend/README.md.")
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["Date.Full"]),
            "station": df["Station.Code"].astype(str),
            "precipitation": df["Data.Precipitation"].astype(float),
            "avg_temp": df["Data.Temperature.Avg Temp"].astype(float),
            "max_temp": df["Data.Temperature.Max Temp"].astype(float),
            "min_temp": df["Data.Temperature.Min Temp"].astype(float),
            "wind_direction": df["Data.Wind.Direction"].astype(float),
            "wind_speed": df["Data.Wind.Speed"].astype(float),
        }
    )
    return out.sort_values(["station", "date"]).reset_index(drop=True)


def load_daily_climate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) from DailyDelhiClimateTrain/Test.csv.

    Columns: date, meantemp, humidity, wind_speed, meanpressure.
    """
    train_path = WEATHER_DIR / "DailyDelhiClimateTrain.csv"
    test_path = WEATHER_DIR / "DailyDelhiClimateTest.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Daily Climate files missing in {WEATHER_DIR}. See backend/README.md."
        )
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    for d in (train, test):
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d.dropna(subset=["date", "meantemp"], inplace=True)
        d.sort_values("date", inplace=True)
        d.reset_index(drop=True, inplace=True)
    # The test set has a few junk rows at the end (all-NaN) on Kaggle — drop them.
    return train, test


def load_seattle() -> pd.DataFrame:
    """Load seattle-weather.csv. Columns: date, precipitation, temp_max,
    temp_min, wind, weather (5-class target)."""
    path = WEATHER_DIR / "seattle-weather.csv"
    if not path.exists():
        raise FileNotFoundError(f"Seattle file missing: {path}. See backend/README.md.")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ── Dispatch helper used by /api/train and the CLI runner ──────────────────
def load(task: str, dataset_id: str):
    if task == "speech" and dataset_id == "tess":
        return load_tess()
    if task == "sentiment" and dataset_id == "twitter":
        return load_twitter()
    if task == "timeseries":
        if dataset_id == "corgis":
            return load_corgis()
        if dataset_id == "climate":
            return load_daily_climate()
        if dataset_id == "seattle":
            return load_seattle()
    raise ValueError(f"Unknown (task, dataset): ({task!r}, {dataset_id!r})")
