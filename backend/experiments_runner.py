"""Experiment runner — invokes train.run_training with various hyperparameter
overrides, logs train/val metrics to experiments.md, and writes a JSON dump
of every run to results/experiments_log.json.

Run from the backend directory:
    python experiments_runner.py <preset>
where preset is one of:
    twitter_clean | twitter_hp | seattle_cw | tess_aug |
    climate_long | corgis_long | all
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

# Quiet TF a bit
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Force UTF-8 stdout so unicode arrows etc. don't crash on Windows cp1252.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

from src.train import run_training  # noqa: E402

LOG_PATH = BACKEND.parent / "experiments.md"
RUN_LOG = BACKEND / "results" / "experiments_log.json"


def _append_md(line: str) -> None:
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _record(label: str, task: str, ds: str, mdl: str, hp: dict, suffix: str) -> dict:
    print(f"\n>>> [{label}] {task}/{ds}/{mdl} suffix={suffix} hp={hp}")
    t0 = time.time()
    res = run_training(
        task, ds, mdl,
        hyperparams=hp,
        force_retrain=True,
        result_suffix=suffix,
        save_plots=False,
    )
    dt = time.time() - t0
    metrics = res["metrics"]
    curve = res.get("curve", [])
    last = curve[-1] if curve else {}
    classify = res.get("classify", False)
    if classify:
        train_acc = last.get("trainAcc", 0)
        val_acc = last.get("valAcc", 0)
        msg = (
            f"- **{label}** `{task}/{ds}/{mdl}` -> "
            f"acc={metrics.get('accuracy')} f1={metrics.get('f1')} "
            f"trainAcc(last)={train_acc:.3f} valAcc(last)={val_acc:.3f} "
            f"epochs={len(curve)} time={dt:.1f}s"
        )
    else:
        msg = (
            f"- **{label}** `{task}/{ds}/{mdl}` -> "
            f"R²={metrics.get('accuracy')} mae={metrics.get('mae')} rmse={metrics.get('rmse')} "
            f"trainLoss(last)={last.get('trainLoss')} valLoss(last)={last.get('valLoss')} "
            f"epochs={len(curve)} time={dt:.1f}s"
        )
    print(msg)
    _append_md(msg)

    log_entry = {
        "label": label,
        "task": task, "datasetId": ds, "model": mdl,
        "suffix": suffix,
        "hp_overrides": hp,
        "metrics": metrics,
        "epochs": len(curve),
        "wall_seconds": round(dt, 1),
    }
    log_entries = []
    if RUN_LOG.exists():
        try:
            log_entries = json.loads(RUN_LOG.read_text(encoding="utf-8"))
        except Exception:
            log_entries = []
    log_entries.append(log_entry)
    RUN_LOG.write_text(json.dumps(log_entries, indent=2), encoding="utf-8")
    return res


# ── Presets ────────────────────────────────────────────────────────────────
def preset_twitter_clean() -> None:
    _append_md("\n### Experiment block: Twitter — fix val/train leak\n")
    for mdl in ("RNN", "LSTM", "GRU"):
        _record(f"twitter-resplit", "sentiment", "twitter", mdl,
                hp={"resplit": True}, suffix="resplit")


def preset_twitter_hp() -> None:
    _append_md("\n### Experiment block: Twitter - LSTM/GRU with longer patience + LR plateau on clean split\n")
    # Resplit + dedupe + extended patience: LSTM/GRU plateau around the
    # class prior for several epochs before learning kicks in. Patience=5
    # was too short and LSTM was killed at epoch 7.
    common = {
        "resplit": True,
        "epochs": 30,
        "early_stopping_patience": 8,
        "reduce_lr_on_plateau": True,
        "reduce_lr_patience": 3,
    }
    for mdl in ("RNN", "LSTM", "GRU"):
        _record("twitter-resplit-patient", "sentiment", "twitter", mdl,
                hp=common, suffix="resplit2")


def preset_twitter_hp2() -> None:
    _append_md("\n### Experiment block: Twitter - resplit + masked Embedding + long patience\n")
    common = {
        "resplit": True,
        "embedding_mask_zero": True,
        "epochs": 30,
        "early_stopping_patience": 8,
        "reduce_lr_on_plateau": True,
        "reduce_lr_patience": 3,
    }
    for mdl in ("RNN", "LSTM", "GRU"):
        _record("twitter-resplit-masked", "sentiment", "twitter", mdl,
                hp=common, suffix="resplitm")


def preset_seattle_cw() -> None:
    _append_md("\n### Experiment block: Seattle - longer training, no class weight\n")
    common = {"epochs": 80,
              "early_stopping_patience": 15,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5,
              "monitor": "val_loss"}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        _record(f"seattle-long", "timeseries", "seattle", mdl,
                hp=common, suffix="long")
    _append_md("\n### Experiment block: Seattle - longer training + sqrt class weight\n")
    common2 = dict(common)
    common2["use_class_weight"] = "sqrt"
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        _record(f"seattle-sqrtcw", "timeseries", "seattle", mdl,
                hp=common2, suffix="sqrtcw")


def preset_tess_aug() -> None:
    _append_md("\n### Experiment block: TESS — SpecAugment + LR plateau\n")
    common = {"specaug_copies": 3, "epochs": 60,
              "early_stopping_patience": 10,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 4,
              "min_lr": 1e-5}
    for mdl in ("RNN", "LSTM", "GRU"):
        hp = dict(common)
        # LSTM specifically benefits from a slightly lower LR with augmentation
        if mdl == "LSTM":
            hp["learning_rate"] = 2e-3
        _record(f"tess-specaug", "speech", "tess", mdl,
                hp=hp, suffix="specaug")


def preset_climate_long() -> None:
    _append_md("\n### Experiment block: Climate — longer training + LR plateau\n")
    common = {"epochs": 80, "early_stopping_patience": 12,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        _record("climate-long", "timeseries", "climate", mdl,
                hp=common, suffix="long")


def preset_corgis_long() -> None:
    _append_md("\n### Experiment block: CORGIS — longer training + LR plateau\n")
    common = {"epochs": 80, "early_stopping_patience": 12,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        _record("corgis-long", "timeseries", "corgis", mdl,
                hp=common, suffix="long")


def preset_tess_rnn_mild() -> None:
    _append_md("\n### Experiment block: TESS RNN - mild specaug + longer training\n")
    # RNN got slightly worse with full specaug; try fewer copies + smaller masks.
    hp = {
        "specaug_copies": 2,
        "specaug_time_mask": 12,
        "specaug_freq_mask": 4,
        "specaug_n_time": 1,
        "specaug_n_freq": 1,
        "epochs": 60,
        "early_stopping_patience": 12,
        "reduce_lr_on_plateau": True,
        "reduce_lr_patience": 5,
        "min_lr": 1e-5,
    }
    _record("tess-mildaug", "speech", "tess", "RNN",
            hp=hp, suffix="mildaug")


def preset_climate_pool() -> None:
    _append_md("\n### Experiment block: Climate - RNN/LSTM/GRU with pool_over_time\n")
    common = {"pool_over_time": True, "epochs": 80,
              "early_stopping_patience": 12,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5,
              "learning_rate": 3e-3}
    for mdl in ("RNN", "LSTM", "GRU"):
        _record("climate-pool", "timeseries", "climate", mdl,
                hp=common, suffix="pool")


def preset_seattle_pool() -> None:
    _append_md("\n### Experiment block: Seattle - pool_over_time + longer\n")
    common = {"pool_over_time": True, "epochs": 80,
              "early_stopping_patience": 15,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5,
              "learning_rate": 3e-3}
    for mdl in ("RNN", "LSTM", "GRU"):
        _record("seattle-pool", "timeseries", "seattle", mdl,
                hp=common, suffix="pool")


def preset_corgis_cyclic() -> None:
    _append_md("\n### Experiment block: CORGIS - cyclic week-of-year features\n")
    common = {"add_cyclic_time": True, "epochs": 60,
              "early_stopping_patience": 10,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        _record("corgis-cyclic", "timeseries", "corgis", mdl,
                hp=common, suffix="cyclic")


def preset_corgis_cyclic_pool() -> None:
    _append_md("\n### Experiment block: CORGIS - cyclic + pool_over_time + lr 3e-3\n")
    common = {"add_cyclic_time": True, "pool_over_time": True,
              "epochs": 60, "early_stopping_patience": 10,
              "reduce_lr_on_plateau": True, "reduce_lr_patience": 5,
              "learning_rate": 3e-3}
    for mdl in ("RNN", "LSTM", "GRU"):
        _record("corgis-cyclic-pool", "timeseries", "corgis", mdl,
                hp=common, suffix="cycpool")


def preset_climate_cyclic() -> None:
    _append_md("\n### Experiment block: Climate - cyclic + pool + longer window\n")
    common = {"add_cyclic_time": True, "pool_over_time": True,
              "epochs": 80, "early_stopping_patience": 12,
              "reduce_lr_on_plateau": True,
              "reduce_lr_patience": 5,
              "learning_rate": 3e-3}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        # ANN doesn't use pool_over_time — make a per-model copy
        hp = dict(common)
        if mdl == "ANN":
            hp.pop("pool_over_time", None)
        _record("climate-cyclic", "timeseries", "climate", mdl,
                hp=hp, suffix="cyclic")


def preset_climate_w60() -> None:
    _append_md("\n### Experiment block: Climate - cyclic + window=60\n")
    common = {"add_cyclic_time": True, "pool_over_time": True,
              "window_size": 60,
              "epochs": 80, "early_stopping_patience": 12,
              "reduce_lr_on_plateau": True, "reduce_lr_patience": 5,
              "learning_rate": 3e-3}
    for mdl in ("ANN", "RNN", "LSTM", "GRU"):
        hp = dict(common)
        if mdl == "ANN":
            hp.pop("pool_over_time", None)
        _record("climate-w60", "timeseries", "climate", mdl,
                hp=hp, suffix="w60")


def preset_tess_rnn_dropout() -> None:
    _append_md("\n### Experiment block: TESS RNN - higher dropout + smaller model\n")
    hp = {"dropout": 0.5, "hidden_units": {"RNN": 64},
          "epochs": 60, "early_stopping_patience": 10,
          "reduce_lr_on_plateau": True, "reduce_lr_patience": 4,
          "specaug_copies": 1, "specaug_time_mask": 8, "specaug_freq_mask": 3,
          "specaug_n_time": 1, "specaug_n_freq": 1,
          "min_lr": 1e-5}
    _record("tess-rnn-drop", "speech", "tess", "RNN", hp=hp, suffix="drop")


PRESETS = {
    "twitter_clean": preset_twitter_clean,
    "twitter_hp": preset_twitter_hp,
    "twitter_hp2": preset_twitter_hp2,
    "seattle_cw": preset_seattle_cw,
    "tess_aug": preset_tess_aug,
    "tess_rnn_mild": preset_tess_rnn_mild,
    "tess_rnn_dropout": preset_tess_rnn_dropout,
    "climate_long": preset_climate_long,
    "climate_pool": preset_climate_pool,
    "climate_cyclic": preset_climate_cyclic,
    "climate_w60": preset_climate_w60,
    "corgis_cyclic": preset_corgis_cyclic,
    "corgis_cyclic_pool": preset_corgis_cyclic_pool,
    "seattle_pool": preset_seattle_pool,
    "corgis_long": preset_corgis_long,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("preset", choices=list(PRESETS.keys()) + ["all"])
    args = p.parse_args()
    if args.preset == "all":
        for fn in PRESETS.values():
            fn()
    else:
        PRESETS[args.preset]()


if __name__ == "__main__":
    main()
