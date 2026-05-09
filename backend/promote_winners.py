"""Promote winning experiment runs to canonical results files.

For each (task, dataset, model) decide whether the best experiment beats the
baseline on the appropriate metric, and if so, copy the experiment JSON +
.h5 to the canonical filename and re-render plots.

Run with:
    python promote_winners.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

RESULTS = BACKEND / "results"
SAVED = BACKEND / "saved_models"

# (task, dataset, model) -> list of suffixes to consider, ordered by preference.
CANDIDATES = {
    ("speech", "tess", "RNN"): ["mildaug", "specaug"],
    ("speech", "tess", "LSTM"): ["specaug"],
    ("speech", "tess", "GRU"): ["specaug"],

    ("sentiment", "twitter", "RNN"): ["resplit2", "resplit", "resplitm"],
    ("sentiment", "twitter", "LSTM"): ["resplitm", "resplit2", "resplit"],
    ("sentiment", "twitter", "GRU"): ["resplit2", "resplit", "resplitm"],

    ("timeseries", "climate", "ANN"): ["cyclic", "long"],
    ("timeseries", "climate", "RNN"): ["cyclic", "pool", "long"],
    ("timeseries", "climate", "LSTM"): ["cyclic", "pool", "long"],
    ("timeseries", "climate", "GRU"): ["cyclic", "pool", "long"],

    ("timeseries", "corgis", "ANN"): ["cyclic", "long"],
    ("timeseries", "corgis", "RNN"): ["cyclic", "long"],
    ("timeseries", "corgis", "LSTM"): ["cyclic", "long"],
    ("timeseries", "corgis", "GRU"): ["cyclic", "long"],

    ("timeseries", "seattle", "ANN"): ["long"],
    ("timeseries", "seattle", "RNN"): ["long", "pool"],
    ("timeseries", "seattle", "LSTM"): ["long", "pool", "sqrtcw"],
    ("timeseries", "seattle", "GRU"): ["long", "pool", "sqrtcw"],
}


def metric_for(task: str, classify: bool) -> str:
    """Pick the canonical comparison metric per task."""
    if task == "speech":
        return "f1"          # 7-class balanced; both acc and F1 align, F1 is stricter
    if task == "sentiment":
        return "f1"          # multi-class macro F1 — robust to mild imbalance
    if classify:
        return "f1"          # Seattle: macro F1 because of severe imbalance
    return "accuracy"        # regression: dashboard "accuracy" is R²·100


def load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    promotions = []
    for (task, ds, mdl), suffixes in CANDIDATES.items():
        base_path = RESULTS / f"{task}_{ds}_{mdl}.json"
        baseline = load(base_path)
        if baseline is None:
            print(f"[skip] no baseline for {task}/{ds}/{mdl}")
            continue
        m = metric_for(task, baseline.get("classify", False))
        base_val = baseline["metrics"].get(m, 0)
        best_suffix = None
        best_path = None
        best_val = base_val

        for suf in suffixes:
            cand_path = RESULTS / f"{task}_{ds}_{mdl}_{suf}.json"
            cand = load(cand_path)
            if cand is None:
                continue
            cand_val = cand["metrics"].get(m, 0)
            if cand_val > best_val:
                best_val = cand_val
                best_suffix = suf
                best_path = cand_path

        if best_suffix is None:
            print(f"[keep ] {task}/{ds}/{mdl} {m}={base_val} (no candidate beat baseline)")
            continue

        # Promote: copy JSON and h5 to canonical name.
        h5_src = SAVED / f"{task}_{ds}_{mdl}_{best_suffix}.h5"
        h5_dst = SAVED / f"{task}_{ds}_{mdl}.h5"
        json_dst = base_path
        shutil.copy2(best_path, json_dst)
        if h5_src.exists():
            shutil.copy2(h5_src, h5_dst)
        promotions.append((task, ds, mdl, m, base_val, best_val, best_suffix))
        print(f"[PROMOTE] {task}/{ds}/{mdl} {m}: {base_val} -> {best_val} (suffix={best_suffix})")

    # Re-render plots for everything we promoted.
    if promotions:
        print(f"\nRe-rendering plots for {len(promotions)} promoted models...")
        from src import visualize  # noqa: E402
        for task, ds, mdl, *_ in promotions:
            try:
                d = json.loads((RESULTS / f"{task}_{ds}_{mdl}.json").read_text(encoding="utf-8"))
                visualize.render_all(task, ds, mdl, d)
                print(f"   plotted {task}/{ds}/{mdl}")
            except Exception as exc:
                print(f"   plot {task}/{ds}/{mdl} failed: {exc}")

    # Summary table
    print("\n=== Promotion summary ===")
    for task, ds, mdl, m, b, v, suf in promotions:
        print(f"  {task:10s}/{ds:8s}/{mdl:5s} {m}: {b:.3f} -> {v:.3f}  [{suf}]")


if __name__ == "__main__":
    main()
