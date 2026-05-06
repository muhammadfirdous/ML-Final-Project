"""Sanity-check that every dataset is in place before training kicks off.

Run this manually: ``python verify_data.py``. The Flask app also imports
``check_all`` and uses it to gate /api/train.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from config import (
    EMOTION_LABEL_MAP,
    EXPECTED_FILES,
    TESS_DIR,
    TESS_EMOTIONS,
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _check_tess() -> Check:
    if not TESS_DIR.exists():
        return Check("tess", False, f"Directory not found: {TESS_DIR}")

    # TESS layout: <ACTRESS>_<emotion>/*.wav, e.g. OAF_angry/, YAF_happy/.
    found_emotions: set[str] = set()
    wav_count = 0
    for sub in TESS_DIR.iterdir():
        if not sub.is_dir():
            continue
        # Extract emotion from suffix after the actress prefix; map known
        # folder-name variants (e.g. "Pleasant_surprise") to canonical keys.
        parts = sub.name.lower().split("_", 1)
        if len(parts) != 2:
            continue
        suffix = parts[1]
        canonical = EMOTION_LABEL_MAP.get(suffix, "").lower()
        if canonical:
            found_emotions.add(canonical)
            wav_count += sum(1 for _ in sub.glob("*.wav"))

    missing = set(TESS_EMOTIONS) - found_emotions
    if missing:
        return Check(
            "tess",
            False,
            f"Missing emotion folders: {sorted(missing)}. "
            f"Expected ACTRESS_<emotion> dirs in {TESS_DIR}.",
        )
    if wav_count < 100:
        return Check(
            "tess",
            False,
            f"Only {wav_count} .wav files found across {TESS_DIR} — expected ~2,800.",
        )
    return Check("tess", True, f"{wav_count} .wav files across {len(found_emotions)} emotions.")


def _check_files(task: str, paths: list[Path]) -> Check:
    missing = [p for p in paths if not p.exists()]
    if missing:
        return Check(task, False, "Missing: " + ", ".join(str(p) for p in missing))
    return Check(task, True, f"All {len(paths)} files present.")


def check_all() -> list[Check]:
    checks: list[Check] = []
    checks.append(_check_tess())
    for task, paths in EXPECTED_FILES.items():
        checks.append(_check_files(task, paths))
    return checks


def main() -> int:
    checks = check_all()
    width = max(len(c.name) for c in checks)
    print("Dataset verification:\n")
    all_ok = True
    for c in checks:
        marker = "OK   " if c.ok else "FAIL "
        print(f"  [{marker}] {c.name.ljust(width)}  {c.detail}")
        all_ok = all_ok and c.ok
    print()
    if all_ok:
        print("All datasets present. Ready to train.")
        return 0
    print("One or more datasets missing. See README for download + placement.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
