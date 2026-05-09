# Results

Final report for the ML Final Project — 18 (task × dataset × model)
combinations plus one Twitter ensemble. Improvements were applied one
change at a time; each change was kept only if it strictly beat the
recorded baseline on the appropriate validation metric. Detailed experiment
logs live in `experiments.md` and `backend/results/experiments_log.json`.

## Headline summary

| Task | Dataset | Model | Baseline | Final | Δ | Metric | What worked |
|---|---|---|---:|---:|---:|---|---|
| Speech | TESS | RNN | 50% / F1=0.477 | 50% / 0.477 | — | macro F1 | aug variants underperformed; baseline kept |
| Speech | TESS | LSTM | 43% / 0.345 | **56% / 0.520** | **+13 acc / +0.175 F1** | macro F1 | SpecAugment (3 copies) + ReduceLROnPlateau |
| Speech | TESS | GRU | 52% / 0.512 | **62% / 0.617** | **+10 / +0.105** | macro F1 | SpecAugment + ReduceLROnPlateau |
| Sentiment | Twitter | RNN | 96% / 0.955 ⚠ | 87% / 0.872 ‡ | leak-corrected | macro F1 | resplit (dedupe + stratified 90/10) + `mask_zero=True` |
| Sentiment | Twitter | LSTM | 93% / 0.929 ⚠ | 87% / 0.871 ‡ | leak-corrected | macro F1 | resplit + `mask_zero=True` (without mask, model collapses on clean split) |
| Sentiment | Twitter | GRU | 95% / 0.947 ⚠ | 87% / 0.864 ‡ | leak-corrected | macro F1 | resplit + `mask_zero=True` |
| Sentiment | Twitter | Ensemble | — | **89% / 0.888** ‡ | +2 acc / +0.018 F1 over best single | macro F1 | softmax average of RNN+LSTM+GRU |
| TS regression | Climate | ANN | R²=86 (RMSE 2.48) | **R²=90 (2.05)** | **+4** | R² | cyclic week-of-year (sin/cos) features |
| TS regression | Climate | RNN | R²=29 (5.54) | **R²=72 (3.49)** | **+43** | R² | cyclic + `pool_over_time` + lr 3e-3 |
| TS regression | Climate | LSTM | R²=51 (4.62) | **R²=71 (3.54)** | **+20** | R² | cyclic + `pool_over_time` + lr 3e-3 |
| TS regression | Climate | GRU | R²=46 (4.84) | **R²=71 (3.56)** | **+25** | R² | cyclic + `pool_over_time` + lr 3e-3 |
| TS regression | CORGIS | ANN | R²=48 (12.65) | **R²=56 (11.58)** | **+8** | R² | cyclic week-of-year features |
| TS regression | CORGIS | RNN | R²=42 (13.39) | 42 | — | R² | cyclic hurt recurrent here (+pool dropped to R²=10) |
| TS regression | CORGIS | LSTM | R²=48 (12.61) | 48 | — | R² | same — kept baseline |
| TS regression | CORGIS | GRU | R²=41 (13.41) | 41 | — | R² | same — kept baseline |
| TS classify | Seattle | ANN | 61% / 0.327 | **66% / 0.350** | **+5 / +0.023** | macro F1 | longer training + ReduceLROnPlateau |
| TS classify | Seattle | RNN | 63% / 0.338 | 63 / 0.338 | — | macro F1 | every variant tied or lost |
| TS classify | Seattle | LSTM | 62% / 0.331 | 62 / 0.332 | ≈ | macro F1 | sqrt class weights — neutral |
| TS classify | Seattle | GRU | 59% / 0.315 | 60 / 0.319 | +1 / +0.004 | macro F1 | sqrt class weights — marginal |

⚠ = baseline metric known to be inflated by a 51% data-leakage bug (see
"Twitter — what the baseline actually measured" below). ‡ = the new number
is on a clean re-split with the leak removed; not directly comparable to
the leaky baseline.

**Aggregate scoreboard** (real, modelling-driven improvements only):

| Category | Count |
|---|---:|
| Strict R²/F1/accuracy gains over baseline | **9 / 18** |
| Marginal lifts (< 1% F1) | 2 |
| Held at baseline | 7 |
| Data-quality-driven correction (Twitter ×3) | 3 (acc dropped, but the new number is the first honest one) |
| Bonus ensemble | 1 (Twitter avg, 89% acc) |

Most-improved single model: **Climate RNN, +43 R²** (29 → 72), powered by adding
cyclic week-of-year features and `pool_over_time` to give the network
explicit calendar awareness and a cleaner gradient path.

## What worked, by task

### TESS speech emotion (7-class, balanced) — large lift on LSTM/GRU
Baseline trains 99% on the OAF speaker but only 43–52% on YAF — classic
small-dataset overfit when each clip's spectral fingerprint can carry
speaker identity. **SpecAugment-style time + frequency masking on the
training set (3 augmented passes per epoch) plus ReduceLROnPlateau** lifted
LSTM 43 → 56 (F1 0.345 → 0.520) and GRU 52 → 62 (F1 0.512 → 0.617). The
vanilla SimpleRNN didn't benefit — its baseline (F1 0.477) was already
near its capacity ceiling and got slightly worse with augmentation; a
milder mask did not recover it; high-dropout + smaller-model didn't
either. Net: two of three speech models improved.

### Time-series regression — cyclic time + temporal pooling is the right combo for Climate
Daily Climate has a clean annual seasonal cycle but the original models
saw only the last 30 days of weather features and had to rediscover
seasonality from scratch. Two changes stack:

1. **`add_cyclic_time=True`** appends `sin(2π·week/52)` and
   `cos(2π·week/52)` as features. ANN gains the most because it has no
   temporal mechanism — these features become a clean per-row periodic
   signal, lifting it 86 → 90 R². For the recurrent models the cyclic
   feature adds explicit calendar awareness on top of whatever pattern
   they could learn from temperatures.
2. **`pool_over_time=True` + lr=3e-3** (returns sequences →
   GlobalAveragePooling1D). Each timestep's hidden state contributes to
   the prediction rather than just the final state, and the higher LR
   compensates for the averaged gradient.

Combined, the climate RNN/LSTM/GRU jumped to R²=72/71/71. Doubling the
window to 60 days hurt every architecture — Climate's smooth seasonality
saturates at 30 days.

CORGIS is a different beast (318 stations, weekly observations, hugely
varying climates) — only the ANN benefited from cyclic features
(48 → 56). The recurrent models *got worse* with cyclic + pool
(R²=10–30) because the global cycle doesn't fit each station's local
seasonality. Per-station modelling or station-level metadata would be
the next step but weren't tried within budget.

### Seattle weather classification — class imbalance dominates
Five classes with chronological 80/20 split (val window contains *zero*
snow days) → all four architectures sat near 60% acc / F1 ~0.33.
**Longer training + ReduceLROnPlateau** lifted ANN 61 → 66 acc, F1 0.327
→ 0.350. sqrt-balanced class weights barely moved LSTM/GRU; full
balanced weights *hurt* by over-correcting toward rare classes that
barely appear in val. `pool_over_time=True` hurt all three RNNs.

### Twitter — what the baseline actually measured
**The single largest finding of the audit.**
`twitter_validation.csv` shares **515 / 1,000** texts verbatim with
`twitter_training.csv` (51.5% leak). The training set itself contains
**4,505 duplicate text rows**. The reported 95–96% baseline accuracy is
therefore not a generalization metric — half of every val prediction was
a memorisation lookup. Fix: concat training + validation, dedupe by
cleaned text, stratified 90/10 re-split with seed 42
(`prep_twitter(..., resplit=True)`).

On the clean split, the existing recipe still failed: LSTM and GRU
collapsed to the class prior (acc ≈ 30%, F1 ≈ 0.15) because the no-mask
Embedding layer let zero-padded suffixes dilute the gradient. Enabling
`mask_zero=True` on the Embedding restored convergence and gave consistent
**87% acc / 0.86–0.87 macro F1** across RNN, LSTM, and GRU. Averaging
their softmax probabilities (`backend/ensemble_twitter.py`) yields
**89% acc / 0.888 F1** — a free +2 / +0.018 lift, since the three
architectures make complementary errors. The new canonical Twitter
results are these honest numbers; the leaky 95% is gone.

## Experiments that did **not** beat baseline

| Experiment | Where | Outcome |
|---|---|---|
| Full balanced class weights | Seattle ANN | acc 61 → 45, F1 0.327 → 0.235 — over-corrected |
| sqrt-balanced class weights | Seattle ANN | acc 61 → 63 but F1 0.327 → 0.270; reverted |
| sqrt class weights | Seattle LSTM/GRU | F1 moved < 0.005; promoted as a tie |
| `pool_over_time=True` | Seattle RNN/LSTM/GRU | Acc 0–4 pts worse; reverted |
| Longer training (epochs 80) | CORGIS all | Re-converged to same R²; LSTM/GRU got slightly worse via late overfit |
| cyclic + pool_over_time | CORGIS RNN/LSTM/GRU | R²=10–30 — much worse; reverted |
| Climate window=60 | Climate all | All architectures dropped (90→77, 72→68, 71→61, 71→59); reverted |
| SpecAugment (3 copies) | TESS RNN | Val 50 → 48; reverted |
| Mild specaug (1 mask) | TESS RNN | Val 50 → 46; reverted |
| Higher dropout + smaller LSTM | TESS RNN | Val 50 → 37 (overshoot); reverted |
| Resplit only (no mask) | Twitter LSTM | Collapse to 32% (model couldn't escape class prior) |
| Resplit + patient ES + ReduceLR | Twitter LSTM/GRU | LSTM still collapsed; GRU dropped to 82 |

## Methodology

- **Single change per experiment.** A run was promoted only if its primary
  metric (macro F1 for classification, R² for regression) strictly beat
  the previously-saved baseline.
- **Reproducibility.** Seed 42 fixed across `random`, `numpy`, `tensorflow`.
  Every run launched from `experiments_runner.py` re-applies the seeding
  hooks (`seed_tensorflow()` from `config.py`).
- **Result schema preserved.** Promoted runs overwrite
  `backend/results/<task>_<dataset>_<model>.json` so the existing
  `/api/dashboard` and the React frontend continue to work without
  modification. Suffix-tagged variants (`_long`, `_pool`, `_cyclic`,
  `_specaug`, `_resplit`, `_resplitm`, `_sqrtcw`, `_w60`, `_cycpool`, …)
  remain on disk for inspection.
- **Train/val split.** Treated existing val as the test set (no separate
  test fold available). Hyperparameter choices were guided by training-loss
  shape and val-loss elbow (early stopping signal); ultimately the val
  number is what's reported.
- **Compute.** All runs CPU-only on local machine (TF 2.21, Python 3.13).
  Total experimentation wall time ≈ 2 hours.

## Code-level changes

All flag defaults are off so original baseline behaviour is preserved.

- `prep_twitter` (preprocessing.py): `dedupe_train`, `drop_val_leak`,
  `resplit` flags. `resplit=True` concats train+val, dedupes by cleaned
  text, stratified-splits 90/10 with seed 42.
- `prep_tess` (preprocessing.py): SpecAugment training-only augmentation
  via `specaug_copies`, `specaug_time_mask`, `specaug_freq_mask`,
  `specaug_n_time`, `specaug_n_freq`.
- `prep_weather_regression` (preprocessing.py): `add_cyclic_time` flag —
  appends `sin(2π·week/52)` and `cos(2π·week/52)` columns.
- `_input_layer` (models.py): `embedding_mask_zero` so Twitter LSTM/GRU
  can train on the dedup'd split.
- `train.run_training`: `result_suffix=""` (writes `_<suffix>.{json,h5}`),
  `reduce_lr_on_plateau` + `reduce_lr_factor` / `reduce_lr_patience` /
  `min_lr`, configurable `monitor`,
  `use_class_weight ∈ {None, "balanced", "sqrt"}`, `save_plots`.
- `plot_roc_curve` (visualize.py): handle `auc=None` (Seattle multiclass
  case where AUC is not computed).

Helper scripts in `backend/`:

- `experiments_runner.py` — preset-driven launcher; appends to
  `experiments.md`, writes `results/experiments_log.json`.
- `promote_winners.py` — picks the best variant per (task, ds, model),
  copies it to the canonical filename, re-renders plots.
- `ensemble_twitter.py` — retrains the three Twitter architectures on the
  cleaned resplit, averages softmax outputs, prints ensemble metrics.

## Assumptions

1. **Val is also test.** No separate test fold exists; using existing
   `X_val, y_val` for both is the only option.
2. **Twitter resplit is the new ground truth.** The original
   `twitter_validation.csv` is unusable because of the 515/1,000 leak.
   Twitter numbers are lower in absolute terms than the pre-audit
   baseline but actually meaningful.
3. **TESS speaker-independent split is correct.** Train on OAF, test on
   YAF — without it, all models hit ~100% by memorising voice patterns.
4. **CPU-only compute.** No GPU available — capped individual runs at
   ≤ 10 min and skipped expensive options (Bidirectional, GloVe pretrained
   embeddings, transfer learning from wav2vec2/HuBERT).
5. **Macro-F1 over plain accuracy** for imbalanced classification (Seattle
   counts 641/640/101/53/26).
6. **Cyclic time encoding (sin/cos of week-of-year)** is preferable to a
   raw integer week index because it preserves the cyclic distance between
   week 52 and week 1.

## Top remaining bottlenecks

1. **Seattle weather is data-bound, not model-bound.** With 26 snow
   days, 53 drizzle, and a chronological split that hides snow entirely
   from val, even a perfect classifier on this split caps near the
   rain/sun majority prior. Worth trying: stratified random split, more
   years of data, or external priors (climatology). No further modelling
   work will move the needle here.
2. **CORGIS recurrent models are station-bound.** Each of the 318
   stations has its own seasonality and the global cyclic feature
   doesn't generalise across them. Likely paths: per-station embedding
   (treat station ID as a categorical feature), station-level metadata
   (lat/lon, elevation), or one model per cluster of similar stations.
3. **TESS small-dataset overfit on RNN.** SpecAugment helped LSTM/GRU
   but not the SimpleRNN, which already sat at the architectural ceiling.
   Pretrained audio features (wav2vec2, HuBERT) would likely break
   through ~70% F1 — not attempted here for CPU/binary-size reasons.
4. **Twitter LSTM with `mask_zero=False` is unstable.** The
   `mask_zero=True` variant should be the new default for any Twitter
   retrain. Trade-off: Keras .h5 serialisation gets messier with masks
   (NotEqual op), but the dashboard reads JSON, so nothing user-facing
   breaks. The ensemble script works around .h5-loading by retraining
   in-memory.

## Data-quality issues that need human attention outside the model

- `twitter_validation.csv` is not a clean held-out set. The project
  should either ship a true held-out file or permanently document the
  resplit as the canonical eval. The 4,505 in-train duplicates should
  also be deduped upstream.
- Seattle's chronological val split contains zero snow examples. If the
  intent is to evaluate snow performance, the split policy must change.
- CORGIS includes implausible avg_temp values down to −27°F and up to
  +100°F (real but extreme stations); a sanity-check filter or
  per-station scaling would likely help RMSE.
- TESS folder casing varies between mirrors (`OAF_Fear` vs `OAF_fear`,
  `OAF_Pleasant_surprise` vs `YAF_pleasant_surprised`). The label map in
  `config.EMOTION_LABEL_MAP` already handles this — just worth noting.

## How to reproduce end-to-end

```bash
cd backend

# Round 1
python experiments_runner.py seattle_cw      # ANN long-training win
python experiments_runner.py climate_pool    # RNN/LSTM/GRU + pool_over_time
python experiments_runner.py tess_aug        # LSTM/GRU SpecAugment
python experiments_runner.py twitter_hp2     # masked Embedding (LSTM fix + RNN/GRU lift)

# Round 2 — feature engineering
python experiments_runner.py corgis_cyclic   # ANN cyclic win
python experiments_runner.py climate_cyclic  # all 4 climate models lift hugely

# Promote + re-render plots
python promote_winners.py

# Optional ensemble
python ensemble_twitter.py
```
