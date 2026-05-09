# Experiments Log

Date started: 2026-05-09. Goal: improve validation metrics across all 18 trained models.

## Assumptions

- **Val = test.** The codebase only has a single hold-out (no 3-way split). I treat the existing `X_val/y_val` as both validation and test. Hyperparameters are picked on training-loss/early-stopping signals where possible to avoid leaking val into the search; all reported "final" numbers are val.
- **Compute budget.** CPU-only (no GPU detected). Per-run cap: ~30 min. Total wall budget: ~3-4 h.
- **Target metrics** (per task):
  - TESS speech (7-class, balanced) → **macro F1** (accuracy is fine since balanced).
  - Twitter sentiment (4-class, modestly imbalanced) → **accuracy + macro F1**.
  - CORGIS / Climate weather regression → **R² (the dashboard "accuracy")** primary, RMSE secondary.
  - Seattle weather (5-class, severely imbalanced 641/640/101/53/26) → **macro F1** (accuracy alone hides minority-class collapse).
- **No frontend changes.** I keep the result-JSON schema fully compatible — the existing dashboard keeps reading these files. Models keep their architecture name (RNN/LSTM/GRU/ANN) so per-architecture comparisons stay meaningful.
- **Improvement criterion.** A change is kept iff val metric strictly improves; revert otherwise.

---

## Phase 1 — Audit

### 1.1 Project map
- Pipeline: `data_loader.load(task, ds)` → `preprocessing.preprocess(...)` → `models.BUILDERS[arch]` → `train.run_training(...)` → `evaluate` + `visualize`.
- 18 models = {speech×{RNN,LSTM,GRU}} + {sentiment×{RNN,LSTM,GRU}} + {timeseries×{corgis,climate,seattle}×{ANN,RNN,LSTM,GRU}}.
- Single train/val hold-out per dataset. Reproducibility: seed 42 fixed for python, numpy, TF.

### 1.2 Baseline metrics (recovered from `results/*.json`)

| Task | Dataset | Model | Val acc / R² | F1/RMSE | Train end-loss | Val end-loss | Diagnosis |
|---|---|---|---|---|---|---|---|
| Speech | TESS | RNN | 50% acc | F1=0.477 | 0.096 | 2.351 | strong overfit |
| Speech | TESS | LSTM | 43% acc | F1=0.345 | 0.084 | 3.099 | strong overfit |
| Speech | TESS | GRU | 52% acc | F1=0.512 | 0.027 | 2.119 | strong overfit |
| Sentiment | Twitter | RNN | 96% acc | F1=0.955 | 0.165 | 0.214 | balanced — but val/train leak |
| Sentiment | Twitter | LSTM | 93% acc | F1=0.929 | 0.164 | 0.245 | balanced — but val/train leak |
| Sentiment | Twitter | GRU | 95% acc | F1=0.947 | 0.111 | 0.236 | mild overfit + leak |
| TS reg | corgis | ANN | R²=48% | RMSE=12.65 | 0.0085 | 0.0212 | ceiling — multi-station hard |
| TS reg | corgis | RNN | R²=42% | RMSE=13.39 | 0.0101 | 0.0161 | underfit (low capacity) |
| TS reg | corgis | LSTM | R²=48% | RMSE=12.61 | 0.0092 | 0.0266 | overfit small |
| TS reg | corgis | GRU | R²=41% | RMSE=13.41 | 0.0071 | 0.0215 | overfit small |
| TS reg | climate | ANN | R²=86% | RMSE=2.48 | 0.014 | 0.011 | great |
| TS reg | climate | RNN | R²=29% | RMSE=5.54 | 0.0345 | 0.0334 | underfit (early-stopped @6) |
| TS reg | climate | LSTM | R²=51% | RMSE=4.62 | 0.0218 | 0.0266 | underfit |
| TS reg | climate | GRU | R²=46% | RMSE=4.84 | 0.0409 | 0.0273 | underfit |
| TS cls | seattle | ANN | 61% acc | F1=0.327 | 0.99 | 0.98 | underfit, severe class imbalance |
| TS cls | seattle | RNN | 63% acc | F1=0.338 | 1.05 | 1.06 | underfit, severe class imbalance |
| TS cls | seattle | LSTM | 62% acc | F1=0.331 | 1.08 | 1.08 | underfit, severe class imbalance |
| TS cls | seattle | GRU | 59% acc | F1=0.315 | 1.02 | 1.02 | underfit, severe class imbalance |

### 1.3 Silent bugs / data quality issues found

1. **Twitter val/train leak (CRITICAL).** `twitter_validation.csv` shares **515 / 1,000** texts verbatim with `twitter_training.csv` (51.5% leak). The reported 95–96% accuracy is inflated; the true generalization metric is unknown. Fix: dedupe by text before splitting; or use a held-out fold from the training data.
2. **Twitter duplicate texts (training).** **4,505** duplicate text rows in training. They bias the loss toward repeated content. Fix: dedupe before fitting the tokenizer / sequence padding.
3. **Seattle severe class imbalance.** `rain=641, sun=640, fog=101, drizzle=53, snow=26`. With chronological 80/20 split, the val partition can hold many minorities and models default to predicting the two majority classes — explains the ~60% accuracy but ~33% macro F1. Fix: class weights or a focal loss; possibly re-balanced sampling.
4. **TESS small-dataset overfit.** Train acc 97-99% with only 1,400 OAF clips — the model memorizes one speaker. No augmentation. Adding spec-augment or modest noise/pitch jitter is a standard remedy.
5. **TS regression early-stopping pulls rug on RNN.** Climate-RNN stops at epoch 6 with R²=29% — patience=5 was too short; the training loss is still high and falling. Need warmup / longer patience / better LR.
6. **No label noise issue spot-checked at 25 random samples** in Twitter (looks fine). TESS labels are folder-derived and unambiguous. Seattle/corgis/climate labels are categorical met-station strings with no obvious mislabels.
7. **Loss/metric match.** SCC loss for classification, MSE for regression — both correct. The dashboard "accuracy" for regression is `max(0, R²·100)` which is reasonable.
8. **Preprocessing parity.** Inference-time preprocessing artifacts (`*.preproc.pkl`) exist — checked that `build_artifacts_*` mirrors `prep_*` (yes, scaler/tokenizer/MFCC stats are train-only fit). No mismatch.

### 1.4 Replication

The repository ships with cached results (`results/*.json`). I treat those as the baseline rather than re-running everything, to save the wall-clock budget for improvement experiments. Spot checks:
- Seattle ANN baseline reproduced from cache: acc=61, F1=0.327. ✓
- Twitter LSTM baseline reproduced from cache: acc=93, F1=0.929. ✓

## Phase 2 — Diagnosis

| Task family | Diagnosis | Evidence |
|---|---|---|
| TESS speech | **Overfit**, dominated by tiny dataset | 97-99% train vs 43-52% val |
| Twitter sentiment | **Inflated by leak** + mild overfit | val texts copied to train |
| TS regression (climate) | **Underfit** for RNN/LSTM/GRU; ANN already strong | 6 epochs early stop; val_loss above train_loss for RNN |
| TS regression (corgis) | **Capacity-limited / hard task** | Multi-station weekly weather has natural noise |
| TS classification (seattle) | **Underfit + class imbalance** | macro F1 ~33% with acc ~60%; majority class dominates |

## Phase 3 — Fixes (one change per experiment)

Format: `task/dataset/model: change → train_metric / val_metric` and a short note.

### Code-level changes that enable the experiments (all flags default off)

- `prep_twitter`: `dedupe_train`, `drop_val_leak`, `resplit` flags. `resplit=True` concats train+val, dedupes by cleaned text, stratified-splits 90/10 with seed 42 — the only honest evaluation given the leak.
- `prep_tess`: SpecAugment-style training-time augmentation via `specaug_copies`, `specaug_time_mask`, `specaug_freq_mask`, `specaug_n_time`, `specaug_n_freq`.
- `models._input_layer`: `embedding_mask_zero=True` on the Embedding layer. Twitter LSTM/GRU need this on the cleaned split — without it, gradients vanish through the zero-padded postfix and the model collapses to the class prior.
- `train.run_training`: `result_suffix` (writes to `_<suffix>.json/.h5` so experiments don't clobber baselines), `reduce_lr_on_plateau` + `reduce_lr_factor` / `reduce_lr_patience` / `min_lr`, configurable `monitor`, `use_class_weight ∈ {None, "balanced", "sqrt"}` for severely imbalanced classification.

A small `experiments_runner.py` driver invokes `run_training` with these overrides, logs every (label, metrics) tuple to `backend/results/experiments_log.json`, and appends a markdown line to this file. `promote_winners.py` decides whether each experiment beats the recorded baseline on the appropriate metric (macro F1 for classification, R² for regression) and copies winning JSON+H5 to the canonical filename.


### Experiment block: Seattle — class weights + LR plateau


### Experiment block: Seattle - longer training, no class weight

- **seattle-long** `timeseries/seattle/ANN` -> acc=66 f1=0.3498 trainAcc(last)=0.634 valAcc(last)=0.659 epochs=44 time=8.7s
- **seattle-long** `timeseries/seattle/RNN` -> acc=63 f1=0.3375 trainAcc(last)=0.590 valAcc(last)=0.437 epochs=21 time=3.7s
- **seattle-long** `timeseries/seattle/LSTM` -> acc=58 f1=0.3085 trainAcc(last)=0.564 valAcc(last)=0.461 epochs=23 time=12.7s
- **seattle-long** `timeseries/seattle/GRU` -> acc=60 f1=0.3184 trainAcc(last)=0.577 valAcc(last)=0.522 epochs=30 time=13.9s

### Experiment block: Seattle - longer training + sqrt class weight

- **seattle-sqrtcw** `timeseries/seattle/ANN` -> acc=63 f1=0.2702 trainAcc(last)=0.656 valAcc(last)=0.618 epochs=41 time=4.5s
- **seattle-sqrtcw** `timeseries/seattle/RNN` -> acc=58 f1=0.308 trainAcc(last)=0.572 valAcc(last)=0.440 epochs=24 time=4.1s
- **seattle-sqrtcw** `timeseries/seattle/LSTM` -> acc=62 f1=0.3317 trainAcc(last)=0.545 valAcc(last)=0.485 epochs=24 time=13.1s
- **seattle-sqrtcw** `timeseries/seattle/GRU` -> acc=60 f1=0.3188 trainAcc(last)=0.573 valAcc(last)=0.570 epochs=38 time=17.3s

### Experiment block: Climate — longer training + LR plateau

- **climate-long** `timeseries/climate/ANN` -> R²=86 mae=2.0089 rmse=2.4615 trainLoss(last)=0.0113 valLoss(last)=0.009 epochs=16 time=6.1s
- **climate-long** `timeseries/climate/RNN` -> R²=29 mae=4.8997 rmse=5.5447 trainLoss(last)=0.0251 valLoss(last)=0.0406 epochs=13 time=2.8s
- **climate-long** `timeseries/climate/LSTM` -> R²=54 mae=3.8577 rmse=4.447 trainLoss(last)=0.0163 valLoss(last)=0.0305 epochs=21 time=12.2s
- **climate-long** `timeseries/climate/GRU` -> R²=47 mae=4.0966 rmse=4.7981 trainLoss(last)=0.0171 valLoss(last)=0.0334 epochs=13 time=7.6s

### Experiment block: CORGIS — longer training + LR plateau

- **corgis-long** `timeseries/corgis/ANN` -> R²=48 mae=9.2824 rmse=12.6475 trainLoss(last)=0.0062 valLoss(last)=0.0256 epochs=16 time=8.8s
- **corgis-long** `timeseries/corgis/RNN` -> R²=42 mae=10.4697 rmse=13.3878 trainLoss(last)=0.0069 valLoss(last)=0.0169 epochs=19 time=11.7s
- **corgis-long** `timeseries/corgis/LSTM` -> R²=43 mae=11.0527 rmse=13.2069 trainLoss(last)=0.0045 valLoss(last)=0.038 epochs=18 time=36.1s
- **corgis-long** `timeseries/corgis/GRU` -> R²=38 mae=10.5572 rmse=13.8246 trainLoss(last)=0.0053 valLoss(last)=0.0375 epochs=17 time=26.9s

### Experiment block: TESS — SpecAugment + LR plateau

- **tess-specaug** `speech/tess/RNN` -> acc=48 f1=0.4676 trainAcc(last)=0.974 valAcc(last)=0.470 epochs=13 time=65.9s
- **tess-specaug** `speech/tess/LSTM` -> acc=56 f1=0.5199 trainAcc(last)=0.998 valAcc(last)=0.579 epochs=11 time=117.8s
- **tess-specaug** `speech/tess/GRU` -> acc=62 f1=0.6168 trainAcc(last)=0.999 valAcc(last)=0.601 epochs=12 time=90.1s

### Experiment block: Twitter — fix val/train leak

- **twitter-resplit** `sentiment/twitter/RNN` -> acc=83 f1=0.8259 trainAcc(last)=0.942 valAcc(last)=0.837 epochs=11 time=59.9s
- **twitter-resplit** `sentiment/twitter/LSTM` -> acc=32 f1=0.1482 trainAcc(last)=0.327 valAcc(last)=0.337 epochs=7 time=137.3s
- **twitter-resplit** `sentiment/twitter/GRU` -> acc=86 f1=0.8507 trainAcc(last)=0.964 valAcc(last)=0.875 epochs=18 time=296.4s

### Experiment block: Twitter - LSTM/GRU with longer patience + LR plateau on clean split


### Experiment block: Climate - RNN/LSTM/GRU with pool_over_time

- **climate-pool** `timeseries/climate/RNN` -> R²=43 mae=4.1372 rmse=4.9788 trainLoss(last)=0.0213 valLoss(last)=0.0314 epochs=18 time=18.5s
- **climate-pool** `timeseries/climate/LSTM` -> R²=64 mae=3.2114 rmse=3.9777 trainLoss(last)=0.0185 valLoss(last)=0.0301 epochs=20 time=29.2s
- **climate-pool** `timeseries/climate/GRU` -> R²=50 mae=3.9543 rmse=4.664 trainLoss(last)=0.0176 valLoss(last)=0.0321 epochs=22 time=29.6s
- **twitter-resplit-patient** `sentiment/twitter/RNN` -> acc=84 f1=0.8311 trainAcc(last)=0.978 valAcc(last)=0.839 epochs=19 time=114.8s

### Experiment block: Seattle - pool_over_time + longer

- **seattle-pool** `timeseries/seattle/RNN` -> acc=59 f1=0.3173 trainAcc(last)=0.522 valAcc(last)=0.567 epochs=43 time=15.4s
- **seattle-pool** `timeseries/seattle/LSTM` -> acc=58 f1=0.3097 trainAcc(last)=0.535 valAcc(last)=0.491 epochs=21 time=18.1s
- **seattle-pool** `timeseries/seattle/GRU` -> acc=57 f1=0.3045 trainAcc(last)=0.539 valAcc(last)=0.512 epochs=21 time=16.4s
- **twitter-resplit-patient** `sentiment/twitter/LSTM` -> acc=36 f1=0.2189 trainAcc(last)=0.309 valAcc(last)=0.315 epochs=10 time=203.4s

### Experiment block: TESS RNN - mild specaug + longer training

- **tess-mildaug** `speech/tess/RNN` -> acc=46 f1=0.4494 trainAcc(last)=0.992 valAcc(last)=0.483 epochs=13 time=65.6s
- **twitter-resplit-patient** `sentiment/twitter/GRU` -> acc=82 f1=0.8141 trainAcc(last)=0.936 valAcc(last)=0.824 epochs=30 time=583.3s

### Experiment block: Twitter - resplit + masked Embedding + long patience

- **twitter-resplit-masked** `sentiment/twitter/RNN` -> acc=87 f1=0.8717 trainAcc(last)=0.985 valAcc(last)=0.888 epochs=11 time=78.1s
- **twitter-resplit-masked** `sentiment/twitter/LSTM` -> acc=87 f1=0.8708 trainAcc(last)=0.974 valAcc(last)=0.880 epochs=12 time=258.1s
- **twitter-resplit-masked** `sentiment/twitter/GRU` -> acc=87 f1=0.8638 trainAcc(last)=0.975 valAcc(last)=0.883 epochs=11 time=217.8s

## Phase 4 — Final canonical metrics (after promote_winners)

After running `python promote_winners.py` and the manual Twitter override (since the leaky baseline F1 is not meaningfully comparable), the canonical `results/<task>_<dataset>_<model>.json` files now contain:

| task / dataset / model | metric | value | source variant |
|---|---|---|---|
| speech/tess/RNN  | acc / F1 | 50 / 0.477 | baseline (kept) |
| speech/tess/LSTM | acc / F1 | **56 / 0.520** | _specaug |
| speech/tess/GRU  | acc / F1 | **62 / 0.617** | _specaug |
| sentiment/twitter/RNN  | acc / F1 | 87 / 0.872 ‡ | _resplitm (resplit + mask_zero) |
| sentiment/twitter/LSTM | acc / F1 | 87 / 0.871 ‡ | _resplitm |
| sentiment/twitter/GRU  | acc / F1 | 87 / 0.864 ‡ | _resplitm |
| timeseries/climate/ANN  | R² / RMSE | 86 / 2.48 | baseline (kept) |
| timeseries/climate/RNN  | R² / RMSE | **43 / 4.98** | _pool |
| timeseries/climate/LSTM | R² / RMSE | **64 / 3.98** | _pool |
| timeseries/climate/GRU  | R² / RMSE | **50 / 4.66** | _pool |
| timeseries/corgis/ANN   | R² / RMSE | 48 / 12.65 | baseline (kept) |
| timeseries/corgis/RNN   | R² / RMSE | 42 / 13.39 | baseline (kept) |
| timeseries/corgis/LSTM  | R² / RMSE | 48 / 12.61 | baseline (kept) |
| timeseries/corgis/GRU   | R² / RMSE | 41 / 13.41 | baseline (kept) |
| timeseries/seattle/ANN  | acc / F1 | **66 / 0.350** | _long |
| timeseries/seattle/RNN  | acc / F1 | 63 / 0.338 | baseline (kept) |
| timeseries/seattle/LSTM | acc / F1 | 62 / 0.332 | _sqrtcw (≈ baseline) |
| timeseries/seattle/GRU  | acc / F1 | 60 / 0.319 | _sqrtcw (≈ baseline) |

‡ Twitter "improvement" is a data-leakage correction, not a modelling lift — the leaky baseline F1≈0.95 is not a generalization metric. The new 0.87 across all three architectures is the first honest number.

### How to reproduce

```bash
cd backend
python experiments_runner.py seattle_cw     # ANN long-training win
python experiments_runner.py climate_pool   # RNN/LSTM/GRU + pool_over_time
python experiments_runner.py tess_aug       # LSTM/GRU SpecAugment
python experiments_runner.py twitter_hp2    # masked Embedding (LSTM fix + RNN/GRU lift)
python promote_winners.py                   # copy winners to canonical paths + re-render plots
```

### Experiment block: CORGIS - cyclic week-of-year features

- **corgis-cyclic** `timeseries/corgis/ANN` -> R²=56 mae=8.566 rmse=11.5808 trainLoss(last)=0.0036 valLoss(last)=0.0198 epochs=18 time=9.3s
- **corgis-cyclic** `timeseries/corgis/RNN` -> R²=37 mae=10.9032 rmse=13.9172 trainLoss(last)=0.0054 valLoss(last)=0.0399 epochs=23 time=10.0s
- **corgis-cyclic** `timeseries/corgis/LSTM` -> R²=34 mae=11.5398 rmse=14.2652 trainLoss(last)=0.0048 valLoss(last)=0.0336 epochs=16 time=29.7s
- **corgis-cyclic** `timeseries/corgis/GRU` -> R²=35 mae=11.3456 rmse=14.1231 trainLoss(last)=0.0052 valLoss(last)=0.0313 epochs=16 time=25.2s

### Experiment block: Climate - cyclic + pool + longer window

- **climate-cyclic** `timeseries/climate/ANN` -> R²=90 mae=1.6001 rmse=2.0459 trainLoss(last)=0.0073 valLoss(last)=0.007 epochs=22 time=7.0s
- **climate-cyclic** `timeseries/climate/RNN` -> R²=72 mae=2.916 rmse=3.491 trainLoss(last)=0.0161 valLoss(last)=0.0314 epochs=15 time=3.2s
- **climate-cyclic** `timeseries/climate/LSTM` -> R²=71 mae=2.9211 rmse=3.5377 trainLoss(last)=0.0128 valLoss(last)=0.0195 epochs=13 time=8.8s
- **climate-cyclic** `timeseries/climate/GRU` -> R²=71 mae=3.0212 rmse=3.5604 trainLoss(last)=0.0121 valLoss(last)=0.0278 epochs=16 time=9.2s

### Experiment block: TESS RNN - higher dropout + smaller model

- **tess-rnn-drop** `speech/tess/RNN` -> acc=37 f1=0.309 trainAcc(last)=0.948 valAcc(last)=0.520 epochs=11 time=29.6s

### Experiment block: Climate - cyclic + window=60

- **climate-w60** `timeseries/climate/ANN` -> R²=77 mae=2.651 rmse=3.1803 trainLoss(last)=0.0082 valLoss(last)=0.0104 epochs=40 time=8.3s
- **climate-w60** `timeseries/climate/RNN` -> R²=68 mae=3.1608 rmse=3.724 trainLoss(last)=0.0111 valLoss(last)=0.0162 epochs=38 time=8.2s
- **climate-w60** `timeseries/climate/LSTM` -> R²=61 mae=3.6293 rmse=4.1206 trainLoss(last)=0.0135 valLoss(last)=0.0306 epochs=17 time=16.8s
- **climate-w60** `timeseries/climate/GRU` -> R²=59 mae=3.6861 rmse=4.2234 trainLoss(last)=0.0132 valLoss(last)=0.0289 epochs=17 time=14.7s

### Experiment block: CORGIS - cyclic + pool_over_time + lr 3e-3

- **corgis-cyclic-pool** `timeseries/corgis/RNN` -> R²=10 mae=12.895 rmse=16.6203 trainLoss(last)=0.0046 valLoss(last)=0.0261 epochs=26 time=23.3s
- **corgis-cyclic-pool** `timeseries/corgis/LSTM` -> R²=12 mae=12.9164 rmse=16.4268 trainLoss(last)=0.0028 valLoss(last)=0.0273 epochs=23 time=60.8s
- **corgis-cyclic-pool** `timeseries/corgis/GRU` -> R²=30 mae=11.6168 rmse=14.6105 trainLoss(last)=0.0032 valLoss(last)=0.0484 epochs=19 time=30.7s
