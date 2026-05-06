# ML Dashboard — Flask Backend

Serves the data the React dashboard at `../ml-presentation/` consumes. Phase 2
(current) returns mock data shaped exactly like `data.jsx`. Real model results
overwrite the mock as Phase 3+ training fills in
`results/<task>_<dataset>_<model>.json`.

---

## 1. Install

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`tensorflow`, `librosa`, and `nltk` are heavy. Install times of 5–10 min on a
fresh box are normal.

After install (one-time), download the NLTK stopwords corpus:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## 2. Datasets — download + place

Place files in this exact tree:

```
backend/data/
├── tess/
│   ├── OAF_angry/   *.wav
│   ├── OAF_disgust/ *.wav
│   ├── ... (all 14 ACTRESS_<emotion>/ folders, 2,800 .wav total)
│   └── YAF_sad/     *.wav
├── twitter/
│   ├── twitter_training.csv
│   └── twitter_validation.csv
└── weather/
    ├── corgis_weather.csv
    ├── DailyDelhiClimateTrain.csv
    ├── DailyDelhiClimateTest.csv
    └── seattle-weather.csv
```

### Direct dataset links

| Dataset | URL |
|---|---|
| Toronto Emotional Speech Set (TESS) | https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess |
| Twitter Entity Sentiment Analysis | https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis |
| Daily Climate Time Series Data | https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data |
| Seattle Weather Prediction | https://www.kaggle.com/datasets/petalme/seattle-weather-prediction-dataset |
| CORGIS Weather (no Kaggle account needed) | https://corgis-edu.github.io/corgis/csv/weather/ |

### Kaggle CLI (recommended for the Kaggle ones)

```bash
# Configure once: place kaggle.json in ~/.kaggle/ (or %USERPROFILE%\.kaggle\)
pip install kaggle

cd backend/data

kaggle datasets download -d ejlok1/toronto-emotional-speech-set-tess -p tess --unzip
kaggle datasets download -d jp797498e/twitter-entity-sentiment-analysis -p twitter --unzip
kaggle datasets download -d sumanthvrao/daily-climate-time-series-data -p weather --unzip
kaggle datasets download -d petalme/seattle-weather-prediction-dataset -p weather --unzip
```

After unzip:

- **TESS**: the archive contains a top-level `TESS Toronto emotional speech set data/`
  directory; move (or symlink) the 14 `*_<emotion>/` folders inside it directly into
  `backend/data/tess/`.
- **Twitter**: leave `twitter_training.csv` / `twitter_validation.csv` at the top of `twitter/`.
- **Weather**: the Daily Climate archive has its CSVs nested one level deep; flatten them.
- **CORGIS**: download `weather.csv` directly from the URL above and rename to `corgis_weather.csv`.

### Verify placement

```bash
python verify_data.py
```

Output is per-dataset OK/FAIL. Backend won't run real training until this is green.

---

## 3. Run the backend

```bash
python app.py
```

Listens on `http://127.0.0.1:5050`. CORS is open to `http://localhost:5173` and
`http://127.0.0.1:5173` (the Vite dev server).

### Quick smoke test

```bash
curl http://127.0.0.1:5050/api/health
curl http://127.0.0.1:5050/api/verify
curl http://127.0.0.1:5050/api/dashboard | python -m json.tool | head -40
```

---

## 4. Run frontend + backend together

In two terminals:

```bash
# terminal 1
cd backend
python app.py

# terminal 2
cd ml-presentation
npm run dev
```

Open `http://localhost:5173`. The dashboard fetches `/api/dashboard` on mount.
A small "Backend offline — showing cached data" badge appears in the bottom-left
if the API is unreachable.

To point the frontend at a different backend host/port, set `VITE_API_BASE`
in `ml-presentation/.env.local`, e.g.:

```
VITE_API_BASE=http://localhost:8080
```

---

## 5. Endpoints

| Method | Path | Returns |
|---|---|---|
| GET  | `/api/health` | `{status, phase, message}` |
| GET  | `/api/verify` | `{ok, checks: [{name, ok, detail}]}` |
| GET  | `/api/dashboard` | Bundle: `algoColors, algoFull, heroStats, projectMap, team, summaryRows, heatmapData, tasks: {speech, sentiment, timeseries}` |
| GET  | `/api/datasets` | Array of `{task, datasetId, name, source, target, classify, stats, classDist?}` |
| GET  | `/api/dataset/<task>/<datasetId>/summary` | Stats + class distribution + pipeline + classes |
| GET  | `/api/metrics/<task>/<datasetId>/<model>` | Per-model: metrics + curve + roc/confusion or predVsActual |
| GET  | `/api/compare/<task>/<datasetId>` | All models for one dataset, with bestId |
| GET  | `/api/plots/<task>/<datasetId>/<model>` | `{plots: {loss, acc, roc, confusion, predVsActual}}` (only those that exist on disk) |
| GET  | `/static/plots/<filename>` | Raw PNG |
| POST | `/api/train` | `{task, datasetId, model, hyperparams?, force_retrain?}` → `{status, runId, ...}` |
| POST | `/api/predict` | Single-sample inference (Phase 7) |

JSON shapes match `data.jsx` / `mockData.js` field-for-field — see Phase 1
contract document for details.

---

## 6. Folder structure

```
backend/
├── app.py                  Flask routes + CORS
├── config.py               Paths, seeds, hyperparameter defaults, dataset registry
├── verify_data.py          Pre-flight check: every dataset file present?
├── requirements.txt
├── README.md               (this file)
├── data/
│   ├── tess/               14 ACTRESS_<emotion>/ folders, 2,800 .wav
│   ├── twitter/            twitter_training.csv, twitter_validation.csv
│   └── weather/            corgis_weather.csv, DailyDelhiClimate{Train,Test}.csv, seattle-weather.csv
├── src/
│   ├── mock_data.py        Python port of mockData.js — fallback when no result cache
│   ├── data_loader.py      Per-dataset raw loaders
│   ├── preprocessing.py    MFCC / tokenisation / windowing
│   ├── models.py           build_ann/rnn/lstm/gru
│   ├── train.py            Generic loop + EarlyStopping + cache (also CLI runner)
│   ├── evaluate.py         Classification + regression metrics
│   └── visualize.py        PNG plots (loss / acc / ROC / confusion / pred-vs-actual)
├── saved_models/           18 *.h5 (one per task/dataset/model)
├── results/                18 per-model JSON files
└── plots/                  PNG plots for every model
```

---

## 7. Phase status — all 18 models trained ✅

| Phase | Status |
|---|---|
| 1. API contract | ✅ |
| 2. Backend skeleton + mock endpoints + frontend wiring | ✅ |
| 3. Task 3 — Weather (CORGIS, Daily Climate, Seattle) × ANN/RNN/LSTM/GRU (12 models) | ✅ |
| 4. Task 2 — Twitter sentiment × RNN/LSTM/GRU (3 models) | ✅ |
| 5. Task 1 — TESS speech emotion × RNN/LSTM/GRU (3 models) | ✅ |
| 6. Evaluation & visualization layer | ✅ |
| 7. End-to-end verification — dashboard shows real numbers | ✅ |

### Final results (18 models on a CPU, ~30 min total)

| Task | Dataset | ANN | RNN | LSTM | GRU | Best | Headline |
|---|---|---|---|---|---|---|---|
| Speech | TESS (speaker-independent) | — | 50% | 43% | **52%** | GRU | Train on OAF, test on YAF — GRU generalizes best across voices |
| Sentiment | Twitter | — | **96%** | 93% | 95% | RNN | Textbook short text; all three are strong |
| Time Series | CORGIS Weather | 48% | 42% | **48%** | 41% | LSTM (RMSE 12.6°F) | Multi-station weekly is genuinely hard |
| Time Series | Daily Climate | **86%** | 29% | 51% | 46% | ANN (RMSE 2.5°C) | Smooth seasonal data → MLP > recurrent |
| Time Series | Seattle Weather | 61% | **63%** | 62% | 59% | RNN | All cluster ~60%; class imbalance dominates |

Accuracy is rendered as **R² for regression** (CORGIS, Daily Climate) and standard accuracy for classification (TESS, Twitter, Seattle). Full per-model metrics including precision/recall/F1/AUC/MAE/RMSE/MAPE live in `results/<task>_<dataset>_<model>.json`.

### Architectural & methodology notes worth knowing

The training pipeline is configured cleanly in `config.py`, but a few task-specific tweaks were needed:

1. **TESS uses a speaker-independent split.** TESS contains only 2 actresses (OAF, YAF) reading the same 200 sentences across 7 emotions. A naive stratified random split lets both speakers leak into train and val, and every model hits ~100% by memorising voice + sentence patterns rather than emotion features. We hold YAF out entirely: train on OAF, test on YAF. MFCC z-score statistics are also computed from the train (OAF) split only. Results drop from a misleading 100% to a realistic 43–52%, and the comparative story between RNN/LSTM/GRU becomes meaningful.
2. **TESS (speech) needs `pool_over_time=True` and `lr=5e-3`.** With 174 timesteps of 40-D MFCC, the default `lr=1e-3` + final-hidden-state classification leaves GRU stuck at chance. Switching to `return_sequences=True` + `GlobalAveragePooling1D` + 5× higher LR fixes it. Applied to all 3 RNN-family speech models for a fair comparison.
3. **MFCC features need z-score normalization.** Raw MFCCs span roughly `[-700, 200]` and the first coefficient (energy) has std 214 — large enough to saturate the gating sigmoids on a long sequence. `preprocessing.prep_tess` standardises per coefficient before training, using train-speaker stats only.
4. **No L2 regularization on the LSTM kernel.** Combined with the no-mask Embedding for the Twitter task, an L2 of 1e-4 pushes the kernel toward zero before it can learn — model collapses to the class prior (27% acc). With L2 removed all three sentiment models train cleanly.

Time-series tasks use the default LR / hp without per-task overrides.

### Retraining

```bash
# Single combo:
python -m src.train timeseries/climate/LSTM

# All four algorithms on one dataset:
python -m src.train timeseries/corgis/all

# Force retrain (overrides the ``results/*.json`` cache):
python -m src.train sentiment/twitter/LSTM --force

# Everything (~30 min on CPU):
python -m src.train all --force
```

`/api/train` accepts the same combo and the same `force_retrain` flag, plus `hyperparams` overrides for one-off experiments.

---

## 8. Reproducibility

- `PYTHONHASHSEED=42` (set in `config.py`)
- `numpy.random.seed(42)`, `random.seed(42)`
- `tf.random.set_seed(42)` + `tf.keras.utils.set_random_seed(42)` (called by
  `config.seed_tensorflow()` from training entrypoints)
- Dropout 0.3 across every model
- EarlyStopping `patience=5, restore_best_weights=True`
- Per-task hyperparameters frozen in `config.HYPERPARAMS`; `/api/train` accepts
  per-call overrides but defaults to these.
