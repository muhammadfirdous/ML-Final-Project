# Machine Learning Comparison Dashboard

A machine learning project comparing ANN, RNN, LSTM, and GRU models across speech emotion recognition, text sentiment, and weather tasks. A React dashboard presents results from a Flask API, with training code, saved metrics, and experiment reports in the repository.

## Tasks

| Area | Datasets | Models |
| --- | --- | --- |
| Speech emotion | Toronto Emotional Speech Set (TESS) | RNN, LSTM, GRU |
| Sentiment analysis | Twitter Entity Sentiment Analysis | RNN, LSTM, GRU, ensemble |
| Weather regression | CORGIS Weather, Daily Delhi Climate | ANN, RNN, LSTM, GRU |
| Weather classification | Seattle Weather | ANN, RNN, LSTM, GRU |

## Start here

- [Results and methodology corrections](RESULTS.md): the latest project report, including the Twitter data leakage correction.
- [Experiment log](experiments.md): the changes explored and their recorded outcomes.
- [Backend guide](backend/README.md): dataset placement, training commands, and API details.

Use `RESULTS.md` when reviewing reported outcomes; the backend guide also retains earlier baseline results. Classification accuracy and regression R² are different metrics and should not be compared as a single accuracy score.

## Run locally

Requirements: Python, Node.js, and npm. TensorFlow and the audio-processing dependencies may require platform-specific setup.

```bash
git clone https://github.com/muhammadfirdous/ML-Final-Project.git
cd ML-Final-Project
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
cd backend
python app.py
```

The API listens on `http://127.0.0.1:5050`. In a second terminal, from the repository root:

```bash
cd ml-presentation
npm ci
npm run dev
```

Open the URL printed by Vite. The frontend uses cached fallback data when the backend is unavailable; verify the API connection before interpreting the dashboard as a live result.

## Data and training

Follow the [backend dataset setup](backend/README.md) before retraining. From `backend/`, with the Python environment active:

```bash
python verify_data.py
python -m src.train timeseries/climate/LSTM
```

Saved result JSON files are under `backend/results/`; evaluation plots are under `backend/plots/`. Training can take substantial time and is separate from starting the dashboard.

## Repository map

| Path | Purpose |
| --- | --- |
| `ml-presentation/` | React dashboard connected to the API |
| `backend/app.py` | Flask endpoints |
| `backend/src/` | Loading, preprocessing, training, evaluation, and inference |
| `backend/results/` | Recorded metrics and experiment results |
| `backend/plots/` | Generated evaluation figures |
| `backend/saved_models/` | Saved models and preprocessing artifacts |
| `project/`, `ml-final-project/` | Additional frontend prototypes |

## Stack

Python, TensorFlow/Keras, Flask, React, Vite, and Recharts.

This is an educational comparison. Reported results depend on the datasets, splits, and experiments documented in the repository; they are not deployment performance guarantees.
