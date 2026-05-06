// Talks to the Flask backend at /api/dashboard.
// Falls back to mockData.js when the backend is unreachable so the
// presentation still renders even if the user forgets to start it.

import * as mock from './mockData.js';

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://127.0.0.1:5050';

function buildMockPayload() {
  return {
    algoColors: mock.ALGO_COLORS,
    algoFull: {
      ANN: 'Artificial Neural Network',
      RNN: 'Recurrent Neural Network',
      LSTM: 'Long Short-Term Memory',
      GRU: 'Gated Recurrent Unit',
    },
    heroStats: mock.HERO_STATS,
    projectMap: mock.PROJECT_MAP,
    team: mock.TEAM,
    summaryRows: mock.SUMMARY_ROWS,
    heatmapData: mock.HEATMAP_DATA,
    tasks: {
      speech: mock.TASK_SPEECH,
      sentiment: mock.TASK_SENTIMENT,
      timeseries: mock.TASK_TIMESERIES,
    },
  };
}

export async function fetchDashboard({ signal } = {}) {
  try {
    const res = await fetch(`${API_BASE}/api/dashboard`, { signal });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    return { data, source: 'api', error: null };
  } catch (err) {
    if (err.name === 'AbortError') throw err;
    return { data: buildMockPayload(), source: 'mock', error: err };
  }
}

export async function postTrain(body) {
  const res = await fetch(`${API_BASE}/api/train`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function predictText({ task, model, text }) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task, model, text }),
  });
  if (!res.ok) throw new Error((await res.json()).message || `HTTP ${res.status}`);
  return res.json();
}

export async function predictAudio({ model, file }) {
  const fd = new FormData();
  fd.append('task', 'speech');
  fd.append('model', model);
  fd.append('audio', file);
  const res = await fetch(`${API_BASE}/api/predict`, { method: 'POST', body: fd });
  if (!res.ok) throw new Error((await res.json()).message || `HTTP ${res.status}`);
  return res.json();
}

export async function listTSSamples(datasetId) {
  const res = await fetch(`${API_BASE}/api/predict/samples/${datasetId}`);
  if (!res.ok) return [];
  return res.json();
}

export async function predictTSSample({ datasetId, model, sampleId }) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task: 'timeseries', datasetId, model, sampleId }),
  });
  if (!res.ok) throw new Error((await res.json()).message || `HTTP ${res.status}`);
  return res.json();
}
