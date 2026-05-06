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
