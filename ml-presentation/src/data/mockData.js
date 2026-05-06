// Placeholder data for the ML dashboard.
// TODO: swap every block tagged below with real values after training.

// ──────────────────────────────────────────────────────────────────────────
// Algorithm color palette — consistent across ALL charts so a user can
// recognize the same model instantly across tasks.
// ──────────────────────────────────────────────────────────────────────────
export const ALGO_COLORS = {
  ANN:  '#fbbf24', // amber
  RNN:  '#60a5fa', // blue
  LSTM: '#22d3ee', // teal (primary accent)
  GRU:  '#c084fc', // purple
};

export const ALGO_FULL = {
  ANN:  'Artificial Neural Network',
  RNN:  'Recurrent Neural Network',
  LSTM: 'Long Short-Term Memory',
  GRU:  'Gated Recurrent Unit',
};

// ──────────────────────────────────────────────────────────────────────────
// Curve generators — plausible-but-clean training dynamics.
// ──────────────────────────────────────────────────────────────────────────
function makeRand(seed) {
  let s = seed;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
}

// Loss + accuracy curves: monotonic-ish convergence to (finalLoss, finalAcc).
function genTraining(epochs, finalLoss, finalAcc, seed) {
  const r = makeRand(seed);
  const startLoss = Math.max(0.6, finalLoss + 0.7);
  const startAcc = Math.max(0.2, finalAcc - 0.4);
  const out = [];
  for (let i = 0; i < epochs; i++) {
    const t = i / (epochs - 1);
    const decay = Math.pow(1 - t, 2.0);
    const trainLoss = finalLoss + (startLoss - finalLoss) * decay + (r() - 0.5) * 0.012;
    const valLoss   = finalLoss * 1.12 + (startLoss * 1.05 - finalLoss * 1.12) * decay + (r() - 0.5) * 0.025;
    const trainAcc  = startAcc + (finalAcc - startAcc) * (1 - decay) + (r() - 0.5) * 0.006;
    const valAcc    = startAcc * 0.95 + (finalAcc * 0.97 - startAcc * 0.95) * (1 - decay) + (r() - 0.5) * 0.018;
    out.push({
      epoch: i + 1,
      trainLoss: +Math.max(0, trainLoss).toFixed(4),
      valLoss:   +Math.max(0, valLoss).toFixed(4),
      trainAcc:  +Math.min(0.99, Math.max(0, trainAcc)).toFixed(4),
      valAcc:    +Math.min(0.99, Math.max(0, valAcc)).toFixed(4),
    });
  }
  return out;
}

// ROC: TPR vs FPR; higher AUC = curve hugs top-left.
function genROC(auc, seed, steps = 30) {
  const r = makeRand(seed);
  const power = 1 / Math.max(0.05, auc * 2 - 0.7);
  const pts = [];
  for (let i = 0; i <= steps; i++) {
    const fpr = i / steps;
    const tpr = Math.min(1, Math.pow(fpr, 1 / power) * (auc * 1.05) + (r() - 0.5) * 0.012);
    pts.push({ fpr: +fpr.toFixed(3), tpr: +Math.max(0, Math.min(1, tpr)).toFixed(3) });
  }
  pts[0] = { fpr: 0, tpr: 0 }; pts[pts.length - 1] = { fpr: 1, tpr: 1 };
  return pts;
}

// Merge multiple ROC series keyed by algo.
function buildROC(aucMap) {
  const series = {};
  Object.keys(aucMap).forEach((k, i) => { series[k] = genROC(aucMap[k], 13 + i * 7); });
  const merged = [];
  for (let i = 0; i <= 30; i++) {
    const row = { fpr: i / 30 };
    Object.keys(series).forEach(k => { row[k] = series[k][i].tpr; });
    merged.push(row);
  }
  return { merged, aucMap };
}

// Time-series prediction line: actual + predicted over N days.
function genPredVsActual(n, rmse, seed) {
  const r = makeRand(seed);
  const out = [];
  let base = 18;
  for (let i = 0; i < n; i++) {
    const seasonal = Math.sin(i / n * Math.PI * 4) * 6 + Math.sin(i / n * Math.PI * 8) * 2;
    const drift = i / n * 4 - 2;
    const noise = (r() - 0.5) * 1.2;
    const actual = base + seasonal + drift + noise;
    const predErr = (r() - 0.5) * rmse * 1.6;
    const pred = actual + predErr;
    out.push({ day: i + 1, actual: +actual.toFixed(2), predicted: +pred.toFixed(2) });
  }
  return out;
}

// Confusion matrix for K classes: diag-heavy, plausible off-diag.
function genConfusion(classes, accuracy, seed) {
  const r = makeRand(seed);
  const k = classes.length;
  const total = 80;
  const m = Array.from({ length: k }, () => Array(k).fill(0));
  for (let i = 0; i < k; i++) {
    const correct = Math.round(total * accuracy + (r() - 0.5) * 4);
    m[i][i] = Math.max(40, Math.min(total - 4, correct));
    let remaining = total - m[i][i];
    for (let j = 0; j < k; j++) {
      if (j === i) continue;
      const share = remaining / (k - 1);
      const v = Math.max(0, Math.round(share + (r() - 0.5) * share));
      m[i][j] = v;
    }
    const rowSum = m[i].reduce((a, b) => a + b, 0);
    const diff = total - rowSum;
    for (let j = k - 1; j >= 0 && j !== i; j--) {
      if (m[i][j] + diff >= 0) { m[i][j] += diff; break; }
    }
  }
  return m;
}

// ──────────────────────────────────────────────────────────────────────────
// TASK 1 — Speech Emotion Classification (TESS)
// Algorithms: RNN, LSTM, GRU
// ──────────────────────────────────────────────────────────────────────────
const EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']; // TODO: confirm 7 classes

export const TASK_SPEECH = {
  id: 'speech',
  num: '01',
  kicker: 'Task 1',
  title: 'Speech Emotion Classification',
  subtitle: 'Recognising 7 emotional states from short Toronto Emotional Speech Set (TESS) recordings.',
  iconName: 'AudioWaveform',
  dataset: {
    name: 'Toronto Emotional Speech Set (TESS)',
    source: 'kaggle.com/ejlok1/toronto-emotional-speech-set-tess',
    type: 'Audio (.wav)',
    target: '7 emotion classes',
    stats: [
      { label: 'Audio clips',       value: '2,800', icon: 'Database' }, // TODO
      { label: 'Emotion classes',   value: '7',     icon: 'Layers' },
      { label: 'Sample rate',       value: '24 kHz', icon: 'Activity' },
      { label: 'Avg clip length',   value: '2.1 s', icon: 'Clock' },
      { label: 'Speakers',          value: '2',     icon: 'Users' },
      { label: 'MFCC coefficients', value: '40',    icon: 'BarChart3', mono: true },
    ],
    classDist: EMOTION_CLASSES.map((c, i) => ({ name: c, value: 400 + (i * 7) % 30 })), // TODO
  },
  pipeline: [
    { id: 'load',     title: 'Load Audio',    icon: 'FolderOpen',
      desc: 'Read .wav files with librosa, preserve original sample rates, attach emotion labels from filename.' },
    { id: 'resample', title: 'Resample',      icon: 'Repeat',
      desc: 'Resample every clip to a uniform 24 kHz so feature extraction is comparable across speakers.' },
    { id: 'mfcc',     title: 'MFCC Extract',  icon: 'AudioWaveform',
      desc: '40 Mel-frequency cepstral coefficients per frame — captures timbre, the cue most correlated with emotion.' },
    { id: 'pad',      title: 'Pad / Truncate', icon: 'Maximize2',
      desc: 'Pad shorter clips and truncate longer ones to 174 frames so the model sees fixed-length sequences.' },
    { id: 'split',    title: 'Train / Test',   icon: 'SplitSquareHorizontal',
      desc: 'Stratified 80/20 split by emotion class — keeps the class balance identical between train and test.' },
  ],
  algos: ['RNN', 'LSTM', 'GRU'],
  // TODO: replace with real metrics
  metrics: {
    RNN:  { accuracy: 78, precision: 0.77, recall: 0.76, f1: 0.76, auc: 0.84, params: '58,247',  trainingTime: '6m 12s' },
    LSTM: { accuracy: 88, precision: 0.89, recall: 0.87, f1: 0.88, auc: 0.94, params: '142,983', trainingTime: '11m 04s', best: true },
    GRU:  { accuracy: 86, precision: 0.86, recall: 0.85, f1: 0.85, auc: 0.92, params: '108,601', trainingTime: '8m 47s' },
  },
  arch: {
    RNN: [
      { name: 'Input',     shape: '(174, 40)', type: 'input' },
      { name: 'SimpleRNN', shape: '128',       type: 'rnn', act: 'tanh' },
      { name: 'Dropout',   shape: '0.3',       type: 'reg' },
      { name: 'Dense',     shape: '7',         type: 'dense', act: 'softmax' },
    ],
    LSTM: [
      { name: 'Input',   shape: '(174, 40)', type: 'input' },
      { name: 'LSTM',    shape: '128',       type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.4',       type: 'reg' },
      { name: 'LSTM',    shape: '64',        type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '32',        type: 'dense', act: 'ReLU' },
      { name: 'Dense',   shape: '7',         type: 'dense', act: 'softmax' },
    ],
    GRU: [
      { name: 'Input',   shape: '(174, 40)', type: 'input' },
      { name: 'GRU',     shape: '96',        type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.3',       type: 'reg' },
      { name: 'GRU',     shape: '48',        type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '7',         type: 'dense', act: 'softmax' },
    ],
  },
  archMeta: {
    RNN:  { activation: 'tanh + softmax',                   regularization: 'Dropout (0.3), Early Stopping (p=10)' },
    LSTM: { activation: 'tanh, sigmoid, ReLU, softmax',     regularization: 'Dropout (0.4), L2 (1e-4), Early Stopping (p=12)' },
    GRU:  { activation: 'tanh, sigmoid, softmax',           regularization: 'Dropout (0.3), Early Stopping (p=10)' },
  },
  curves: {
    RNN:  genTraining(40, 0.55, 0.78, 11),
    LSTM: genTraining(40, 0.32, 0.88, 23),
    GRU:  genTraining(40, 0.38, 0.86, 41),
  },
  roc: buildROC({ RNN: 0.84, LSTM: 0.94, GRU: 0.92 }),
  confusion: {
    RNN:  genConfusion(EMOTION_CLASSES, 0.78, 11),
    LSTM: genConfusion(EMOTION_CLASSES, 0.88, 23),
    GRU:  genConfusion(EMOTION_CLASSES, 0.86, 41),
  },
  classes: EMOTION_CLASSES,
};

// ──────────────────────────────────────────────────────────────────────────
// TASK 2 — Sentiment Analysis (Twitter Entity Sentiment)
// Algorithms: RNN, LSTM, GRU
// ──────────────────────────────────────────────────────────────────────────
const SENTIMENT_CLASSES = ['Positive', 'Negative', 'Neutral', 'Irrelevant'];

export const TASK_SENTIMENT = {
  id: 'sentiment',
  num: '02',
  kicker: 'Task 2',
  title: 'Sentiment Analysis',
  subtitle: 'Classifying tweets into Positive, Negative, Neutral, or Irrelevant given a target entity.',
  iconName: 'MessageSquare',
  dataset: {
    name: 'Twitter Entity Sentiment Analysis',
    source: 'kaggle.com/jp797498e/twitter-entity-sentiment-analysis',
    type: 'Text',
    target: '4 sentiment classes',
    stats: [
      { label: 'Total tweets',       value: '74,682', icon: 'Database' }, // TODO
      { label: 'Sentiment classes',  value: '4',      icon: 'Layers' },
      { label: 'Unique entities',    value: '32',     icon: 'Tag' },
      { label: 'Avg tweet length',   value: '18 tok', icon: 'Type' },
      { label: 'Vocab size',         value: '24,500', icon: 'Hash' },
      { label: 'Embedding dim',      value: '128',    icon: 'BarChart3', mono: true },
    ],
    classDist: [ // TODO
      { name: 'Positive',   value: 20832 },
      { name: 'Negative',   value: 22542 },
      { name: 'Neutral',    value: 18318 },
      { name: 'Irrelevant', value: 12990 },
    ],
  },
  pipeline: [
    { id: 'lower',    title: 'Lowercase',     icon: 'CaseLower',
      desc: 'Force every character to lowercase so "Apple" and "apple" share a single embedding.' },
    { id: 'tokenize', title: 'Tokenize',      icon: 'Type',
      desc: 'Whitespace + punctuation tokenizer; URLs, mentions, and hashtags collapsed to special tokens.' },
    { id: 'stop',     title: 'Stopword Filter', icon: 'Filter',
      desc: 'Drop common English stopwords ("the", "is", "and") that contribute no sentiment signal.' },
    { id: 'embed',    title: 'Embedding',     icon: 'Hash',
      desc: '128-dim trainable embedding initialised on the training vocabulary — 24,500 unique tokens.' },
    { id: 'pad',      title: 'Pad Sequences', icon: 'Maximize2',
      desc: 'Pad/truncate every tokenised tweet to length 50 — 95th percentile of the training distribution.' },
  ],
  algos: ['RNN', 'LSTM', 'GRU'],
  metrics: { // TODO
    RNN:  { accuracy: 72, precision: 0.71, recall: 0.70, f1: 0.70, auc: 0.78, params: '3.21M',  trainingTime: '4m 50s' },
    LSTM: { accuracy: 84, precision: 0.85, recall: 0.83, f1: 0.84, auc: 0.91, params: '3.42M',  trainingTime: '9m 18s', best: true },
    GRU:  { accuracy: 82, precision: 0.83, recall: 0.81, f1: 0.82, auc: 0.89, params: '3.38M',  trainingTime: '7m 22s' },
  },
  arch: {
    RNN: [
      { name: 'Input',     shape: '(50,)',     type: 'input' },
      { name: 'Embedding', shape: '24500×128', type: 'dense' },
      { name: 'SimpleRNN', shape: '64',        type: 'rnn', act: 'tanh' },
      { name: 'Dropout',   shape: '0.3',       type: 'reg' },
      { name: 'Dense',     shape: '4',         type: 'dense', act: 'softmax' },
    ],
    LSTM: [
      { name: 'Input',     shape: '(50,)',     type: 'input' },
      { name: 'Embedding', shape: '24500×128', type: 'dense' },
      { name: 'LSTM',      shape: '128',       type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout',   shape: '0.4',       type: 'reg' },
      { name: 'Dense',     shape: '32',        type: 'dense', act: 'ReLU' },
      { name: 'Dense',     shape: '4',         type: 'dense', act: 'softmax' },
    ],
    GRU: [
      { name: 'Input',     shape: '(50,)',     type: 'input' },
      { name: 'Embedding', shape: '24500×128', type: 'dense' },
      { name: 'GRU',       shape: '128',       type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout',   shape: '0.3',       type: 'reg' },
      { name: 'Dense',     shape: '4',         type: 'dense', act: 'softmax' },
    ],
  },
  archMeta: {
    RNN:  { activation: 'tanh + softmax',                regularization: 'Dropout (0.3), Early Stopping (p=8)' },
    LSTM: { activation: 'tanh, sigmoid, ReLU, softmax',  regularization: 'Dropout (0.4), Early Stopping (p=10)' },
    GRU:  { activation: 'tanh, sigmoid, softmax',        regularization: 'Dropout (0.3), Early Stopping (p=10)' },
  },
  curves: {
    RNN:  genTraining(30, 0.62, 0.72, 51),
    LSTM: genTraining(30, 0.38, 0.84, 53),
    GRU:  genTraining(30, 0.42, 0.82, 55),
  },
  roc: buildROC({ RNN: 0.78, LSTM: 0.91, GRU: 0.89 }),
  classes: SENTIMENT_CLASSES,
};

// ──────────────────────────────────────────────────────────────────────────
// TASK 3 — Time Series Forecasting (3 datasets)
// Algorithms: ANN, RNN, LSTM, GRU (all four)
// ──────────────────────────────────────────────────────────────────────────
const TS_PIPELINE = [
  { id: 'raw',    title: 'Raw Data',      icon: 'Database',
    desc: 'Daily observations across one or more climate variables loaded from CSV.' },
  { id: 'clean',  title: 'Cleaning',      icon: 'Sparkles',
    desc: 'Outliers detected via IQR, duplicate timestamps removed, anomalous readings clipped to 4σ.' },
  { id: 'norm',   title: 'Normalization', icon: 'Sliders',
    desc: 'MinMax scaling per feature → every channel lives in [0, 1] so magnitude bias dies.' },
  { id: 'window', title: 'Windowing',     icon: 'LayoutGrid',
    desc: 'Sliding window of 30 days predicts day 31. Stride = 1. Sequence length is uniform across models.' },
  { id: 'split',  title: 'Train / Test',  icon: 'SplitSquareHorizontal',
    desc: 'Chronological 80/20 split — no leakage from future into past.' },
];

export const TS_ALGOS = ['ANN', 'RNN', 'LSTM', 'GRU'];

// Generic architecture for tabular time-series tasks
function tsArch() {
  return {
    ANN: [
      { name: 'Input',   shape: '(150,)', type: 'input' },
      { name: 'Dense',   shape: '128',    type: 'dense', act: 'ReLU' },
      { name: 'Dropout', shape: '0.2',    type: 'reg' },
      { name: 'Dense',   shape: '64',     type: 'dense', act: 'ReLU' },
      { name: 'Dense',   shape: '1',      type: 'dense', act: 'Linear' },
    ],
    RNN: [
      { name: 'Input',     shape: '(30, F)', type: 'input' },
      { name: 'SimpleRNN', shape: '64',      type: 'rnn', act: 'tanh' },
      { name: 'Dropout',   shape: '0.3',     type: 'reg' },
      { name: 'Dense',     shape: '1',       type: 'dense', act: 'Linear' },
    ],
    LSTM: [
      { name: 'Input',   shape: '(30, F)', type: 'input' },
      { name: 'LSTM',    shape: '128',     type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.3',     type: 'reg' },
      { name: 'LSTM',    shape: '64',      type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '1',       type: 'dense', act: 'Linear' },
    ],
    GRU: [
      { name: 'Input',   shape: '(30, F)', type: 'input' },
      { name: 'GRU',     shape: '96',      type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.3',     type: 'reg' },
      { name: 'GRU',     shape: '48',      type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '1',       type: 'dense', act: 'Linear' },
    ],
  };
}
const TS_ARCH = tsArch();
const TS_ARCH_META = {
  ANN:  { activation: 'ReLU + Linear',           regularization: 'Dropout (0.2), Early Stopping (p=8)' },
  RNN:  { activation: 'tanh + Linear',           regularization: 'Dropout (0.3), Early Stopping (p=10)' },
  LSTM: { activation: 'tanh, sigmoid, Linear',   regularization: 'Dropout (0.3), L2 (1e-4), ES (p=12)' },
  GRU:  { activation: 'tanh, sigmoid, Linear',   regularization: 'Dropout (0.3), Early Stopping (p=10)' },
};

// Build a complete TS-task object from a small spec.
function buildTSTask({ id, name, source, target, type, stats, classify, metrics, predSeed }) {
  const curves = {};
  TS_ALGOS.forEach(a => {
    const finalLoss = classify
      ? 1 - (metrics[a].accuracy / 100) + 0.05
      : Math.min(0.6, metrics[a].rmse / 10);
    const finalAcc = classify ? metrics[a].accuracy / 100 : 0.5 + (1 - finalLoss) * 0.5;
    curves[a] = genTraining(40, finalLoss, finalAcc, predSeed + a.charCodeAt(0));
  });
  const predVsActual = {};
  TS_ALGOS.forEach(a => { predVsActual[a] = genPredVsActual(60, metrics[a].rmse || 2.5, predSeed + a.charCodeAt(0) * 3); });

  const bestId = TS_ALGOS.reduce((best, a) => {
    const better = classify ? metrics[a].accuracy > metrics[best].accuracy : metrics[a].rmse < metrics[best].rmse;
    return better ? a : best;
  }, TS_ALGOS[0]);

  return {
    id, name, source, target, type, stats, classify, metrics, bestId,
    curves, predVsActual,
    arch: TS_ARCH, archMeta: TS_ARCH_META,
    pipeline: TS_PIPELINE, algos: TS_ALGOS,
  };
}

// TODO: every metric block below is placeholder data.
const TS_DATASETS = [
  buildTSTask({
    id: 'corgis',
    name: 'CORGIS Weather',
    source: 'corgis-edu.github.io/corgis/csv/weather',
    type: 'Tabular time series',
    target: 'Mean temperature (°C)',
    stats: [
      { label: 'Records',      value: '16,743',  icon: 'Database' },
      { label: 'Features',     value: '14',      icon: 'Layers' },
      { label: 'Stations',     value: '49',      icon: 'MapPin' },
      { label: 'Time span',    value: '2016',    icon: 'Calendar' },
      { label: 'Frequency',    value: 'Daily',   icon: 'Clock' },
      { label: 'Target',       value: 'Mean Temp', icon: 'Target', mono: true },
    ],
    metrics: {
      ANN:  { accuracy: 78, mae: 2.94, mape: 6.2,  rmse: 3.8, params: '52,481',  trainingTime: '2m 44s' },
      RNN:  { accuracy: 82, mae: 2.43, mape: 5.1,  rmse: 3.2, params: '64,193',  trainingTime: '5m 18s' },
      LSTM: { accuracy: 88, mae: 1.92, mape: 4.0,  rmse: 2.6, params: '128,917', trainingTime: '8m 32s' },
      GRU:  { accuracy: 87, mae: 2.05, mape: 4.2,  rmse: 2.7, params: '96,801',  trainingTime: '6m 41s' },
    },
    predSeed: 71,
  }),
  buildTSTask({
    id: 'climate',
    name: 'Daily Climate',
    source: 'kaggle.com/sumanthvrao/daily-climate-time-series-data',
    type: 'Daily climate measurements',
    target: 'Mean temperature (°C)',
    stats: [
      { label: 'Records',     value: '1,576',     icon: 'Database' },
      { label: 'Features',    value: '5',         icon: 'Layers' },
      { label: 'Time range',  value: '2013–2017', icon: 'Calendar' },
      { label: 'Frequency',   value: 'Daily',     icon: 'Clock' },
      { label: 'Missing',     value: '0',         icon: 'CheckCircle2' },
      { label: 'Target',      value: 'meantemp',  icon: 'Target', mono: true },
    ],
    metrics: {
      ANN:  { accuracy: 81, mae: 2.21, mape: 7.8,  rmse: 2.9, params: '48,193',  trainingTime: '2m 14s' },
      RNN:  { accuracy: 85, mae: 1.81, mape: 6.4,  rmse: 2.4, params: '62,401',  trainingTime: '4m 32s' },
      LSTM: { accuracy: 92, mae: 1.42, mape: 4.9,  rmse: 1.9, params: '124,673', trainingTime: '7m 48s' },
      GRU:  { accuracy: 90, mae: 1.51, mape: 5.2,  rmse: 2.0, params: '94,209',  trainingTime: '5m 56s' },
    },
    predSeed: 41,
  }),
  buildTSTask({
    id: 'seattle',
    name: 'Seattle Weather',
    source: 'kaggle.com/petalme/seattle-weather-prediction-dataset',
    type: 'Daily weather records',
    target: 'Weather category (5 classes)',
    classify: true,
    stats: [
      { label: 'Records',     value: '1,461',     icon: 'Database' },
      { label: 'Features',    value: '5',         icon: 'Layers' },
      { label: 'Classes',     value: '5',         icon: 'Tag' },
      { label: 'Time range',  value: '2012–2015', icon: 'Calendar' },
      { label: 'Frequency',   value: 'Daily',     icon: 'Clock' },
      { label: 'Target',      value: 'weather',   icon: 'Target', mono: true },
    ],
    metrics: {
      ANN:  { accuracy: 74, precision: 0.73, recall: 0.71, f1: 0.71, rmse: 0,  params: '46,721',  trainingTime: '2m 02s' },
      RNN:  { accuracy: 78, precision: 0.77, recall: 0.76, f1: 0.76, rmse: 0,  params: '60,201',  trainingTime: '4m 11s' },
      LSTM: { accuracy: 85, precision: 0.86, recall: 0.84, f1: 0.85, rmse: 0,  params: '120,481', trainingTime: '7m 24s' },
      GRU:  { accuracy: 83, precision: 0.84, recall: 0.82, f1: 0.83, rmse: 0,  params: '92,018',  trainingTime: '5m 41s' },
    },
    predSeed: 89,
  }),
];

export const TASK_TIMESERIES = {
  id: 'timeseries',
  num: '03',
  kicker: 'Task 3',
  title: 'Time-Series Forecasting',
  subtitle: 'Three weather datasets, the full ANN/RNN/LSTM/GRU lineup, head-to-head on each.',
  iconName: 'LineChart',
  datasets: TS_DATASETS,
};

// ──────────────────────────────────────────────────────────────────────────
// CROSS-TASK SUMMARY
// ──────────────────────────────────────────────────────────────────────────
export const SUMMARY_ROWS = [
  { task: 'Speech Emotion',  taskId: 'speech',     dataset: 'TESS',            best: 'LSTM', metric: 'Accuracy', score: '88%',   classify: true },
  { task: 'Sentiment',       taskId: 'sentiment',  dataset: 'Twitter',         best: 'LSTM', metric: 'Accuracy', score: '84%',   classify: true },
  { task: 'Time Series',     taskId: 'timeseries', dataset: 'CORGIS Weather',  best: 'LSTM', metric: 'RMSE',     score: '2.6',   classify: false },
  { task: 'Time Series',     taskId: 'timeseries', dataset: 'Daily Climate',   best: 'LSTM', metric: 'RMSE',     score: '1.9',   classify: false },
  { task: 'Time Series',     taskId: 'timeseries', dataset: 'Seattle Weather', best: 'LSTM', metric: 'Accuracy', score: '85%',   classify: true },
];

// Heatmap: rows = datasets, cols = algos, value = primary metric (normalized 0-1 for color)
export const HEATMAP_DATA = [
  { dataset: 'TESS (Speech)',          ANN: null, RNN: 78, LSTM: 88, GRU: 86, metric: 'Acc' },
  { dataset: 'Twitter (Sentiment)',    ANN: null, RNN: 72, LSTM: 84, GRU: 82, metric: 'Acc' },
  { dataset: 'CORGIS Weather',         ANN: 78,   RNN: 82, LSTM: 88, GRU: 87, metric: 'Acc*' },
  { dataset: 'Daily Climate',          ANN: 81,   RNN: 85, LSTM: 92, GRU: 90, metric: 'Acc*' },
  { dataset: 'Seattle Weather',        ANN: 74,   RNN: 78, LSTM: 85, GRU: 83, metric: 'Acc' },
];

// ──────────────────────────────────────────────────────────────────────────
// TEAM
// ──────────────────────────────────────────────────────────────────────────
export const TEAM = [
  {
    name: 'Muhammad Firdous',
    role: 'Sole author — full pipeline',
    initials: 'MF',
    contribution: 'Built the full pipeline end-to-end: TESS audio preprocessing & MFCC extraction, Twitter text tokenisation & embedding training, three time-series datasets, all 18 model trainings (ANN/RNN/LSTM/GRU), and the comparative analysis writeup.',
    worked: ['TESS · RNN/LSTM/GRU', 'Twitter · RNN/LSTM/GRU', 'CORGIS · ANN/RNN/LSTM/GRU', 'Daily Climate · ANN/RNN/LSTM/GRU', 'Seattle · ANN/RNN/LSTM/GRU'],
  },
];

// ──────────────────────────────────────────────────────────────────────────
// PROJECT MAP — for the hero tree diagram
// ──────────────────────────────────────────────────────────────────────────
export const PROJECT_MAP = [
  { task: 'Speech',     icon: 'AudioWaveform', datasets: ['TESS'],                                       algos: ['RNN', 'LSTM', 'GRU'] },
  { task: 'Sentiment',  icon: 'MessageSquare', datasets: ['Twitter Entity'],                             algos: ['RNN', 'LSTM', 'GRU'] },
  { task: 'Time Series', icon: 'LineChart',    datasets: ['CORGIS', 'Daily Climate', 'Seattle Weather'], algos: ['ANN', 'RNN', 'LSTM', 'GRU'] },
];

// Total models trained: 3 + 3 + (4×3) = 18.
export const HERO_STATS = [
  { value: 3,  suffix: '',  label: 'Task categories' },
  { value: 5,  suffix: '',  label: 'Datasets' },
  { value: 4,  suffix: '',  label: 'Algorithm families' },
  { value: 18, suffix: '',  label: 'Models trained' },
];
