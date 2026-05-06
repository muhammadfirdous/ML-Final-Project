import { useEffect, useMemo, useState } from 'react';
import { Card, SectionHeader, Reveal } from './components.jsx';
import { Icon } from './Icon.jsx';
import { useDashboard } from './data/DashboardContext.jsx';
import {
  predictText, predictAudio, predictTSSample, listTSSamples,
} from './data/dashboardApi.js';

// ─────────────────────────────────────────────────────────────────────────
// Shared bits
// ─────────────────────────────────────────────────────────────────────────
function ProbBars({ probabilities, predictedClass }) {
  const { data } = useDashboard();
  const accent = data.algoColors.LSTM;
  const entries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
  return (
    <div className="space-y-2 mt-4">
      {entries.map(([cls, p]) => {
        const pct = (p * 100).toFixed(1);
        const isWinner = cls === predictedClass;
        return (
          <div key={cls}>
            <div className="flex justify-between text-xs mb-1">
              <span className={isWinner ? 'font-semibold' : ''}
                    style={{ color: isWinner ? accent : 'var(--text-muted)' }}>
                {cls}{isWinner && ' ← prediction'}
              </span>
              <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{pct}%</span>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--bg-elev)' }}>
              <div className="h-full transition-all duration-700"
                   style={{ width: `${pct}%`,
                            background: isWinner ? accent : 'var(--text-dim)' }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ModelPicker({ algos, value, onChange }) {
  const { data } = useDashboard();
  return (
    <div className="inline-flex items-center gap-1 p-1 rounded-xl"
         style={{ background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
      {algos.map(a => {
        const active = a === value;
        return (
          <button key={a} onClick={() => onChange(a)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2"
            style={{
              background: active ? 'var(--bg-card)' : 'transparent',
              color: active ? 'var(--text)' : 'var(--text-muted)',
              border: '1px solid ' + (active ? 'var(--border-strong)' : 'transparent'),
            }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: data.algoColors[a] }} />
            {a}
          </button>
        );
      })}
    </div>
  );
}

function ResultStatus({ loading, error }) {
  if (loading) {
    return (
      <div className="flex items-center gap-2 mt-4 text-xs font-mono"
           style={{ color: 'var(--text-muted)' }}>
        <span className="w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
        Predicting…
      </div>
    );
  }
  if (error) {
    return (
      <div className="mt-4 text-xs font-mono px-3 py-2 rounded-lg"
           style={{ color: '#fda4af', background: 'rgba(244, 63, 94, 0.08)',
                    border: '1px solid rgba(244, 63, 94, 0.25)' }}>
        ✕ {error}
      </div>
    );
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────
// Sentiment — text input
// ─────────────────────────────────────────────────────────────────────────
const SENTIMENT_EXAMPLES = [
  'This game is absolutely amazing — best update in years.',
  "I hate this update, completely broken and unplayable.",
  'Saw the announcement at the conference. Anyone got more details?',
  'Finally fixed the matchmaking bug. Devs deserve a round of applause.',
];

export function TryItSentiment({ algos }) {
  const [model, setModel] = useState(algos.find(a => a === 'LSTM') || algos[0]);
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submit = async () => {
    if (!text.trim()) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await predictText({ task: 'sentiment', model, text });
      setResult(r);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card padding="p-7">
      <div className="flex items-start justify-between mb-5 gap-4 flex-wrap">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider mb-1"
               style={{ color: 'var(--accent)' }}>Try it yourself</div>
          <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>
            Run a tweet through the trained model
          </div>
        </div>
        <ModelPicker algos={algos} value={model} onChange={setModel} />
      </div>

      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        rows={3}
        placeholder="Type or paste a tweet here…"
        className="w-full px-4 py-3 rounded-lg font-sans text-sm resize-none focus:outline-none"
        style={{
          background: 'var(--bg-elev)', color: 'var(--text)',
          border: '1px solid var(--border)',
        }}
      />

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button onClick={submit} disabled={loading || !text.trim()}
          className="px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: text.trim() && !loading ? 'var(--accent)' : 'var(--bg-elev)',
            color: text.trim() && !loading ? 'var(--bg)' : 'var(--text-muted)',
            cursor: text.trim() && !loading ? 'pointer' : 'not-allowed',
          }}>
          Predict sentiment
        </button>
        <span className="text-xs" style={{ color: 'var(--text-dim)' }}>or try:</span>
        {SENTIMENT_EXAMPLES.map((ex, i) => (
          <button key={i} onClick={() => setText(ex)}
            className="text-xs px-2 py-1 rounded-md font-mono transition-colors"
            style={{ background: 'var(--bg-elev)', color: 'var(--text-muted)',
                     border: '1px solid var(--border)' }}>
            example {i + 1}
          </button>
        ))}
      </div>

      <ResultStatus loading={loading} error={error} />
      {result && (
        <div className="mt-5 pt-5" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="text-xs font-mono uppercase tracking-wider mb-1"
               style={{ color: 'var(--text-muted)' }}>Cleaned input → model</div>
          <div className="text-sm font-mono px-3 py-2 rounded-md mb-1"
               style={{ background: 'var(--bg-elev)', color: 'var(--text)' }}>
            {result.cleaned || '(empty after cleaning)'}
          </div>
          <ProbBars probabilities={result.probabilities} predictedClass={result.predictedClass} />
        </div>
      )}
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Speech — wav upload
// ─────────────────────────────────────────────────────────────────────────
export function TryItSpeech({ algos }) {
  const [model, setModel] = useState(algos.find(a => a === 'LSTM') || algos[0]);
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!file) { setAudioUrl(null); return; }
    const url = URL.createObjectURL(file);
    setAudioUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const onFile = (e) => {
    const f = e.target.files?.[0] || null;
    setFile(f);
    setResult(null);
    setError(null);
  };

  const submit = async () => {
    if (!file) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await predictAudio({ model, file });
      setResult(r);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card padding="p-7">
      <div className="flex items-start justify-between mb-5 gap-4 flex-wrap">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider mb-1"
               style={{ color: 'var(--accent)' }}>Try it yourself</div>
          <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>
            Upload a .wav clip and classify the emotion
          </div>
        </div>
        <ModelPicker algos={algos} value={model} onChange={setModel} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="px-4 py-2 rounded-lg text-sm font-medium cursor-pointer transition-colors flex items-center gap-2"
          style={{ background: 'var(--bg-elev)', color: 'var(--text)',
                   border: '1px solid var(--border-strong)' }}>
          <Icon name="upload" size={14} />
          {file ? 'Replace file' : 'Choose .wav'}
          <input type="file" accept=".wav,audio/wav" onChange={onFile} className="hidden" />
        </label>
        {file && (
          <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
            {file.name} · {(file.size / 1024).toFixed(0)} KB
          </span>
        )}
        <button onClick={submit} disabled={loading || !file}
          className="ml-auto px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: file && !loading ? 'var(--accent)' : 'var(--bg-elev)',
            color: file && !loading ? 'var(--bg)' : 'var(--text-muted)',
            cursor: file && !loading ? 'pointer' : 'not-allowed',
          }}>
          Predict emotion
        </button>
      </div>

      {audioUrl && (
        <audio controls src={audioUrl} className="w-full mt-4" style={{ borderRadius: 8 }} />
      )}

      <ResultStatus loading={loading} error={error} />
      {result && (
        <div className="mt-5 pt-5" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="text-xs font-mono uppercase tracking-wider mb-2"
               style={{ color: 'var(--text-muted)' }}>
            Audio: {result.audioDurationSec}s
          </div>
          <ProbBars probabilities={result.probabilities} predictedClass={result.predictedClass} />
        </div>
      )}
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────────────
// Time-series — pick a sample window
// ─────────────────────────────────────────────────────────────────────────
export function TryItTimeSeries({ datasetId, algos, classify }) {
  const [model, setModel] = useState(algos.find(a => a === 'LSTM') || algos[0]);
  const [samples, setSamples] = useState([]);
  const [sampleId, setSampleId] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    listTSSamples(datasetId).then(s => {
      if (!alive) return;
      setSamples(s);
      setSampleId(s[0]?.id ?? null);
    });
    return () => { alive = false; };
  }, [datasetId]);

  // Reset prediction whenever the dataset / sample / model changes.
  useEffect(() => { setResult(null); setError(null); }, [datasetId, sampleId, model]);

  const submit = async () => {
    if (sampleId == null) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const r = await predictTSSample({ datasetId, model, sampleId });
      setResult(r);
    } catch (e) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card padding="p-7">
      <div className="flex items-start justify-between mb-5 gap-4 flex-wrap">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider mb-1"
               style={{ color: 'var(--accent)' }}>Try it yourself</div>
          <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>
            {classify ? 'Classify a held-out day' : 'Forecast the next step from a held-out window'}
          </div>
        </div>
        <ModelPicker algos={algos} value={model} onChange={setModel} />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs font-mono uppercase tracking-wider"
               style={{ color: 'var(--text-muted)' }}>Sample</label>
        <select value={sampleId ?? ''} onChange={e => setSampleId(e.target.value)}
          className="px-3 py-2 rounded-lg text-sm font-mono"
          style={{ background: 'var(--bg-elev)', color: 'var(--text)',
                   border: '1px solid var(--border)' }}>
          {samples.length === 0 && <option value="">no samples available</option>}
          {samples.map(s => (
            <option key={s.id} value={s.id}>{s.label || s.id}</option>
          ))}
        </select>
        <button onClick={submit} disabled={loading || sampleId == null}
          className="ml-auto px-4 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{
            background: sampleId != null && !loading ? 'var(--accent)' : 'var(--bg-elev)',
            color: sampleId != null && !loading ? 'var(--bg)' : 'var(--text-muted)',
            cursor: sampleId != null && !loading ? 'pointer' : 'not-allowed',
          }}>
          {classify ? 'Predict class' : 'Predict next step'}
        </button>
      </div>

      <ResultStatus loading={loading} error={error} />
      {result && classify && result.probabilities && (
        <div className="mt-5 pt-5" style={{ borderTop: '1px solid var(--border)' }}>
          {result.sample?.actualClass && (
            <div className="text-xs font-mono mb-3"
                 style={{ color: 'var(--text-muted)' }}>
              Actual: <span style={{ color: 'var(--text)' }}>{result.sample.actualClass}</span>
              {result.sample.date && <span> · {result.sample.date}</span>}
            </div>
          )}
          <ProbBars probabilities={result.probabilities} predictedClass={result.predictedClass} />
        </div>
      )}
      {result && !classify && result.predictedValue != null && (
        <div className="mt-5 pt-5 grid grid-cols-1 md:grid-cols-3 gap-5"
             style={{ borderTop: '1px solid var(--border)' }}>
          <div>
            <div className="text-xs font-mono uppercase tracking-wider mb-1"
                 style={{ color: 'var(--text-dim)' }}>Predicted</div>
            <div className="font-display text-2xl font-semibold tracking-tight"
                 style={{ color: 'var(--accent)' }}>{result.predictedValue}</div>
          </div>
          {result.sample?.actual != null && (
            <>
              <div>
                <div className="text-xs font-mono uppercase tracking-wider mb-1"
                     style={{ color: 'var(--text-dim)' }}>Actual</div>
                <div className="font-display text-2xl font-semibold tracking-tight"
                     style={{ color: 'var(--text)' }}>{result.sample.actual}</div>
              </div>
              <div>
                <div className="text-xs font-mono uppercase tracking-wider mb-1"
                     style={{ color: 'var(--text-dim)' }}>Error</div>
                <div className="font-display text-2xl font-semibold tracking-tight"
                     style={{ color: 'var(--text)' }}>
                  {Math.abs(result.predictedValue - result.sample.actual).toFixed(2)}
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </Card>
  );
}
