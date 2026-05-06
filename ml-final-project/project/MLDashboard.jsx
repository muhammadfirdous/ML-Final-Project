// =============================================================================
// ML Final Project — Time-Series Forecasting on Daily Climate Data
// Single-file React component for Vite + Tailwind CSS
// =============================================================================
//
// DEPENDENCIES (install via npm/pnpm/yarn):
//   react, react-dom, recharts, lucide-react
//
// USAGE:
//   import MLDashboard from './MLDashboard';
//   export default function App() { return <MLDashboard />; }
//
// REPLACE-DATA: search for "REPLACE-DATA" markers to swap in real values.
// =============================================================================

import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine,
} from 'recharts';
import {
  Sun, Moon, User, GraduationCap, Calendar, ArrowDown, Layers, Database,
  CheckCircle2, Target, Clock, Sparkles, Sliders, LayoutGrid,
  SplitSquareHorizontal, Table, ChevronRight, Trophy, Lightbulb, AlertTriangle,
} from 'lucide-react';

const THEME_CSS = `
:root { --bg:#070b14; --bg-elev:#0c1322; --bg-card:rgba(18,26,44,0.6); --border:rgba(148,163,184,0.12); --border-strong:rgba(148,163,184,0.22); --text:#e6edf7; --text-muted:#8a96ab; --text-dim:#5d6a82; --accent:#22d3ee; --accent-2:#38bdf8; --accent-glow:rgba(34,211,238,0.18); }
html.light { --bg:#f6f8fc; --bg-elev:#ffffff; --bg-card:rgba(255,255,255,0.75); --border:rgba(15,23,42,0.08); --border-strong:rgba(15,23,42,0.14); --text:#0f172a; --text-muted:#475569; --text-dim:#94a3b8; --accent:#0891b2; --accent-2:#0284c7; --accent-glow:rgba(8,145,178,0.12); }
html { scroll-behavior:smooth; scroll-padding-top:88px; }
body { background:var(--bg); color:var(--text); font-family:Inter,sans-serif; }
.glass { background:var(--bg-card); border:1px solid var(--border); backdrop-filter:blur(12px); }
.card-hover { transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; }
.card-hover:hover { transform: translateY(-2px); border-color: var(--border-strong); box-shadow: 0 8px 32px -8px var(--accent-glow); }
.gradient-text { background: linear-gradient(135deg, var(--text) 0%, var(--accent) 100%); -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.reveal { opacity:0; transform:translateY(24px); transition: opacity .7s ease, transform .7s ease; }
.reveal.in { opacity:1; transform:translateY(0); }
@keyframes wave-drift { 0%{transform:translateX(0) scaleY(1)} 50%{transform:translateX(-40px) scaleY(1.03)} 100%{transform:translateX(-80px) scaleY(1)} }
.hero-wave path { animation: wave-drift 18s ease-in-out infinite; }
.hero-wave path:nth-child(2) { animation-duration:24s; animation-direction:reverse; }
.hero-wave path:nth-child(3) { animation-duration:30s; }
.hero-wave path:nth-child(4) { animation-duration:22s; animation-direction:reverse; }
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.winner-shimmer { background: linear-gradient(110deg, transparent 30%, var(--accent-glow) 50%, transparent 70%); background-size:200% 100%; animation: shimmer 4s ease-in-out infinite; }
@keyframes pulse-glow { 0%,100%{box-shadow:0 0 0 0 var(--accent-glow)} 50%{box-shadow:0 0 0 8px transparent} }
.pulse-dot { animation: pulse-glow 2.4s ease-in-out infinite; }
.recharts-cartesian-grid line { stroke: var(--border); }
.recharts-text { fill: var(--text-muted); font-family: Inter, sans-serif; font-size:12px; }
`;


// =============================================================================

// PLACEHOLDER DATA — REPLACE-DATA when real numbers are ready

// =============================================================================

// Placeholder data for the ML dashboard.
// REPLACE-DATA: swap these blocks with your real values when ready.

const MODELS = [
  { id: 'ANN',  name: 'ANN',  fullName: 'Artificial Neural Network', color: '#a78bfa', accuracy: 82, precision: 0.81, recall: 0.79, f1: 0.80, rmse: 3.42, trainingTime: '2m 14s', params: '48,193' },
  { id: 'RNN',  name: 'RNN',  fullName: 'Recurrent Neural Network', color: '#f472b6', accuracy: 85, precision: 0.84, recall: 0.83, f1: 0.83, rmse: 2.98, trainingTime: '4m 32s', params: '62,401' },
  { id: 'LSTM', name: 'LSTM', fullName: 'Long Short-Term Memory',   color: '#22d3ee', accuracy: 91, precision: 0.92, recall: 0.90, f1: 0.91, rmse: 2.11, trainingTime: '7m 48s', params: '124,673', best: true },
  { id: 'GRU',  name: 'GRU',  fullName: 'Gated Recurrent Unit',     color: '#34d399', accuracy: 89, precision: 0.89, recall: 0.88, f1: 0.88, rmse: 2.34, trainingTime: '5m 56s', params: '94,209' },
];

// Plausible-but-clean training curves: fast initial drop, gentle plateau,
// validation slightly above training and noisier.
function genLossCurves(epochs, finalTrain, finalVal, seed) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const data = [];
  const startTrain = 0.85, startVal = 0.92;
  for (let i = 0; i < epochs; i++) {
    const t = i / (epochs - 1);
    const decay = Math.pow(1 - t, 2.2);
    const trainLoss = finalTrain + (startTrain - finalTrain) * decay + (rand() - 0.5) * 0.012;
    const valLoss   = finalVal   + (startVal - finalVal) * decay + (rand() - 0.5) * 0.028;
    const trainAcc  = 0.55 + (1 - decay) * (1 - finalTrain * 0.6) + (rand() - 0.5) * 0.01;
    const valAcc    = 0.52 + (1 - decay) * (1 - finalVal * 0.55) + (rand() - 0.5) * 0.022;
    data.push({
      epoch: i + 1,
      trainLoss: +trainLoss.toFixed(4),
      valLoss: +valLoss.toFixed(4),
      trainAcc: +Math.min(0.99, Math.max(0, trainAcc)).toFixed(4),
      valAcc:   +Math.min(0.99, Math.max(0, valAcc)).toFixed(4),
    });
  }
  return data;
}

const TRAINING_CURVES = {
  ANN:  genLossCurves(40, 0.18, 0.22, 11),
  RNN:  genLossCurves(40, 0.14, 0.18, 23),
  LSTM: genLossCurves(40, 0.07, 0.10, 41),
  GRU:  genLossCurves(40, 0.09, 0.12, 67),
};

// ROC curve: TPR vs FPR. Generate concave-up curve, better models hug top-left.
function genROC(auc, seed) {
  let s = seed;
  const rand = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const pts = [];
  const power = 1 / (auc * 2 - 0.7); // higher AUC → steeper
  for (let i = 0; i <= 30; i++) {
    const fpr = i / 30;
    const tpr = Math.min(1, Math.pow(fpr, 1 / power) * (auc * 1.05) + (rand() - 0.5) * 0.015);
    pts.push({ fpr: +fpr.toFixed(3), tpr: +Math.max(0, Math.min(1, tpr)).toFixed(3) });
  }
  pts[0] = { fpr: 0, tpr: 0 };
  pts[pts.length - 1] = { fpr: 1, tpr: 1 };
  return pts;
}

const ROC_DATA = (() => {
  const aucMap = { ANN: 0.84, RNN: 0.87, LSTM: 0.94, GRU: 0.92 };
  const series = {};
  Object.keys(aucMap).forEach((k, i) => { series[k] = genROC(aucMap[k], 13 + i * 7); });
  // Merge into a single array indexed by fpr step
  const merged = [];
  for (let i = 0; i <= 30; i++) {
    const row = { fpr: i / 30 };
    Object.keys(series).forEach(k => { row[k] = series[k][i].tpr; });
    merged.push(row);
  }
  return { merged, aucMap };
})();

// Preview table — first 5 rows of Daily Climate dataset
const PREVIEW_ROWS = [
  { date: '2013-01-01', meantemp: 10.00, humidity: 84.50, wind_speed: 0.00, meanpressure: 1015.67 },
  { date: '2013-01-02',  meantemp:  7.40, humidity: 92.00, wind_speed: 2.98, meanpressure: 1017.80 },
  { date: '2013-01-03',  meantemp:  7.17, humidity: 87.00, wind_speed: 4.63, meanpressure: 1018.67 },
  { date: '2013-01-04',  meantemp:  8.67, humidity: 71.33, wind_speed: 1.23, meanpressure: 1017.17 },
  { date: '2013-01-05',  meantemp:  6.00, humidity: 86.83, wind_speed: 3.70, meanpressure: 1016.50 },
];

const FEATURE_TYPES = [
  { name: 'Numeric',     value: 4, color: '#22d3ee' },
  { name: 'Datetime',    value: 1, color: '#a78bfa' },
  { name: 'Categorical', value: 0, color: '#f472b6' },
];

const DATASET_STATS = [
  { label: 'Features',         value: '5',     icon: 'Layers' },
  { label: 'Rows (samples)',   value: '1,576', icon: 'Database' },
  { label: 'Missing values',   value: '0',     icon: 'CheckCircle2' },
  { label: 'Target variable',  value: 'meantemp', icon: 'Target', mono: true },
  { label: 'Time range',       value: "2013–2017", icon: 'Calendar' },
  { label: 'Frequency',        value: 'Daily',  icon: 'Clock' },
];

const PIPELINE_STEPS = [
  { id: 'raw',    title: 'Raw Data',          icon: 'Database',
    desc: '1,576 daily observations spanning four climate variables collected in Delhi between 2013 and 2017.' },
  { id: 'clean',  title: 'Cleaning',          icon: 'Sparkles',
    desc: 'Outliers detected via IQR, duplicate timestamps removed, and a single anomalous pressure reading clipped to 4σ.' },
  { id: 'norm',   title: 'Normalization',     icon: 'Sliders',
    desc: 'MinMax scaling applied per feature so every channel lives in [0, 1] — preserves shape, kills magnitude bias.' },
  { id: 'window', title: 'Sequence Windowing', icon: 'LayoutGrid',
    desc: 'Sliding window of 30 days → predict day 31. Stride = 1. Yields 1,545 (X, y) pairs ready for sequence models.' },
  { id: 'split',  title: 'Train / Test Split', icon: 'SplitSquareHorizontal',
    desc: 'Chronological 80/20 split — no leakage from future into past. 1,236 train / 309 test sequences.' },
];

const ARCHITECTURES = {
  ANN: {
    layers: [
      { name: 'Input',   shape: '(150,)',     type: 'input' },
      { name: 'Dense',   shape: '128',        type: 'dense', act: 'ReLU' },
      { name: 'Dropout', shape: '0.2',        type: 'reg' },
      { name: 'Dense',   shape: '64',         type: 'dense', act: 'ReLU' },
      { name: 'Dense',   shape: '1',          type: 'dense', act: 'Linear' },
    ],
    activation: 'ReLU + Linear output',
    regularization: 'Dropout (0.2), Early Stopping (patience=8)',
  },
  RNN: {
    layers: [
      { name: 'Input',     shape: '(30, 5)',  type: 'input' },
      { name: 'SimpleRNN', shape: '64',       type: 'rnn', act: 'tanh' },
      { name: 'Dropout',   shape: '0.3',      type: 'reg' },
      { name: 'Dense',     shape: '32',       type: 'dense', act: 'ReLU' },
      { name: 'Dense',     shape: '1',        type: 'dense', act: 'Linear' },
    ],
    activation: 'tanh + ReLU + Linear',
    regularization: 'Dropout (0.3), Early Stopping (patience=10)',
  },
  LSTM: {
    layers: [
      { name: 'Input',   shape: '(30, 5)',    type: 'input' },
      { name: 'LSTM',    shape: '128',        type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.3',        type: 'reg' },
      { name: 'LSTM',    shape: '64',         type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '32',         type: 'dense', act: 'ReLU' },
      { name: 'Dense',   shape: '1',          type: 'dense', act: 'Linear' },
    ],
    activation: 'tanh, sigmoid, ReLU, Linear',
    regularization: 'Dropout (0.3), L2 (1e-4), Early Stopping (patience=12)',
  },
  GRU: {
    layers: [
      { name: 'Input',   shape: '(30, 5)',    type: 'input' },
      { name: 'GRU',     shape: '96',         type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dropout', shape: '0.3',        type: 'reg' },
      { name: 'GRU',     shape: '48',         type: 'rnn', act: 'tanh / sigmoid' },
      { name: 'Dense',   shape: '1',          type: 'dense', act: 'Linear' },
    ],
    activation: 'tanh, sigmoid, Linear',
    regularization: 'Dropout (0.3), Early Stopping (patience=10)',
  },
};

const TEAM = [
  // REPLACE-DATA: swap with real team info.
  { name: 'Member 1', role: 'Lead — Modeling & Training', initials: 'M1',
    contribution: 'Designed the LSTM and GRU architectures, ran the hyperparameter sweep, and wrote the training loop with early stopping.' },
];



// =============================================================================

// REUSABLE COMPONENTS

// =============================================================================

// Small reusable components: nav, icons, counters, charts, etc.

// =============================================================================
// Icon — adapter over lucide-react. Accepts kebab-case or PascalCase names.
// =============================================================================
const LUCIDE_MAP = {
  Sun, Moon, User, GraduationCap, Calendar, ArrowDown, Layers, Database,
  CheckCircle2, Target, Clock, Sparkles, Sliders, LayoutGrid,
  SplitSquareHorizontal, Table, ChevronRight, Trophy, Lightbulb, AlertTriangle,
  // kebab-case aliases used in JSX call sites
  'sun': Sun, 'moon': Moon, 'user': User, 'graduation-cap': GraduationCap,
  'calendar': Calendar, 'arrow-down': ArrowDown, 'layers': Layers,
  'database': Database, 'check-circle2': CheckCircle2, 'target': Target,
  'clock': Clock, 'sparkles': Sparkles, 'sliders': Sliders,
  'layout-grid': LayoutGrid, 'split-square-horizontal': SplitSquareHorizontal,
  'table': Table, 'chevron-right': ChevronRight, 'trophy': Trophy,
  'lightbulb': Lightbulb, 'alert-triangle': AlertTriangle,
};
function Icon({ name, size = 18, className = '', strokeWidth = 1.75, style }) {
  const Cmp = LUCIDE_MAP[name];
  if (!Cmp) return null;
  return <Cmp size={size} strokeWidth={strokeWidth} className={className} style={style} />;
}

// ────────────────────────────────────────────────────────────────────────
// Animated counter — counts from 0 to value when in viewport
// ────────────────────────────────────────────────────────────────────────
function Counter({ value, suffix = '', prefix = '', duration = 1400, decimals = 0 }) {
  const [n, setN] = useState(0);
  const ref = useRef(null);
  const triggered = useRef(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting && !triggered.current) {
        triggered.current = true;
        const start = performance.now();
        const tick = (t) => {
          const p = Math.min(1, (t - start) / duration);
          const eased = 1 - Math.pow(1 - p, 3);
          setN(value * eased);
          if (p < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      }
    }, { threshold: 0.4 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, [value, duration]);
  return <span ref={ref}>{prefix}{n.toFixed(decimals)}{suffix}</span>;
}

// ────────────────────────────────────────────────────────────────────────
// Reveal-on-scroll wrapper
// ────────────────────────────────────────────────────────────────────────
function Reveal({ children, delay = 0, className = '' }) {
  const ref = useRef(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const obs = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setVisible(true); obs.disconnect(); }
    }, { threshold: 0.15 });
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className={`reveal ${visible ? 'in' : ''} ${className}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Section header — small uppercase label + display heading + lede
// ────────────────────────────────────────────────────────────────────────
function SectionHeader({ kicker, title, subtitle, num }) {
  return (
    <div className="mb-10">
      <div className="flex items-center gap-3 mb-3 font-mono text-xs tracking-[0.18em] uppercase" style={{ color: 'var(--accent)' }}>
        {num && <span className="opacity-60">{num}</span>}
        <span className="h-px w-8" style={{ background: 'var(--accent)', opacity: 0.4 }} />
        <span>{kicker}</span>
      </div>
      <h2 className="font-display text-3xl md:text-5xl font-semibold leading-[1.05] tracking-tight" style={{ color: 'var(--text)' }}>
        {title}
      </h2>
      {subtitle && (
        <p className="mt-4 max-w-2xl text-base md:text-lg leading-relaxed" style={{ color: 'var(--text-muted)' }}>
          {subtitle}
        </p>
      )}
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Sticky top nav with section anchors
// ────────────────────────────────────────────────────────────────────────
function TopNav({ sections, theme, onToggleTheme }) {
  const [active, setActive] = useState(sections[0].id);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    const obs = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) setActive(e.target.id); });
    }, { rootMargin: '-40% 0px -55% 0px' });
    sections.forEach(s => {
      const el = document.getElementById(s.id);
      if (el) obs.observe(el);
    });
    return () => obs.disconnect();
  }, [sections]);

  return (
    <header
      className="fixed top-0 left-0 right-0 z-40 transition-all duration-300"
      style={{
        background: scrolled ? 'color-mix(in oklab, var(--bg) 80%, transparent)' : 'transparent',
        borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
        backdropFilter: scrolled ? 'blur(16px) saturate(140%)' : 'none',
        WebkitBackdropFilter: scrolled ? 'blur(16px) saturate(140%)' : 'none',
      }}
    >
      <div className="max-w-7xl mx-auto px-6 md:px-10 h-[72px] flex items-center gap-6">
        {/* Logo */}
        <a href="#hero" className="flex items-center gap-2.5 shrink-0">
          <div className="relative w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'var(--accent-glow)', border: '1px solid var(--accent)' }}>
            <div className="w-2 h-2 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
          </div>
          <div className="font-display font-semibold tracking-tight text-[15px]" style={{ color: 'var(--text)' }}>
            climate<span style={{ color: 'var(--accent)' }}>.forecast</span>
          </div>
        </a>

        {/* Section links — desktop */}
        <nav className="hidden lg:flex items-center gap-1 mx-auto no-scrollbar overflow-x-auto">
          {sections.map(s => (
            <a
              key={s.id}
              href={`#${s.id}`}
              className="relative px-3 py-2 text-[13px] font-medium tracking-tight transition-colors"
              style={{ color: active === s.id ? 'var(--text)' : 'var(--text-muted)' }}
            >
              {s.label}
              {active === s.id && (
                <span className="absolute left-3 right-3 -bottom-px h-px" style={{ background: 'var(--accent)' }} />
              )}
            </a>
          ))}
        </nav>

        {/* Spacer for medium screens */}
        <div className="lg:hidden flex-1" />

        {/* Theme toggle */}
        <button
          onClick={onToggleTheme}
          className="w-9 h-9 rounded-lg flex items-center justify-center transition-colors hover:opacity-80"
          style={{ border: '1px solid var(--border)', background: 'var(--bg-elev)', color: 'var(--text-muted)' }}
          aria-label="Toggle theme"
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} />
        </button>
      </div>
    </header>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Glass card — base for every panel
// ────────────────────────────────────────────────────────────────────────
function Card({ children, className = '', hoverable = true, padding = 'p-6', as: Tag = 'div', style, ...rest }) {
  return (
    <Tag
      className={`glass rounded-2xl ${padding} ${hoverable ? 'card-hover' : ''} ${className}`}
      style={style}
      {...rest}
    >
      {children}
    </Tag>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Stat badge — number + label, used in hero
// ────────────────────────────────────────────────────────────────────────
function StatBadge({ value, suffix = '', label, decimals = 0 }) {
  return (
    <div className="glass rounded-xl px-5 py-4 flex items-center gap-4">
      <div className="font-display text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>
        <Counter value={value} suffix={suffix} decimals={decimals} />
      </div>
      <div className="text-xs leading-tight uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
        {label}
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Hero waveform background — multiple time-series lines drifting
// ────────────────────────────────────────────────────────────────────────
function HeroWaveform() {
  // Generate 4 plausible weather-style series
  const paths = useMemo(() => {
    const W = 1600, H = 600;
    const make = (seed, amp, period, yBase, smoothness = 1) => {
      let s = seed;
      const r = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
      const pts = [];
      for (let x = 0; x <= W; x += 4) {
        const t = x / W;
        const noise = (r() - 0.5) * amp * 0.3 * smoothness;
        const wave = Math.sin(t * Math.PI * period) * amp + Math.sin(t * Math.PI * period * 2.3) * amp * 0.4;
        pts.push([x, yBase + wave + noise]);
      }
      // Smooth path with quadratic curves
      let d = `M${pts[0][0]},${pts[0][1].toFixed(1)}`;
      for (let i = 1; i < pts.length - 1; i++) {
        const [x0, y0] = pts[i];
        const [x1, y1] = pts[i + 1];
        const xc = (x0 + x1) / 2, yc = (y0 + y1) / 2;
        d += ` Q${x0.toFixed(1)},${y0.toFixed(1)} ${xc.toFixed(1)},${yc.toFixed(1)}`;
      }
      return d;
    };
    return [
      { d: make(7,  60, 6, 200, 0.6), opacity: 0.55, width: 1.2 },
      { d: make(23, 80, 4, 320, 0.8), opacity: 0.40, width: 1.4 },
      { d: make(41, 100, 3, 420, 1.0), opacity: 0.30, width: 1.6 },
      { d: make(67, 70, 8, 140, 0.5), opacity: 0.25, width: 1.0 },
    ];
  }, []);

  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
      {/* Radial accent glow */}
      <div
        className="absolute -top-32 left-1/2 -translate-x-1/2 w-[1100px] h-[700px] opacity-50"
        style={{ background: 'radial-gradient(ellipse at center, var(--accent-glow) 0%, transparent 60%)' }}
      />
      {/* Grid lines */}
      <svg className="absolute inset-0 w-full h-full opacity-[0.05]" preserveAspectRatio="none" viewBox="0 0 100 100">
        <defs>
          <pattern id="grid" width="6" height="6" patternUnits="userSpaceOnUse">
            <path d="M 6 0 L 0 0 0 6" fill="none" stroke="currentColor" strokeWidth="0.1" />
          </pattern>
        </defs>
        <rect width="100" height="100" fill="url(#grid)" />
      </svg>
      {/* Animated waveforms */}
      <svg className="hero-wave absolute inset-0 w-[120%] h-full -left-[10%]" viewBox="0 0 1600 600" preserveAspectRatio="none">
        <defs>
          <linearGradient id="wave-grad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0" />
            <stop offset="50%" stopColor="var(--accent)" stopOpacity="1" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
        </defs>
        {paths.map((p, i) => (
          <path
            key={i}
            d={p.d}
            stroke="url(#wave-grad)"
            strokeWidth={p.width}
            fill="none"
            opacity={p.opacity}
            strokeLinecap="round"
          />
        ))}
      </svg>
      {/* Bottom fade */}
      <div className="absolute bottom-0 left-0 right-0 h-48" style={{ background: 'linear-gradient(to bottom, transparent, var(--bg))' }} />
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// Custom tooltip for charts
// ────────────────────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label, labelFormatter }) {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="glass rounded-lg px-3 py-2 text-xs" style={{ border: '1px solid var(--border-strong)' }}>
      <div className="font-mono mb-1" style={{ color: 'var(--text-muted)' }}>
        {labelFormatter ? labelFormatter(label) : label}
      </div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full" style={{ background: p.color }} />
          <span style={{ color: 'var(--text)' }}>{p.name}</span>
          <span className="ml-auto font-mono font-medium" style={{ color: 'var(--text)' }}>
            {typeof p.value === 'number' ? p.value.toFixed(p.value < 1 ? 3 : 2) : p.value}
          </span>
        </div>
      ))}
    </div>
  );
}



// =============================================================================

// SECTION COMPONENTS

// =============================================================================

// Section components — composed in app.jsx



// ────────────────────────────────────────────────────────────────────────
// 1. HERO
// ────────────────────────────────────────────────────────────────────────
function HeroSection({ winnerModel }) {
  return (
    <section id="hero" className="relative min-h-screen flex items-center pt-24 pb-32 overflow-hidden">
      <HeroWaveform />
      <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-10 w-full">
        {/* Top meta strip */}
        <Reveal>
          <div className="flex items-center gap-3 mb-8 font-mono text-xs tracking-[0.2em] uppercase" style={{ color: 'var(--accent)' }}>
            <span className="inline-block w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
            <span>Final Term · Machine Learning · Spring 2026</span>
          </div>
        </Reveal>

        {/* Title */}
        <Reveal delay={120}>
          <h1 className="font-display font-semibold text-5xl md:text-7xl lg:text-8xl tracking-[-0.025em] leading-[0.95] max-w-5xl">
            <span style={{ color: 'var(--text)' }}>Time-Series</span>{' '}
            <span className="gradient-text">Forecasting</span>
            <br />
            <span style={{ color: 'var(--text)' }}>on Daily Climate</span>{' '}
            <span style={{ color: 'var(--text-muted)' }}>Data.</span>
          </h1>
        </Reveal>

        <Reveal delay={240}>
          <p className="mt-8 max-w-2xl text-base md:text-lg leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            A side-by-side study of four sequence models — <span style={{ color: 'var(--text)' }}>ANN, RNN, LSTM</span>, and <span style={{ color: 'var(--text)' }}>GRU</span> —
            trained to predict next-day mean temperature. We profile each on accuracy, error, and training cost,
            then dissect why one architecture pulls ahead.
          </p>
        </Reveal>

        {/* Author + course meta */}
        <Reveal delay={340}>
          <div className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-3 text-sm" style={{ color: 'var(--text-muted)' }}>
            <div className="flex items-center gap-2">
              <Icon name="user" size={14} />
              <span>{TEAM[0].name}</span>
            </div>
            <div className="flex items-center gap-2">
              <Icon name="graduation-cap" size={14} />
              <span>University of Central Asia</span>
            </div>
            <div className="flex items-center gap-2">
              <Icon name="calendar" size={14} />
              <span>May 2026</span>
            </div>
          </div>
        </Reveal>

        {/* Stat badges */}
        <Reveal delay={460}>
          <div className="mt-14 grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl">
            <StatBadge value={4} label="Models compared" />
            <StatBadge value={1576} label="Daily samples" />
            <StatBadge value={91} suffix="%" label={`Best accuracy · ${winnerModel}`} />
            <StatBadge value={2.11} decimals={2} label="Best RMSE (°C)" />
          </div>
        </Reveal>

        {/* Scroll cue */}
        <Reveal delay={620}>
          <a href="#dataset" className="mt-16 inline-flex items-center gap-2 text-xs font-mono tracking-[0.18em] uppercase hover:opacity-80 transition-opacity" style={{ color: 'var(--text-muted)' }}>
            <span>Scroll to explore</span>
            <Icon name="arrow-down" size={14} />
          </a>
        </Reveal>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 2. DATASET OVERVIEW
// ────────────────────────────────────────────────────────────────────────
function DatasetSection() {
  return (
    <section id="dataset" className="py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="01"
            kicker="Dataset"
            title="Daily Climate Time Series"
            subtitle="Four climate variables sampled daily in Delhi between 2013 and 2017. Five clean columns, no missing values, one continuous target."
          />
        </Reveal>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Stats grid */}
          <Reveal className="lg:col-span-2">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {DATASET_STATS.map((s, i) => (
                <Card key={i} padding="p-5">
                  <div className="flex items-start justify-between mb-4">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center" style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>
                      <Icon name={s.icon} size={16} />
                    </div>
                  </div>
                  <div className={`font-display text-2xl font-semibold leading-none mb-1.5 ${s.mono ? 'font-mono text-lg' : ''}`} style={{ color: 'var(--text)' }}>
                    {s.value}
                  </div>
                  <div className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{s.label}</div>
                </Card>
              ))}
            </div>
          </Reveal>

          {/* Feature type donut */}
          <Reveal delay={120}>
            <Card padding="p-6" className="h-full flex flex-col">
              <div className="text-xs uppercase tracking-wider mb-1 font-mono" style={{ color: 'var(--text-muted)' }}>Feature types</div>
              <div className="font-display text-lg font-semibold mb-4" style={{ color: 'var(--text)' }}>Distribution</div>
              <div className="flex-1 min-h-[180px]">
                <ResponsiveContainer width="100%" height={180}>
                  <PieChart>
                    <Pie data={FEATURE_TYPES} dataKey="value" innerRadius={48} outerRadius={72} paddingAngle={2} stroke="none" animationDuration={1100}>
                      {FEATURE_TYPES.map((e, i) => <Cell key={i} fill={e.color} />)}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-3 space-y-1.5">
                {FEATURE_TYPES.map(f => (
                  <div key={f.name} className="flex items-center gap-2 text-xs">
                    <span className="w-2 h-2 rounded-full" style={{ background: f.color }} />
                    <span style={{ color: 'var(--text-muted)' }}>{f.name}</span>
                    <span className="ml-auto font-mono font-medium" style={{ color: 'var(--text)' }}>{f.value}</span>
                  </div>
                ))}
              </div>
            </Card>
          </Reveal>
        </div>

        {/* Preview table */}
        <Reveal delay={200}>
          <Card padding="p-0" className="mt-6 overflow-hidden">
            <div className="flex items-center justify-between px-6 py-4" style={{ borderBottom: '1px solid var(--border)' }}>
              <div className="flex items-center gap-3">
                <Icon name="table" size={16} className="opacity-60" />
                <span className="font-display font-semibold text-sm" style={{ color: 'var(--text)' }}>Preview · first 5 rows</span>
              </div>
              <span className="font-mono text-xs" style={{ color: 'var(--text-dim)' }}>climate_train.csv</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                    <th className="text-left px-6 py-3 font-medium">date</th>
                    <th className="text-right px-6 py-3 font-medium">meantemp <span style={{ color: 'var(--accent)' }}>(target)</span></th>
                    <th className="text-right px-6 py-3 font-medium">humidity</th>
                    <th className="text-right px-6 py-3 font-medium">wind_speed</th>
                    <th className="text-right px-6 py-3 font-medium">meanpressure</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-sm" style={{ color: 'var(--text)' }}>
                  {PREVIEW_ROWS.map((row, i) => (
                    <tr key={i} style={{ borderTop: '1px solid var(--border)' }} className="hover:bg-white/[0.02] transition-colors">
                      <td className="px-6 py-3" style={{ color: 'var(--text-muted)' }}>{row.date}</td>
                      <td className="px-6 py-3 text-right font-medium" style={{ color: 'var(--accent)' }}>{row.meantemp.toFixed(2)}</td>
                      <td className="px-6 py-3 text-right">{row.humidity.toFixed(2)}</td>
                      <td className="px-6 py-3 text-right">{row.wind_speed.toFixed(2)}</td>
                      <td className="px-6 py-3 text-right">{row.meanpressure.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </Reveal>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 3. PREPROCESSING PIPELINE
// ────────────────────────────────────────────────────────────────────────
function PipelineSection() {
  const [active, setActive] = useState(0);
  return (
    <section id="pipeline" className="py-24 md:py-32" style={{ background: 'color-mix(in oklab, var(--bg-elev) 30%, var(--bg))' }}>
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="02"
            kicker="Preprocessing"
            title="From raw CSV to training-ready sequences"
            subtitle="Five deterministic steps. Click any card to read what we did and why."
          />
        </Reveal>

        {/* Flow diagram */}
        <Reveal delay={120}>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3 md:gap-2 relative">
            {PIPELINE_STEPS.map((step, i) => {
              const isActive = active === i;
              return (
                <div key={step.id} className="relative">
                  <button
                    onClick={() => setActive(i)}
                    className="w-full text-left group"
                    aria-pressed={isActive}
                  >
                    <Card
                      padding="p-5"
                      className={`h-full transition-all`}
                      style={{
                        borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                        background: isActive ? 'var(--accent-glow)' : 'var(--bg-card)',
                      }}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div
                          className="w-9 h-9 rounded-lg flex items-center justify-center transition-colors"
                          style={{
                            background: isActive ? 'var(--accent)' : 'var(--bg-elev)',
                            color: isActive ? 'var(--bg)' : 'var(--accent)',
                            border: '1px solid ' + (isActive ? 'var(--accent)' : 'var(--border)'),
                          }}
                        >
                          <Icon name={step.icon} size={16} />
                        </div>
                        <span className="font-mono text-xs" style={{ color: 'var(--text-dim)' }}>0{i + 1}</span>
                      </div>
                      <div className="font-display font-semibold text-[15px]" style={{ color: 'var(--text)' }}>{step.title}</div>
                    </Card>
                  </button>
                  {/* Connector arrow */}
                  {i < PIPELINE_STEPS.length - 1 && (
                    <div className="hidden md:flex absolute top-1/2 -right-3 -translate-y-1/2 z-10 w-6 h-6 items-center justify-center pointer-events-none">
                      <Icon name="chevron-right" size={14} className="opacity-40" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </Reveal>

        {/* Active step detail */}
        <Reveal delay={200}>
          <Card padding="p-8" className="mt-8">
            <div className="flex flex-col md:flex-row md:items-start gap-6">
              <div className="shrink-0">
                <div className="w-14 h-14 rounded-xl flex items-center justify-center" style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>
                  <Icon name={PIPELINE_STEPS[active].icon} size={24} />
                </div>
              </div>
              <div className="flex-1">
                <div className="font-mono text-xs uppercase tracking-[0.18em] mb-2" style={{ color: 'var(--accent)' }}>
                  Step 0{active + 1} of 0{PIPELINE_STEPS.length}
                </div>
                <h3 className="font-display font-semibold text-2xl md:text-3xl mb-3 tracking-tight" style={{ color: 'var(--text)' }}>
                  {PIPELINE_STEPS[active].title}
                </h3>
                <p className="text-base leading-relaxed max-w-3xl" style={{ color: 'var(--text-muted)' }}>
                  {PIPELINE_STEPS[active].desc}
                </p>
              </div>
            </div>
          </Card>
        </Reveal>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 4. MODEL ARCHITECTURES
// ────────────────────────────────────────────────────────────────────────
function ArchitectureSection() {
  return (
    <section id="architecture" className="py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="03"
            kicker="Architectures"
            title="Four sequence models, four hypotheses"
            subtitle="Each network sees the same 30-day window. Capacity scales with the model's ability to remember."
          />
        </Reveal>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {MODELS.map((m, idx) => {
            const arch = ARCHITECTURES[m.id];
            return (
              <Reveal key={m.id} delay={idx * 80}>
                <Card padding="p-6" className="h-full">
                  {/* Header */}
                  <div className="flex items-start justify-between mb-5">
                    <div>
                      <div className="flex items-center gap-3 mb-1">
                        <span className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                        <h3 className="font-display font-semibold text-2xl tracking-tight" style={{ color: 'var(--text)' }}>{m.name}</h3>
                        {m.best && (
                          <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full" style={{ background: 'var(--accent-glow)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
                            Best
                          </span>
                        )}
                      </div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{m.fullName}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Params</div>
                      <div className="font-mono text-sm font-medium" style={{ color: 'var(--text)' }}>{m.params}</div>
                    </div>
                  </div>

                  {/* Layer diagram */}
                  <div className="space-y-1.5 mb-5">
                    {arch.layers.map((l, i) => {
                      const colors = {
                        input: { bg: 'var(--bg-elev)', text: 'var(--text-muted)', accent: 'var(--text-dim)' },
                        dense: { bg: 'color-mix(in oklab, ' + m.color + ' 12%, transparent)', text: 'var(--text)', accent: m.color },
                        rnn:   { bg: 'color-mix(in oklab, ' + m.color + ' 22%, transparent)', text: 'var(--text)', accent: m.color },
                        reg:   { bg: 'var(--bg-elev)', text: 'var(--text-muted)', accent: 'var(--text-dim)' },
                      }[l.type];
                      return (
                        <div
                          key={i}
                          className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-mono"
                          style={{ background: colors.bg, border: '1px solid var(--border)' }}
                        >
                          <span className="w-1 h-4 rounded-full" style={{ background: colors.accent }} />
                          <span className="font-medium" style={{ color: colors.text }}>{l.name}</span>
                          <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>{l.shape}</span>
                          {l.act && <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--text-dim)', background: 'var(--bg)' }}>{l.act}</span>}
                        </div>
                      );
                    })}
                  </div>

                  {/* Footer details */}
                  <div className="grid grid-cols-2 gap-3 pt-4" style={{ borderTop: '1px solid var(--border)' }}>
                    <div>
                      <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>Activation</div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{arch.activation}</div>
                    </div>
                    <div>
                      <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>Regularization</div>
                      <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{arch.regularization}</div>
                    </div>
                  </div>
                </Card>
              </Reveal>
            );
          })}
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 5. TRAINING RESULTS
// ────────────────────────────────────────────────────────────────────────
function TrainingSection() {
  const [activeModel, setActiveModel] = useState('LSTM');
  const data = TRAINING_CURVES[activeModel];
  const model = MODELS.find(m => m.id === activeModel);

  return (
    <section id="training" className="py-24 md:py-32" style={{ background: 'color-mix(in oklab, var(--bg-elev) 30%, var(--bg))' }}>
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="04"
            kicker="Training"
            title="Loss & accuracy across 40 epochs"
            subtitle="Train vs validation, with early stopping. Switch models to see how each one converges."
          />
        </Reveal>

        {/* Tab toggle */}
        <Reveal delay={120}>
          <div className="inline-flex items-center gap-1 p-1 rounded-xl mb-6" style={{ background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
            {MODELS.map(m => {
              const isActive = activeModel === m.id;
              return (
                <button
                  key={m.id}
                  onClick={() => setActiveModel(m.id)}
                  className="relative px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2"
                  style={{
                    background: isActive ? 'var(--bg-card)' : 'transparent',
                    color: isActive ? 'var(--text)' : 'var(--text-muted)',
                    border: '1px solid ' + (isActive ? 'var(--border-strong)' : 'transparent'),
                  }}
                >
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: m.color }} />
                  {m.name}
                </button>
              );
            })}
          </div>
        </Reveal>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
          {/* Loss chart */}
          <Reveal delay={200}>
            <Card padding="p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Loss curve</div>
                  <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>MSE · {model.name}</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Final val loss</div>
                  <div className="font-mono text-sm font-medium" style={{ color: model.color }}>{data[data.length - 1].valLoss}</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="epoch" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} domain={[0, 'auto']} />
                  <Tooltip content={<ChartTooltip labelFormatter={v => `Epoch ${v}`} />} />
                  <Line type="monotone" dataKey="trainLoss" name="Train" stroke={model.color} strokeWidth={2} dot={false} animationDuration={1200} />
                  <Line type="monotone" dataKey="valLoss"   name="Validation" stroke={model.color} strokeWidth={1.5} strokeDasharray="4 3" dot={false} animationDuration={1400} />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Reveal>

          {/* Accuracy chart */}
          <Reveal delay={280}>
            <Card padding="p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Accuracy curve</div>
                  <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>R² score · {model.name}</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Final val R²</div>
                  <div className="font-mono text-sm font-medium" style={{ color: model.color }}>{data[data.length - 1].valAcc}</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <defs>
                    <linearGradient id={`acc-grad-${activeModel}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={model.color} stopOpacity={0.3} />
                      <stop offset="100%" stopColor={model.color} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="epoch" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} domain={[0, 1]} />
                  <Tooltip content={<ChartTooltip labelFormatter={v => `Epoch ${v}`} />} />
                  <Area type="monotone" dataKey="trainAcc" name="Train"      stroke={model.color} strokeWidth={2}   fill={`url(#acc-grad-${activeModel})`} animationDuration={1200} />
                  <Line type="monotone" dataKey="valAcc"   name="Validation" stroke={model.color} strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            </Card>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 6. COMPARATIVE ANALYSIS
// ────────────────────────────────────────────────────────────────────────
function ComparisonSection({ winnerId }) {
  const winner = MODELS.find(m => m.id === winnerId) || MODELS.find(m => m.best);
  const barData = MODELS.map(m => ({
    name: m.name, Accuracy: m.accuracy, F1: m.f1 * 100, color: m.color,
  }));

  return (
    <section id="comparison" className="py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="05"
            kicker="Comparative Analysis"
            title="Head-to-head, by the numbers"
            subtitle="Six metrics, one winner. Lower RMSE is better; everything else is higher-is-better."
          />
        </Reveal>

        {/* Winner card */}
        <Reveal delay={80}>
          <Card padding="p-0" className="mb-8 overflow-hidden relative" hoverable={false}>
            <div className="winner-shimmer absolute inset-0 pointer-events-none" />
            <div className="relative grid grid-cols-1 md:grid-cols-12 gap-6 p-8 items-center">
              <div className="md:col-span-5 flex items-center gap-5">
                <div className="w-16 h-16 rounded-2xl flex items-center justify-center shrink-0" style={{ background: `color-mix(in oklab, ${winner.color} 18%, transparent)`, border: `1px solid ${winner.color}` }}>
                  <Icon name="trophy" size={28} style={{ color: winner.color }} />
                </div>
                <div>
                  <div className="font-mono text-xs uppercase tracking-[0.2em] mb-1" style={{ color: winner.color }}>Best performer</div>
                  <div className="font-display text-3xl md:text-4xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>
                    {winner.name}
                  </div>
                  <div className="text-sm" style={{ color: 'var(--text-muted)' }}>{winner.fullName}</div>
                </div>
              </div>
              <div className="md:col-span-7 grid grid-cols-2 md:grid-cols-4 gap-4">
                {[
                  { label: 'Accuracy', value: winner.accuracy + '%' },
                  { label: 'F1', value: winner.f1.toFixed(2) },
                  { label: 'RMSE', value: winner.rmse.toFixed(2) },
                  { label: 'Train time', value: winner.trainingTime },
                ].map(s => (
                  <div key={s.label}>
                    <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>{s.label}</div>
                    <div className="font-display text-2xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>{s.value}</div>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </Reveal>

        {/* Table */}
        <Reveal delay={140}>
          <Card padding="p-0" className="mb-8 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                    <th className="text-left px-6 py-4 font-medium">Model</th>
                    <th className="text-right px-6 py-4 font-medium">Accuracy</th>
                    <th className="text-right px-6 py-4 font-medium">Precision</th>
                    <th className="text-right px-6 py-4 font-medium">Recall</th>
                    <th className="text-right px-6 py-4 font-medium">F1</th>
                    <th className="text-right px-6 py-4 font-medium">RMSE ↓</th>
                    <th className="text-right px-6 py-4 font-medium">Train time</th>
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map(m => {
                    const isWinner = m.id === winner.id;
                    return (
                      <tr
                        key={m.id}
                        style={{
                          borderTop: '1px solid var(--border)',
                          background: isWinner ? 'var(--accent-glow)' : 'transparent',
                        }}
                        className="hover:bg-white/[0.02] transition-colors"
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-3">
                            <span className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                            <span className="font-display font-semibold" style={{ color: 'var(--text)' }}>{m.name}</span>
                            {isWinner && <Icon name="trophy" size={12} style={{ color: 'var(--accent)' }} />}
                          </div>
                        </td>
                        <td className="px-6 py-4 text-right font-mono font-medium" style={{ color: isWinner ? 'var(--accent)' : 'var(--text)' }}>{m.accuracy}%</td>
                        <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text)' }}>{m.precision.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text)' }}>{m.recall.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text)' }}>{m.f1.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text)' }}>{m.rmse.toFixed(2)}</td>
                        <td className="px-6 py-4 text-right font-mono" style={{ color: 'var(--text-muted)' }}>{m.trainingTime}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </Reveal>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-5">
          {/* ROC */}
          <Reveal delay={200} className="lg:col-span-3">
            <Card padding="p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>ROC curve</div>
                  <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>True positive rate vs false positive rate</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <LineChart data={ROC_DATA.merged} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" axisLine={false} tickLine={false} type="number" domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                  <YAxis axisLine={false} tickLine={false} domain={[0, 1]} tickFormatter={v => v.toFixed(1)} />
                  <Tooltip content={<ChartTooltip labelFormatter={v => `FPR ${(+v).toFixed(2)}`} />} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-dim)" strokeDasharray="3 4" strokeWidth={1} />
                  {MODELS.map(m => (
                    <Line key={m.id} type="monotone" dataKey={m.id} name={`${m.name} (AUC ${ROC_DATA.aucMap[m.id]})`} stroke={m.color} strokeWidth={2} dot={false} animationDuration={1400} />
                  ))}
                  <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} iconType="circle" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </Reveal>

          {/* Bar chart */}
          <Reveal delay={280} className="lg:col-span-2">
            <Card padding="p-6">
              <div className="flex items-center justify-between mb-5">
                <div>
                  <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Key metrics</div>
                  <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>Accuracy & F1 (×100)</div>
                </div>
              </div>
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={barData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} />
                  <YAxis axisLine={false} tickLine={false} domain={[0, 100]} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="Accuracy" radius={[6, 6, 0, 0]} animationDuration={1100}>
                    {barData.map((d, i) => <Cell key={i} fill={d.color} />)}
                  </Bar>
                  <Bar dataKey="F1" radius={[6, 6, 0, 0]} animationDuration={1300}>
                    {barData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.45} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 7. DISCUSSION
// ────────────────────────────────────────────────────────────────────────
function DiscussionSection({ winnerId }) {
  const winner = MODELS.find(m => m.id === winnerId) || MODELS.find(m => m.best);
  return (
    <section id="discussion" className="py-24 md:py-32" style={{ background: 'color-mix(in oklab, var(--bg-elev) 30%, var(--bg))' }}>
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="06"
            kicker="Discussion"
            title="Why one network pulled ahead"
            subtitle="Some intuition for the gap between the four architectures, plus what we'd do differently next."
          />
        </Reveal>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Reveal delay={120}>
            <Card padding="p-8">
              <div className="flex items-center gap-2 mb-4 font-mono text-xs uppercase tracking-[0.18em]" style={{ color: 'var(--accent)' }}>
                <Icon name="lightbulb" size={14} />
                <span>Key insight</span>
              </div>
              <h3 className="font-display text-2xl font-semibold tracking-tight mb-5" style={{ color: 'var(--text)' }}>
                Why <span style={{ color: winner.color }}>{winner.name}</span> outperformed the rest
              </h3>
              <div className="space-y-4 text-[15px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                <p>
                  Daily climate data has long, low-frequency seasonal cycles layered on short-term weather noise.
                  The plain ANN treats each day as an independent feature row — it has no notion of order, so it can't
                  separate "yesterday's pressure" from "the pressure 21 days ago."
                </p>
                <p>
                  The simple RNN sees order but suffers vanishing gradients past ~10 timesteps. With a 30-day window,
                  it stops learning from the early portion of every sequence. <span style={{ color: 'var(--text)' }}>LSTM and GRU</span> both
                  carry a gated cell state that lets gradient signal flow through long horizons.
                </p>
                <p>
                  LSTM edges out GRU here because its separate forget/input gates give it finer control over how
                  seasonal context decays. The cost is real — LSTM took <span className="font-mono" style={{ color: 'var(--text)' }}>{winner.trainingTime}</span>,
                  about 30% longer than GRU. For a deployed service that retrains daily, GRU might still be the right pick.
                </p>
              </div>
            </Card>
          </Reveal>

          <Reveal delay={200}>
            <Card padding="p-8">
              <div className="flex items-center gap-2 mb-4 font-mono text-xs uppercase tracking-[0.18em]" style={{ color: 'var(--accent)' }}>
                <Icon name="alert-triangle" size={14} />
                <span>Limitations & next steps</span>
              </div>
              <h3 className="font-display text-2xl font-semibold tracking-tight mb-5" style={{ color: 'var(--text)' }}>
                What we'd improve given more time
              </h3>
              <ul className="space-y-4">
                {[
                  { t: 'Single-city scope', d: 'Delhi-only weather generalises poorly. A multi-city dataset with shared embeddings would test transfer.' },
                  { t: 'Univariate target', d: 'We predicted mean temperature alone. A multi-output head (temp + humidity + pressure) would cost little and reveal coupled dynamics.' },
                  { t: 'No transformer baseline', d: 'A small encoder-only transformer with positional embeddings is a natural next comparator — likely closes the gap on LSTM.' },
                  { t: 'Hyperparameter search', d: 'We hand-tuned. A Bayesian search over hidden size, dropout, and window length would tighten error bars.' },
                ].map((item, i) => (
                  <li key={i} className="flex items-start gap-3">
                    <div className="shrink-0 w-6 h-6 rounded-full flex items-center justify-center font-mono text-[10px] mt-0.5" style={{ background: 'var(--bg-elev)', color: 'var(--accent)', border: '1px solid var(--border)' }}>{i + 1}</div>
                    <div>
                      <div className="font-display font-semibold text-[15px] mb-0.5" style={{ color: 'var(--text)' }}>{item.t}</div>
                      <div className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{item.d}</div>
                    </div>
                  </li>
                ))}
              </ul>
            </Card>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// 8. TEAM
// ────────────────────────────────────────────────────────────────────────
function TeamSection() {
  return (
    <section id="team" className="py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader
            num="07"
            kicker="Credits"
            title="The team behind the project"
            subtitle="Roles and contributions for the final term submission."
          />
        </Reveal>
        <div className={`grid gap-5 ${TEAM.length === 1 ? 'max-w-xl' : TEAM.length === 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
          {TEAM.map((person, i) => (
            <Reveal key={i} delay={i * 80}>
              <Card padding="p-7" className="h-full">
                <div className="flex items-center gap-4 mb-5">
                  <div className="w-14 h-14 rounded-full flex items-center justify-center font-display font-semibold text-lg shrink-0" style={{ background: 'var(--accent-glow)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
                    {person.initials}
                  </div>
                  <div>
                    <div className="font-display font-semibold text-lg leading-tight" style={{ color: 'var(--text)' }}>{person.name}</div>
                    <div className="text-xs font-mono uppercase tracking-wider mt-1" style={{ color: 'var(--accent)' }}>{person.role}</div>
                  </div>
                </div>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{person.contribution}</p>
              </Card>
            </Reveal>
          ))}
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// FOOTER
// ────────────────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer className="py-12 px-6 md:px-10" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} />
          <span className="font-mono text-xs uppercase tracking-[0.18em]" style={{ color: 'var(--text-muted)' }}>
            climate.forecast · Final Term ML Project · 2026
          </span>
        </div>
        <div className="font-mono text-xs" style={{ color: 'var(--text-dim)' }}>University of Central Asia</div>
      </div>
    </footer>
  );
}



// =============================================================================

// APP

// =============================================================================

// App entry — assembles everything, owns theme + tweaks state.

const NAV_SECTIONS = [
  { id: 'hero',         label: 'Overview' },
  { id: 'dataset',      label: 'Dataset' },
  { id: 'pipeline',     label: 'Preprocessing' },
  { id: 'architecture', label: 'Architectures' },
  { id: 'training',     label: 'Training' },
  { id: 'comparison',   label: 'Comparison' },
  { id: 'discussion',   label: 'Discussion' },
  { id: 'team',         label: 'Team' },
];

const TWEAK_DEFAULTS = {
  "theme": "dark",
  "accent": "teal",
  "winnerModel": "LSTM",
  "showDiscussion": true,
  "showTeam": true,
  "density": "spacious"
};

const ACCENT_PRESETS = {
  teal:    { dark: { accent: '#22d3ee', accent2: '#38bdf8', glow: 'rgba(34, 211, 238, 0.18)' },
             light: { accent: '#0891b2', accent2: '#0284c7', glow: 'rgba(8, 145, 178, 0.12)' } },
  indigo:  { dark: { accent: '#818cf8', accent2: '#a78bfa', glow: 'rgba(129, 140, 248, 0.20)' },
             light: { accent: '#4f46e5', accent2: '#7c3aed', glow: 'rgba(79, 70, 229, 0.12)' } },
  emerald: { dark: { accent: '#34d399', accent2: '#10b981', glow: 'rgba(52, 211, 153, 0.18)' },
             light: { accent: '#059669', accent2: '#0d9488', glow: 'rgba(5, 150, 105, 0.12)' } },
  amber:   { dark: { accent: '#fbbf24', accent2: '#f59e0b', glow: 'rgba(251, 191, 36, 0.18)' },
             light: { accent: '#d97706', accent2: '#b45309', glow: 'rgba(217, 119, 6, 0.12)' } },
};

function applyAccent(accent, theme) {
  const preset = ACCENT_PRESETS[accent] || ACCENT_PRESETS.teal;
  const v = preset[theme] || preset.dark;
  document.documentElement.style.setProperty('--accent', v.accent);
  document.documentElement.style.setProperty('--accent-2', v.accent2);
  document.documentElement.style.setProperty('--accent-glow', v.glow);
}

function App() {
  // REPLACE-DATA: Tweaks host integration removed for handoff. State is local now.
  const [tweaks, setTweakState] = useState(TWEAK_DEFAULTS);
  const setTweak = (k, v) => {
    if (typeof k === 'object') setTweakState(prev => ({ ...prev, ...k }));
    else setTweakState(prev => ({ ...prev, [k]: v }));
  };
  const theme = tweaks.theme;

  // Apply theme class
  React.useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light');
    applyAccent(tweaks.accent, theme);
  }, [theme, tweaks.accent]);

  const toggleTheme = () => setTweak('theme', theme === 'dark' ? 'light' : 'dark');

  return (
    <>
      <TopNav sections={NAV_SECTIONS} theme={theme} onToggleTheme={toggleTheme} />

      <main>
        <HeroSection winnerModel={tweaks.winnerModel} />
        <DatasetSection />
        <PipelineSection />
        <ArchitectureSection />
        <TrainingSection />
        <ComparisonSection winnerId={tweaks.winnerModel} />
        {tweaks.showDiscussion && <DiscussionSection winnerId={tweaks.winnerModel} />}
        {tweaks.showTeam && <TeamSection />}
      </main>

      <Footer />

      {/* Tweaks panel */}
      <TweaksPanel title="Tweaks">
        <TweakSection title="Appearance">
          <TweakRadio
            label="Theme"
            value={tweaks.theme}
            onChange={v => setTweak('theme', v)}
            options={[{ value: 'dark', label: 'Dark' }, { value: 'light', label: 'Light' }]}
          />
          <TweakSelect
            label="Accent color"
            value={tweaks.accent}
            onChange={v => setTweak('accent', v)}
            options={[
              { value: 'teal',    label: 'Teal (default)' },
              { value: 'indigo',  label: 'Indigo' },
              { value: 'emerald', label: 'Emerald' },
              { value: 'amber',   label: 'Amber' },
            ]}
          />
        </TweakSection>

        <TweakSection title="Content">
          <TweakSelect
            label="Highlighted best model"
            value={tweaks.winnerModel}
            onChange={v => setTweak('winnerModel', v)}
            options={[
              { value: 'ANN',  label: 'ANN' },
              { value: 'RNN',  label: 'RNN' },
              { value: 'LSTM', label: 'LSTM (default)' },
              { value: 'GRU',  label: 'GRU' },
            ]}
          />
          <TweakToggle
            label="Show Discussion section"
            value={tweaks.showDiscussion}
            onChange={v => setTweak('showDiscussion', v)}
          />
          <TweakToggle
            label="Show Team section"
            value={tweaks.showTeam}
            onChange={v => setTweak('showTeam', v)}
          />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}




// =============================================================================
// Tweaks panel stubs — the in-tool tweaks host is removed for handoff.
// Re-add a settings UI of your own if you want runtime theming on deploy.
// =============================================================================
function TweaksPanel() { return null; }
function TweakSection() { return null; }
function TweakRadio() { return null; }
function TweakSelect() { return null; }
function TweakToggle() { return null; }

// =============================================================================
// Default export — mounts theme CSS once, renders App.
// =============================================================================
export default function MLDashboard() {
  useEffect(() => {
    if (document.getElementById('ml-dashboard-styles')) return;
    const style = document.createElement('style');
    style.id = 'ml-dashboard-styles';
    style.textContent = THEME_CSS;
    document.head.appendChild(style);
  }, []);
  return <App />;
}
