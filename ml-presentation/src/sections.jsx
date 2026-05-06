import { useState, useEffect } from 'react';
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, BarChart, Bar,
  Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ReferenceLine,
} from 'recharts';
import { Icon } from './Icon.jsx';
import {
  Card, Reveal, SectionHeader, StatBadge, HeroWaveform, ChartTooltip,
} from './components.jsx';
import { useDashboard } from './data/DashboardContext.jsx';

// Pull the globals (ALGO_COLORS, ALGO_FULL, TEAM, etc.) once from context so
// every renderer below can keep its existing closed-over names.
function useGlobals() {
  const { data } = useDashboard();
  return {
    ALGO_COLORS: data.algoColors,
    ALGO_FULL: data.algoFull,
    TEAM: data.team,
    PROJECT_MAP: data.projectMap,
    HERO_STATS: data.heroStats,
    SUMMARY_ROWS: data.summaryRows,
    HEATMAP_DATA: data.heatmapData,
  };
}

// ────────────────────────────────────────────────────────────────────────
// Shared bits
// ────────────────────────────────────────────────────────────────────────
function MetricCell({ label, value, color }) {
  return (
    <div>
      <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>{label}</div>
      <div className="font-display text-2xl font-semibold tracking-tight" style={{ color: color || 'var(--text)' }}>{value}</div>
    </div>
  );
}

function LayerStack({ layers, color }) {
  const layerStyle = (l) => ({
    input: { bg: 'var(--bg-elev)', text: 'var(--text-muted)', accent: 'var(--text-dim)' },
    dense: { bg: `color-mix(in oklab, ${color} 12%, transparent)`, text: 'var(--text)', accent: color },
    rnn:   { bg: `color-mix(in oklab, ${color} 22%, transparent)`, text: 'var(--text)', accent: color },
    reg:   { bg: 'var(--bg-elev)', text: 'var(--text-muted)', accent: 'var(--text-dim)' },
  })[l.type];
  return (
    <div className="space-y-1.5">
      {layers.map((l, i) => {
        const c = layerStyle(l);
        return (
          <div key={i} className="flex items-center gap-3 px-3 py-2 rounded-lg text-xs font-mono"
               style={{ background: c.bg, border: '1px solid var(--border)' }}>
            <span className="w-1 h-4 rounded-full" style={{ background: c.accent }} />
            <span className="font-medium" style={{ color: c.text }}>{l.name}</span>
            <span className="ml-auto" style={{ color: 'var(--text-muted)' }}>{l.shape}</span>
            {l.act && <span className="px-1.5 py-0.5 rounded" style={{ color: 'var(--text-dim)', background: 'var(--bg)' }}>{l.act}</span>}
          </div>
        );
      })}
    </div>
  );
}

function AlgoCard({ algo, layers, meta, metrics, primaryMetric = 'accuracy', primaryFormat = (v) => v + '%' }) {
  const { ALGO_COLORS, ALGO_FULL } = useGlobals();
  const color = ALGO_COLORS[algo];
  const isBest = metrics.best;
  const primaryValue = primaryFormat(metrics[primaryMetric]);
  return (
    <Card padding="p-6" className="h-full">
      <div className="flex items-start justify-between mb-5">
        <div>
          <div className="flex items-center gap-3 mb-1">
            <span className="w-2.5 h-2.5 rounded-full" style={{ background: color }} />
            <h3 className="font-display font-semibold text-2xl tracking-tight" style={{ color: 'var(--text)' }}>{algo}</h3>
            {isBest && (
              <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-0.5 rounded-full"
                    style={{ background: `color-mix(in oklab, ${color} 18%, transparent)`, color, border: `1px solid ${color}` }}>
                Best
              </span>
            )}
          </div>
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{ALGO_FULL[algo]}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>{primaryMetric}</div>
          <div className="font-mono text-base font-semibold" style={{ color }}>{primaryValue}</div>
        </div>
      </div>
      <LayerStack layers={layers} color={color} />
      <div className="grid grid-cols-3 gap-3 pt-4 mt-5" style={{ borderTop: '1px solid var(--border)' }}>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>Params</div>
          <div className="font-mono text-xs" style={{ color: 'var(--text)' }}>{metrics.params}</div>
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>Train time</div>
          <div className="font-mono text-xs" style={{ color: 'var(--text)' }}>{metrics.trainingTime}</div>
        </div>
        <div>
          <div className="font-mono text-[10px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-dim)' }}>Activation</div>
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>{meta.activation}</div>
        </div>
      </div>
    </Card>
  );
}

// Static class map so Tailwind JIT picks up these utility classes.
const PIPELINE_COLS = {
  3: 'md:grid-cols-3',
  4: 'md:grid-cols-4',
  5: 'md:grid-cols-5',
  6: 'md:grid-cols-6',
};

function PipelineStrip({ steps }) {
  const [active, setActive] = useState(0);
  const Active = steps[active];
  const colsClass = PIPELINE_COLS[steps.length] || 'md:grid-cols-5';
  return (
    <>
      <div className={`grid grid-cols-1 ${colsClass} gap-3 md:gap-2 relative`}>
        {steps.map((step, i) => {
          const isActive = active === i;
          return (
            <div key={step.id} className="relative">
              <button onClick={() => setActive(i)} className="w-full text-left">
                <Card padding="p-5"
                  style={{
                    borderColor: isActive ? 'var(--accent)' : 'var(--border)',
                    background: isActive ? 'var(--accent-glow)' : 'var(--bg-card)',
                  }}
                  className="h-full">
                  <div className="flex items-center justify-between mb-4">
                    <div className="w-9 h-9 rounded-lg flex items-center justify-center"
                      style={{
                        background: isActive ? 'var(--accent)' : 'var(--bg-elev)',
                        color: isActive ? 'var(--bg)' : 'var(--accent)',
                        border: '1px solid ' + (isActive ? 'var(--accent)' : 'var(--border)'),
                      }}>
                      <Icon name={step.icon} size={16} />
                    </div>
                    <span className="font-mono text-xs" style={{ color: 'var(--text-dim)' }}>0{i + 1}</span>
                  </div>
                  <div className="font-display font-semibold text-[15px]" style={{ color: 'var(--text)' }}>{step.title}</div>
                </Card>
              </button>
              {i < steps.length - 1 && (
                <div className="hidden md:flex absolute top-1/2 -right-3 -translate-y-1/2 z-10 w-6 h-6 items-center justify-center pointer-events-none">
                  <Icon name="chevron-right" size={14} className="opacity-40" />
                </div>
              )}
            </div>
          );
        })}
      </div>
      <Card padding="p-7" className="mt-6">
        <div className="flex flex-col md:flex-row md:items-start gap-6">
          <div className="shrink-0">
            <div className="w-14 h-14 rounded-xl flex items-center justify-center"
                 style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>
              <Icon name={Active.icon} size={24} />
            </div>
          </div>
          <div className="flex-1">
            <div className="font-mono text-xs uppercase tracking-[0.18em] mb-2" style={{ color: 'var(--accent)' }}>
              Step 0{active + 1} of 0{steps.length}
            </div>
            <h3 className="font-display font-semibold text-2xl mb-3 tracking-tight" style={{ color: 'var(--text)' }}>{Active.title}</h3>
            <p className="text-base leading-relaxed max-w-3xl" style={{ color: 'var(--text-muted)' }}>{Active.desc}</p>
          </div>
        </div>
      </Card>
    </>
  );
}

function StatsGrid({ stats, cols = 'md:grid-cols-3' }) {
  return (
    <div className={`grid grid-cols-2 ${cols} gap-3`}>
      {stats.map((s, i) => (
        <Card key={i} padding="p-5">
          <div className="flex items-start justify-between mb-4">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center"
                 style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>
              <Icon name={s.icon} size={16} />
            </div>
          </div>
          <div className={`font-display font-semibold leading-none mb-1.5 ${s.mono ? 'font-mono text-base' : 'text-2xl'}`} style={{ color: 'var(--text)' }}>{s.value}</div>
          <div className="text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>{s.label}</div>
        </Card>
      ))}
    </div>
  );
}

function ModelToggle({ algos, active, onChange }) {
  const { ALGO_COLORS } = useGlobals();
  return (
    <div className="inline-flex items-center gap-1 p-1 rounded-xl" style={{ background: 'var(--bg-elev)', border: '1px solid var(--border)' }}>
      {algos.map(a => {
        const isActive = active === a;
        return (
          <button key={a} onClick={() => onChange(a)}
            className="relative px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2"
            style={{
              background: isActive ? 'var(--bg-card)' : 'transparent',
              color: isActive ? 'var(--text)' : 'var(--text-muted)',
              border: '1px solid ' + (isActive ? 'var(--border-strong)' : 'transparent'),
            }}>
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: ALGO_COLORS[a] }} />
            {a}
          </button>
        );
      })}
    </div>
  );
}

function TaskBanner({ task }) {
  return (
    <div className="relative overflow-hidden rounded-2xl mb-12 glass" style={{ borderColor: 'var(--border-strong)' }}>
      <div className="absolute -top-16 -right-16 w-64 h-64 opacity-50 pointer-events-none"
           style={{ background: 'radial-gradient(circle, var(--accent-glow) 0%, transparent 60%)' }} />
      <div className="relative px-7 py-8 md:px-10 md:py-10 flex flex-col md:flex-row md:items-center gap-6">
        <div className="w-16 h-16 rounded-2xl flex items-center justify-center shrink-0"
             style={{ background: 'var(--accent-glow)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
          <Icon name={task.iconName} size={28} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-2 font-mono text-xs tracking-[0.2em] uppercase" style={{ color: 'var(--accent)' }}>
            <span>{task.num}</span>
            <span className="h-px w-6" style={{ background: 'var(--accent)', opacity: 0.4 }} />
            <span>{task.kicker}</span>
          </div>
          <h2 className="font-display text-3xl md:text-4xl font-semibold tracking-tight mb-2" style={{ color: 'var(--text)' }}>{task.title}</h2>
          <p className="text-sm md:text-base leading-relaxed max-w-3xl" style={{ color: 'var(--text-muted)' }}>{task.subtitle}</p>
        </div>
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────
// HERO
// ────────────────────────────────────────────────────────────────────────
export function HeroSection() {
  const { TEAM, PROJECT_MAP, HERO_STATS, ALGO_COLORS } = useGlobals();
  return (
    <section id="overview" className="relative min-h-screen flex items-center pt-24 pb-32 overflow-hidden">
      <HeroWaveform />
      <div className="relative z-10 max-w-7xl mx-auto px-6 md:px-10 w-full">
        <Reveal>
          <div className="flex items-center gap-3 mb-8 font-mono text-xs tracking-[0.2em] uppercase" style={{ color: 'var(--accent)' }}>
            <span className="inline-block w-1.5 h-1.5 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
            <span>Final Term · Machine Learning · Spring 2026</span>
          </div>
        </Reveal>
        <Reveal delay={120}>
          <h1 className="font-display font-semibold text-5xl md:text-7xl lg:text-8xl tracking-[-0.025em] leading-[0.95] max-w-5xl">
            <span style={{ color: 'var(--text)' }}>Comparing</span>{' '}
            <span className="gradient-text">Deep Learning</span>
            <br />
            <span style={{ color: 'var(--text)' }}>across speech, text,</span>{' '}
            <span style={{ color: 'var(--text-muted)' }}>and time.</span>
          </h1>
        </Reveal>
        <Reveal delay={240}>
          <p className="mt-8 max-w-2xl text-base md:text-lg leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            Three task categories, five datasets, four algorithm families — <span style={{ color: 'var(--text)' }}>ANN, RNN, LSTM, GRU</span> —
            benchmarked on emotion recognition, tweet sentiment, and weather forecasting. Same metrics, same colors, every comparison side-by-side.
          </p>
        </Reveal>
        <Reveal delay={340}>
          <div className="mt-10 flex flex-wrap items-center gap-x-8 gap-y-3 text-sm" style={{ color: 'var(--text-muted)' }}>
            <div className="flex items-center gap-2"><Icon name="user" size={14} /><span>{TEAM[0].name}</span></div>
            <div className="flex items-center gap-2"><Icon name="graduation-cap" size={14} /><span>University of Central Asia</span></div>
            <div className="flex items-center gap-2"><Icon name="calendar" size={14} /><span>May 2026</span></div>
          </div>
        </Reveal>

        <Reveal delay={460}>
          <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-3 max-w-3xl">
            {HERO_STATS.map((s, i) => <StatBadge key={i} {...s} />)}
          </div>
        </Reveal>

        <Reveal delay={580}>
          <div className="mt-16">
            <div className="font-mono text-xs uppercase tracking-[0.2em] mb-5" style={{ color: 'var(--text-muted)' }}>Project map</div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {PROJECT_MAP.map((p, idx) => (
                <Card key={p.task} padding="p-6">
                  <div className="flex items-center gap-3 mb-5">
                    <div className="w-10 h-10 rounded-lg flex items-center justify-center"
                         style={{ background: 'var(--accent-glow)', color: 'var(--accent)' }}>
                      <Icon name={p.icon} size={18} />
                    </div>
                    <div>
                      <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Task 0{idx + 1}</div>
                      <div className="font-display font-semibold text-base" style={{ color: 'var(--text)' }}>{p.task}</div>
                    </div>
                  </div>
                  <div className="text-xs uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>Datasets</div>
                  <div className="flex flex-wrap gap-1.5 mb-4">
                    {p.datasets.map(d => (
                      <span key={d} className="px-2 py-1 rounded-md text-xs font-mono"
                            style={{ background: 'var(--bg-elev)', color: 'var(--text)', border: '1px solid var(--border)' }}>{d}</span>
                    ))}
                  </div>
                  <div className="text-xs uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>Algorithms</div>
                  <div className="flex flex-wrap gap-1.5">
                    {p.algos.map(a => (
                      <span key={a} className="px-2 py-0.5 rounded-md text-xs font-mono font-medium flex items-center gap-1.5"
                            style={{ background: `color-mix(in oklab, ${ALGO_COLORS[a]} 15%, transparent)`, color: ALGO_COLORS[a], border: `1px solid color-mix(in oklab, ${ALGO_COLORS[a]} 40%, transparent)` }}>
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: ALGO_COLORS[a] }} />{a}
                      </span>
                    ))}
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

function ConfusionMatrix({ matrix, classes, color }) {
  const max = Math.max(...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs" style={{ borderCollapse: 'separate', borderSpacing: 2 }}>
        <thead>
          <tr>
            <th></th>
            {classes.map(c => (
              <th key={c} className="font-mono font-medium px-2 py-1 text-center"
                  style={{ color: 'var(--text-muted)', writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="font-mono px-2 py-1 text-right" style={{ color: 'var(--text-muted)' }}>{classes[i]}</td>
              {row.map((v, j) => {
                const intensity = v / max;
                const isDiag = i === j;
                const bg = isDiag
                  ? `color-mix(in oklab, ${color} ${10 + intensity * 70}%, transparent)`
                  : `color-mix(in oklab, ${color} ${intensity * 30}%, transparent)`;
                return (
                  <td key={j} className="font-mono w-9 h-9 text-center rounded"
                      style={{ background: bg, color: intensity > 0.5 ? 'var(--text)' : 'var(--text-muted)' }}>
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LossChart({ data, color, title, finalLoss }) {
  return (
    <Card padding="p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Loss curve</div>
          <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>{title}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Final val loss</div>
          <div className="font-mono text-sm font-medium" style={{ color }}>{finalLoss}</div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="epoch" axisLine={false} tickLine={false} />
          <YAxis axisLine={false} tickLine={false} domain={[0, 'auto']} />
          <Tooltip content={<ChartTooltip labelFormatter={v => `Epoch ${v}`} />} />
          <Line type="monotone" dataKey="trainLoss" name="Train"      stroke={color} strokeWidth={2}   dot={false} animationDuration={1100} />
          <Line type="monotone" dataKey="valLoss"   name="Validation" stroke={color} strokeWidth={1.5} strokeDasharray="4 3" dot={false} animationDuration={1300} />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}

function AccChart({ data, color, title, finalAcc }) {
  const id = 'acc-' + title.replace(/\W+/g, '');
  return (
    <Card padding="p-6">
      <div className="flex items-center justify-between mb-5">
        <div>
          <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Accuracy curve</div>
          <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>{title}</div>
        </div>
        <div className="text-right">
          <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>Final val acc</div>
          <div className="font-mono text-sm font-medium" style={{ color }}>{finalAcc}</div>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={color} stopOpacity={0.3} />
              <stop offset="100%" stopColor={color} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="epoch" axisLine={false} tickLine={false} />
          <YAxis axisLine={false} tickLine={false} domain={[0, 1]} />
          <Tooltip content={<ChartTooltip labelFormatter={v => `Epoch ${v}`} />} />
          <Area type="monotone" dataKey="trainAcc" name="Train" stroke={color} strokeWidth={2} fill={`url(#${id})`} />
          <Line type="monotone" dataKey="valAcc"  name="Validation" stroke={color} strokeWidth={1.5} strokeDasharray="4 3" dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </Card>
  );
}

function ROCChart({ roc, algos }) {
  const { ALGO_COLORS } = useGlobals();
  return (
    <Card padding="p-6">
      <div className="mb-5">
        <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>ROC curve</div>
        <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>True positive rate vs false positive rate</div>
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={roc.merged} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="fpr" type="number" domain={[0, 1]} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(1)} />
          <YAxis domain={[0, 1]} axisLine={false} tickLine={false} tickFormatter={v => v.toFixed(1)} />
          <Tooltip content={<ChartTooltip labelFormatter={v => `FPR ${(+v).toFixed(2)}`} />} />
          <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-dim)" strokeDasharray="3 4" />
          {algos.map(a => (
            <Line key={a} type="monotone" dataKey={a} name={`${a} (AUC ${roc.aucMap[a]})`} stroke={ALGO_COLORS[a]} strokeWidth={2} dot={false} animationDuration={1300} />
          ))}
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} iconType="circle" />
        </LineChart>
      </ResponsiveContainer>
    </Card>
  );
}

function ResultsTable({ algos, metrics, columns, bestId }) {
  const { ALGO_COLORS } = useGlobals();
  return (
    <Card padding="p-0" className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              <th className="text-left px-6 py-4 font-medium">Model</th>
              {columns.map(c => (
                <th key={c.key} className="text-right px-6 py-4 font-medium">{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {algos.map(a => {
              const m = metrics[a];
              const isBest = bestId ? a === bestId : m.best;
              return (
                <tr key={a} style={{ borderTop: '1px solid var(--border)', background: isBest ? 'var(--accent-glow)' : 'transparent' }}>
                  <td className="px-6 py-4">
                    <div className="flex items-center gap-3">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ background: ALGO_COLORS[a] }} />
                      <span className="font-display font-semibold" style={{ color: 'var(--text)' }}>{a}</span>
                      {isBest && <Icon name="trophy" size={12} style={{ color: 'var(--accent)' }} />}
                    </div>
                  </td>
                  {columns.map(c => (
                    <td key={c.key} className="px-6 py-4 text-right font-mono" style={{ color: isBest && c.key === 'accuracy' ? 'var(--accent)' : 'var(--text)' }}>
                      {c.format ? c.format(m[c.key]) : m[c.key]}
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function WinnerBanner({ algo, metrics, primaryLabel, primaryValue }) {
  const { ALGO_COLORS, ALGO_FULL } = useGlobals();
  const color = ALGO_COLORS[algo];
  return (
    <Card padding="p-0" hoverable={false} className="overflow-hidden relative mb-6">
      <div className="winner-shimmer absolute inset-0 pointer-events-none" />
      <div className="relative grid grid-cols-1 md:grid-cols-12 gap-6 p-7 items-center">
        <div className="md:col-span-5 flex items-center gap-5">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center shrink-0"
               style={{ background: `color-mix(in oklab, ${color} 18%, transparent)`, border: `1px solid ${color}` }}>
            <Icon name="trophy" size={24} style={{ color }} />
          </div>
          <div>
            <div className="font-mono text-xs uppercase tracking-[0.2em] mb-1" style={{ color }}>Best on this dataset</div>
            <div className="font-display text-3xl font-semibold tracking-tight" style={{ color: 'var(--text)' }}>{algo}</div>
            <div className="text-sm" style={{ color: 'var(--text-muted)' }}>{ALGO_FULL[algo]}</div>
          </div>
        </div>
        <div className="md:col-span-7 grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCell label={primaryLabel} value={primaryValue} color={color} />
          {metrics.f1 != null && <MetricCell label="F1" value={metrics.f1.toFixed(2)} />}
          {metrics.auc != null && <MetricCell label="AUC" value={metrics.auc.toFixed(2)} />}
          {metrics.rmse != null && metrics.rmse > 0 && <MetricCell label="RMSE" value={metrics.rmse.toFixed(2)} />}
          {metrics.mae != null && <MetricCell label="MAE" value={metrics.mae.toFixed(2)} />}
          <MetricCell label="Train time" value={metrics.trainingTime} />
        </div>
      </div>
    </Card>
  );
}

// ────────────────────────────────────────────────────────────────────────
// CLASSIFICATION TASK SECTION (Speech + Sentiment)
// ────────────────────────────────────────────────────────────────────────
export function ClassificationTaskSection({ task, sectionId }) {
  const { ALGO_COLORS } = useGlobals();
  const [activeModel, setActiveModel] = useState(task.algos.find(a => task.metrics[a].best) || task.algos[0]);
  const winnerId = task.algos.find(a => task.metrics[a].best);
  const winner = task.metrics[winnerId];
  const data = task.curves[activeModel];
  const color = ALGO_COLORS[activeModel];

  const accBarData = task.algos.map(a => ({ name: a, Accuracy: task.metrics[a].accuracy, F1: task.metrics[a].f1 * 100, color: ALGO_COLORS[a] }));

  return (
    <section id={sectionId} className="py-24 md:py-28">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal><TaskBanner task={task} /></Reveal>

        <Reveal>
          <SectionHeader kicker="Dataset" title={task.dataset.name} subtitle={`Source · ${task.dataset.source}`} />
        </Reveal>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-16">
          <Reveal className="lg:col-span-2">
            <StatsGrid stats={task.dataset.stats} />
          </Reveal>
          <Reveal delay={120}>
            <Card padding="p-6" className="h-full">
              <div className="text-xs uppercase tracking-wider mb-1 font-mono" style={{ color: 'var(--text-muted)' }}>Class distribution</div>
              <div className="font-display text-lg font-semibold mb-4" style={{ color: 'var(--text)' }}>{task.dataset.target}</div>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={task.dataset.classDist} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="name" axisLine={false} tickLine={false} fontSize={10} interval={0} />
                  <YAxis axisLine={false} tickLine={false} />
                  <Tooltip content={<ChartTooltip />} />
                  <Bar dataKey="value" radius={[6, 6, 0, 0]} fill="var(--accent)" animationDuration={1000} />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </Reveal>
        </div>

        <Reveal>
          <SectionHeader kicker="Preprocessing" title="From raw input to model-ready tensors" subtitle="Click any step to read its details." />
        </Reveal>
        <Reveal delay={120}><PipelineStrip steps={task.pipeline} /></Reveal>

        <div className="mt-20">
          <Reveal>
            <SectionHeader kicker="Architectures" title="One card per model"
              subtitle={`Each network sees identical input. Capacity scales from ${task.algos[0]} → ${task.algos[task.algos.length - 1]}.`} />
          </Reveal>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {task.algos.map((a, idx) => (
              <Reveal key={a} delay={idx * 80}>
                <AlgoCard algo={a} layers={task.arch[a]} meta={task.archMeta[a]} metrics={task.metrics[a]} primaryMetric="accuracy" primaryFormat={v => v + '%'} />
              </Reveal>
            ))}
          </div>
        </div>

        <div className="mt-20">
          <Reveal>
            <SectionHeader kicker="Training" title="Loss & accuracy across epochs" subtitle="Switch models to see how each one converges." />
          </Reveal>
          <Reveal delay={120}>
            <ModelToggle algos={task.algos} active={activeModel} onChange={setActiveModel} />
          </Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-6">
            <Reveal delay={200}><LossChart data={data} color={color} title={`MSE · ${activeModel}`} finalLoss={data[data.length - 1].valLoss} /></Reveal>
            <Reveal delay={280}><AccChart  data={data} color={color} title={`Accuracy · ${activeModel}`} finalAcc={data[data.length - 1].valAcc} /></Reveal>
          </div>
        </div>

        {task.confusion && (
          <div className="mt-12">
            <Reveal>
              <Card padding="p-6">
                <div className="flex items-center justify-between mb-5">
                  <div>
                    <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Confusion matrix</div>
                    <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>Predictions vs ground truth · {activeModel}</div>
                  </div>
                </div>
                <ConfusionMatrix matrix={task.confusion[activeModel]} classes={task.classes} color={color} />
                <div className="mt-3 text-xs" style={{ color: 'var(--text-dim)' }}>Rows = actual, columns = predicted. Diagonal saturated by accuracy.</div>
              </Card>
            </Reveal>
          </div>
        )}

        <div className="mt-20">
          <Reveal>
            <SectionHeader kicker="Comparison" title="Head-to-head on this dataset" subtitle="Six metrics, one winner." />
          </Reveal>
          <Reveal delay={80}>
            <WinnerBanner algo={winnerId} metrics={winner} primaryLabel="Accuracy" primaryValue={winner.accuracy + '%'} />
          </Reveal>
          <Reveal delay={140}>
            <ResultsTable
              algos={task.algos}
              metrics={task.metrics}
              bestId={winnerId}
              columns={[
                { key: 'accuracy',     label: 'Accuracy',  format: v => v + '%' },
                { key: 'precision',    label: 'Precision', format: v => v.toFixed(2) },
                { key: 'recall',       label: 'Recall',    format: v => v.toFixed(2) },
                { key: 'f1',           label: 'F1',        format: v => v.toFixed(2) },
                { key: 'auc',          label: 'AUC',       format: v => v.toFixed(2) },
                { key: 'trainingTime', label: 'Train time' },
              ]}
            />
          </Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 mt-6">
            <Reveal delay={200} className="lg:col-span-3"><ROCChart roc={task.roc} algos={task.algos} /></Reveal>
            <Reveal delay={280} className="lg:col-span-2">
              <Card padding="p-6">
                <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Key metrics</div>
                <div className="font-display font-semibold text-lg mb-4" style={{ color: 'var(--text)' }}>Accuracy & F1 (×100)</div>
                <ResponsiveContainer width="100%" height={280}>
                  <BarChart data={accBarData} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" axisLine={false} tickLine={false} />
                    <YAxis axisLine={false} tickLine={false} domain={[0, 100]} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="Accuracy" radius={[6, 6, 0, 0]} animationDuration={900}>
                      {accBarData.map((d, i) => <Cell key={i} fill={d.color} />)}
                    </Bar>
                    <Bar dataKey="F1" radius={[6, 6, 0, 0]} animationDuration={1100}>
                      {accBarData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.45} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Reveal>
          </div>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// TIME SERIES TASK — multi-dataset, tabbed
// ────────────────────────────────────────────────────────────────────────
export function TimeSeriesTaskSection({ task }) {
  const { ALGO_COLORS } = useGlobals();
  const [activeDs, setActiveDs] = useState(task.datasets[0].id);
  const ds = task.datasets.find(d => d.id === activeDs);
  const [activeModel, setActiveModel] = useState(ds.bestId);
  useEffect(() => { setActiveModel(ds.bestId); }, [activeDs, ds.bestId]);

  const winnerId = ds.bestId;
  const winner = ds.metrics[winnerId];
  const color = ALGO_COLORS[activeModel];
  const isClassify = ds.classify;

  const cols = isClassify
    ? [
        { key: 'accuracy',     label: 'Accuracy',  format: v => v + '%' },
        { key: 'precision',    label: 'Precision', format: v => v.toFixed(2) },
        { key: 'recall',       label: 'Recall',    format: v => v.toFixed(2) },
        { key: 'f1',           label: 'F1',        format: v => v.toFixed(2) },
        { key: 'trainingTime', label: 'Train time' },
      ]
    : [
        { key: 'rmse',         label: 'RMSE ↓',   format: v => v.toFixed(2) },
        { key: 'mae',          label: 'MAE ↓',    format: v => v.toFixed(2) },
        { key: 'mape',         label: 'MAPE ↓',   format: v => v.toFixed(1) + '%' },
        { key: 'accuracy',     label: 'R² Acc',   format: v => v + '%' },
        { key: 'params',       label: 'Params' },
        { key: 'trainingTime', label: 'Train time' },
      ];

  return (
    <section id="task-timeseries" className="py-24 md:py-28">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal><TaskBanner task={task} /></Reveal>

        <Reveal>
          <div className="mb-8">
            <div className="font-mono text-xs uppercase tracking-[0.2em] mb-3" style={{ color: 'var(--text-muted)' }}>Dataset</div>
            <div className="flex flex-wrap gap-2">
              {task.datasets.map(d => {
                const isActive = d.id === activeDs;
                return (
                  <button key={d.id} onClick={() => setActiveDs(d.id)}
                    className="text-left px-5 py-3 rounded-xl transition-all"
                    style={{
                      background: isActive ? 'var(--accent-glow)' : 'var(--bg-elev)',
                      border: '1px solid ' + (isActive ? 'var(--accent)' : 'var(--border)'),
                      color: isActive ? 'var(--text)' : 'var(--text-muted)',
                    }}>
                    <div className="font-display font-semibold text-base">{d.name}</div>
                    <div className="font-mono text-[11px] mt-0.5" style={{ color: 'var(--text-dim)' }}>{d.target}</div>
                  </button>
                );
              })}
            </div>
          </div>
        </Reveal>

        <Reveal key={'ov-' + activeDs}>
          <SectionHeader kicker="Overview" title={ds.name} subtitle={`Source · ${ds.source}`} />
        </Reveal>
        <Reveal delay={80} key={'stats-' + activeDs}>
          <StatsGrid stats={ds.stats} />
        </Reveal>

        <Reveal delay={140} key={'plot-' + activeDs}>
          <Card padding="p-6" className="mt-6">
            <div className="flex items-center justify-between mb-5">
              <div>
                <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Target variable</div>
                <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>{ds.target} · raw signal (test window)</div>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={ds.predVsActual[winnerId]} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                <defs>
                  <linearGradient id={`raw-${ds.id}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.25} />
                    <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="day" axisLine={false} tickLine={false} />
                <YAxis axisLine={false} tickLine={false} />
                <Tooltip content={<ChartTooltip labelFormatter={v => `Day ${v}`} />} />
                <Area type="monotone" dataKey="actual" name="Actual" stroke="var(--accent)" strokeWidth={2} fill={`url(#raw-${ds.id})`} />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Reveal>

        <div className="mt-16">
          <Reveal><SectionHeader kicker="Preprocessing" title="From raw CSV to training-ready sequences" /></Reveal>
          <Reveal delay={120}><PipelineStrip steps={ds.pipeline} /></Reveal>
        </div>

        <div className="mt-20">
          <Reveal><SectionHeader kicker="Architectures" title="Four sequence models, four hypotheses" /></Reveal>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            {ds.algos.map((a, idx) => (
              <Reveal key={a} delay={idx * 70}>
                <AlgoCard
                  algo={a} layers={ds.arch[a]} meta={ds.archMeta[a]}
                  metrics={{ ...ds.metrics[a], best: a === winnerId }}
                  primaryMetric={isClassify ? 'accuracy' : 'rmse'}
                  primaryFormat={isClassify ? (v => v + '%') : (v => v.toFixed(2))}
                />
              </Reveal>
            ))}
          </div>
        </div>

        <div className="mt-20">
          <Reveal><SectionHeader kicker="Training" title="Convergence per model" /></Reveal>
          <Reveal delay={100}><ModelToggle algos={ds.algos} active={activeModel} onChange={setActiveModel} /></Reveal>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mt-6">
            <Reveal delay={200} key={'lc-' + activeDs + activeModel}>
              <LossChart data={ds.curves[activeModel]} color={color} title={`${isClassify ? 'Cross-entropy' : 'MSE'} · ${activeModel}`} finalLoss={ds.curves[activeModel][ds.curves[activeModel].length - 1].valLoss} />
            </Reveal>
            <Reveal delay={280} key={'pa-' + activeDs + activeModel}>
              <Card padding="p-6">
                <div className="flex items-center justify-between mb-5">
                  <div>
                    <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Predicted vs actual</div>
                    <div className="font-display font-semibold text-lg" style={{ color: 'var(--text)' }}>{activeModel} · 60-day test window</div>
                  </div>
                  {!isClassify && (
                    <div className="text-right">
                      <div className="font-mono text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-dim)' }}>RMSE</div>
                      <div className="font-mono text-sm font-medium" style={{ color }}>{ds.metrics[activeModel].rmse.toFixed(2)}</div>
                    </div>
                  )}
                </div>
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={ds.predVsActual[activeModel]} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="day" axisLine={false} tickLine={false} />
                    <YAxis axisLine={false} tickLine={false} />
                    <Tooltip content={<ChartTooltip labelFormatter={v => `Day ${v}`} />} />
                    <Line type="monotone" dataKey="actual"    name="Actual"    stroke="var(--text-muted)" strokeWidth={1.5} dot={false} />
                    <Line type="monotone" dataKey="predicted" name="Predicted" stroke={color} strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Reveal>
          </div>
        </div>

        <div className="mt-20">
          <Reveal><SectionHeader kicker="Comparison" title="Head-to-head on this dataset" /></Reveal>
          <Reveal delay={80} key={'wb-' + activeDs}>
            <WinnerBanner
              algo={winnerId} metrics={winner}
              primaryLabel={isClassify ? 'Accuracy' : 'RMSE'}
              primaryValue={isClassify ? winner.accuracy + '%' : winner.rmse.toFixed(2)}
            />
          </Reveal>
          <Reveal delay={140} key={'rt-' + activeDs}>
            <ResultsTable algos={ds.algos} metrics={ds.metrics} bestId={winnerId} columns={cols} />
          </Reveal>
        </div>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// CROSS-TASK SUMMARY
// ────────────────────────────────────────────────────────────────────────
export function SummarySection() {
  const { ALGO_COLORS, SUMMARY_ROWS, HEATMAP_DATA } = useGlobals();
  return (
    <section id="summary" className="py-24 md:py-32" style={{ background: 'color-mix(in oklab, var(--bg-elev) 30%, var(--bg))' }}>
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader num="04" kicker="Cross-task summary" title="Which architecture won where"
            subtitle="The full picture across all five datasets, side-by-side. LSTM dominates — but the gap to GRU is small." />
        </Reveal>

        <Reveal delay={80}>
          <Card padding="p-0" className="overflow-hidden mb-10">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="font-mono text-xs uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
                    <th className="text-left px-6 py-4 font-medium">Task</th>
                    <th className="text-left px-6 py-4 font-medium">Dataset</th>
                    <th className="text-left px-6 py-4 font-medium">Best model</th>
                    <th className="text-left px-6 py-4 font-medium">Metric</th>
                    <th className="text-right px-6 py-4 font-medium">Score</th>
                  </tr>
                </thead>
                <tbody>
                  {SUMMARY_ROWS.map((r, i) => (
                    <tr key={i} style={{ borderTop: '1px solid var(--border)' }}>
                      <td className="px-6 py-4 font-display font-semibold" style={{ color: 'var(--text)' }}>{r.task}</td>
                      <td className="px-6 py-4 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.dataset}</td>
                      <td className="px-6 py-4">
                        <span className="inline-flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-mono font-semibold"
                              style={{ background: `color-mix(in oklab, ${ALGO_COLORS[r.best]} 18%, transparent)`, color: ALGO_COLORS[r.best], border: `1px solid color-mix(in oklab, ${ALGO_COLORS[r.best]} 40%, transparent)` }}>
                          <span className="w-1.5 h-1.5 rounded-full" style={{ background: ALGO_COLORS[r.best] }} />
                          {r.best}
                        </span>
                      </td>
                      <td className="px-6 py-4 font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{r.metric}</td>
                      <td className="px-6 py-4 text-right font-mono font-semibold" style={{ color: 'var(--accent)' }}>{r.score}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </Reveal>

        <Reveal delay={140}>
          <Card padding="p-7">
            <div className="font-mono text-xs uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>Performance heatmap</div>
            <div className="font-display font-semibold text-lg mb-6" style={{ color: 'var(--text)' }}>Primary metric per dataset (higher = darker)</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm" style={{ borderCollapse: 'separate', borderSpacing: 4 }}>
                <thead>
                  <tr>
                    <th className="text-left font-mono text-xs uppercase tracking-wider px-3 py-2" style={{ color: 'var(--text-muted)' }}>Dataset</th>
                    {['ANN', 'RNN', 'LSTM', 'GRU'].map(a => (
                      <th key={a} className="text-center font-mono text-xs uppercase tracking-wider px-3 py-2 font-semibold" style={{ color: ALGO_COLORS[a] }}>
                        <div className="flex items-center justify-center gap-2">
                          <span className="w-2 h-2 rounded-full" style={{ background: ALGO_COLORS[a] }} />{a}
                        </div>
                      </th>
                    ))}
                    <th className="text-left font-mono text-xs uppercase tracking-wider px-3 py-2" style={{ color: 'var(--text-muted)' }}>Metric</th>
                  </tr>
                </thead>
                <tbody>
                  {HEATMAP_DATA.map(row => {
                    const vals = ['ANN', 'RNN', 'LSTM', 'GRU'].map(a => row[a]).filter(v => v != null);
                    const min = Math.min(...vals), max = Math.max(...vals);
                    const winner = ['ANN', 'RNN', 'LSTM', 'GRU'].reduce((b, a) => (row[a] != null && (row[b] == null || row[a] > row[b])) ? a : b, 'ANN');
                    return (
                      <tr key={row.dataset}>
                        <td className="px-3 py-2 font-display font-semibold text-sm" style={{ color: 'var(--text)' }}>{row.dataset}</td>
                        {['ANN', 'RNN', 'LSTM', 'GRU'].map(a => {
                          const v = row[a];
                          if (v == null) return (
                            <td key={a} className="text-center px-3 py-3 rounded font-mono text-xs"
                                style={{ background: 'var(--bg-elev)', color: 'var(--text-dim)' }}>—</td>
                          );
                          const norm = (v - min) / Math.max(0.0001, max - min);
                          const isWin = a === winner;
                          return (
                            <td key={a} className="text-center px-3 py-3 rounded font-mono font-semibold"
                                style={{
                                  background: `color-mix(in oklab, ${ALGO_COLORS[a]} ${15 + norm * 65}%, transparent)`,
                                  color: norm > 0.4 ? 'var(--text)' : 'var(--text-muted)',
                                  border: isWin ? `1px solid ${ALGO_COLORS[a]}` : '1px solid transparent',
                                }}>
                              {v}{row.metric.includes('Acc') ? '%' : ''}
                              {isWin && <span className="ml-1.5 opacity-70">★</span>}
                            </td>
                          );
                        })}
                        <td className="px-3 py-2 font-mono text-xs" style={{ color: 'var(--text-dim)' }}>{row.metric}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="mt-4 text-xs" style={{ color: 'var(--text-dim)' }}>★ = best algorithm for that dataset. *R² accuracy used for regression rows.</div>
          </Card>
        </Reveal>

        <Reveal delay={220}>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mt-10">
            <Card padding="p-7">
              <div className="flex items-center gap-2 mb-3 font-mono text-xs uppercase tracking-[0.18em]" style={{ color: ALGO_COLORS.LSTM }}>
                <Icon name="trophy" size={14} /><span>When LSTM wins</span>
              </div>
              <h3 className="font-display text-lg font-semibold mb-3" style={{ color: 'var(--text)' }}>Long-context, complex signal</h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                LSTM took the top spot on every dataset. Separate forget/input gates give it the finest control over how
                far back to look — most important on TESS (174-frame audio sequences) and Daily Climate (30-day windows).
                The cost: ~30% more training time than GRU.
              </p>
            </Card>
            <Card padding="p-7">
              <div className="flex items-center gap-2 mb-3 font-mono text-xs uppercase tracking-[0.18em]" style={{ color: ALGO_COLORS.GRU }}>
                <Icon name="zap" size={14} /><span>When GRU is enough</span>
              </div>
              <h3 className="font-display text-lg font-semibold mb-3" style={{ color: 'var(--text)' }}>Fewer parameters, almost the same score</h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                GRU sat 1–2 percentage points behind LSTM on every dataset while training noticeably faster. For a
                deployed service that retrains daily — or any project on a single GPU — GRU is the pragmatic pick.
              </p>
            </Card>
            <Card padding="p-7">
              <div className="flex items-center gap-2 mb-3 font-mono text-xs uppercase tracking-[0.18em]" style={{ color: ALGO_COLORS.ANN }}>
                <Icon name="cpu" size={14} /><span>When ANN suffices</span>
              </div>
              <h3 className="font-display text-lg font-semibold mb-3" style={{ color: 'var(--text)' }}>Tabular regression, short horizons</h3>
              <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
                A plain ANN can&apos;t sequence-model, but on flat tabular regression (CORGIS, Daily Climate, Seattle) it still
                hit 74–81% R². If your features already encode time (lags, rolling stats), an MLP is the cheapest baseline
                worth running.
              </p>
            </Card>
          </div>
        </Reveal>
      </div>
    </section>
  );
}

// ────────────────────────────────────────────────────────────────────────
// TEAM
// ────────────────────────────────────────────────────────────────────────
export function TeamSection() {
  const { TEAM } = useGlobals();
  return (
    <section id="team" className="py-24 md:py-32">
      <div className="max-w-7xl mx-auto px-6 md:px-10">
        <Reveal>
          <SectionHeader num="05" kicker="Credits" title="The team behind the project" subtitle="Per-member responsibilities and which datasets / algorithms they touched." />
        </Reveal>
        <div className={`grid gap-5 ${TEAM.length === 1 ? 'max-w-2xl' : TEAM.length === 2 ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
          {TEAM.map((person, i) => (
            <Reveal key={i} delay={i * 80}>
              <Card padding="p-7">
                <div className="flex items-center gap-4 mb-5">
                  <div className="w-14 h-14 rounded-full flex items-center justify-center font-display font-semibold text-lg shrink-0"
                       style={{ background: 'var(--accent-glow)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
                    {person.initials}
                  </div>
                  <div>
                    <div className="font-display font-semibold text-lg leading-tight" style={{ color: 'var(--text)' }}>{person.name}</div>
                    <div className="text-xs font-mono uppercase tracking-wider mt-1" style={{ color: 'var(--accent)' }}>{person.role}</div>
                  </div>
                </div>
                <p className="text-sm leading-relaxed mb-5" style={{ color: 'var(--text-muted)' }}>{person.contribution}</p>
                {person.worked && (
                  <div>
                    <div className="font-mono text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-dim)' }}>Worked on</div>
                    <div className="flex flex-wrap gap-1.5">
                      {person.worked.map((w, k) => (
                        <span key={k} className="px-2 py-1 rounded-md text-xs font-mono"
                              style={{ background: 'var(--bg-elev)', color: 'var(--text-muted)', border: '1px solid var(--border)' }}>{w}</span>
                      ))}
                    </div>
                  </div>
                )}
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
export function Footer() {
  return (
    <footer className="py-12 px-6 md:px-10" style={{ borderTop: '1px solid var(--border)' }}>
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} />
          <span className="font-mono text-xs uppercase tracking-[0.18em]" style={{ color: 'var(--text-muted)' }}>
            ML Final Term · 3 tasks · 5 datasets · 2026
          </span>
        </div>
        <div className="font-mono text-xs" style={{ color: 'var(--text-dim)' }}>University of Central Asia</div>
      </div>
    </footer>
  );
}
