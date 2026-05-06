import { useState, useEffect } from 'react';
import { TopNav } from './components.jsx';
import {
  HeroSection, ClassificationTaskSection, TimeSeriesTaskSection,
  SummarySection, TeamSection, Footer,
} from './sections.jsx';
import { DashboardProvider, useDashboard } from './data/DashboardContext.jsx';

const NAV_SECTIONS = [
  { id: 'overview',         label: 'Overview' },
  { id: 'task-speech',      label: 'Speech' },
  { id: 'task-sentiment',   label: 'Sentiment' },
  { id: 'task-timeseries',  label: 'Time Series' },
  { id: 'summary',          label: 'Summary' },
  { id: 'team',             label: 'Team' },
];

function LoadingScreen() {
  return (
    <div className="fixed inset-0 flex items-center justify-center" style={{ background: 'var(--bg)' }}>
      <div className="flex items-center gap-3" style={{ color: 'var(--text-muted)' }}>
        <span className="w-2 h-2 rounded-full pulse-dot" style={{ background: 'var(--accent)' }} />
        <span className="font-mono text-xs uppercase tracking-[0.2em]">Loading dashboard…</span>
      </div>
    </div>
  );
}

function SourceBadge({ source }) {
  if (source !== 'mock') return null;
  return (
    <div
      className="fixed bottom-4 left-4 z-50 glass rounded-lg px-3 py-2 text-xs font-mono"
      style={{ color: 'var(--text-muted)', border: '1px solid var(--border-strong)' }}
    >
      ⚠ Backend offline — showing cached data
    </div>
  );
}

function Dashboard() {
  const { data, source, loading } = useDashboard();
  const [theme, setTheme] = useState('dark');

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light');
  }, [theme]);

  if (loading || !data) return <LoadingScreen />;

  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  return (
    <>
      <TopNav sections={NAV_SECTIONS} theme={theme} onToggleTheme={toggleTheme} />

      <main>
        <HeroSection />
        <ClassificationTaskSection task={data.tasks.speech}    sectionId="task-speech" />
        <ClassificationTaskSection task={data.tasks.sentiment} sectionId="task-sentiment" />
        <TimeSeriesTaskSection     task={data.tasks.timeseries} />
        <SummarySection />
        <TeamSection />
      </main>

      <Footer />
      <SourceBadge source={source} />
    </>
  );
}

export default function App() {
  return (
    <DashboardProvider>
      <Dashboard />
    </DashboardProvider>
  );
}
