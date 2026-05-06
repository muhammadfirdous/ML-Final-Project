// App entry — assembles every section, owns theme + tweaks state.

const NAV_SECTIONS = [
  { id: 'overview',         label: 'Overview' },
  { id: 'task-speech',      label: 'Speech' },
  { id: 'task-sentiment',   label: 'Sentiment' },
  { id: 'task-timeseries',  label: 'Time Series' },
  { id: 'summary',          label: 'Summary' },
  { id: 'team',             label: 'Team' },
];

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "accent": "teal",
  "showSummary": true,
  "showTeam": true
}/*EDITMODE-END*/;

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
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const theme = tweaks.theme;

  React.useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light');
    applyAccent(tweaks.accent, theme);
  }, [theme, tweaks.accent]);

  const toggleTheme = () => setTweak('theme', theme === 'dark' ? 'light' : 'dark');

  return (
    <>
      <TopNav sections={NAV_SECTIONS} theme={theme} onToggleTheme={toggleTheme} />

      <main>
        <HeroSection />
        <ClassificationTaskSection task={TASK_SPEECH}    sectionId="task-speech" />
        <ClassificationTaskSection task={TASK_SENTIMENT} sectionId="task-sentiment" />
        <TimeSeriesTaskSection     task={TASK_TIMESERIES} />
        {tweaks.showSummary && <SummarySection />}
        {tweaks.showTeam && <TeamSection />}
      </main>

      <Footer />

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

        <TweakSection title="Sections">
          <TweakToggle label="Show Summary"  value={tweaks.showSummary} onChange={v => setTweak('showSummary', v)} />
          <TweakToggle label="Show Team"     value={tweaks.showTeam}    onChange={v => setTweak('showTeam', v)} />
        </TweakSection>
      </TweaksPanel>
    </>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
