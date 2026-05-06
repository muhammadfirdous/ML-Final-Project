// Small reusable components: nav, icons, counters, charts, etc.

const { useState, useEffect, useRef, useMemo } = React;
const {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ReferenceLine,
} = Recharts;

// ────────────────────────────────────────────────────────────────────────
// Lucide icon — renders an SVG by name with consistent sizing
// ────────────────────────────────────────────────────────────────────────
// Lucide UMD exposes icons as PascalCase keys (e.g. "Database", "ChevronRight")
const KEBAB_TO_PASCAL = (s) => s.split(/[-_\s]/).map(p => p ? p[0].toUpperCase() + p.slice(1) : '').join('');
function Icon({ name, size = 18, className = '', strokeWidth = 1.75, style: extraStyle }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current || !window.lucide || !window.lucide.icons) return;
    const icons = window.lucide.icons;
    const key = icons[name] ? name
      : icons[KEBAB_TO_PASCAL(name)] ? KEBAB_TO_PASCAL(name)
      : null;
    const iconData = key ? icons[key] : null;
    if (iconData) {
      ref.current.innerHTML = '';
      const svg = window.lucide.createElement(iconData);
      svg.setAttribute('width', size);
      svg.setAttribute('height', size);
      svg.setAttribute('stroke-width', strokeWidth);
      ref.current.appendChild(svg);
    } else {
      ref.current.innerHTML = '';
    }
  }, [name, size, strokeWidth]);
  return <span ref={ref} className={`inline-flex items-center justify-center ${className}`} style={{ width: size, height: size, ...extraStyle }} />;
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

Object.assign(window, {
  Icon, Counter, Reveal, SectionHeader, TopNav, Card, StatBadge,
  HeroWaveform, ChartTooltip,
});
