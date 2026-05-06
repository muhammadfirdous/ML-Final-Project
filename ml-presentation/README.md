# ML Final Term — Group Presentation

A static, single-page presentation site comparing ANN, RNN, LSTM, and GRU across speech emotion (TESS), tweet sentiment (Twitter Entity), and weather forecasting (CORGIS, Daily Climate, Seattle Weather).

## Run locally

```bash
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

## Where to drop in real model results

All placeholder numbers, training curves, ROC/AUC inputs, confusion matrices, dataset stats, and team info live in:

```
src/data/mockData.js
```

Search the file for `// TODO` markers to find every block that needs to be swapped after training. The exported objects (`TASK_SPEECH`, `TASK_SENTIMENT`, `TASK_TIMESERIES`, `TEAM`, `HEATMAP_DATA`, `SUMMARY_ROWS`, etc.) are consumed directly by the section components — no other file needs editing to update results.

## Tech stack

- **Vite** — build tool and dev server
- **React 19** — UI
- **Tailwind CSS v4** (via `@tailwindcss/vite`) — styling
- **Recharts** — every chart
- **lucide-react** — icons (loaded dynamically by name)
- **Framer Motion** — installed and available for future animations

## Project structure

```
src/
├── App.jsx              composes the page (nav, sections, footer)
├── main.jsx             React entry
├── index.css            Tailwind import + theme tokens + custom styles
├── Icon.jsx             lucide-react wrapper that accepts PascalCase names
├── components.jsx       TopNav, Card, Counter, Reveal, hero waveform, etc.
├── sections.jsx         Hero, classification task, time-series task, summary, team, footer
└── data/
    └── mockData.js      ALL placeholder data — edit this when results land
```
