import { createContext, useContext, useEffect, useState } from 'react';
import { fetchDashboard } from './dashboardApi.js';

const DashboardContext = createContext(null);

export function DashboardProvider({ children }) {
  const [state, setState] = useState({
    data: null,
    source: null,         // 'api' | 'mock'
    loading: true,
    error: null,
  });

  useEffect(() => {
    const ctrl = new AbortController();
    fetchDashboard({ signal: ctrl.signal })
      .then(({ data, source, error }) => {
        setState({ data, source, loading: false, error });
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        setState({ data: null, source: null, loading: false, error: err });
      });
    return () => ctrl.abort();
  }, []);

  return (
    <DashboardContext.Provider value={state}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  const ctx = useContext(DashboardContext);
  if (!ctx) throw new Error('useDashboard must be used inside <DashboardProvider>');
  return ctx;
}
