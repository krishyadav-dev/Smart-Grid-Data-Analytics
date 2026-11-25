import { Component, useEffect, useState, type ReactNode } from "react";
import { useApi, type Json } from "./api";
import { useMode, type Mode } from "./theme";
import Overview from "./pages/Overview";
import Faults from "./pages/Faults";
import Theft from "./pages/Theft";
import Thermal from "./pages/Thermal";
import PowerQuality from "./pages/PowerQuality";
import System from "./pages/System";

export type Ctx = { mode: Mode; config: Json; layout: Json; meta: Json };

const I = (d: string) => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
    <path d={d} />
  </svg>
);

const PAGES: { id: string; label: string; icon: ReactNode; render: (c: Ctx) => ReactNode }[] = [
  { id: "map", label: "Grid map", icon: I("M9 4 3 6v14l6-2 6 2 6-2V4l-6 2-6-2zM9 4v14M15 6v14"), render: (c) => <Overview {...c} /> },
  { id: "faults", label: "Faults", icon: I("M13 2 4 14h7l-1 8 9-12h-7l1-8z"), render: (c) => <Faults {...c} /> },
  { id: "theft", label: "Energy & theft", icon: I("M3 12h4l3-8 4 16 3-8h4"), render: (c) => <Theft {...c} /> },
  { id: "thermal", label: "Transformer thermal", icon: I("M14 14.8V4a2 2 0 1 0-4 0v10.8a4 4 0 1 0 4 0z"), render: (c) => <Thermal {...c} /> },
  { id: "pq", label: "Power quality", icon: I("M2 12c2.5-6 5-6 7.5 0s5 6 7.5 0 3.5-3 5-3"), render: (c) => <PowerQuality {...c} /> },
  { id: "system", label: "System", icon: I("M4 6h16M4 12h16M4 18h10"), render: (c) => <System {...c} /> },
];

function useHash(): string {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "map";
  const [h, setH] = useState(read);
  useEffect(() => {
    const on = () => setH(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return h;
}

class PageBoundary extends Component<{ children: ReactNode; page: string }, { error: string | null }> {
  state = { error: null as string | null };
  static getDerivedStateFromError(e: Error) {
    return { error: String(e?.message || e) };
  }
  componentDidUpdate(prev: { page: string }) {
    if (prev.page !== this.props.page && this.state.error) this.setState({ error: null });
  }
  render() {
    return this.state.error
      ? <div className="empty">This page failed to render: {this.state.error}. Other pages still work.</div>
      : this.props.children;
  }
}

export default function App() {
  const [mode, setMode] = useMode();
  const route = useHash();
  const config = useApi("/api/config");
  const layout = useApi("/api/geo/layout");
  const meta = useApi("/api/meta");
  const page = PAGES.find((x) => x.id === route) ?? PAGES[0];

  useEffect(() => {
    document.title = `${page.label} · Feeder Dashboard`;
  }, [page]);

  const err = config.error || layout.error || meta.error;
  const ready = config.data && layout.data && meta.data;

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <div className="brand-mark">{I("M13 2 4 14h7l-1 8 9-12h-7l1-8z")}</div>
          <div>
            <b>Feeder Dashboard</b>
            <small>{config.data?.feeder_id ?? "ieee13_11kv"} · report</small>
          </div>
        </div>
        <nav className="nav" aria-label="Pages">
          {PAGES.map((x) => (
            <a key={x.id} href={`#/${x.id}`} aria-current={x.id === page.id ? "page" : undefined}>
              {x.icon}
              {x.label}
            </a>
          ))}
        </nav>
        <div className="side-foot">
          <button className="btn" style={{ width: "100%" }} onClick={() => setMode(mode === "dark" ? "light" : "dark")}
            aria-label="Toggle colour theme">
            {mode === "dark" ? "☀ Light theme" : "☾ Dark theme"}
          </button>
          <p style={{ marginTop: 10 }}>
            Report mode: frozen Layer 3 results and the simulated 2024 year. Illustrative Delhi placement.
          </p>
        </div>
      </aside>
      <main className="main">
        {err && <div className="empty">The report API is not reachable ({err}). Start it with <code>python -m src.dashboard</code>.</div>}
        {!err && !ready && <div className="skeleton" style={{ height: 480 }} />}
        {ready && <PageBoundary page={page.id}>{page.render({ mode, config: config.data, layout: layout.data, meta: meta.data })}</PageBoundary>}
      </main>
    </div>
  );
}
