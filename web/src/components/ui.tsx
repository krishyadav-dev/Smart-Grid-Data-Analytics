import type { ReactNode } from "react";
import type { Json, Load } from "../api";

export function Card({ title, sub, right, children, className = "" }: {
  title: string; sub?: ReactNode; right?: ReactNode; children: ReactNode; className?: string;
}) {
  return (
    <section className={`card ${className}`}>
      <div className="card-head">
        <div>
          <h2>{title}</h2>
          {sub && <p>{sub}</p>}
        </div>
        {right}
      </div>
      {children}
    </section>
  );
}

export function Kpi({ label, value, unit, sub }: { label: string; value: ReactNode; unit?: string; sub?: ReactNode }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className="value">
        {value}
        {unit && <small>{unit}</small>}
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export type Status = "good" | "warning" | "serious" | "critical" | "neutral";
const ICON: Record<Status, string> = { good: "✓", warning: "!", serious: "!", critical: "✕", neutral: "•" };

/** Status = colour + icon + label, never colour alone. */
export function Pill({ status, children }: { status: Status; children: ReactNode }) {
  return (
    <span className={`pill ${status}`}>
      <span className="dot" aria-hidden />
      <span aria-hidden>{ICON[status]}</span>
      {children}
    </span>
  );
}

export function Unavailable({ d }: { d: Json }) {
  return (
    <div className="empty">
      Not available in this checkout{d?.file ? <> (<code>{d.file}</code> missing)</> : null}.
      {d?.how && (
        <>
          {" "}Generate it with <code>{d.how}</code>.
        </>
      )}
    </div>
  );
}

/** Loading / error / unavailable gate around a view that needs one API response. */
export function Gate({ s, children, h = 120 }: { s: Load; children: (d: Json) => ReactNode; h?: number }) {
  if (s.error) return <div className="empty">Could not load: {s.error}</div>;
  if (!s.data) return <div className="skeleton" style={{ height: h }} />;
  if (s.data.available === false) return <Unavailable d={s.data} />;
  return <>{children(s.data)}</>;
}

export function Seg<T extends string>({ value, options, onChange, label }: {
  value: T; options: { v: T; label: string }[]; onChange: (v: T) => void; label: string;
}) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.v} aria-pressed={o.v === value} onClick={() => onChange(o.v)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function PageHead({ title, sub, right }: { title: string; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="topbar">
      <div>
        <h1>{title}</h1>
        {sub && <p>{sub}</p>}
      </div>
      {right}
    </div>
  );
}
