// Report-mode API client (src/dashboard/app.py).  Responses are typed loosely:
// the server shapes them from the frozen results files and the pages read the
// fields they need.
import { useEffect, useState } from "react";

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Json = any;

const cache = new Map<string, Promise<Json>>();

export function get(path: string): Promise<Json> {
  let p = cache.get(path);
  if (!p) {
    p = fetch(path).then(async (r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`);
      return r.json();
    });
    p.catch(() => cache.delete(path));
    cache.set(path, p);
  }
  return p;
}

export type Load<T = Json> = { data?: T; error?: string; loading: boolean };

export function useApi<T = Json>(path: string | null): Load<T> {
  const [state, setState] = useState<Load<T>>({ loading: !!path });
  useEffect(() => {
    if (!path) {
      setState({ loading: false });
      return;
    }
    let live = true;
    setState((s) => ({ data: s.data, loading: true }));
    get(path)
      .then((d) => live && setState({ data: d as T, loading: false }))
      .catch((e) => live && setState({ error: String(e.message || e), loading: false }));
    return () => {
      live = false;
    };
  }, [path]);
  return state;
}

export const fmt = {
  n(v: number | null | undefined, d = 1): string {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return v.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d });
  },
  pct(v: number | null | undefined, d = 1): string {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return `${(v * 100).toFixed(d)} %`;
  },
  int(v: number | null | undefined): string {
    if (v === null || v === undefined || !isFinite(v)) return "–";
    return Math.round(v).toLocaleString("en-IN");
  },
};

/** Proportions come either as a float or as {k, n, p, lo, hi}. */
export function p(x: Json): number | null {
  if (x === null || x === undefined) return null;
  if (typeof x === "number") return x;
  return typeof x.p === "number" ? x.p : null;
}

export function dayLabel(d: string): string {
  const t = new Date(d + "T00:00:00");
  return t.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** Category x-axis over ISO days with one label per month. */
export function monthAxis(days: string[]) {
  return {
    type: "category",
    data: days,
    axisLabel: { interval: 0, formatter: (d: string) => (d.endsWith("-01") ? MONTHS[Number(d.slice(5, 7)) - 1] : "") },
  };
}
