// Dataviz reference palette (validated; see the dataviz skill's palette.md).
// Light and dark are each selected steps of the same hues, not a flip.
import { useEffect, useState } from "react";

export type Mode = "light" | "dark";

export const SERIES: Record<Mode, string[]> = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
};

// Sequential: one hue (blue), light -> dark.  Bins for maps and heatmaps.
export const BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
// Second sequential context (orange), used when blue is already on screen as magnitude.
export const ORANGE_RAMP = ["#fde0d2", "#f8b89c", "#f29066", "#eb6834", "#c4501f", "#9a3c14", "#6e2a0c"];

export const STATUS = { good: "#0ca30c", warning: "#fab219", serious: "#ec835a", critical: "#d03b3b" };
// Diverging blue <-> red with a neutral gray midpoint.
export const DIVERGING: Record<Mode, { neg: string; mid: string; pos: string }> = {
  light: { neg: "#2a78d6", mid: "#f0efec", pos: "#e34948" },
  dark: { neg: "#3987e5", mid: "#383835", pos: "#e66767" },
};

export const CHROME: Record<Mode, { surface: string; ink: string; ink2: string; muted: string; grid: string; axis: string }> = {
  light: { surface: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7" },
  dark: { surface: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835" },
};

function resolve(): Mode {
  const t = document.documentElement.dataset.theme;
  if (t === "light" || t === "dark") return t;
  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

export function useMode(): [Mode, (m: Mode) => void] {
  const [mode, setMode] = useState<Mode>(resolve);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setMode(resolve());
    mq.addEventListener("change", on);
    window.addEventListener("sg-theme", on);
    return () => {
      mq.removeEventListener("change", on);
      window.removeEventListener("sg-theme", on);
    };
  }, []);
  const set = (m: Mode) => {
    document.documentElement.dataset.theme = m;
    try {
      localStorage.setItem("sg-theme", m);
    } catch {
      /* storage blocked: theme still applies for this page */
    }
    window.dispatchEvent(new Event("sg-theme"));
  };
  return [mode, set];
}

/** Base ECharts option for a mode: recessive hairline axes, system sans, palette order. */
export function baseOption(mode: Mode) {
  const c = CHROME[mode];
  const axis = {
    axisLine: { lineStyle: { color: c.axis } },
    axisTick: { show: false },
    axisLabel: { color: c.muted, fontSize: 11 },
    splitLine: { lineStyle: { color: c.grid, type: "solid" as const } },
    nameTextStyle: { color: c.muted, fontSize: 11 },
  };
  return {
    color: SERIES[mode],
    backgroundColor: "transparent",
    textStyle: { fontFamily: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: c.ink2 },
    grid: { left: 48, right: 16, top: 28, bottom: 32, containLabel: false },
    legend: { textStyle: { color: c.ink2, fontSize: 12 }, icon: "roundRect", itemWidth: 10, itemHeight: 10, top: 0, right: 0 },
    tooltip: {
      backgroundColor: c.surface,
      borderColor: mode === "dark" ? "rgba(255,255,255,0.12)" : "rgba(11,11,11,0.12)",
      textStyle: { color: c.ink, fontSize: 12 },
      extraCssText: "border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.15);",
    },
    xAxis: { ...axis, splitLine: { show: false } },
    yAxis: { ...axis },
  };
}

export function binColor(v: number | null | undefined, lo: number, hi: number, ramp = BLUE_RAMP): string | null {
  if (v === null || v === undefined || !isFinite(v)) return null;
  const f = hi > lo ? (v - lo) / (hi - lo) : 0;
  const k = Math.max(0, Math.min(ramp.length - 1, Math.floor(f * ramp.length)));
  return ramp[k];
}
