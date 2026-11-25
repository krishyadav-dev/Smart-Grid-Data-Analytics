import { useEffect, useRef } from "react";
import * as echarts from "echarts";
import { baseOption, type Mode } from "../theme";
import type { Json } from "../api";

function isObj(v: unknown): v is Record<string, unknown> {
  return !!v && typeof v === "object" && !Array.isArray(v);
}

function merge(a: Json, b: Json): Json {
  if (!isObj(a) || !isObj(b)) return b === undefined ? a : b;
  const out: Record<string, unknown> = { ...a };
  for (const [k, v] of Object.entries(b)) out[k] = isObj(v) && isObj(a[k]) ? merge(a[k], v) : v;
  return out;
}

/** Page option layered on the mode's base option; axis arrays get the base axis style each. */
export function themed(mode: Mode, option: Json): Json {
  const base = baseOption(mode);
  const out = merge(base, { ...option, xAxis: undefined, yAxis: undefined });
  if (option.legend === undefined) delete out.legend;            // one series: the title names it
  for (const k of ["xAxis", "yAxis"] as const) {
    const v = option[k];
    if (v === undefined) delete out[k];
    else out[k] = Array.isArray(v) ? v.map((x) => merge(base[k], x)) : merge(base[k], v);
  }
  return roomForAxisNames(out);
}

/** Axis names (units) were drawn outside the grid and clipped: x names go under the
 *  axis, centred, and the grid grows to fit both. */
function roomForAxisNames(o: Json): Json {
  if (!o.grid || Array.isArray(o.grid)) return o;
  const g = { ...o.grid };
  const list = (a: Json) => (a === undefined ? [] : Array.isArray(a) ? a : [a]);
  const xs = list(o.xAxis).map((x: Json) => {
    if (!x.name || x.nameLocation === "start") return x;
    const y = { ...x, nameLocation: "middle", nameGap: x.nameGap ?? 24 };
    g.bottom = Math.max(Number(g.bottom ?? 0), y.nameGap + 18);
    return y;
  });
  const ys = list(o.yAxis).map((y: Json) => {
    if (!y.name || !(y.nameLocation === undefined || y.nameLocation === "end") || y.inverse) return y;
    g.top = Math.max(Number(g.top ?? 0), 28);
    // centred on the axis line, a long name runs off the left edge: start it at the axis
    return y.position === "right" ? y : { ...y, nameTextStyle: { align: "left", ...y.nameTextStyle }, nameGap: y.nameGap ?? 10 };
  });
  return {
    ...o, grid: g,
    ...(o.xAxis !== undefined ? { xAxis: Array.isArray(o.xAxis) ? xs : xs[0] } : {}),
    ...(o.yAxis !== undefined ? { yAxis: Array.isArray(o.yAxis) ? ys : ys[0] } : {}),
  };
}

type Props = {
  mode: Mode;
  option: Json;
  className?: string;
  onClick?: (p: Json) => void;
  ariaLabel: string;
};

export default function Chart({ mode, option, className = "chart", onClick, ariaLabel }: Props) {
  const el = useRef<HTMLDivElement>(null);
  const inst = useRef<echarts.ECharts | null>(null);
  const clickRef = useRef(onClick);
  clickRef.current = onClick;

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: "canvas" });
    inst.current = c;
    c.on("click", (p) => clickRef.current?.(p));
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      inst.current = null;
    };
  }, []);

  useEffect(() => {
    inst.current?.setOption(themed(mode, option), true);
  }, [mode, option]);

  return <div ref={el} className={className} role="img" aria-label={ariaLabel} />;
}
