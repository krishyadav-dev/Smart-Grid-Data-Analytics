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
  return out;
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
