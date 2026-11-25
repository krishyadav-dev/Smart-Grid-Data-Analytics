// No-token fallback for the feeder map: a single-line diagram of the same layers.
// Mapbox GL JS v3 cannot load any style without a token, and drawn to scale the feeder
// is a thin strip (the 650-632 trunk is 4.3 km, 671-692 is 60 m), so this view is
// topological: the textbook IEEE 13-bus arrangement, drawn at the container's real
// pixel size so labels stay readable.
import { useLayoutEffect, useMemo, useRef, useState } from "react";
import type { Json } from "../api";
import { CHROME, STATUS, type Mode } from "../theme";
import type { Highlight } from "./FeederMap";

type Props = {
  mode: Mode;
  layout: Json;
  dtColor?: Record<string, string | null>;
  dtInfo?: Record<string, unknown>;
  areaColor?: Record<string, string | null>;
  highlight?: Highlight | null;
  trueBus?: string | null;
  selectedDt?: string | null;
  onSelectDt?: (id: string) => void;
};

// Bus positions in grid units (column, row): the usual IEEE 13-bus drawing.
const BUS: Record<string, [number, number]> = {
  "650": [2, 0],
  "646": [0, 1], "645": [1, 1], "632": [2, 1], "633": [3, 1], "634": [4, 1],
  "670": [2, 2],
  "611": [0, 3], "684": [1, 3], "671": [2, 3], "692": [3, 3], "675": [4, 3],
  "652": [1, 4], "680": [2, 4],
};
// Where each bus's DTs hang (offset of the first DT, step between DTs), in grid units.
const DT_PLACE: Record<string, { at: [number, number]; step: [number, number] }> = {
  "634": { at: [0, 0.55], step: [0.3, 0] },
  "645": { at: [0, 0.55], step: [0.3, 0] },
  "646": { at: [0, 0.55], step: [0.3, 0] },
  "670": { at: [0.55, 0], step: [0, 0.3] },
  "671": { at: [-0.9, -0.55], step: [0.34, 0] },       // up-left: the only free side of 671
  "692": { at: [0, -0.55], step: [0.3, 0] },
  "675": { at: [-0.34, 0.55], step: [0.34, 0] },
  "611": { at: [0, 0.55], step: [0.3, 0] },
  "652": { at: [0, 0.55], step: [0.3, 0] },
};
const DEFAULT_PLACE = { at: [0, 0.55] as [number, number], step: [0.3, 0] as [number, number] };
const PAD = { l: 28, r: 44, t: 30, b: 34 };

export default function SvgFeeder(p: Props) {
  const c = CHROME[p.mode];
  const box = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [hover, setHover] = useState<{ x: number; y: number; q: Json } | null>(null);

  useLayoutEffect(() => {
    const el = box.current;
    if (!el) return;
    const on = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    on();
    const ro = new ResizeObserver(on);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const feats: Json[] = p.layout.features;

  // DT positions in grid units, fanned out from their bus.
  const dts = useMemo(() => {
    const byBus: Record<string, Json[]> = {};
    feats.filter((f) => f.properties.kind === "dt").forEach((f) => (byBus[f.properties.bus] ??= []).push(f.properties));
    const out: { q: Json; bus: [number, number]; at: [number, number] }[] = [];
    for (const [bus, list] of Object.entries(byBus)) {
      const b = BUS[bus];
      if (!b) continue;
      const pl = DT_PLACE[bus] ?? DEFAULT_PLACE;
      list.sort((a, z) => String(a.load_id).localeCompare(String(z.load_id))).forEach((q, i) =>
        out.push({ q, bus: b, at: [b[0] + pl.at[0] + i * pl.step[0], b[1] + pl.at[1] + i * pl.step[1]] }));
    }
    return out;
  }, [feats]);

  // Grid units -> pixels: fill the box, with at most 1.5x stretch between axes.
  const proj = useMemo(() => {
    const xs = [...Object.values(BUS).map((b) => b[0]), ...dts.map((d) => d.at[0])];
    const ys = [...Object.values(BUS).map((b) => b[1]), ...dts.map((d) => d.at[1])];
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const aw = Math.max(1, size.w - PAD.l - PAD.r), ah = Math.max(1, size.h - PAD.t - PAD.b);
    let ux = aw / (x1 - x0), uy = ah / (y1 - y0);
    ux = Math.min(ux, uy * 1.5);
    uy = Math.min(uy, ux * 1.5);
    const ox = PAD.l + (aw - (x1 - x0) * ux) / 2, oy = PAD.t + (ah - (y1 - y0) * uy) / 2;
    return { pt: ([x, y]: [number, number]): [number, number] => [ox + (x - x0) * ux, oy + (y - y0) * uy], ux };
  }, [size, dts]);

  const hl = new Set(p.highlight?.buses ?? []);
  const hlColor = p.highlight ? (p.highlight.status === "neutral" ? "#6250d6" : STATUS[p.highlight.status]) : c.ink2;
  const neutralDt = p.mode === "dark" ? "#5f5e5a" : "#b9b8b1";
  const ring = p.mode === "dark" ? "#1a1a19" : "#ffffff";
  const small = 0.34 * proj.ux < 36;                          // DTs too close for their labels
  const halo = { stroke: c.surface, strokeWidth: 4, paintOrder: "stroke" as const, strokeLinejoin: "round" as const };

  const show = (ev: React.MouseEvent, q: Json) => {
    const r = box.current?.getBoundingClientRect();
    if (r) setHover({ x: ev.clientX - r.left, y: ev.clientY - r.top, q });
  };

  const ready = size.w > 0 && size.h > 0;
  return (
    <div ref={box} style={{ position: "absolute", inset: 0 }} onMouseLeave={() => setHover(null)}>
      {ready && (
        <svg width={size.w} height={size.h} role="img" aria-label="Feeder single-line diagram (schematic, not to scale)">
          {/* predicted-zone underlay */}
          {feats.filter((f) => f.properties.kind === "segment" && hl.has(f.properties.to) && BUS[f.properties.from] && BUS[f.properties.to]).map((f) => {
            const [ax, ay] = proj.pt(BUS[f.properties.from]), [bx, by] = proj.pt(BUS[f.properties.to]);
            return <line key={`hl-${f.properties.id}`} x1={ax} y1={ay} x2={bx} y2={by} stroke={hlColor} strokeOpacity={0.4} strokeWidth={16} strokeLinecap="round" />;
          })}

          {/* DT stubs */}
          {dts.map(({ q, bus, at }) => {
            const [ax, ay] = proj.pt(bus), [bx, by] = proj.pt(at);
            return <line key={`stub-${q.id}`} x1={ax} y1={ay} x2={bx} y2={by} stroke={c.muted} strokeOpacity={0.7} strokeWidth={1.5} />;
          })}

          {/* segments: line, transformer (two coils), switch (box) */}
          {feats.filter((f) => f.properties.kind === "segment" && BUS[f.properties.from] && BUS[f.properties.to]).map((f) => {
            const q = f.properties;
            const [ax, ay] = proj.pt(BUS[q.from]), [bx, by] = proj.pt(BUS[q.to]);
            const mx = (ax + bx) / 2, my = (ay + by) / 2;
            const tip = `${q.segment}: ${q.element}, ${Math.round(q.model_length_m)} m in the model`;
            return (
              <g key={q.id}>
                <line x1={ax} y1={ay} x2={bx} y2={by} stroke={c.ink2} strokeWidth={q.element === "line" ? 3 : 2} strokeLinecap="round">
                  <title>{tip}</title>
                </line>
                {q.element === "transformer" && (
                  <g fill={c.surface} stroke={c.ink2} strokeWidth={2}>
                    <circle cx={mx - 5} cy={my} r={7} /><circle cx={mx + 5} cy={my} r={7} fillOpacity={0} />
                    <title>{tip}</title>
                  </g>
                )}
                {q.element === "switch" && (
                  <rect x={mx - 6} y={my - 6} width={12} height={12} rx={2} fill={c.surface} stroke={c.ink2} strokeWidth={2}><title>{tip}</title></rect>
                )}
              </g>
            );
          })}

          {/* buses */}
          {feats.filter((f) => f.properties.kind === "bus" && BUS[f.properties.bus]).map((f) => {
            const q = f.properties;
            const [x, y] = proj.pt(BUS[q.bus]);
            const sub = q.role === "substation";
            return (
              <g key={q.id} onMouseMove={(ev) => show(ev, q)} onMouseLeave={() => setHover(null)}>
                {q.bus === p.trueBus && <circle cx={x} cy={y} r={17} fill="none" stroke={c.ink} strokeWidth={2.5} />}
                {sub
                  ? <rect x={x - 11} y={y - 11} width={22} height={22} rx={4} fill={hl.has(q.bus) ? hlColor : c.ink2} stroke={ring} strokeWidth={2.5} />
                  : <circle cx={x} cy={y} r={6.5} fill={hl.has(q.bus) ? hlColor : c.ink2} stroke={ring} strokeWidth={2.5} />}
                <text x={x + (sub ? 16 : 10)} y={y - (sub ? 4 : 9)} fontSize={13} fontWeight={600} fill={c.ink} {...halo}>
                  {q.bus}
                  <tspan fontSize={11} fontWeight={400} fill={c.muted}> {sub ? `substation · zone ${q.zone}` : q.zone}</tspan>
                </text>
              </g>
            );
          })}

          {/* DTs */}
          {dts.map(({ q, bus, at }) => {
            const [x, y] = proj.pt(at);
            const above = at[1] < bus[1];                        // label on the side away from the bus
            const sel = q.load_id === p.selectedDt;
            const r = Math.max(4, Math.min(7 + Math.min(1, Math.max(0, (q.rating_kva - 160) / 470)) * 5, 0.34 * proj.ux / 2 - 2));
            const area = p.areaColor?.[q.load_id];
            return (
              <g key={q.id} style={{ cursor: p.onSelectDt ? "pointer" : "default" }}
                onClick={() => p.onSelectDt?.(q.load_id)} onMouseMove={(ev) => show(ev, q)} onMouseLeave={() => setHover(null)}>
                {area && <circle cx={x} cy={y} r={r + 7} fill={area} fillOpacity={0.35} />}
                <circle cx={x} cy={y} r={r} fill={p.dtColor?.[q.load_id] ?? neutralDt} stroke={sel ? c.ink : ring} strokeWidth={sel ? 3 : 2} />
                {(!small || sel) && (
                  <text x={x} y={above ? y - r - 6 : y + r + 13} textAnchor="middle" fontSize={10.5} fill={sel ? c.ink : c.ink2} fontWeight={sel ? 600 : 400} {...halo}>
                    {String(q.load_id).replace(/^dt_/, "")}
                  </text>
                )}
              </g>
            );
          })}

          <text x={size.w - 12} y={size.h - 12} textAnchor="end" fontSize={11} fill={c.muted}>single-line diagram · not to scale</text>
        </svg>
      )}
      {hover && (
        <div className="mapboxgl-popup-content popup"
          style={{ position: "absolute", left: Math.min(hover.x + 14, Math.max(0, size.w - 220)), top: hover.y + 10, padding: "8px 10px", pointerEvents: "none", zIndex: 3, minWidth: 190 }}>
          {hover.q.kind === "dt" ? (
            <>
              <b>{hover.q.load_id}</b>
              <div className="row"><span>Meter</span><span>{hover.q.meter_id}</span></div>
              <div className="row"><span>Bus / zone</span><span>{hover.q.bus} / {hover.q.zone}</span></div>
              <div className="row"><span>Rating</span><span>{hover.q.rating_kva} kVA</span></div>
              <div className="row"><span>Households</span><span>{hover.q.households ?? "–"}</span></div>
              {p.dtInfo?.[hover.q.load_id] != null && <div className="row"><span>Selected metric</span><span>{String(p.dtInfo[hover.q.load_id])}</span></div>}
            </>
          ) : (
            <>
              <b>{hover.q.role === "substation" ? hover.q.label : `Bus ${hover.q.bus}`}</b>
              <div className="row"><span>Zone</span><span>{hover.q.zone}</span></div>
            </>
          )}
        </div>
      )}
    </div>
  );
}
