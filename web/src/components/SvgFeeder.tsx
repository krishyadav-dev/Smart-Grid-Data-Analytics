// No-token fallback for the feeder map: the same layers drawn as SVG in a local
// equirectangular projection (Mapbox GL JS v3 cannot load any style without a token).
import { useMemo, useState } from "react";
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

const W = 1000, H = 700, PAD = 40;

export default function SvgFeeder(p: Props) {
  const c = CHROME[p.mode];
  const [hover, setHover] = useState<{ x: number; y: number; q: Json } | null>(null);

  const proj = useMemo(() => {
    let w = 180, s = 90, e = -180, n = -90;
    const visit = (a: Json) => {
      if (typeof a[0] === "number") { w = Math.min(w, a[0]); e = Math.max(e, a[0]); s = Math.min(s, a[1]); n = Math.max(n, a[1]); }
      else a.forEach(visit);
    };
    p.layout.features.forEach((f: Json) => visit(f.geometry.coordinates));
    const kx = Math.cos(((s + n) / 2) * Math.PI / 180);          // metres per degree of longitude shrink
    const sx = (W - 2 * PAD) / ((e - w) * kx), sy = (H - 2 * PAD) / (n - s);
    const k = Math.min(sx, sy);
    const ox = (W - (e - w) * kx * k) / 2, oy = (H - (n - s) * k) / 2;
    const pt = (lon: number, lat: number): [number, number] => [ox + (lon - w) * kx * k, oy + (n - lat) * k];
    const mPerPx = 111_320 / k;
    return { pt, mPerPx };
  }, [p.layout]);

  const hl = new Set(p.highlight?.buses ?? []);
  const hlColor = p.highlight ? (p.highlight.status === "neutral" ? "#6250d6" : STATUS[p.highlight.status]) : c.ink2;
  const feats: Json[] = p.layout.features;
  const path = (coords: number[][]) => coords.map(([lo, la], i) => `${i ? "L" : "M"}${proj.pt(lo, la).map((v) => v.toFixed(1)).join(",")}`).join("");
  const neutralDt = p.mode === "dark" ? "#5f5e5a" : "#b9b8b1";
  const ring = p.mode === "dark" ? "#1a1a19" : "#ffffff";
  const bar = 1000 / proj.mPerPx;                                 // 1 km scale bar

  return (
    <div style={{ position: "absolute", inset: 0 }} onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet" role="img"
        aria-label="Feeder single-line map (schematic, no base map)">
        {feats.filter((f) => f.properties.kind === "service_area").map((f) => (
          <path key={f.properties.id} d={path(f.geometry.coordinates[0]) + "Z"}
            fill={p.areaColor?.[f.properties.load_id] ?? "transparent"} fillOpacity={0.5} stroke={c.axis} strokeWidth={1} />
        ))}
        {feats.filter((f) => f.properties.kind === "segment" && hl.has(f.properties.to)).map((f) => (
          <path key={`hl-${f.properties.id}`} d={path(f.geometry.coordinates)} stroke={hlColor} strokeOpacity={0.45} strokeWidth={14} strokeLinecap="round" fill="none" />
        ))}
        {feats.filter((f) => f.properties.kind === "segment").map((f) => (
          <path key={f.properties.id} d={path(f.geometry.coordinates)} stroke={c.ink2} strokeWidth={f.properties.element === "line" ? 3 : 2}
            strokeLinecap="round" fill="none"><title>{`${f.properties.segment}: ${f.properties.model_length_m} m (${f.properties.element})`}</title></path>
        ))}
        {feats.filter((f) => f.properties.kind === "bus").map((f) => {
          const [x, y] = proj.pt(...(f.geometry.coordinates as [number, number]));
          const q = f.properties;
          return (
            <g key={q.id}>
              {q.bus === p.trueBus && <circle cx={x} cy={y} r={18} fill="none" stroke={c.ink} strokeWidth={3} />}
              <circle cx={x} cy={y} r={q.role === "substation" ? 10 : 6} fill={hl.has(q.bus) ? hlColor : c.ink2} stroke={ring} strokeWidth={2.5}
                onMouseMove={(ev) => setHover({ x: ev.nativeEvent.offsetX, y: ev.nativeEvent.offsetY, q })} />
              <text x={x} y={y - 14} textAnchor="middle" fontSize={14} fill={c.ink} stroke={c.surface} strokeWidth={4} paintOrder="stroke">{q.bus}</text>
            </g>
          );
        })}
        {feats.filter((f) => f.properties.kind === "dt").map((f) => {
          const [x, y] = proj.pt(...(f.geometry.coordinates as [number, number]));
          const q = f.properties;
          const sel = q.load_id === p.selectedDt;
          const r = 7 + Math.min(1, Math.max(0, (q.rating_kva - 160) / 470)) * 6;
          return (
            <circle key={q.id} cx={x} cy={y} r={r} fill={p.dtColor?.[q.load_id] ?? neutralDt}
              stroke={sel ? c.ink : ring} strokeWidth={sel ? 3.5 : 2.5} style={{ cursor: "pointer" }}
              onClick={() => p.onSelectDt?.(q.load_id)}
              onMouseMove={(ev) => setHover({ x: ev.nativeEvent.offsetX, y: ev.nativeEvent.offsetY, q })} onMouseLeave={() => setHover(null)} />
          );
        })}
        <g transform={`translate(${W - PAD - bar}, ${H - 18})`}>
          <line x1={0} x2={bar} y1={0} y2={0} stroke={c.ink2} strokeWidth={2} />
          <text x={bar / 2} y={-6} textAnchor="middle" fontSize={12} fill={c.muted}>1 km</text>
        </g>
      </svg>
      {hover && (
        <div className="mapboxgl-popup-content popup" style={{ position: "absolute", left: hover.x + 14, top: hover.y + 8, padding: "8px 10px", pointerEvents: "none", zIndex: 3, minWidth: 170 }}>
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
