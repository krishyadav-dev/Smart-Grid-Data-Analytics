import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import mapboxgl from "mapbox-gl";
import "mapbox-gl/dist/mapbox-gl.css";
import type { Json } from "../api";
import SvgFeeder from "./SvgFeeder";
import { CHROME, STATUS, type Mode } from "../theme";

export type Highlight = { buses: string[]; status: "good" | "critical" | "neutral"; label?: string };

type Props = {
  mode: Mode;
  layout: Json;                                  // config/geo_layout.geojson
  token: string | null;
  note?: string;
  dtColor?: Record<string, string | null>;       // DT marker fill by load_id
  dtInfo?: Record<string, ReactNode | string>;   // extra popup line per DT (plain text)
  areaColor?: Record<string, string | null>;     // service-area fill by load_id
  highlight?: Highlight | null;                  // predicted zone (status colour)
  trueBus?: string | null;                       // ground-truth fault bus (ring)
  selectedDt?: string | null;
  onSelectDt?: (loadId: string) => void;
  legend?: ReactNode;
  tall?: boolean;
};

type Basemap = "streets" | "satellite";

const BLANK = (mode: Mode): mapboxgl.StyleSpecification => ({
  version: 8,
  sources: {},
  layers: [{ id: "bg", type: "background", paint: { "background-color": mode === "dark" ? "#141413" : "#f3f2ee" } }],
});

function styleFor(mode: Mode, token: string | null, base: Basemap): string | mapboxgl.StyleSpecification {
  if (!token) return BLANK(mode);
  if (base === "satellite") return "mapbox://styles/mapbox/satellite-streets-v12";
  return mode === "dark" ? "mapbox://styles/mapbox/dark-v11" : "mapbox://styles/mapbox/light-v11";
}

function bbox(fc: Json): [[number, number], [number, number]] {
  let w = 180, s = 90, e = -180, n = -90;
  const visit = (c: Json) => {
    if (typeof c[0] === "number") {
      w = Math.min(w, c[0]); e = Math.max(e, c[0]); s = Math.min(s, c[1]); n = Math.max(n, c[1]);
    } else c.forEach(visit);
  };
  fc.features.forEach((f: Json) => visit(f.geometry.coordinates));
  return [[w, s], [e, n]];
}

function MapboxView(p: Props & { onFail: (msg: string) => void }) {
  const el = useRef<HTMLDivElement>(null);
  const map = useRef<mapboxgl.Map | null>(null);
  const [base, setBase] = useState<Basemap>("streets");
  const [ready, setReady] = useState(0);
  const propsRef = useRef(p);
  propsRef.current = p;

  // Feature collection with the display properties baked in (geometry sent once per change).
  const data = useMemo(() => {
    const hl = new Set(p.highlight?.buses ?? []);
    const hlColor = p.highlight ? (p.highlight.status === "neutral" ? "#6250d6" : STATUS[p.highlight.status]) : null;
    return {
      type: "FeatureCollection",
      features: p.layout.features.map((f: Json) => {
        const q = f.properties;
        const props: Json = { ...q };
        if (q.kind === "dt") {
          props.fill = p.dtColor?.[q.load_id] ?? null;
          props.selected = q.load_id === p.selectedDt;
          props.hl = hl.has(q.bus);
        }
        if (q.kind === "service_area") props.fill = p.areaColor?.[q.load_id] ?? null;
        if (q.kind === "segment") props.hl = hl.has(q.to);           // the edge feeding a zone bus
        if (q.kind === "bus") {
          props.hl = hl.has(q.bus);
          props.truth = q.bus === p.trueBus;
        }
        props.hlColor = hlColor;
        return { ...f, properties: props };
      }),
    };
  }, [p.layout, p.dtColor, p.areaColor, p.highlight, p.trueBus, p.selectedDt]);
  const dataRef = useRef(data);
  dataRef.current = data;

  // create once
  useEffect(() => {
    if (!el.current) return;
    if (p.token) mapboxgl.accessToken = p.token;
    let m: mapboxgl.Map;
    try {
      m = new mapboxgl.Map({
        container: el.current,
        style: styleFor(p.mode, p.token, base),
        bounds: bbox(p.layout),
        fitBoundsOptions: { padding: 48 },
        attributionControl: true,
        cooperativeGestures: false,
      });
    } catch (e) {
      p.onFail(String((e as Error).message || e));
      return;
    }
    map.current = m;
    m.addControl(new mapboxgl.NavigationControl({ showCompass: false }), "top-left");
    m.addControl(new mapboxgl.ScaleControl({ unit: "metric" }), "bottom-right");
    m.on("style.load", () => setReady((r) => r + 1));
    m.on("error", (ev) => {
      const err = ev?.error as { message?: string; status?: number } | undefined;
      const msg = String(err?.message || "");
      if (err?.status === 401 || err?.status === 403 || /access token|unauthori[sz]ed|forbidden/i.test(msg)) propsRef.current.onFail("Mapbox rejected the token (check MAPBOX_TOKEN and its URL restrictions); showing the feeder without a base map.");
    });
    const popup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false, offset: 10, maxWidth: "260px" });
    m.on("mousemove", "dts", (ev) => {
      const f = ev.features?.[0];
      if (!f) return;
      m.getCanvas().style.cursor = "pointer";
      const q = f.properties as Json;
      const extra = propsRef.current.dtInfo?.[q.load_id];
      popup
        .setLngLat((f.geometry as Json).coordinates)
        .setHTML(
          `<div class="popup"><b>${q.load_id}</b>` +
            `<div class="row"><span>Meter</span><span>${q.meter_id}</span></div>` +
            `<div class="row"><span>Bus / zone</span><span>${q.bus} / ${q.zone}</span></div>` +
            `<div class="row"><span>Rating</span><span>${q.rating_kva} kVA</span></div>` +
            `<div class="row"><span>Households</span><span>${q.households ?? "–"}</span></div>` +
            (extra ? `<div class="row"><span>Selected metric</span><span>${extra}</span></div>` : "") +
            `</div>`,
        )
        .addTo(m);
    });
    m.on("mouseleave", "dts", () => {
      m.getCanvas().style.cursor = "";
      popup.remove();
    });
    m.on("click", "dts", (ev) => {
      const id = (ev.features?.[0]?.properties as Json)?.load_id;
      if (id) propsRef.current.onSelectDt?.(id);
    });
    const busPopup = new mapboxgl.Popup({ closeButton: false, closeOnClick: false, offset: 8 });
    m.on("mousemove", "buses", (ev) => {
      const q = ev.features?.[0]?.properties as Json;
      if (!q) return;
      busPopup.setLngLat((ev.features![0].geometry as Json).coordinates)
        .setHTML(`<div class="popup"><b>${q.role === "substation" ? q.label : `Bus ${q.bus}`}</b>` +
          `<div class="row"><span>Zone</span><span>${q.zone}</span></div></div>`).addTo(m);
    });
    m.on("mouseleave", "buses", () => busPopup.remove());
    return () => {
      popup.remove();
      m.remove();
      map.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // theme / basemap change -> new style (layers re-added on style.load)
  const first = useRef(true);
  useEffect(() => {
    if (first.current) {
      first.current = false;
      return;
    }
    map.current?.setStyle(styleFor(p.mode, p.token, base));
  }, [p.mode, base, p.token]);

  // (re)add layers after each style load
  useEffect(() => {
    const m = map.current;
    if (!m || !ready) return;
    const c = CHROME[p.mode];
    const ring = p.mode === "dark" ? "#1a1a19" : "#ffffff";
    const neutralDt = p.mode === "dark" ? "#5f5e5a" : "#b9b8b1";
    const line = base === "satellite" ? "#ffffff" : p.mode === "dark" ? "#c3c2b7" : "#52514e";
    if (m.getSource("feeder")) return;
    m.addSource("feeder", { type: "geojson", data: dataRef.current as Json });
    m.addLayer({
      id: "areas", type: "fill", source: "feeder", filter: ["==", ["get", "kind"], "service_area"],
      paint: { "fill-color": ["coalesce", ["get", "fill"], "rgba(0,0,0,0)"], "fill-opacity": 0.45 },
    });
    m.addLayer({
      id: "areas-line", type: "line", source: "feeder", filter: ["==", ["get", "kind"], "service_area"],
      paint: { "line-color": c.axis, "line-width": 1 },
    });
    m.addLayer({
      id: "segments-hl", type: "line", source: "feeder",
      filter: ["all", ["==", ["get", "kind"], "segment"], ["==", ["get", "hl"], true]],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": ["coalesce", ["get", "hlColor"], line], "line-width": 9, "line-opacity": 0.45 },
    });
    m.addLayer({
      id: "segments", type: "line", source: "feeder", filter: ["==", ["get", "kind"], "segment"],
      layout: { "line-cap": "round", "line-join": "round" },
      paint: { "line-color": line, "line-width": ["match", ["get", "element"], "line", 2.5, 1.5] },
    });
    m.addLayer({
      id: "buses", type: "circle", source: "feeder", filter: ["==", ["get", "kind"], "bus"],
      paint: {
        "circle-radius": ["case", ["==", ["get", "role"], "substation"], 8, 4],
        "circle-color": ["case", ["==", ["get", "hl"], true], ["coalesce", ["get", "hlColor"], line], line],
        "circle-stroke-color": ring, "circle-stroke-width": 2,
      },
    });
    m.addLayer({
      id: "truth", type: "circle", source: "feeder",
      filter: ["all", ["==", ["get", "kind"], "bus"], ["==", ["get", "truth"], true]],
      paint: { "circle-radius": 14, "circle-color": "rgba(0,0,0,0)", "circle-stroke-color": c.ink, "circle-stroke-width": 2.5 },
    });
    m.addLayer({
      id: "dts", type: "circle", source: "feeder", filter: ["==", ["get", "kind"], "dt"],
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["get", "rating_kva"], 160, 6, 630, 10],
        "circle-color": ["coalesce", ["get", "fill"], neutralDt],
        "circle-stroke-color": ["case", ["==", ["get", "selected"], true], c.ink, ring],
        "circle-stroke-width": ["case", ["==", ["get", "selected"], true], 3, 2],
      },
    });
    if (p.token) {
      m.addLayer({
        id: "bus-labels", type: "symbol", source: "feeder", filter: ["==", ["get", "kind"], "bus"],
        layout: { "text-field": ["get", "bus"], "text-size": 11, "text-offset": [0, -1.3], "text-font": ["DIN Pro Medium", "Arial Unicode MS Regular"] },
        paint: { "text-color": base === "satellite" ? "#ffffff" : c.ink, "text-halo-color": base === "satellite" ? "#000" : c.surface, "text-halo-width": 1.5 },
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready]);

  // values change -> setData only
  useEffect(() => {
    const src = map.current?.getSource("feeder") as mapboxgl.GeoJSONSource | undefined;
    src?.setData(data as Json);
  }, [data, ready]);

  return (
    <div className={`mapbox ${p.tall ? "tall" : ""}`}>
      <div ref={el} style={{ position: "absolute", inset: 0 }} />
      {(
        <div className="map-notice" style={{ padding: 4 }}>
          <div className="seg" role="group" aria-label="Base map">
            {(["streets", "satellite"] as Basemap[]).map((b) => (
              <button key={b} aria-pressed={b === base} onClick={() => setBase(b)}>
                {b === "streets" ? "Streets" : "Satellite"}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="map-legend">
        {p.legend}
        {p.note && <div className="fine">{p.note}</div>}
      </div>
    </div>
  );
}

/** Mapbox GL map when a token is set; an SVG drawing of the same layers otherwise. */
export default function FeederMap(p: Props) {
  const [failed, setFailed] = useState<string | null>(null);
  if (p.token && !failed) return <MapboxView {...p} onFail={setFailed} />;
  return (
    <div className={`mapbox schematic ${p.tall ? "tall" : ""}`}>
      <div className="schematic-draw">
        <SvgFeeder {...p} />
      </div>
      <div className="map-legend">
        <div>{p.legend}</div>
        <div className="fine">
          {p.note && <div>{p.note}</div>}
          <div title="Set MAPBOX_TOKEN=pk… in .env and restart the service for street and satellite base maps.">
            {failed ?? "Schematic view: no Mapbox token set, so no street or satellite base map."}
          </div>
        </div>
      </div>
    </div>
  );
}

export function RampLegend({ title, ramp, lo, hi, unit }: { title: string; ramp: string[]; lo: string; hi: string; unit?: string }) {
  return (
    <div>
      <b>{title}</b>
      <div className="ramp">{ramp.map((c) => <span key={c} style={{ background: c }} />)}</div>
      <div className="ends"><span>{lo}</span><span>{hi}{unit ? ` ${unit}` : ""}</span></div>
    </div>
  );
}
