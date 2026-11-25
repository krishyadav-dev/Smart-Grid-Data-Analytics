import { useMemo, useState } from "react";
import type { Ctx } from "../App";
import { dayLabel, fmt, monthAxis, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import FeederMap, { RampLegend } from "../components/FeederMap";
import { Card, Gate, Kpi, PageHead, Seg } from "../components/ui";
import { BLUE_RAMP, binColor, CHROME, SERIES } from "../theme";

type Metric = "load" | "hotspot" | "theft" | "vu";

const METRICS: Record<Metric, { label: string; unit: string; lo: number; hi: number; title: string }> = {
  load: { label: "Loading", unit: "%", lo: 0, hi: 140, title: "DT loading, daily max of the worst phase (% of rated current)" },
  hotspot: { label: "Hot-spot", unit: "°C", lo: 20, hi: 105, title: "Hot-spot temperature, daily max (IS 2026-7 reference)" },
  theft: { label: "Theft est.", unit: "kWh", lo: 0, hi: 60, title: "Engine NTL estimate per DT-day (netted, λ = 0)" },
  vu: { label: "Unbalance", unit: "%", lo: 0, hi: 2.5, title: "Voltage unbalance, daily max (limit 2 %)" },
};

export default function Overview({ mode, config, layout, meta }: Ctx) {
  const grid = useApi("/api/results/grid/daily");
  const thermal = useApi("/api/results/thermal/daily");
  const theft = useApi("/api/results/theft/dt-daily");
  const vu = useApi("/api/results/unbalance/daily");
  const [metric, setMetric] = useState<Metric>("load");
  const [dayIdx, setDayIdx] = useState<number | null>(null);
  const [dt, setDt] = useState<string>("dt_634_1");

  const days: string[] = grid.data?.days ?? thermal.data?.days ?? [];
  const peakIdx = useMemo(() => {
    const h: number[] = grid.data?.head_peak_mw ?? [];
    return h.length ? h.indexOf(Math.max(...h)) : 175;
  }, [grid.data]);
  const k = dayIdx ?? peakIdx;
  const day = days[k];
  const dayData = useApi(day ? `/api/results/grid/day/${day}` : null);

  // per-DT series for the selected metric: {load_id: number[] over days}
  const series: Record<string, (number | null)[]> | null = useMemo(() => {
    if (metric === "load") return grid.data?.available ? grid.data.dt_load_pct_max : null;
    if (metric === "hotspot") return thermal.data?.available ? thermal.data.theta_h_max_c : null;
    if (metric === "theft") return theft.data?.available ? theft.data.estimate_kwh : null;
    if (metric === "vu" && vu.data?.available) return vu.data.max_pct;
    return null;
  }, [metric, grid.data, thermal.data, theft.data, vu.data]);

  const m = METRICS[metric];
  const dtColor: Record<string, string | null> = {};
  const dtInfo: Record<string, string> = {};
  if (series) {
    for (const [id, v] of Object.entries(series)) {
      const x = v[k];
      dtColor[id] = binColor(x, m.lo, m.hi);
      dtInfo[id] = x === null || x === undefined ? "–" : `${fmt.n(x, metric === "vu" ? 2 : 1)} ${m.unit}`;
    }
  }
  const metricMissing = !series;

  const c = CHROME[mode];
  const dtYear = useMemo(() => {
    if (!series || !series[dt]) return null;
    const lim = metric === "vu" ? 2 : metric === "hotspot" ? 98 : metric === "load" ? 100 : null;
    return {
      grid: { left: 44, right: 12, top: 16, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => (v == null ? "–" : `${fmt.n(v, 1)} ${m.unit}`) },
      xAxis: monthAxis(days, 2),
      yAxis: { type: "value", name: m.unit, scale: metric === "hotspot" },
      series: [{
        type: "line", name: dt, data: series[dt], showSymbol: false, lineStyle: { width: 2 }, color: SERIES[mode][0],
        markLine: {
          silent: true, symbol: "none",
          data: [
            { xAxis: day, lineStyle: { color: c.ink2, width: 1, type: "solid" }, label: { show: false } },
            ...(lim ? [{ yAxis: lim, lineStyle: { color: "#d03b3b", width: 1, type: "solid" }, label: { formatter: metric === "load" ? "rated" : `limit ${lim}`, position: "insideEndTop", color: c.muted, fontSize: 10 } }] : []),
          ],
        },
      }],
    };
  }, [series, dt, days, day, metric, mode]);

  const dayProfile = useMemo(() => {
    const d = dayData.data;
    if (!d?.available) return null;
    return {
      grid: { left: 44, right: 12, top: 16, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 2)} MW` },
      xAxis: { type: "category", data: d.times, axisLabel: { interval: 23 } },
      yAxis: { type: "value", name: "MW", scale: true },
      series: [{ type: "line", name: "Feeder head P", data: d.p_head_mw, showSymbol: false, lineStyle: { width: 2 }, areaStyle: { opacity: 0.08 }, color: SERIES[mode][0] }],
    };
  }, [dayData.data, mode]);

  const g: Json = grid.data;
  const dtMeta = meta.meters.find((x: Json) => x.load_id === dt);

  return (
    <>
      <PageHead
        title="Grid map"
        sub="The Stage B 11 kV feeder (IEEE 13-bus, converted) placed in Delhi for illustration. Pick a day of the simulated 2024 year and a metric to colour the distribution transformers."
        right={<Seg label="Map metric" value={metric} onChange={setMetric}
          options={(Object.keys(METRICS) as Metric[]).map((v) => ({ v, label: METRICS[v].label }))} />}
      />

      <div className="card" style={{ flexDirection: "row", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <div className="slider">
          <label htmlFor="day" style={{ color: "var(--ink-2)", fontSize: 12 }}>Day</label>
          <input id="day" type="range" min={0} max={Math.max(0, days.length - 1)} value={k}
            onChange={(e) => setDayIdx(Number(e.target.value))} aria-valuetext={day} />
          <b className="mono" style={{ minWidth: 104 }}>{day ? dayLabel(day) : "–"}</b>
        </div>
        <div className="toolbar">
          <button className="btn" onClick={() => setDayIdx(Math.max(0, k - 1))} aria-label="Previous day">‹</button>
          <button className="btn" onClick={() => setDayIdx(Math.min(days.length - 1, k + 1))} aria-label="Next day">›</button>
          <button className="btn" onClick={() => setDayIdx(peakIdx)}>Peak day</button>
        </div>
      </div>

      <Gate s={grid} h={84}>
        {() => (
          <div className="kpis">
            <Kpi label="Feeder head peak" value={fmt.n(g.head_peak_mw[k], 2)} unit="MW" sub={`year peak ${fmt.n(Math.max(...g.head_peak_mw), 2)} MW`} />
            <Kpi label="Head energy" value={fmt.n(g.head_energy_mwh[k], 1)} unit="MWh" />
            <Kpi label="Losses" value={fmt.n(g.losses_mwh[k], 2)} unit="MWh"
              sub={`${fmt.n((100 * g.losses_mwh[k]) / g.head_energy_mwh[k], 1)} % of head energy`} />
            <Kpi label="LV voltage, min / max" value={`${fmt.n(g.v_min_lv_pu[k], 3)} / ${fmt.n(g.v_max_lv_pu[k], 3)}`} unit="pu" sub="over the DT LV buses" />
            <Kpi label="Ambient max (ERA5)" value={fmt.n(g.ambient_max_c?.[k], 1)} unit="°C" sub="28.6 N, 77.2 E" />
            <Kpi label="OLTC tap moves" value={fmt.int(g.oltc_tap_moves[k])} />
          </div>
        )}
      </Gate>

      <div className="grid g3">
        <div className="span2">
          <FeederMap
            tall
            mode={mode}
            layout={layout}
            token={config.mapbox_token}
            dtColor={dtColor}
            dtInfo={dtInfo}
            selectedDt={dt}
            onSelectDt={setDt}
            note={config.map_note}
            legend={metricMissing
              ? <b>{m.label}: data file not present</b>
              : <RampLegend title={m.title} ramp={BLUE_RAMP} lo={`${m.lo}`} hi={`${m.hi}`} unit={m.unit} />}
          />
        </div>
        <div className="grid" style={{ alignContent: "start" }}>
          <Card title={dt} sub={dtMeta ? `${dtMeta.meter_id} · bus ${dtMeta.bus} · zone ${dtMeta.zone} · ${dtMeta.rating_kva} kVA. Click a DT on the map to switch.` : undefined}
            right={<select className="select" value={dt} onChange={(e) => setDt(e.target.value)} aria-label="Transformer">
              {meta.meters.filter((x: Json) => x.role === "dt").map((x: Json) => <option key={x.load_id}>{x.load_id}</option>)}
            </select>}>
            <h3>{m.title}, 2024</h3>
            {dtYear ? <Chart mode={mode} option={dtYear} className="chart short" ariaLabel={`${m.label} over 2024 for ${dt}`} />
              : <div className="empty">No data for this metric in this checkout.</div>}
          </Card>
          <Card title="Feeder head, selected day" sub="15-min active power at the 11 kV head (m10)">
            <Gate s={dayData}>{() => <Chart mode={mode} option={dayProfile} className="chart short" ariaLabel="Feeder head power over the selected day" />}</Gate>
          </Card>
        </div>
      </div>
    </>
  );
}
