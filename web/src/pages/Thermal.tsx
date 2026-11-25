import { useMemo, useState } from "react";
import type { Ctx } from "../App";
import { dayLabel, fmt, monthAxis, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import { Card, Gate, Kpi, PageHead, Pill } from "../components/ui";
import { BLUE_RAMP, CHROME, SERIES } from "../theme";

const MODELS: [string, string][] = [["persistence", "Persistence"], ["seasonal_naive", "Seasonal naive"], ["lstm", "LSTM (3-seed mean)"]];

export default function Thermal({ mode, meta }: Ctx) {
  const sum = useApi("/api/results/thermal");
  const daily = useApi("/api/results/thermal/daily");
  const [dt, setDt] = useState("dt_634_1");
  const [day, setDay] = useState<string | null>(null);
  const c = CHROME[mode];
  const S = SERIES[mode];
  const d: Json = daily.data;
  const s: Json = sum.data;

  const peakDay = useMemo(() => {
    if (!d?.available) return null;
    const v: number[] = d.theta_h_max_c[dt];
    return d.days[v.indexOf(Math.max(...v))];
  }, [d, dt]);
  const shownDay = day ?? peakDay ?? "2024-06-25";
  const dayData = useApi(`/api/results/thermal/day/${shownDay}`);

  const heat = useMemo(() => {
    if (!d?.available) return null;
    const data: [number, number, number | null][] = [];
    d.dts.forEach((k: string, y: number) => d.theta_h_max_c[k].forEach((v: number | null, x: number) => data.push([x, y, v])));
    return {
      grid: { left: 70, right: 16, top: 8, bottom: 56 },
      tooltip: { formatter: (q: Json) => `${d.dts[q.value[1]]}, ${dayLabel(d.days[q.value[0]])}<br/>hot-spot max <b>${fmt.n(q.value[2], 1)} °C</b>` },
      xAxis: { ...monthAxis(d.days), splitArea: { show: false } },
      yAxis: { type: "category", data: d.dts, inverse: true, splitLine: { show: false } },
      visualMap: { min: 20, max: 100, calculable: false, orient: "horizontal", left: "center", bottom: 0, itemWidth: 10, itemHeight: 160,
        text: ["100 °C", "20 °C"], textStyle: { color: c.muted, fontSize: 11 }, inRange: { color: BLUE_RAMP } },
      series: [{ type: "heatmap", data, progressive: 0, emphasis: { itemStyle: { borderColor: c.ink, borderWidth: 1 } } }],
    };
  }, [d, mode]);

  const traj = useMemo(() => {
    const x: Json = dayData.data;
    if (!x?.available) return null;
    const series: Json[] = [{ type: "line", name: "Reference (metered load)", data: x.theta_h_c[dt], showSymbol: false, lineStyle: { width: 2.5 }, color: S[0], z: 3 }];
    if (x.forecast) MODELS.forEach(([m, label], i) => series.push({ type: "line", name: label, data: x.forecast[m][dt], showSymbol: false, lineStyle: { width: 1.5 }, color: S[i + 1] }));
    series[0].markLine = { silent: true, symbol: "none", data: [{ yAxis: 98, label: { formatter: "98 °C", color: c.muted, fontSize: 10 }, lineStyle: { color: "#d03b3b", width: 1, type: "solid" } }] };
    return {
      grid: { left: 44, right: 40, top: 52, bottom: 28 },
      legend: {},
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 1)} °C` },
      xAxis: { type: "category", data: x.times, axisLabel: { interval: 15 } },
      yAxis: { type: "value", name: "°C", scale: true, max: (v: Json) => Math.max(100, Math.ceil(v.max / 5) * 5) },
      series,
    };
  }, [dayData.data, dt, mode]);

  const ambient = useMemo(() => {
    const x: Json = dayData.data;
    if (!x?.available || !x.ambient_c) return null;
    return {
      grid: { left: 44, right: 16, top: 16, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 1)} °C` },
      xAxis: { type: "category", data: x.times, axisLabel: { interval: 15 } },
      yAxis: { type: "value", name: "°C", scale: true },
      series: [{ type: "line", name: "Ambient (ERA5)", data: x.ambient_c, showSymbol: false, lineStyle: { width: 2 }, color: S[1] }],
    };
  }, [dayData.data, mode]);

  const rmse = (m: string, h: string) => s?.forecast?.models?.[m]?.hot_spot_pooled?.[h]?.rmse;
  const boot = s?.forecast?.bootstrap?.rmse_persistence_minus_lstm_k;
  const lol = useMemo(() => {
    if (!d?.available) return null;
    const tot = d.dts.map((k: string) => (d.loss_of_life_h[k] as number[]).reduce((a, b) => a + (b ?? 0), 0));
    return {
      grid: { left: 70, right: 24, top: 8, bottom: 28 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 0)} h` },
      xAxis: { type: "value", name: "loss of life (hours)" },
      yAxis: { type: "category", data: d.dts, inverse: true, splitLine: { show: false } },
      series: [{ type: "bar", name: "Loss of life 2024", data: tot, barMaxWidth: 12, itemStyle: { borderRadius: [0, 4, 4, 0] }, color: S[0] }],
    };
  }, [d, mode]);

  return (
    <>
      <PageHead title="Transformer thermal"
        sub="IS 2026-7 hot-spot and top-oil model driven by the metered DT load and ERA5 ambient air temperature, plus a day-ahead load forecast turned into a hot-spot forecast." />
      <Gate s={sum} h={84}>
        {(t) => (
          <div className="kpis">
            <Kpi label="Annex E acceptance" value={t.acceptance.passes ? <Pill status="good">Pass</Pill> : <Pill status="critical">Fail</Pill>}
              sub={`θh(30 min) ${fmt.n(t.acceptance.result?.theta_h_c?.["30"], 2)} °C vs ${t.acceptance.expected_theta_h_c?.["30"]}`} />
            <Kpi label="Peak hot-spot 2024" value={fmt.n(t.reference.main.peak_theta_h?.value_c, 1)} unit="°C"
              sub={`${t.reference.main.peak_theta_h?.dt} at ${String(t.reference.main.peak_theta_h?.at_step).slice(0, 16)}`} />
            <Kpi label="DT-days above 98 / 120 / 140 °C" value={`${t.reference.main.event_dt_days_total?.["2024"]?.["98"] ?? "–"} / ${t.reference.main.event_dt_days_total?.["2024"]?.["120"] ?? "–"} / ${t.reference.main.event_dt_days_total?.["2024"]?.["140"] ?? "–"}`} />
            <Kpi label="24 h hot-spot RMSE" value={fmt.n(rmse("lstm", "24h"), 2)} unit="K" sub={`LSTM; persistence ${fmt.n(rmse("persistence", "24h"), 2)} K`} />
            <Kpi label="Persistence − LSTM RMSE" value={fmt.n(boot?.point, 3)} unit="K"
              sub={boot ? `95 % CI ${fmt.n(boot.ci95[0], 2)} to ${fmt.n(boot.ci95[1], 2)}: LSTM ${boot.lstm_better ? "better" : "not better"}` : undefined} />
          </div>
        )}
      </Gate>

      <Card title="Daily maximum hot-spot, every DT, 2024" sub="Click a cell to open that DT and day below.">
        <Gate s={daily}>
          {() => <Chart mode={mode} option={heat} className="chart" ariaLabel="Heatmap of daily max hot-spot per DT"
            onClick={(q) => { if (q?.value) { setDt(d.dts[q.value[1]]); setDay(d.days[q.value[0]]); } }} />}
        </Gate>
      </Card>

      <div className="grid g3">
        <Card className="span2" title={`${dt}, ${dayLabel(shownDay)}`}
          sub={dayData.data?.forecast ? dayData.data.forecast_note : "Reference trajectory. Forecasts exist for test origins 1 May – 30 Dec."}
          right={<div className="toolbar">
            <select className="select" value={dt} onChange={(e) => setDt(e.target.value)} aria-label="Transformer">
              {meta.meters.filter((x: Json) => x.role === "dt").map((x: Json) => <option key={x.load_id}>{x.load_id}</option>)}
            </select>
            <input className="select" type="date" min="2024-01-01" max="2024-12-31" value={shownDay} onChange={(e) => e.target.value && setDay(e.target.value)} aria-label="Day" />
          </div>}>
          <Gate s={dayData}>{() => <Chart mode={mode} option={traj} className="chart tall" ariaLabel="Hot-spot trajectory with forecasts" />}</Gate>
          <div className="honesty">{s?.forecast?.honesty}</div>
        </Card>
        <div className="grid" style={{ alignContent: "start" }}>
          <Card title="Ambient air, same day" sub="ERA5 2 m temperature, 28.6 N 77.2 E (IST)">
            <Gate s={dayData}>{() => ambient ? <Chart mode={mode} option={ambient} className="chart short" ariaLabel="Ambient temperature" /> : <div className="empty">ERA5 file not present.</div>}</Gate>
          </Card>
          <Card title="Hot-spot forecast RMSE" sub="Pooled over 244 test origins and 13 DTs (K)">
            <Gate s={sum}>
              {() => (
                <table>
                  <thead><tr><th>Model</th><th className="num">1 h</th><th className="num">6 h</th><th className="num">24 h</th></tr></thead>
                  <tbody>
                    {MODELS.map(([m, label]) => (
                      <tr key={m}><td>{label}</td>{["1h", "6h", "24h"].map((h) => <td key={h} className="num">{fmt.n(rmse(m, h), 2)}</td>)}</tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Gate>
          </Card>
        </div>
      </div>

      <Card title="Loss of life, 2024" sub="IS 2026-7 ageing hours per DT, summed over the year (reference trajectory).">
        <Gate s={daily}>{() => <Chart mode={mode} option={lol} className="chart" ariaLabel="Loss of life per DT" />}</Gate>
      </Card>
    </>
  );
}
