import { useMemo, useState } from "react";
import type { Ctx } from "../App";
import { dayLabel, fmt, monthAxis, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import { Card, Gate, Kpi, PageHead, Pill, Seg } from "../components/ui";
import { BLUE_RAMP, CHROME, SERIES } from "../theme";

type Stat = "max_pct" | "p95_pct";

export default function PowerQuality({ mode }: Ctx) {
  const daily = useApi("/api/results/unbalance/daily");
  const sum = useApi("/api/results/unbalance");
  const [stat, setStat] = useState<Stat>("p95_pct");
  const c = CHROME[mode];
  const d: Json = daily.data;
  const s: Json = sum.data;

  const heat = useMemo(() => {
    if (!d?.available) return null;
    const ids: string[] = d.meters.map((m: Json) => m.load_id);
    const labels = d.meters.map((m: Json) => `${m.meter_id} ${m.load_id}`);
    const data: [number, number, number | null][] = [];
    ids.forEach((k, y) => (d[stat][k] as (number | null)[]).forEach((v, x) => data.push([x, y, v])));
    return {
      grid: { left: 130, right: 16, top: 8, bottom: 56 },
      tooltip: { formatter: (q: Json) => `${labels[q.value[1]]}, ${dayLabel(d.days[q.value[0]])}<br/>VU ${stat === "max_pct" ? "max" : "p95"} <b>${fmt.n(q.value[2], 2)} %</b> (limit ${d.limit_pct} %)` },
      xAxis: { ...monthAxis(d.days), splitArea: { show: false } },
      yAxis: { type: "category", data: labels, inverse: true, splitLine: { show: false } },
      visualMap: {
        type: "piecewise", orient: "horizontal", left: "center", bottom: 0, itemWidth: 14, itemHeight: 10,
        textStyle: { color: c.ink2, fontSize: 11 },
        pieces: [
          { lt: 0.5, color: BLUE_RAMP[0], label: "< 0.5 %" },
          { gte: 0.5, lt: 1, color: BLUE_RAMP[2], label: "0.5–1 %" },
          { gte: 1, lt: 1.5, color: BLUE_RAMP[3], label: "1–1.5 %" },
          { gte: 1.5, lt: 2, color: BLUE_RAMP[5], label: "1.5–2 %" },
          { gte: 2, color: "#d03b3b", label: "≥ 2 % (above limit)" },
        ],
      },
      series: [{ type: "heatmap", data, progressive: 0 }],
    };
  }, [d, stat, mode]);

  const scen = useMemo(() => {
    if (!s?.per_s_per_meter) return null;
    const ss = Object.keys(s.per_s_per_meter).sort((a, b) => Number(a) - Number(b));
    const dts = Object.keys(s.per_s_per_meter[ss[0]]);
    const pick = ["dt_634_1", "dt_675_1", "dt_611_1"].filter((x) => dts.includes(x));
    return {
      grid: { left: 44, right: 16, top: 32, bottom: 36 },
      legend: {},
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 2)} %` },
      xAxis: { type: "category", data: ss, name: "scenario severity s", nameLocation: "middle", nameGap: 24 },
      yAxis: { type: "value", name: "VU max %" },
      series: [
        ...pick.map((k, i) => ({ type: "line", name: k, data: ss.map((x) => s.per_s_per_meter[x][k]?.vu_max_pct ?? null), showSymbol: true, symbolSize: 8, lineStyle: { width: 2 }, color: SERIES[mode][i] })),
      ].map((x, i) => (i === 0 ? { ...x, markLine: { silent: true, symbol: "none", data: [{ yAxis: s.limit_pct, label: { formatter: `limit ${s.limit_pct} %`, color: c.muted, fontSize: 10 }, lineStyle: { color: "#d03b3b", width: 1, type: "solid" } }] } } : x)),
    };
  }, [s, mode]);

  const per = s?.year_2024_per_meter?.per_meter as Record<string, Json> | undefined;

  return (
    <>
      <PageHead title="Power quality"
        sub="Voltage unbalance (IS 17036:2018 Cl. 4.2.4: 2 % for 95 % of each week) at the feeder head and the 13 DT LV supply points, from the metered 15-min voltages of the simulated year."
        right={<Seg label="Daily statistic" value={stat} onChange={setStat} options={[{ v: "p95_pct", label: "Daily p95" }, { v: "max_pct", label: "Daily max" }]} />} />

      <Gate s={sum} h={84}>
        {(u) => (
          <div className="kpis">
            <Kpi label="Feeder head VU max, 2024" value={fmt.n(u.year_2024_feeder_head?.truth_vu_max_pct, 2)} unit="%"
              sub={`${u.year_2024_feeder_head?.truth_weeks_exceeding ?? "–"} of ${u.year_2024_feeder_head?.weeks ?? "–"} weeks exceeding`} />
            <Kpi label="Engine weekly alarms, 2024" value={per ? fmt.int(Object.values(per).reduce((a, v) => a + v.engine_weekly_alarms, 0)) : "–"}
              sub={per ? `truth weeks exceeding: ${Object.values(per).reduce((a, v) => a + v.truth_weeks_exceeding, 0)}` : undefined} />
            <Kpi label="Scenario week" value={u.n_scenarios} unit="scenarios" sub={`${String(u.week?.[0]).slice(0, 10)}; baseline check ${u.baseline_check_passed ? "✓" : "✕"}`} />
            <Kpi label="Scenario feeder-head VU max" value={fmt.n(u.feeder_head_vu_max_pct, 2)} unit="%" sub="worst injected scenario" />
          </div>
        )}
      </Gate>

      <Card title="Voltage unbalance per meter and day, 2024" sub={d?.method}>
        <Gate s={daily}>{() => <Chart mode={mode} option={heat} className="chart tall" ariaLabel="Voltage unbalance heatmap" />}</Gate>
      </Card>

      <div className="grid g2">
        <Card title="Weekly compliance per meter, 2024" sub={s?.year_2024_per_meter?.label}>
          <Gate s={sum}>
            {() => per ? (
              <div className="tablewrap">
                <table>
                  <thead><tr><th>Meter</th><th className="num">Weeks</th><th className="num">VU max %</th><th className="num">Engine alarms</th><th className="num">Truth exceeding</th><th>Agreement</th></tr></thead>
                  <tbody>
                    {Object.entries(per).map(([k, v]) => (
                      <tr key={k}><td>{k}</td><td className="num">{v.weeks}</td><td className="num">{fmt.n(v.vu_max_pct_engine, 2)}</td>
                        <td className="num">{v.engine_weekly_alarms}</td><td className="num">{v.truth_weeks_exceeding}</td>
                        <td>{v.agreement_weeks === v.weeks ? <Pill status="good">{v.agreement_weeks}/{v.weeks}</Pill> : <Pill status="warning">{v.agreement_weeks}/{v.weeks}</Pill>}</td></tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <div className="empty">Per-meter 2024 summary not present.</div>}
          </Gate>
        </Card>
        <Card title="Injected unbalance scenarios" sub={s?.note}>
          <Gate s={sum}>{() => <Chart mode={mode} option={scen} className="chart tall" ariaLabel="VU max against scenario severity" />}</Gate>
        </Card>
      </div>
    </>
  );
}
