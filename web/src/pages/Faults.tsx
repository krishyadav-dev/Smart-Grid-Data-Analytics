import { useMemo, useState } from "react";
import type { Ctx } from "../App";
import { fmt, p, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import FeederMap from "../components/FeederMap";
import { Card, Gate, Kpi, PageHead, Pill, Seg } from "../components/ui";
import { BLUE_RAMP, CHROME, SERIES, type Mode } from "../theme";

type Filter = "all" | "hit" | "miss";

function confusion(mode: Mode, m: Record<string, Record<string, number>>, order: string[], title: string) {
  const c = CHROME[mode];
  const data: [number, number, number][] = [];
  let max = 0;
  order.forEach((t, y) => order.forEach((pr, x) => {
    const v = m[t]?.[pr] ?? 0;
    max = Math.max(max, v);
    data.push([x, y, v]);
  }));
  return {
    grid: { left: 56, right: 12, top: 12, bottom: 44 },
    tooltip: { formatter: (q: Json) => `true <b>${order[q.value[1]]}</b> → predicted <b>${order[q.value[0]]}</b>: ${q.value[2]}` },
    xAxis: { type: "category", data: order, name: "predicted", nameLocation: "middle", nameGap: 26, splitArea: { show: false } },
    yAxis: { type: "category", data: order, name: "true", nameLocation: "middle", nameGap: 40, inverse: true, splitLine: { show: false } },
    visualMap: { show: false, min: 0, max: Math.max(1, max), inRange: { color: [mode === "dark" ? "#232322" : "#f3f2ee", ...BLUE_RAMP.slice(2)] } },
    series: [{
      type: "heatmap", name: title, data,
      label: { show: true, color: c.ink, fontSize: 11, formatter: (q: Json) => (q.value[2] ? q.value[2] : "") },
      itemStyle: { borderColor: c.surface, borderWidth: 2, borderRadius: 3 },
    }],
  };
}

export default function Faults({ mode, config, layout, meta }: Ctx) {
  const sum = useApi("/api/results/faults");
  const ev = useApi("/api/results/faults/events");
  const [filter, setFilter] = useState<Filter>("all");
  const [ftype, setFtype] = useState<string>("all");
  const [sel, setSel] = useState<number | null>(null);
  const [meter, setMeter] = useState<string | null>(null);

  const events: Json[] = ev.data?.events ?? [];
  const truth = !!ev.data?.truth_available;
  const rows = events.filter((e) =>
    (ftype === "all" || e.ftype === ftype) &&
    (filter === "all" || !truth || (filter === "hit" ? e.zone_hit : !e.zone_hit)));
  const cur = events.find((e) => e.matched === sel) ?? rows[0] ?? null;
  const zoneBuses: Record<string, string[]> = Object.fromEntries(meta.zones.map((z: Json) => [z.id, z.buses]));

  const burst = useApi(cur && truth ? `/api/results/faults/bursts/${cur.matched}` : null);
  const meters: Json[] = burst.data?.available ? burst.data.meters : [];
  const deepest = useMemo(() => {
    let best: string | null = null, v = Infinity;
    for (const m of meters) for (const ph of ["a", "b", "c"]) {
      // ignore de-energised samples (after the breaker clears every meter reads ~0)
      const live = (m.v_pu[ph] as (number | null)[]).filter((y): y is number => y != null && y > 0.05);
      const x = live.length ? Math.min(...live) : Infinity;
      if (x < v) { v = x; best = m.load_id; }
    }
    return best;
  }, [meters]);
  const shown = meters.find((m) => m.load_id === meter) ?? meters.find((m) => m.load_id === deepest) ?? meters[0];

  const burstOpt = (kind: "v_pu" | "i_a") => shown && {
    grid: { left: 52, right: 12, top: 28, bottom: 30 },
    legend: { data: ["Phase a", "Phase b", "Phase c"] },
    tooltip: { trigger: "axis", valueFormatter: (v: number) => (kind === "v_pu" ? `${fmt.n(v, 3)} pu` : `${fmt.n(v, 0)} A`) },
    xAxis: { type: "category", data: burst.data.t_ms, name: "ms", axisLabel: { interval: 19 } },
    yAxis: { type: "value", name: kind === "v_pu" ? "V (pu)" : "I (A)", scale: kind === "v_pu" },
    series: ["a", "b", "c"].map((ph, j) => ({
      type: "line", name: `Phase ${ph}`, data: shown[kind][ph], showSymbol: false, lineStyle: { width: 2 }, color: SERIES[mode][j],
    })),
  };

  const s: Json = sum.data;
  const hl = cur ? { buses: zoneBuses[cur.zone] ?? [], status: !truth ? "neutral" : cur.zone_hit ? "good" : "critical" } as const : null;

  const perZone = useMemo(() => {
    if (!s?.zone) return null;
    const zs = Object.keys(s.zone.per_true_zone);
    return {
      grid: { left: 40, right: 12, top: 16, bottom: 28 },
      tooltip: {
        trigger: "axis",
        formatter: (q: Json) => {
          const z = zs[q[0].dataIndex], v = s.zone.per_true_zone[z];
          return `Zone <b>${z}</b>: ${v.k}/${v.n} (${fmt.pct(v.p, 0)})`;
        },
      },
      xAxis: { type: "category", data: zs.map((z) => `${z} (n=${s.zone.per_true_zone[z].n})`) },
      yAxis: { type: "value", max: 1, axisLabel: { formatter: (v: number) => `${Math.round(v * 100)}%` } },
      series: [{
        type: "bar", name: "Engine", data: zs.map((z) => p(s.zone.per_true_zone[z])), barMaxWidth: 22,
        itemStyle: { borderRadius: [4, 4, 0, 0] }, color: SERIES[mode][0],
      }],
    };
  }, [s, mode]);

  return (
    <>
      <PageHead title="Fault events"
        sub="Rule-based IS 17036 engine on the frozen test half (100 faults + 100 quiet windows, scored once). The engine outputs a zone, not a point: the map highlights the whole predicted zone." />
      <Gate s={sum} h={84}>
        {(d) => (
          <div className="kpis">
            <Kpi label="Detection recall" value={fmt.pct(p(d.detection.recall), 0)} sub={`${d.detection.recall.k}/${d.detection.recall.n} faults`} />
            <Kpi label="Quiet-window alarms" value={fmt.int(d.detection.quiet_windows_with_alarm.k)} sub={`of ${d.detection.quiet_windows_with_alarm.n} windows`} />
            <Kpi label="Type accuracy" value={fmt.pct(p(d.type.accuracy_4class), 0)} sub={`LLG ${fmt.pct(p(d.type.per_type_accuracy.LLG), 0)} is the weak spot`} />
            <Kpi label="Zone accuracy" value={fmt.pct(p(d.zone.accuracy), 0)} sub={`deepest-dip baseline ${fmt.pct(p(d.zone.baseline), 0)}`} />
            <Kpi label="Live alarm latency p95" value={fmt.n(d.live?.pooled?.["900x"]?.alarm_latency_ms?.p95, 1)} unit="ms"
              sub={d.live ? `900x replay, live = batch ${d.live.pooled?.["900x"]?.live_batch_identical ? "✓" : "✕"}` : "live summary not present"} />
          </div>
        )}
      </Gate>

      <div className="grid g2">
        <Card title="Predicted zone on the map"
          sub={cur ? `Event ${cur.matched}: ${cur.ftype} · predicted zone ${cur.zone} (${(zoneBuses[cur.zone] ?? []).join(", ")})${truth ? ` · true bus ${cur.true_bus} (zone ${cur.true_zone})` : ""}` : "Select an event"}
          right={cur && truth ? (cur.zone_hit ? <Pill status="good">Zone hit</Pill> : <Pill status="critical">Zone miss</Pill>) : null}>
          <FeederMap mode={mode} layout={layout} token={config.mapbox_token} highlight={hl} trueBus={truth ? cur?.true_bus : null}
            note={config.map_note}
            legend={<div className="legend-inline" style={{ flexDirection: "column", gap: 4 }}>
              <span><i style={{ background: "#0ca30c" }} />✓ predicted zone, correct</span>
              <span><i style={{ background: "#d03b3b" }} />✕ predicted zone, wrong</span>
              {truth && <span><i style={{ border: `2px solid ${CHROME[mode].ink}`, borderRadius: "50%" }} />true fault bus (simulation ground truth)</span>}
            </div>} />
        </Card>
        <Card title="Test events" sub={truth ? "Joined to the fault set (results/faults_2024): true bus and type are simulation ground truth." : ev.data?.truth_how ? `Truth not present: ${ev.data.truth_how}` : undefined}
          right={<div className="toolbar">
            <select className="select" value={ftype} onChange={(e) => setFtype(e.target.value)} aria-label="Fault type">
              {["all", "SLG", "LL", "LLG", "3PH"].map((t) => <option key={t} value={t}>{t === "all" ? "All types" : t}</option>)}
            </select>
            {truth && <Seg label="Zone result" value={filter} onChange={setFilter} options={[{ v: "all", label: "All" }, { v: "hit", label: "Hits" }, { v: "miss", label: "Misses" }]} />}
          </div>}>
          <Gate s={ev}>
            {() => (
              <div className="tablewrap" style={{ maxHeight: 460 }}>
                <table>
                  <thead><tr>
                    <th>Onset (IST)</th><th>Type</th>{truth && <th>True</th>}<th>Zone</th>{truth && <th>True bus</th>}
                    {truth && <th>Rf</th>}<th className="num">Residual %</th><th className="num">Clear ms</th>{truth && <th>Result</th>}
                  </tr></thead>
                  <tbody>
                    {rows.map((e) => (
                      <tr key={e.matched} className="clickable" aria-selected={cur?.matched === e.matched} onClick={() => { setSel(e.matched); setMeter(null); }}>
                        <td className="mono">{String(e.onset).slice(0, 19)}</td>
                        <td>{e.ftype}</td>
                        {truth && <td>{e.true_ftype}{e.type_hit ? "" : " ✕"}</td>}
                        <td>{e.zone}</td>
                        {truth && <td>{e.true_bus} ({e.true_zone})</td>}
                        {truth && <td>{e.rf_class === "arc" ? "arc" : `${e.rf_class} Ω`}</td>}
                        <td className="num">{fmt.n(e.residual_pct, 1)}</td>
                        <td className="num">{fmt.int(e.clearing_ms)}</td>
                        {truth && <td>{e.zone_hit ? <Pill status="good">hit</Pill> : <Pill status="critical">miss</Pill>}</td>}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Gate>
          <p className="note">{rows.length} of {events.length} events shown.</p>
        </Card>
      </div>

      <Card title="Fault burst" sub="Half-cycle RMS voltage and current at one DT meter (10 ms steps). Defaults to the meter with the deepest dip while energised; readings fall to 0 once the protection clears."
        right={meters.length ? <select className="select" value={shown?.load_id} onChange={(e) => setMeter(e.target.value)} aria-label="Meter">
          {meters.map((m) => <option key={m.load_id} value={m.load_id}>{m.load_id}{m.load_id === deepest ? " (deepest dip)" : ""}</option>)}
        </select> : null}>
        {!truth ? <div className="empty">Burst records need the fault set: <code>python scripts/run_faults.py</code>.</div> : (
          <Gate s={burst}>
            {() => (
              <div className="grid g2">
                <Chart mode={mode} option={burstOpt("v_pu")} ariaLabel="Burst voltage per phase" />
                <Chart mode={mode} option={burstOpt("i_a")} ariaLabel="Burst current per phase" />
              </div>
            )}
          </Gate>
        )}
      </Card>

      <Gate s={sum}>
        {(d) => (
          <div className="grid g3">
            <Card title="Type confusion" sub="Rows: true type, columns: predicted (test half)">
              <Chart mode={mode} option={confusion(mode, d.type.confusion_true_rows, ["SLG", "LL", "LLG", "3PH"], "type")} ariaLabel="Fault type confusion matrix" />
            </Card>
            <Card title="Zone confusion" sub="Rows: true zone, columns: predicted. Zone E (671, 680, 692) cannot be split.">
              <Chart mode={mode} option={confusion(mode, d.zone.confusion_true_rows, "ABCDEFG".split(""), "zone")} ariaLabel="Zone confusion matrix" />
            </Card>
            <Card title="Zone accuracy by true zone" sub="Small n per zone: read with the counts.">
              <Chart mode={mode} option={perZone} ariaLabel="Zone accuracy per true zone" />
            </Card>
          </div>
        )}
      </Gate>
    </>
  );
}
