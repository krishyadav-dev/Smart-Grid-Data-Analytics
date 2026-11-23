import { useEffect, useMemo, useState } from "react";
import type { Ctx } from "../App";
import { fmt, get, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import { Card, Gate, Kpi, PageHead, Pill } from "../components/ui";
import { SERIES } from "../theme";

const SOURCES: [string, string][] = [
  ["/api/results/faults/events", "Fault events + fault set (truth)"],
  ["/api/results/grid/daily", "Simulated year state (results/year_2024)"],
  ["/api/results/unbalance/daily", "Metered meter frame (results/year_2024)"],
  ["/api/results/theft/dt-daily", "Theft per DT-day"],
  ["/api/results/theft/inspection?limit=1", "Jokar test scores"],
  ["/api/results/thermal/daily", "Thermal reference trajectories"],
];

export default function System({ mode, config, layout }: Ctx) {
  const ing = useApi("/api/results/ingestion");
  const faults = useApi("/api/results/faults");
  const [avail, setAvail] = useState<Record<string, Json>>({});

  useEffect(() => {
    SOURCES.forEach(([u]) => get(u).then((d) => setAvail((a) => ({ ...a, [u]: d }))).catch((e) => setAvail((a) => ({ ...a, [u]: { available: false, how: String(e) } }))));
  }, []);

  const rows = useMemo(() => {
    const out: Json[] = [];
    for (const r of ing.data?.runs ?? []) for (const x of r.results) out.push({ file: r.file, os: r.metadata?.os, profile: r.metadata?.profiles, ...x });
    return out;
  }, [ing.data]);

  const lat = useMemo(() => {
    if (!rows.length) return null;
    const cats = rows.map((r) => `${r.file.replace("bench_", "").slice(0, 8)} · ${r.playback_speed}x`);
    const q = (k: string) => rows.map((r) => r.latency_ms?.total?.[k] ?? null);
    return {
      grid: { left: 52, right: 16, top: 32, bottom: 48 },
      legend: {},
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 1)} ms` },
      xAxis: { type: "category", data: cats, axisLabel: { rotate: 0, fontSize: 10 } },
      yAxis: { type: "value", name: "ms, poll → subscriber" },
      series: [["p50", 0], ["p95", 1], ["p99", 2]].map(([k, i]) => ({
        type: "bar", name: k, data: q(k as string), barMaxWidth: 14, barGap: "15%", itemStyle: { borderRadius: [4, 4, 0, 0] }, color: SERIES[mode][i as number],
      })),
    };
  }, [rows, mode]);

  const live: Json = faults.data?.live;

  return (
    <>
      <PageHead title="System" sub="Ingestion benchmarks (Layer 2), the live-replay check of the fault engine, and which regenerable data files this checkout has." />

      <div className="kpis">
        <Kpi label="Mapbox token" value={config.mapbox_token ? <Pill status="good">Set</Pill> : <Pill status="neutral">Not set</Pill>}
          sub={config.mapbox_token ? "base maps on" : "feeder layers only"} />
        <Kpi label="Map layout" value={layout.metadata?.mode === "mapbox" ? "Road-routed" : "Schematic"}
          sub={`${layout.features.length} features, worst length ratio ${fmt.n(Math.max(...Object.values(layout.metadata?.length_ratio ?? { x: 1 }).map((v) => Math.abs((v as number) - 1))) * 100, 1)} %`} />
        <Kpi label="Live replay" value={live ? (live.passed ? <Pill status="good">Passed</Pill> : <Pill status="critical">Failed</Pill>) : "–"}
          sub={live ? `${Object.keys(live.pooled).join(", ")} replays` : "summary not present"} />
        <Kpi label="Mode" value="Report" sub="live mode (MQTT + WebSocket) is phase P1" />
      </div>

      <div className="grid g2">
        <Card title="Ingestion latency" sub="Gateway poll to validated subscriber, per benchmark run and playback speed.">
          <Gate s={ing}>{() => <Chart mode={mode} option={lat} className="chart tall" ariaLabel="Ingestion latency percentiles" />}</Gate>
        </Card>
        <Card title="Data files in this checkout" sub="Report pages degrade to a notice when a git-ignored file is missing.">
          <table>
            <thead><tr><th>Source</th><th>Status</th><th>How to make it</th></tr></thead>
            <tbody>
              {SOURCES.map(([u, label]) => {
                const a = avail[u];
                return (
                  <tr key={u}><td>{label}</td>
                    <td>{!a ? <Pill status="neutral">checking</Pill> : a.available === false ? <Pill status="warning">missing</Pill> : <Pill status="good">present</Pill>}</td>
                    <td className="note" style={{ whiteSpace: "normal" }}>{a?.available === false ? a.how : ""}</td></tr>
                );
              })}
            </tbody>
          </table>
        </Card>
      </div>

      <Card title="Benchmark runs">
        <Gate s={ing}>
          {() => (
            <div className="tablewrap">
              <table>
                <thead><tr><th>Run</th><th>Profile</th><th className="num">Speed</th><th className="num">Expected</th><th className="num">Accepted</th><th className="num">Loss</th>
                  <th className="num">Msgs/s</th><th className="num">p50 ms</th><th className="num">p95 ms</th><th className="num">p99 ms</th><th>OS</th></tr></thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}><td className="mono">{r.file}</td><td>{r.profile}</td><td className="num">{r.playback_speed}x</td>
                      <td className="num">{fmt.int(r.expected_messages)}</td><td className="num">{fmt.int(r.accepted)}</td><td className="num">{fmt.pct(r.loss_fraction, 2)}</td>
                      <td className="num">{fmt.n(r.throughput_msgs_per_s, 1)}</td>
                      <td className="num">{fmt.n(r.latency_ms?.total?.p50, 1)}</td><td className="num">{fmt.n(r.latency_ms?.total?.p95, 1)}</td><td className="num">{fmt.n(r.latency_ms?.total?.p99, 1)}</td>
                      <td className="note">{r.os}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Gate>
      </Card>

      {live && (
        <Card title="Fault engine live replay" sub="Recorded weeks replayed through MQTT at 1x and 900x; live events must equal the batch run.">
          <div className="tablewrap">
            <table>
              <thead><tr><th>Speed</th><th className="num">Faults</th><th className="num">Alarms</th><th className="num">Bursts</th><th className="num">Telemetry</th><th className="num">Alarm p50 ms</th><th className="num">Alarm p95 ms</th><th>Live = batch</th></tr></thead>
              <tbody>
                {Object.entries(live.pooled).map(([k, v]: [string, Json]) => (
                  <tr key={k}><td>{k}</td><td className="num">{v.faults}</td><td className="num">{v.alarms}</td>
                    <td className="num">{v.bursts}/{v.bursts_expected}</td><td className="num">{v.telemetry}/{v.telemetry_expected}</td>
                    <td className="num">{fmt.n(v.alarm_latency_ms?.p50, 1)}</td><td className="num">{fmt.n(v.alarm_latency_ms?.p95, 1)}</td>
                    <td>{v.live_batch_identical ? <Pill status="good">identical</Pill> : <Pill status="critical">differs</Pill>}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}
