import { useMemo, useState } from "react";
import type { Ctx } from "../App";
import { fmt, monthAxis, p, useApi, type Json } from "../api";
import Chart from "../components/Chart";
import FeederMap, { RampLegend } from "../components/FeederMap";
import { Card, Gate, Kpi, PageHead, Pill } from "../components/ui";
import { CHROME, ORANGE_RAMP, SERIES, binColor } from "../theme";

const mean = (a: (number | null)[]) => {
  const v = a.filter((x): x is number => x !== null && isFinite(x));
  return v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
};

export default function Theft({ mode, config, layout, meta }: Ctx) {
  const sum = useApi("/api/results/theft");
  const daily = useApi("/api/results/theft/dt-daily");
  const [dt, setDt] = useState("dt_634_1");
  const insp = useApi(`/api/results/theft/inspection?dt=${dt}&limit=25`);
  const c = CHROME[mode];
  const S = SERIES[mode];
  const s: Json = sum.data;
  const d: Json = daily.data;

  // test-window mean daily estimate per DT -> service-area shading
  const area = useMemo(() => {
    if (!d?.available) return null;
    const i0 = d.days.indexOf(d.test_window[0]);
    const est: Record<string, number | null> = {};
    for (const k of d.dts) est[k] = mean(d.estimate_kwh[k].slice(i0));
    const hi = Math.max(...Object.values(est).map((x) => x ?? 0));
    return { est, hi: Math.ceil(hi / 10) * 10 };
  }, [d]);

  const sankey = useMemo(() => {
    if (!s?.worked_example) return null;
    const w = s.worked_example;
    const root = "Unbilled (head − billed)";
    const nodes = [root, "Line losses (est.)", "DT losses (est.)", "Offline (est.)", "NTL estimate"];
    const vals = [w.technical_losses_lines, w.technical_losses_dts, w.offline, w.netted_estimate];
    const col = [S[0], c.axis, c.axis, S[3], S[1]];
    return {
      tooltip: { trigger: "item", valueFormatter: (v: number) => `${fmt.n(v, 1)} MWh` },
      series: [{
        type: "sankey", left: 150, right: 170, top: 12, bottom: 12, nodeWidth: 12, nodeGap: 18, draggable: false,
        layoutIterations: 0,
        data: nodes.map((n, i) => ({ name: n, itemStyle: { color: col[i], borderWidth: 0 },
          label: i === 0 ? { position: "left" } : undefined })),
        links: vals.map((v, i) => ({ source: root, target: nodes[i + 1], value: v })),
        lineStyle: { color: "target", opacity: 0.3, curveness: 0.5 },
        label: { color: c.ink, fontSize: 12, formatter: (q: Json) => `${q.name}\n${fmt.n(q.value, 0)} MWh` },
        emphasis: { focus: "adjacency" },
      }],
    };
  }, [s, mode]);

  const dtChart = useMemo(() => {
    if (!d?.available) return null;
    const series: Json[] = [
      { type: "line", name: "Engine estimate (netted, λ = 0)", data: d.estimate_kwh[dt], showSymbol: false, lineStyle: { width: 2 }, color: S[0],
        markArea: { silent: true, itemStyle: { color: mode === "dark" ? "rgba(255,255,255,0.04)" : "rgba(11,11,11,0.035)" },
          label: { show: true, position: "insideTopLeft", color: c.muted, fontSize: 10, formatter: "test window (scored once)" },
          data: [[{ xAxis: d.test_window[0] }, { xAxis: d.test_window[1] }]] } },
    ];
    if (d.truth_available) series.push({ type: "line", name: "True energy theft h1–h4 (ground truth)", data: d.true_theft_kwh[dt], showSymbol: false, lineStyle: { width: 2 }, color: S[1] });
    return {
      grid: { left: 48, right: 12, top: 32, bottom: 28 },
      legend: {},
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 1)} kWh` },
      xAxis: monthAxis(d.days),
      yAxis: { type: "value", name: "kWh/day" },
      series,
    };
  }, [d, dt, mode]);

  const lambda = useMemo(() => {
    if (!s?.lambda_sweep_dt) return null;
    const ks = Object.keys(s.lambda_sweep_dt);
    return {
      grid: { left: 52, right: 12, top: 16, bottom: 36 },
      tooltip: { trigger: "axis", valueFormatter: (v: number) => `${fmt.n(v, 1)} kWh` },
      xAxis: { type: "category", data: ks, name: "assumed λ (true λ = 0)", nameLocation: "middle", nameGap: 24 },
      yAxis: { type: "value", name: "bias, kWh per DT-day" },
      series: [{ type: "bar", name: "Bias", data: ks.map((k) => s.lambda_sweep_dt[k].bias_kwh), barMaxWidth: 26,
        itemStyle: { borderRadius: 4 }, color: S[0] }],
    };
  }, [s, mode]);

  const ins: Json = insp.data;

  return (
    <>
      <PageHead title="Energy balance and theft"
        sub={<>Engine energy balance (head → billed + technical losses + offline + theft) and the Jokar consumer detector. {s ? <span className="tag">{s.injection === "ntl_inj2" ? "theft injection 2 (Stage 4)" : "theft injection 1"}</span> : null}</>} />

      <Gate s={sum} h={84}>
        {(t) => (
          <div className="kpis">
            <Kpi label="Energy identity, Jul–Dec" value={t.energy_identity.closes ? "Closes" : "Open"}
              sub={`residual ${fmt.n(t.energy_identity.residual_kwh, 2)} kWh over ${fmt.int(t.energy_identity.steps)} steps`} />
            {t.annual_mwh?.jul_dec_test && <Kpi label="Theft estimate, Jul–Dec" value={fmt.n(t.annual_mwh.jul_dec_test.bottom_up_primary, 0)} unit="MWh"
              sub={`true ${fmt.n(t.annual_mwh.jul_dec_test.true_energy_theft_h1_h4, 0)} MWh (ground truth)`} />}
            {(() => {
              const fd = t.feeder.jul_dec_test.bottom_up_primary ?? t.feeder.jul_dec_test.netted;
              return <Kpi label="Feeder daily MAE" value={fmt.n(fd?.mae_kwh, 0)} unit="kWh"
                sub={`${t.feeder.jul_dec_test.bottom_up_primary ? "bottom-up Σ DT" : "head netted"}, corr ${fmt.n(fd?.corr, 2)}`} />;
            })()}
            <Kpi label="DT-day MAE" value={fmt.n(t.dt.jul_dec_test.netted_lambda_0?.mae_kwh, 1)} unit="kWh"
              sub={`corr ${fmt.n(t.dt.jul_dec_test.netted_lambda_0?.corr, 2)}, n = ${fmt.int(t.dt.jul_dec_test.netted_lambda_0?.n)}`} />
            <Kpi label="Jokar precision h1–h4" value={fmt.pct(t.jokar.macro_energy_theft_h1_h4.precision, 1)}
              sub={`recall ${fmt.pct(t.jokar.macro_energy_theft_h1_h4.recall, 0)} (macro, ${t.jokar.variant}, ${t.jokar.gate})`} />
            {t.zheng && <Kpi label="Zheng CNN AUC (SGCC)" value={fmt.n(t.zheng.mean_and_range.auc.mean, 3)}
              sub={`range ${fmt.n(t.zheng.mean_and_range.auc.min, 3)}–${fmt.n(t.zheng.mean_and_range.auc.max, 3)}; paper ${t.zheng.paper.auc}`} />}
          </div>
        )}
      </Gate>

      <div className="grid g2">
        <Card title="Where the unbilled energy goes, 2024"
          sub={s ? `Head ${fmt.n(s.worked_example.head_energy, 0)} MWh, billed ${fmt.n(s.worked_example.household_metered, 0)} MWh (${fmt.pct(s.worked_example.household_metered / s.worked_example.head_energy, 1)}). The engine splits the rest; the NTL estimate is what remains. True energy theft h1–h4: ${fmt.n(s.worked_example.true_energy_theft_h1_h4, 0)} MWh (ground truth).` : undefined}>
          <Gate s={sum}>{() => <Chart mode={mode} option={sankey} className="chart tall" ariaLabel="Energy flow from feeder head" />}</Gate>
        </Card>
        <Card title="Estimated theft by DT area" sub="Service areas shaded by the mean daily engine estimate over the test window. Click a DT to inspect it.">
          <FeederMap mode={mode} layout={layout} token={config.mapbox_token} note={config.map_note}
            areaColor={area ? Object.fromEntries(Object.entries(area.est).map(([k, v]) => [k, binColor(v, 0, area.hi, ORANGE_RAMP)])) : undefined}
            dtColor={area ? Object.fromEntries(Object.entries(area.est).map(([k, v]) => [k, binColor(v, 0, area.hi, ORANGE_RAMP)])) : undefined}
            dtInfo={area ? Object.fromEntries(Object.entries(area.est).map(([k, v]) => [k, `${fmt.n(v, 1)} kWh/day (est.)`])) : undefined}
            selectedDt={dt} onSelectDt={setDt}
            legend={area ? <RampLegend title="Mean NTL estimate, Jul–Dec" ramp={ORANGE_RAMP} lo="0" hi={`${area.hi}`} unit="kWh/day" /> : <b>Daily DT estimates not present</b>} />
        </Card>
      </div>

      <Card title={`Daily NTL at ${dt}`}
        sub={d?.truth_available ? d.truth_note : "Engine estimate only (ground-truth labels not present)."}
        right={<select className="select" value={dt} onChange={(e) => setDt(e.target.value)} aria-label="Transformer">
          {meta.meters.filter((x: Json) => x.role === "dt").map((x: Json) => <option key={x.load_id}>{x.load_id}</option>)}
        </select>}>
        <Gate s={daily}>{() => <Chart mode={mode} option={dtChart} className="chart tall" ariaLabel={`Daily theft estimate and truth for ${dt}`} />}</Gate>
      </Card>

      <div className="grid g2">
        <Card title={`Inspection list, ${dt}`} sub={ins?.rule ?? "Jokar SVM ranking over the test window"}>
          <div className="honesty">{ins?.honesty ?? "Precision is about 2 %: an order to inspect in, not a verdict."}</div>
          <Gate s={insp}>
            {(r) => (
              <>
                {r.per_dt?.[dt] && <p className="note">
                  {r.per_dt[dt].flagged_by_any_fold} of {r.per_dt[dt].households_scored} households flagged by at least one fold, {r.per_dt[dt].flagged_by_all_folds} by all four
                  {r.truth_available ? `; ${r.per_dt[dt].true_energy_thieves} true energy thieves in this area (ground truth)` : ""}.
                </p>}
                <div className="tablewrap">
                  <table>
                    <thead><tr><th>#</th><th>Household</th><th className="num">Folds flagging</th>
                      {r.folds.map((f: string) => <th key={f} className="num">{f}</th>)}
                      {r.truth_available && <th>Truth</th>}</tr></thead>
                    <tbody>
                      {r.rows.map((x: Json, i: number) => (
                        <tr key={x.household}>
                          <td className="mono">{i + 1}</td><td className="mono">{x.household}</td>
                          <td className="num">{x.folds_flagging}/4</td>
                          {r.folds.map((f: string) => <td key={f} className="num">{x.scores[f] == null ? "–" : fmt.n(x.scores[f], 2)}</td>)}
                          {r.truth_available && <td>{x.true_attack ? <Pill status="serious">{x.true_attack}</Pill> : <span className="note">honest</span>}</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="note">Fold score = share of online test days flagged; q per fold: {Object.entries(r.q).map(([f, q]) => `${f} ${fmt.n(q as number, 2)}`).join(", ")}.</p>
              </>
            )}
          </Gate>
        </Card>
        <div className="grid" style={{ alignContent: "start" }}>
          <Card title="Jokar detector per fold" sub={s?.jokar?.label}>
            <Gate s={sum}>
              {(t) => (
                <div className="tablewrap">
                  <table>
                    <thead><tr><th>Fold</th><th className="num">Positives</th><th className="num">TP</th><th className="num">FP</th><th className="num">Precision</th><th className="num">Recall</th></tr></thead>
                    <tbody>
                      {Object.entries(t.jokar.per_fold).map(([f, v]: [string, Json]) => (
                        <tr key={f}><td>{f}{["h5", "h6"].includes(f) ? " (tariff shift)" : ""}</td><td className="num">{v.positives}</td><td className="num">{v.tp}</td><td className="num">{v.fp}</td>
                          <td className="num">{fmt.pct(p(v.precision), 1)}</td><td className="num">{fmt.pct(p(v.recall), 0)}</td></tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </Gate>
            {s?.jokar?.b8_note && <p className="note">{s.jokar.b8_note}</p>}
          </Card>
          <Card title="Sensitivity to the assumed loss factor λ" sub="Bias of the DT-day estimate on the test window when the engine assumes λ but the truth is 0.">
            <Gate s={sum}>{() => <Chart mode={mode} option={lambda} className="chart short" ariaLabel="Bias against assumed lambda" />}</Gate>
          </Card>
          {s?.zheng && (
            <Card title="Zheng Wide & Deep CNN" sub={s.zheng.label}>
              <div className="kpis">
                <Kpi label="AUC" value={fmt.n(s.zheng.mean_and_range.auc.mean, 3)} sub={`paper ${s.zheng.paper.auc}`} />
                <Kpi label="MAP@100" value={fmt.n(s.zheng.mean_and_range.map_at_100?.mean, 3)} sub={`paper ${s.zheng.paper.map_at_100}`} />
                <Kpi label="MAP@200" value={fmt.n(s.zheng.mean_and_range.map_at_200?.mean, 3)} sub={`paper ${s.zheng.paper.map_at_200}`} />
              </div>
              <p className="note">{s.zheng.note}</p>
            </Card>
          )}
        </div>
      </div>
    </>
  );
}
