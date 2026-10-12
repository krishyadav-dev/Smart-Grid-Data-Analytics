"""One-off cross-check of src/data_source/ieee13_data.py against the IEEE package.

Parses the local (git-ignored) plain-text copies in docs/References/_text/IEEE13/
and compares every transcribed value.  Prints one line per value (OK / MISMATCH)
and a summary; exit code 1 on any mismatch.  Not a pytest test: the reference
files are not in the repository.

    python scripts/check_ieee13_transcription.py
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_source import ieee13_data as ieee  # noqa: E402

REF = ROOT / "docs" / "References" / "_text" / "IEEE13"
TXT = REF / "IEEE 13 Node Test Feeder.txt"

results: list[tuple[bool, str]] = []


def check(label: str, got, expected, tol: float = 1e-9) -> None:
    if isinstance(expected, (int, float)) and isinstance(got, (int, float)):
        ok = abs(float(got) - float(expected)) <= tol
    else:
        ok = got == expected
    results.append((ok, f"{'OK      ' if ok else 'MISMATCH'} {label}: "
                        f"module={got!r} source={expected!r}"))


def nums(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d*\.?\d+", s)]


def read_csv(name: str) -> list[list[str]]:
    with open(REF / name, newline="", encoding="utf-8-sig") as fh:
        return [row for row in csv.reader(fh)]


def main() -> int:
    lines = TXT.read_text(encoding="utf-8-sig").splitlines()
    line = lambda n: lines[n - 1]  # 1-based, as cited in ieee13_data  # noqa: E731

    # ---- line configurations: Z and B matrices --------------------------- #
    for cfg, ((l0, _), z_up, b_up) in ieee.LINE_CONFIGS.items():
        check(f"config {cfg} header line {l0}", line(l0).strip(), f"Configuration {cfg}:")
        start = next(i for i in range(l0, l0 + 4) if "Z (R +jX)" in line(i))
        zr = nums(line(start + 1)) + nums(line(start + 2)) + nums(line(start + 3))
        z_src = [complex(zr[i], zr[i + 1]) for i in range(0, 12, 2)]
        bstart = next(i for i in range(start, start + 6) if "B in micro" in line(i))
        b_src = nums(line(bstart + 1)) + nums(line(bstart + 2)) + nums(line(bstart + 3))
        for k, lab in enumerate(["aa", "ab", "ac", "bb", "bc", "cc"]):
            check(f"config {cfg} Z{lab}", z_up[k], z_src[k], 0)
            check(f"config {cfg} B{lab}", b_up[k], b_src[k], 0)

    # ---- phasing (l. 28-40) --------------------------------------------- #
    for cfg, phases in ieee.CONFIG_PHASES.items():
        src = next(l for l in lines[26:40] if l.strip().startswith(str(cfg)))
        ph = "".join(sorted(p.lower() for p in src.split("\t")[1].split() if p != "N"))
        check(f"config {cfg} phases", phases, ph)

    # ---- line segments (Line Data.csv) ---------------------------------- #
    rows = [r for r in read_csv("Line Data.csv")[3:] if r and r[0]]
    src = [(r[0], r[1], int(r[2]), int(r[3]) if r[3].isdigit() else r[3]) for r in rows]
    check("line segment count", len(ieee.LINE_SEGMENTS), len(src))
    for mod, s in zip(ieee.LINE_SEGMENTS, src):
        check(f"segment {s[0]}-{s[1]}", mod, s)

    # ---- transformers (Transformer data.csv) ----------------------------- #
    t = {r[0].strip(): r for r in read_csv("Transformer data.csv")[3:] if r and r[0]}
    for key, mod in (("Substation:", ieee.SUBSTATION_TRANSFORMER), ("XFM -1", ieee.XFM1)):
        r = t[key]
        check(f"{key} kVA", mod["kva"], float(r[1]))
        check(f"{key} kV high", mod["kv_high"], nums(r[2])[0])
        check(f"{key} kV low", mod["kv_low"], nums(r[3])[0])
        check(f"{key} R %", mod["r_pct"], float(r[4]))
        check(f"{key} X %", mod["x_pct"], float(r[5]))

    # ---- capacitors (Cap Data.csv) --------------------------------------- #
    for r in read_csv("Cap Data.csv")[4:]:
        if r and r[0] in ieee.CAPACITORS:
            src_q = tuple(float(x) if x else 0.0 for x in r[1:4])
            check(f"capacitor {r[0]}", ieee.CAPACITORS[r[0]], src_q)

    # ---- regulator (Regulator Data.csv + taps l. 259-261) ---------------- #
    reg = {r[0].strip(): r[1:] for r in read_csv("Regulator Data.csv") if r and r[0]}
    check("regulator segment", "%s - %s" % ieee.REGULATOR["segment"], reg["Line Segment:"][0])
    check("regulator bandwidth V", ieee.REGULATOR["bandwidth_v"], nums(reg["Bandwidth:"][0])[0])
    check("regulator PT ratio", ieee.REGULATOR["pt_ratio"], float(reg["PT Ratio:"][0]))
    check("regulator CT A", ieee.REGULATOR["ct_primary_a"], float(reg["Primary CT Rating:"][0]))
    check("regulator R setting", ieee.REGULATOR["r_setting_v"], tuple(float(x) for x in reg["R - Setting:"]))
    check("regulator X setting", ieee.REGULATOR["x_setting_v"], tuple(float(x) for x in reg["X - Setting:"]))
    check("regulator voltage level", ieee.REGULATOR["voltage_level_v"],
          tuple(float(x) for x in reg["Volltage Level:"]))
    taps = tuple(int(nums(line(n))[-1]) for n in (259, 260, 261))
    check("regulator published taps (l. 259-261)", ieee.REGULATOR["published_taps"], taps)

    # ---- loads (Spot Load Data.csv, Distributed Load Data.csv) ----------- #
    for r in read_csv("Spot Load Data.csv")[4:]:
        if not r or not r[0]:
            if r and r[1] == "TOTAL":
                v = [float(x) for x in r[2:8]]
                check("spot load totals", ieee.SPOT_LOAD_TOTALS,
                      ((v[0], v[2], v[4]), (v[1], v[3], v[5])))
            continue
        conn, model = r[1].split("-")
        v = [float(x) for x in r[2:8]]
        check(f"spot load {r[0]}", ieee.SPOT_LOADS[r[0]],
              (conn, model, (v[0], v[2], v[4]), (v[1], v[3], v[5])))
    r = read_csv("Distributed Load Data.csv")[4]
    conn, model = r[2].split("-")
    v = [float(x) for x in r[3:9]]
    check("distributed load", ieee.DISTRIBUTED_LOAD,
          (r[0], r[1], conn, model, (v[0], v[2], v[4]), (v[1], v[3], v[5])))

    # ---- published results ----------------------------------------------- #
    kw, kvar = nums(line(198)), nums(line(199))
    check("input kW per phase", ieee.PUBLISHED_INPUT_KW, tuple(kw[:3]))
    check("input kW total", ieee.PUBLISHED_INPUT_KW_TOTAL, kw[3])
    check("input kvar per phase", ieee.PUBLISHED_INPUT_KVAR, tuple(kvar[:3]))
    check("input kvar total", ieee.PUBLISHED_INPUT_KVAR_TOTAL, kvar[3])
    check("load kW total", ieee.PUBLISHED_LOAD_KW_TOTAL, nums(line(205))[-1])
    check("load kvar total", ieee.PUBLISHED_LOAD_KVAR_TOTAL, nums(line(208))[-1])
    loss = nums(line(217))
    check("loss kW per phase", ieee.PUBLISHED_LOSS_KW, tuple(loss[:3]))
    check("loss kW total", ieee.PUBLISHED_LOSS_KW_TOTAL, loss[3])
    check("loss kvar total", ieee.PUBLISHED_LOSS_KVAR_TOTAL, nums(line(218))[-1])
    cap = nums(line(226))
    check("capacitor actual kvar per phase", ieee.PUBLISHED_CAP_KVAR, tuple(cap[:3]))
    check("capacitor actual kvar total", ieee.PUBLISHED_CAP_KVAR_TOTAL, cap[3])

    # ---- voltage profile (l. 235-249) ------------------------------------ #
    profile = {}
    for l in lines[234:249]:
        cells = l.split("|")
        node = cells[0].strip()
        phases = []
        for c in cells[1:4]:
            m = re.search(r"(-?\d*\.\d+)\s+at\s+(-?\d*\.\d+)", c)
            phases.append((float(m.group(1)), float(m.group(2))) if m else None)
        profile[node] = tuple(phases)
    for node, mod in ieee.PUBLISHED_VOLTAGES.items():
        for ph, mv, sv in zip("abc", mod, profile[node]):
            check(f"voltage {node} {ph} (mag, angle)", mv, sv)

    for _, msg in results:
        print(msg)
    bad = sum(1 for ok, _ in results if not ok)
    print(f"\n{len(results)} values checked, {len(results) - bad} OK, {bad} MISMATCH")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
