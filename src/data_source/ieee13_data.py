"""
Published IEEE 13-node test feeder data (Stage A source of truth).

Transcribed from the IEEE PES Distribution System Analysis Subcommittee package
"IEEE 13 Node Test Feeder" (2004).  Every block cites its line number in the
plain-text copy ``docs/References/_text/IEEE13/IEEE 13 Node Test Feeder.txt``
(local, git-ignored).  ``scripts/check_ieee13_transcription.py`` cross-checks
every value here against that file and the CSV tables next to it.

Units are as published: impedances in ohm/mile, susceptances in uS/mile,
lengths in feet, powers in kW / kvar, voltages in pu on the node's base.
Phase order of every matrix and tuple is (a, b, c).
"""

from __future__ import annotations

SOURCE_FILE = "IEEE 13 Node Test Feeder.txt"

# Base voltages (l. 272 "kVll 4.160"; l. 310 "kVll .480")
V_PRIMARY_KV = 4.16
V_XFM1_LV_KV = 0.48
FREQUENCY_HZ = 60.0

# --------------------------------------------------------------------------- #
# Line configurations                                                          #
# --------------------------------------------------------------------------- #
# Phases present per config (l. 28-40: phasing column; N = neutral, Kron-reduced).
CONFIG_PHASES = {
    601: "abc",  # l. 30 "B A C N"
    602: "abc",  # l. 31 "C A B N"
    603: "bc",   # l. 32 "C B N"
    604: "ac",   # l. 33 "A C N"
    605: "c",    # l. 34 "C N"
    606: "abc",  # l. 39 "A B C N"
    607: "a",    # l. 40 "A N"
}

# Upper triangle (aa, ab, ac, bb, bc, cc) of Z = R + jX in ohm/mile and of
# B in uS/mile.  {config: (source lines, Z upper triangle, B upper triangle)}
LINE_CONFIGS = {
    601: ((113, 122),
          (0.3465 + 1.0179j, 0.1560 + 0.5017j, 0.1580 + 0.4236j,
           0.3375 + 1.0478j, 0.1535 + 0.3849j, 0.3414 + 1.0348j),
          (6.2998, -1.9958, -1.2595, 5.9597, -0.7417, 5.6386)),
    602: ((124, 133),
          (0.7526 + 1.1814j, 0.1580 + 0.4236j, 0.1560 + 0.5017j,
           0.7475 + 1.1983j, 0.1535 + 0.3849j, 0.7436 + 1.2112j),
          (5.6990, -1.0817, -1.6905, 5.1795, -0.6588, 5.4246)),
    603: ((135, 144),
          (0j, 0j, 0j,
           1.3294 + 1.3471j, 0.2066 + 0.4591j, 1.3238 + 1.3569j),
          (0.0, 0.0, 0.0, 4.7097, -0.8999, 4.6658)),
    604: ((146, 155),
          (1.3238 + 1.3569j, 0j, 0.2066 + 0.4591j,
           0j, 0j, 1.3294 + 1.3471j),
          (4.6658, 0.0, -0.8999, 0.0, 0.0, 4.7097)),
    605: ((156, 165),
          (0j, 0j, 0j,
           0j, 0j, 1.3292 + 1.3475j),
          (0.0, 0.0, 0.0, 0.0, 0.0, 4.5193)),
    606: ((167, 176),
          (0.7982 + 0.4463j, 0.3192 + 0.0328j, 0.2849 - 0.0143j,
           0.7891 + 0.4041j, 0.3192 + 0.0328j, 0.7982 + 0.4463j),
          (96.8897, 0.0, 0.0, 96.8897, 0.0, 96.8897)),
    607: ((178, 187),
          (1.3425 + 0.5124j, 0j, 0j,
           0j, 0j, 0j),
          (88.9912, 0.0, 0.0, 0.0, 0.0, 0.0)),
}

# Ampacity in kA per config: Kersting, "Radial Distribution Test Feeders",
# Table 3 (overhead ACSR at 50 degC), Table 5 (250 kcmil AA CN cable in duct),
# Table 6 (1/0 AA tape-shielded cable); conductor per config from l. 28-40.
# See docs/References/README.md, Verified locations.
CONFIG_AMPACITY_KA = {
    601: 0.730,  # 556.5 kcmil ACSR
    602: 0.340,  # 4/0 ACSR
    603: 0.230,  # 1/0 ACSR
    604: 0.230,
    605: 0.230,
    606: 0.260,  # 250 kcmil AA, concentric neutral
    607: 0.165,  # 1/0 AA, tape shielded
}

# --------------------------------------------------------------------------- #
# Line segments (l. 42-56): (node A, node B, length ft, config or element)     #
# --------------------------------------------------------------------------- #
LINE_SEGMENTS = [
    ("632", "645", 500, 603),          # l. 45
    ("632", "633", 500, 602),          # l. 46
    ("633", "634", 0, "XFM-1"),        # l. 47
    ("645", "646", 300, 603),          # l. 48
    ("650", "632", 2000, 601),         # l. 49
    ("684", "652", 800, 607),          # l. 50
    ("632", "671", 2000, 601),         # l. 51
    ("671", "684", 300, 604),          # l. 52
    ("671", "680", 1000, 601),         # l. 53
    ("671", "692", 0, "Switch"),       # l. 54
    ("684", "611", 300, 605),          # l. 55
    ("692", "675", 500, 606),          # l. 56
]

# --------------------------------------------------------------------------- #
# Transformers (l. 58-62): kVA, kV high, kV low, R %, X %, connection          #
# --------------------------------------------------------------------------- #
SUBSTATION_TRANSFORMER = {  # l. 61; used from Stage B step B4 only
    "kva": 5000, "kv_high": 115.0, "kv_low": 4.16, "r_pct": 1.0, "x_pct": 8.0,
    "connection": "D / Gr.Y",
}
XFM1 = {  # l. 62
    "kva": 500, "kv_high": 4.16, "kv_low": 0.48, "r_pct": 1.1, "x_pct": 2.0,
    "connection": "Gr.W / Gr.W",
}

# --------------------------------------------------------------------------- #
# Capacitors (l. 64-70): kvar per phase (a, b, c)                              #
# --------------------------------------------------------------------------- #
CAPACITORS = {
    "675": (200.0, 200.0, 200.0),   # l. 68
    "611": (0.0, 0.0, 100.0),       # l. 69
}

# --------------------------------------------------------------------------- #
# Regulator (l. 73-87) and published final taps (l. 258-261)                   #
# --------------------------------------------------------------------------- #
REGULATOR = {
    "segment": ("650", "632"),      # l. 76
    "bandwidth_v": 2.0,             # l. 81
    "pt_ratio": 20.0,               # l. 82
    "ct_primary_a": 700.0,          # l. 83
    "r_setting_v": (3.0, 3.0, 3.0), # l. 85
    "x_setting_v": (9.0, 9.0, 9.0), # l. 86
    "voltage_level_v": (122.0, 122.0, 122.0),  # l. 87
    "published_taps": (10, 8, 11),  # l. 259-261
}
# Step size: standard 32-step +/-10 % regulator, 0.625 %/step.  Not printed in
# the package; implied by the published RG60 voltages (l. 236): 1 + 10*0.00625
# = 1.0625, 1 + 8*0.00625 = 1.0500, 1 + 11*0.00625 = 1.06875 -> 1.0687.
REGULATOR_STEP_PU = 0.00625

# --------------------------------------------------------------------------- #
# Loads                                                                         #
# --------------------------------------------------------------------------- #
# Spot loads (l. 89-101): node -> (connection, model, (kW a, b, c), (kvar a, b, c))
# For delta loads Ph-1/2/3 are AB/BC/CA.
SPOT_LOADS = {
    "634": ("Y", "PQ", (160.0, 120.0, 120.0), (110.0, 90.0, 90.0)),  # l. 93
    "645": ("Y", "PQ", (0.0, 170.0, 0.0), (0.0, 125.0, 0.0)),        # l. 94
    "646": ("D", "Z", (0.0, 230.0, 0.0), (0.0, 132.0, 0.0)),         # l. 95
    "652": ("Y", "Z", (128.0, 0.0, 0.0), (86.0, 0.0, 0.0)),          # l. 96
    "671": ("D", "PQ", (385.0, 385.0, 385.0), (220.0, 220.0, 220.0)),# l. 97
    "675": ("Y", "PQ", (485.0, 68.0, 290.0), (190.0, 60.0, 212.0)),  # l. 98
    "692": ("D", "I", (0.0, 0.0, 170.0), (0.0, 0.0, 151.0)),         # l. 99
    "611": ("Y", "I", (0.0, 0.0, 170.0), (0.0, 0.0, 80.0)),          # l. 100
}
SPOT_LOAD_TOTALS = ((1158.0, 973.0, 1135.0), (606.0, 627.0, 753.0))  # l. 101

# Distributed load on 632-671 (l. 103-107), lumped at bus 670 (Phase 1).
DISTRIBUTED_LOAD = ("632", "671", "Y", "PQ",
                    (17.0, 66.0, 117.0), (10.0, 38.0, 68.0))  # l. 107

# --------------------------------------------------------------------------- #
# Published power-flow results                                                  #
# --------------------------------------------------------------------------- #
PUBLISHED_INPUT_KW = (1251.398, 977.332, 1348.461)   # l. 198
PUBLISHED_INPUT_KW_TOTAL = 3577.191                  # l. 198
PUBLISHED_INPUT_KVAR = (681.570, 373.418, 669.784)   # l. 199
PUBLISHED_INPUT_KVAR_TOTAL = 1724.772                # l. 199
PUBLISHED_LOAD_KW_TOTAL = 3466.128                   # l. 205
PUBLISHED_LOAD_KVAR_TOTAL = 2101.586                 # l. 208
PUBLISHED_LOSS_KW = (39.107, -4.697, 76.653)         # l. 217
PUBLISHED_LOSS_KW_TOTAL = 111.063                    # l. 217
PUBLISHED_LOSS_KVAR_TOTAL = 324.653                  # l. 218
PUBLISHED_CAP_KVAR = (193.443, 222.747, 285.276)     # l. 225-226 (actual kVA)
PUBLISHED_CAP_KVAR_TOTAL = 701.466                   # l. 226

# Voltage profile (l. 235-249): node -> ((mag, angle deg) or None) per phase a, b, c.
PUBLISHED_VOLTAGES = {
    "650":  ((1.0000, 0.00), (1.0000, -120.00), (1.0000, 120.00)),     # l. 235
    "RG60": ((1.0625, 0.00), (1.0500, -120.00), (1.0687, 120.00)),     # l. 236
    "632":  ((1.0210, -2.49), (1.0420, -121.72), (1.0174, 117.83)),    # l. 237
    "633":  ((1.0180, -2.56), (1.0401, -121.77), (1.0148, 117.82)),    # l. 238
    "634":  ((0.9940, -3.23), (1.0218, -122.22), (0.9960, 117.34)),    # l. 240
    "645":  (None, (1.0329, -121.90), (1.0155, 117.86)),               # l. 241
    "646":  (None, (1.0311, -121.98), (1.0134, 117.90)),               # l. 242
    "671":  ((0.9900, -5.30), (1.0529, -122.34), (0.9778, 116.02)),    # l. 243
    "680":  ((0.9900, -5.30), (1.0529, -122.34), (0.9778, 116.02)),    # l. 244
    "684":  ((0.9881, -5.32), None, (0.9758, 115.92)),                 # l. 245
    "611":  (None, None, (0.9738, 115.78)),                            # l. 246
    "652":  ((0.9825, -5.25), None, None),                             # l. 247
    "692":  ((0.9900, -5.31), (1.0529, -122.34), (0.9777, 116.02)),    # l. 248
    "675":  ((0.9835, -5.56), (1.0553, -122.52), (0.9758, 116.03)),    # l. 249
}
