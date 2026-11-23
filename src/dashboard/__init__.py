"""Layer 4: dashboard service (plan.md).

Report mode (P0) serves the frozen Layer 3 results and the illustrative map
layout of the feeder; live mode (P1+) will subscribe to the Layer 2 validated
MQTT topics.  The service adds presentation only: every number it returns is
read from a results file written by Layers 1-3 or from ``config/``.
"""
