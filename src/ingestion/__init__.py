"""Ingestion & Broker Layer (Layer 2).

gateway   — polls the Layer 1 Modbus meters, decodes, publishes to MQTT.
listener  — FastAPI app: validates, enriches and hands off the stream.

Layer 2 never imports ``feeder_model`` or ``pandapower`` at runtime.
"""
