"""
Data Source Layer (Layer 1) — Smart Grid Data Analytics.

Modules:
    feeder_model    — IEEE 13-bus unbalanced power-flow simulation at 11 kV.
    meter_server    — Modbus TCP smart-meter server (DLMS/COSEM data-model aligned).
    dataset_loader  — SGCC theft-dataset loader and profile sampler.
    theft_injector  — Synthetic electricity-theft attack functions (Jokar et al. 2016).
    config          — Centralised configuration constants.
"""
