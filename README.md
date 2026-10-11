# Smart Grid Data Analytics

---

## Table of Contents
- [About](#about)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Security](#security)
- [How to Contribute?](#how-to-contribute)
- [What's Next?](#whats-next)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Author](#author)

---

## About
Smart Grid Data Analytics is a robust framework designed to bridge unbalanced power-flow simulations with real-world electricity theft datasets. By integrating an IEEE 13-bus test feeder simulation with the State Grid Corporation of China (SGCC) theft dataset, this project creates a high-fidelity testbed for evaluating advanced metering infrastructure (AMI) security, electricity theft detection algorithms, and grid anomaly analysis. It natively exposes the simulated grid state via a Modbus TCP server for integration with external OSI layers or real-world networking equipment.

---

## Features
- **Unbalanced Feeder Simulation**: Complete implementation of the Kersting 1991 IEEE 13-bus test feeder using `pandapower` for full 3-phase unbalanced power-flow (`runpp_3ph`).
- **Fault Engine**: Simulates Single-Line-to-Ground (SLG), Line-to-Line (LL), and 3-Phase faults using IEC 60909 standards.
- **Synthetic Dataset Bridging**: Integrates daily SGCC data into 15-minute feeder intervals via a localized double-hump intra-day shape equation.
- **Theft Injector Engine**: Dynamically synthesizes 5 types of electricity theft attacks (based on Jokar et al., 2016) directly into the power streams.
- **Modbus TCP Meter Server**: Simulates real-time smart meters using `pymodbus`, encoding telemetry (voltage, current, power) across 32-bit registers for live network querying.

---

## Tech Stack
- **Python 3.8+**
- **Simulation**: `pandapower` (Unbalanced load flow & short circuit analysis)
- **Data Engineering**: `pandas`, `numpy` (Vectorized attack injections and time-series manipulation)
- **Networking**: `pymodbus` (Modbus TCP server implementation)
- **Testing**: `pytest`

---

## Architecture
The system operates as the **Data Source Layer (Layer 1)** for a broader security or analytics architecture.
1. **Physical Grid Layer (`feeder_model.py`)**: Solves the electrical state of the network.
2. **Data Layer (`dataset_loader.py` & `theft_injector.py`)**: Maps historical datasets onto the grid nodes and synthetically manipulates consumption profiles to simulate theft.
3. **Communication Layer (`meter_server.py`)**: Packages the state of individual loads into DLMS/COSEM-like Modbus registers, exposing them asynchronously over TCP to client applications.

---

## Project Structure
```text
Smart Grid Data Analytics/
├── data/               # Raw and processed SGCC datasets
├── docs/               # Detailed architectural and API documentation
├── src/
│   └── data_source/
│       ├── dataset_loader.py   # SGCC data extraction and interpolation
│       ├── feeder_model.py     # pandapower IEEE 13-bus implementation
│       ├── meter_server.py     # Modbus TCP Server
│       └── theft_injector.py   # Synthetics attack vectors
├── tests/              # Comprehensive pytest suite
├── .gitignore          # Repository git exclusions
└── requirements.txt    # Project dependencies
```

---

## Getting Started

### Prerequisites
1. Python 3.8 or higher.
2. Ensure you have `pip` and a virtual environment set up.

### Installation
1. Clone the repository.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset Preparation
The system attempts to automatically download the SGCC dataset from GitHub on first run. If you are in a restricted environment or the download fails, you can download the dataset separately and place it in the fallback directory.

1. Download the SGCC dataset manually.
2. Extract the dataset and place the CSV file in the following structure:
   ```text
   data/
   └── fallback/
       └── SGCC_theft_detecton_data/
           ├── data set.csv        # Full dataset
           └── datasetsmall.csv    # (Optional) Smaller subset for quick testing
   ```

### Running the System
**1. Run a Grid Simulation:**
```python
from src.data_source.feeder_model import run_timeseries
timeseries_df = run_timeseries(start="2024-01-01", end="2024-01-02", freq="15min")
```

**2. Start the Modbus TCP Server:**
```python
import asyncio
from src.data_source.meter_server import start_meter_server
# Expose the simulation data to localhost:5020
asyncio.run(start_meter_server(timeseries_data=timeseries_df, playback_speed=1.0))
```

### Ingestion Layer (Layer 2)
Gateway (Modbus → MQTT) → Mosquitto → FastAPI listener (validate, enrich, hand off).
```bash
docker compose up -d mosquitto                       # or a native Mosquitto 2.x
python -m src.ingestion.listener --port 8000         # REST: /health, /api/v1/stats, /api/v1/meters/...
python -m src.ingestion.gateway --config config/gateway.yaml
python scripts/run_meter_server.py --port 5020 --speed 10   # add --profiles sgcc for SGCC load shapes
```
Validated stream: MQTT `sg/v1/ieee13_11kv/validated/+`, or the in-process `TelemetryHub`.
Details: `docs/telemetry_schema.md`, `docs/ingestion_ops.md`, `docs/build_report_C.md`, `docs/build_report_D.md`.

### Testing
To run the full suite of 79 integration and unit tests:
```bash
pytest tests/ -v
```

---

## Configuration
Currently, configurations for the feeder model and dataset paths can be directly passed into their respective functions. For future enhancements, environment variables or a `.env` file can be utilized to set IP addresses, Modbus port numbers (default `5020`), and dataset remote URLs.

---

## Security
- **Simulation Environment**: The Modbus TCP server is designed for simulation and testing. By default, Modbus is an unencrypted, unauthenticated protocol. 
- **Network Exposure**: Do not bind the meter server to a public IP address (`0.0.0.0`) on untrusted networks unless you are intentionally creating a honeypot. It is recommended to bind strictly to `localhost` (`127.0.0.1`).

---

## How to Contribute?
1. Fork the project.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Ensure all tests pass (`pytest tests/`).
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5. Push to the branch (`git push origin feature/AmazingFeature`).
6. Open a Pull Request.

---

## What's Next?
- **Layer 2 Integration**: Developing an Intrusion Detection System (IDS) layer that pulls from the Modbus server to classify the injected theft in real-time.
- **Dynamic Topology**: Allowing dynamic topology changes (e.g., line switching, capacitor bank toggling) mid-simulation.
- **Distributed Energy Resources (DERs)**: Integrating solar PV profiles and battery storage models into the feeder.

---

## License
Distributed under the MIT License. See `LICENSE` for more information.

---

## Acknowledgements
- [pandapower](https://pandapower.readthedocs.io/) for their exceptional open-source power system analysis framework.
- State Grid Corporation of China (SGCC) for the foundational dataset used in electricity theft modeling.
- Jokar et al. (2016) for foundational research on consumption pattern-based theft attacks.

---

## Author
Krish Yadav
