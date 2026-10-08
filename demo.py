import asyncio
import pandas as pd
from src.data_source.feeder_model import run_timeseries
from src.data_source.meter_server import start_meter_server

async def main():
    print("1. Starting 24-hour grid simulation (Please wait...)")
    # Run a simulation for one day with 15-minute intervals
    timeseries_df = run_timeseries(start="2024-01-01", end="2024-01-02", freq="15min")
    
    print(f"\n2. Simulation complete! Generated {len(timeseries_df)} data points.")
    print("\nSample Data:")
    print(timeseries_df[['timestamp', 'bus', 'phase', 'voltage_v', 'active_power_kw']].head())

    print("\n3. Spinning up Modbus TCP Smart Meters...")
    print("Meters are now live on localhost:5020 and playing back the simulation at 100x speed.")
    print("Press Ctrl+C to stop.")
    
    # Start the server and play back the simulated data fast (e.g. 100x real-time)
    await start_meter_server(timeseries_data=timeseries_df, playback_speed=100.0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nDemo stopped.")
