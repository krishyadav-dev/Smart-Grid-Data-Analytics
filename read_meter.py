import time
from pymodbus.client import ModbusTcpClient

# Helper to decode two 16-bit registers into a 32-bit unsigned integer
def decode_u32(registers):
    return (registers[0] << 16) | registers[1]

print("Connecting to Modbus Server at 127.0.0.1:5020...")
client = ModbusTcpClient('127.0.0.1', port=5020)

if client.connect():
    print("Connected Successfully!\n")
    
    # We will poll Load Point 634 (Unit ID 1)
    UNIT_ID = 1
    print(f"Monitoring Smart Meter (Unit ID {UNIT_ID})... Press Ctrl+C to stop.\n")
    
    try:
        # Loop to show the data updating in real-time as the simulation plays
        for i in range(10): 
            print("-" * 75)
            print(f"Poll {i+1}/10: Unit ID {UNIT_ID}")
            # Read Input Registers (fc=4) starting at address 0, count 28
            # This covers Voltage, Current, Power, Frequency, and Reactive Power
            ir_response = client.read_input_registers(address=0, count=28, device_id=UNIT_ID)
            
            # Read Holding Registers (fc=3) at address 0. This holds Total Energy
            hr_response = client.read_holding_registers(address=0, count=2, device_id=UNIT_ID)
            
            if not ir_response.isError() and not hr_response.isError():
                # Extract and decode the 32-bit values from Input Registers
                regs = ir_response.registers
                v_a = decode_u32(regs[0:2]) / 100.0
                v_b = decode_u32(regs[2:4]) / 100.0
                v_c = decode_u32(regs[4:6]) / 100.0
                
                i_a = decode_u32(regs[6:8]) / 1000.0
                i_b = decode_u32(regs[8:10]) / 1000.0
                i_c = decode_u32(regs[10:12]) / 1000.0
                
                p_a = decode_u32(regs[12:14]) / 1000.0
                p_b = decode_u32(regs[14:16]) / 1000.0
                p_c = decode_u32(regs[16:18]) / 1000.0
                
                freq = decode_u32(regs[18:20]) / 100.0
                
                q_a = decode_u32(regs[20:22]) / 1000.0
                q_b = decode_u32(regs[22:24]) / 1000.0
                q_c = decode_u32(regs[24:26]) / 1000.0
                q_tot = decode_u32(regs[26:28]) / 1000.0
                
                # Extract Total Energy from Holding Registers
                energy = decode_u32(hr_response.registers[0:2])
                
                print(f"Voltage (V)       : Ph-A {v_a:7.2f} | Ph-B {v_b:7.2f} | Ph-C {v_c:7.2f}")
                print(f"Current (A)       : Ph-A {i_a:7.2f} | Ph-B {i_b:7.2f} | Ph-C {i_c:7.2f}")
                print(f"Active Pwr (kW)   : Ph-A {p_a:7.2f} | Ph-B {p_b:7.2f} | Ph-C {p_c:7.2f}")
                print(f"React. Pwr (kvar) : Ph-A {q_a:7.2f} | Ph-B {q_b:7.2f} | Ph-C {q_c:7.2f}")
                print(f"Grid Freq: {freq:5.2f} Hz | Total Reactive: {q_tot:7.2f} kvar | Total Energy: {energy} Wh")
            else:
                print("Failed to read registers from the server.")
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    finally:
        client.close()
        print("\nDisconnected.")
else:
    print("Failed to connect to the server. Is demo.py running?")
