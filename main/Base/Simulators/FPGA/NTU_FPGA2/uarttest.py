# example code to communicate NTU state machine.
# this code demonstrate how to send data to State machine, BRAM, and register in execution unit.
"""
QAOA Simulation State Machine Communication Via URAT
============================================
This module demonstrates how to communicate with the FPGA QAOA hardware
via UART serial protocol. It shows:
- Data type conversions for UART communication
- Register operations (rA, rB, rT, rU)
- BRAM read/write operations
- QAOA system commands
- Arithmetic operations on FPGA

Hardware Architecture:
---------------------
PC <--> UART <--> State Machine <--> BRAM <--> QAOA System

Registers:
    rT: Temporary register (receives all data from PC)
    rA: Address register (used for BRAM addressing)
    rB: General purpose register (arithmetic operations)
    rU: Output register (all data sent to PC must be in rU)

BRAM Banks:
    0x0100000000000000: Parameters (β, γ, etc.)
    0x0200000000000000: Cost function
    0x0400000000000000: Trigonometric values (sin/cos)
    0x1000000000000000: State vector (main BRAM)

Protocol:
    - All sent data goes to rT first → move to target register
    - All fetched data must be in rU → fetch to PC
    - Fire-and-forget: No acknowledgment, use timing delays

Author: Amir Alizadeh, Hiroki Shabata
Date: 2025
"""
import serial
import struct
import time
from serial.tools import list_ports
import struct

# ==============================================
# Data type conversions for UART communication
# ==============================================

def fp64b(f):
    """
    Convert Python float to 64-bit IEEE 754 little-endian bytes.
    
    Args:
        f: Float value to convert
        
    Returns:
        8 bytes representing the float in little-endian format
        
    Example:
        >>> fp64b(3.14159)
        b'\\x18-DT\\xfb!\\t@'
    """

    return struct.pack('<d', f)

def bfp64(b):
    """
    Convert 64-bit little-endian bytes to Python float.
    
    Args:
        b: 8 bytes in little-endian format
        
    Returns:
        Decoded float value
        
    Example:
        >>> bfp64(b'\\x18-DT\\xfb!\\t@')
        3.14159
    """
    return struct.unpack('<d', b)[0]

def ib8(i):
    """
    Convert integer to 8-byte little-endian format.
    
    Args:
        i: Integer value (typically used for BRAM addresses)
        
    Returns:
        8 bytes in little-endian format
        
    Example:
        >>> ib8(0x1000000000000000).hex()
        '0000000000000010'
    """
    return i.to_bytes(8, "little")

def ib1(i):
    """
    Convert integer to 1-byte format.
    
    Args:
        i: Integer value 0-255 (typically opcodes)
        
    Returns:
        Single byte
        
    Example:
        >>> ib1(111).hex()
        '6f'
    """
    return i.to_bytes(1, "little")
#=============================================
# Operation Codes (Opcodes)
# =============================================

# --- Data Transfer Opcodes ---
OP_NONE        = ib1(0) # Send: 1, Res: 0.
OP_NONE8        = ib8(0) # Send: 1, Res: 0.
OP_SEND1T      = ib1(1) # Send: 0, Res: 1.
OP_SEND8T      = ib1(2) # Send: 0, Res: 1.

# --- Register Move Opcodes ---
OP_MOV_T2A     = ib1(3) # Send: 1, Res: 0.
OP_MOV_T2B     = ib1(4) # Send: 1, Res: 0.
OP_MOV_A2U     = ib1(5) # Send: 1, Res: 0.
OP_MOV_A2B     = ib1(6) # Send: 0, Res: 1.
OP_MOV_Info2U  = ib1(7) # Send: 0, Res: 1.

# --- Data Fetch Opcodes ---
OP_FETCH1U     = ib1(60) # Send: 0, Res: 8.
OP_FETCH8U     = ib1(61) # Send: 0, Res: 8.

# --- Arithmetic Opcodes ---
OP_ADD_B2A     = ib1(80) # Send: 0, Res: 0.
OP_MUL_B2A     = ib1(81) # Send: 0, Res: 0.
OP_ADDFP_B2A   = ib1(82) # Send: 0, Res: 0.
OP_MULFP_B2A   = ib1(83) # Send: 0, Res: 0.
OP_INC_A       = ib1(84) # Send: 0, Res: 0.

# --- BRAM Access Opcodes ---
OP_WRITE_T2RAM = ib1(111) # Send: 0, Res: 0.
OP_READ_RAM2U  = ib1(112) # Send: 0, Res: 0.

# --- QAOA System Control ---
OP_SEND_CMD    = ib1(118) # Send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv

 # seems below are extra
OP_RUN_MIXER   = ib1(100) # Send: 0, Res: 0. Run the mixer operation 1 step.
OP_RUN_COST    = ib1(101) # Send: 0, Res: 0. Run the cost operation 1 step. 
OP_RUN_CONTINUOUS = ib1(103) # Send: 0, Res: 0. Run the cost operation 1 step. 
OP_ENABLE_INTRUPTION = ib1(121) # Make the state machine send an 1 byte message to PC (always 43), if rS64 was changed. Note that rS64 cannot be changed from PC, so this event occurs from FPGA side only. This intruption may minimize the waiting time of the culculation. You can implement a function so that it will be invoked with this signal, by checking reading the UART port always, and immediately take an action if the state of FPGA changed spontaneously. Such function is called "callback function" or "event handler". event handler minimize the wainting time of external process without wasting computing resouces. Default of this mode is off.
# ---------------------------------------------------

"""
Enable interrupt mode: State machine sends 1-byte message (value 43) 
to PC when rS64 register changes. Useful for event-driven programming 
to minimize waiting time. Requires implementing callback/event handler.
Default: OFF
"""

# ============================================================================
# QAOA System Commands (used with OP_SEND_CMD)
# ============================================================================
qa_WAIT = ib1(1) # Put QAOA system in wait state
qa_RUN = ib1(2) # Start QAOA execution
qa_INIT = ib1(4)    # Initialize QAOA system

def main():
    """
    Demonstrate complete FPGA communication workflow:
    1. Version check
    2. Register arithmetic test
    3. BRAM write operations
    4. BRAM read operations
    """
    # Test Execution
    v1 = 0.83134910
    v2 = 0.13134750
    v3 = 0.43248110
    bv1 = fp64b(v1)
    v1t = bfp64(bv1)
    print(bv1.hex(), v1t, v1)

    uart_port = 'COM3'
    # uart_port = "/dev/ttyUSB0"
    # uart_port = None
    baud_rate = 115200
    ser = serial.Serial(port = uart_port, baudrate = baud_rate, timeout=None)

    # there are registers, rA, rB, rU, rT in the state machine
    # all data sent to state machine will be firstly stored into rT. you need to move value in rT to write anoter register. e.g., call OP_MOV_T2A
    # all data sent from state machine must be in rU. i.e., You need to move any value to rU before issue fetch command. e.g., OP_MOV_A2U if you want the value of rA then call OP_FETCH8U. 
    # OP_FETCH8U fetch 64 bit of rU. OP_FETCH1U fetch first 8 bit in rU. For OP_SEND1T, 8T similarly.

    data_array = [
        OP_SEND1T,
        ib1(12), OP_MOV_T2A, OP_MOV_A2U, OP_FETCH1U,
        OP_MOV_A2B, OP_ADD_B2A,OP_MOV_A2U, OP_FETCH1U, OP_MUL_B2A,
        OP_MOV_A2U, OP_FETCH1U,
        OP_SEND1T, qa_WAIT,
        OP_SEND_CMD, 
        OP_SEND8T, ib8(0x0100000000000000),  # write BRAM, like, beta, gamma, and so on. Parameters for the algorithm.
        OP_MOV_T2A,
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_SEND8T, ib8(0x0200000000000000),  # write BRAM0, for cost function?
        OP_MOV_T2A,
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_SEND8T, ib8(0x0400000000000000),  # write BRAM1, for sin(gamma H), cos(gamma H)?. These 2 value also can be computed inside the FPGA.
        OP_MOV_T2A,
        OP_SEND8T, fp64b(v1),
        OP_WRITE_T2RAM,
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_SEND8T, ib8(0x1000000000000000), # write main BRAM, intended to state vector.
        OP_MOV_T2A,
        OP_SEND8T, fp64b(v1),
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM. OP_INC_A increment rA by 1, like rA++; in C.
        OP_SEND8T, fp64b(v2),
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM
        OP_SEND8T, fp64b(v3),
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM
        OP_WRITE_T2RAM,
        OP_INC_A, # to next address of BRAM
        OP_WRITE_T2RAM,
        OP_SEND8T, ib8(0x1000000000000000), # read address of BRAM
        OP_MOV_T2A,
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, # to next address of BRAM. This OP increment rA by 1, like rA++; in C.
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, # to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U,
        OP_INC_A, #  to next address of BRAM
        OP_READ_RAM2U, 
        OP_FETCH8U
    ]
    # --- Version Check ---
    print("\n=== FPGA Version Check ===")
    ser.write(OP_MOV_Info2U + OP_FETCH8U);  # check the version.
    version_bytes = ser.read(8)
    dr = ser.read(8)
    version_str = version_bytes.decode('ascii', errors='ignore').strip('\x00')
    print(f"FPGA Version: {version_str}")
    print("version", dr.decode()) 

    # --- Execute Command Sequence ---
    print("\n=== Executing Command Sequence ===")
    for b in data_array:
        ser.write(b); time.sleep(1e-5)  # recommend to wait tiny period of time, to prevent UART buffer overflow.

        if b == OP_FETCH8U:
            dr = ser.read(8)
            fr = bfp64(dr)
            print(fr, dr.hex())
        elif b == OP_FETCH1U:
            dr = ser.read(1)
            print(dr.hex(), int.from_bytes(dr, "little"))

    print("\n=== Expected Output ===")
    print("  8-bit read:   12  (hex: 0c)")
    print("  8-bit read:   24  (hex: 18)")
    print("  8-bit read:   32  (hex: 20)")
    print("  64-bit read: 4.4e-323  (BRAM params - uninitialized)")
    print("  64-bit read: 2.38e-299 (BRAM cost - uninitialized)")
    print("  64-bit read: 0.8313491 (BRAM trig - wrote v1)")
    print("  64-bit read: 0.8313491 (BRAM[0] - wrote v1)")
    print("  64-bit read: 0.1313475 (BRAM[1] - wrote v2)")
    print("  64-bit read: 0.4324811 (BRAM[2] - wrote v3)")
    print("  64-bit read: 0.0       (BRAM[3-8] - wrote zeros)")

    ser.close()
    print("\n✓ Serial port closed")
# output should be the following.
# version NTUSMv01
# 0c 12
# 18 24
# 20 32
# 4.4e-323 0900000000000000
# 2.3843434815971076e-299 aaaa80aa80efef01
# 0.8313491 9ae3816d699aea3f
# 0.8313491 9ae3816d699aea3f
# 0.1313475 72a774b0fecfc03f
# 0.4324811 b9d62835c5addb3f
# 0.0 0000000000000000


if __name__ == "__main__":
    main()