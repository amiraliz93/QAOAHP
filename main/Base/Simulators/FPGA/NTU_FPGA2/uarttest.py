# example code to communicate NTU state machine.
# this code demonstrate how to send data to State machine, BRAM, and register in execution unit.
import serial
import struct
import time
from serial.tools import list_ports
import struct

def fp64b(f):
    return struct.pack('<d', f)
def bfp64(b):
    return struct.unpack('<d', b)[0]
def ib8(i):
    return i.to_bytes(8, "little")
def ib1(i):
    return i.to_bytes(1, "little")

OP_NONE        = ib1(0) # Send: 1, Res: 0.
OP_NONE8        = ib8(0) # Send: 1, Res: 0.
OP_SEND1T      = ib1(1) # Send: 0, Res: 1.
OP_SEND8T      = ib1(2) # Send: 0, Res: 1.
OP_MOV_T2A     = ib1(3) # Send: 1, Res: 0.
OP_MOV_T2B     = ib1(4) # Send: 1, Res: 0.
OP_MOV_A2U     = ib1(5) # Send: 1, Res: 0.
OP_MOV_A2B     = ib1(6) # Send: 0, Res: 1.
OP_MOV_Info2U  = ib1(7) # Send: 0, Res: 1.
OP_FETCH1U     = ib1(60) # Send: 0, Res: 8.
OP_FETCH8U     = ib1(61) # Send: 0, Res: 8.
OP_ADD_B2A     = ib1(80) # Send: 0, Res: 0.
OP_MUL_B2A     = ib1(81) # Send: 0, Res: 0.
OP_ADDFP_B2A   = ib1(82) # Send: 0, Res: 0.
OP_MULFP_B2A   = ib1(83) # Send: 0, Res: 0.
OP_INC_A       = ib1(84) # Send: 0, Res: 0.
OP_WRITE_T2RAM = ib1(111) # Send: 0, Res: 0.
OP_READ_RAM2U  = ib1(112) # Send: 0, Res: 0.
OP_SEND_CMD    = ib1(118) # Send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv
OP_RUN_MIXER   = ib1(100) # Send: 0, Res: 0. Run the mixer operation 1 step.
OP_RUN_COST    = ib1(101) # Send: 0, Res: 0. Run the cost operation 1 step. 
OP_RUN_CONTINUOUS = ib1(103) # Send: 0, Res: 0. Run the cost operation 1 step. 
OP_ENABLE_INTRUPTION = ib1(121) # Make the state machine send an 1 byte message to PC (always 43), if rS64 was changed. Note that rS64 cannot be changed from PC, so this event occurs from FPGA side only. This intruption may minimize the waiting time of the culculation. You can implement a function so that it will be invoked with this signal, by checking reading the UART port always, and immediately take an action if the state of FPGA changed spontaneously. Such function is called "callback function" or "event handler". event handler minimize the wainting time of external process without wasting computing resouces. Default of this mode is off.

qa_WAIT = ib1(1)
qa_RUN = ib1(2)
qa_INIT = ib1(4)

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
ser.write(OP_MOV_Info2U + OP_FETCH8U);  # check the version.
dr = ser.read(8)
print("version", dr.decode())

for b in data_array:
    ser.write(b); time.sleep(1e-5)  # recommend to wait tiny period of time, to prevent UART buffer overflow.
    if b == OP_FETCH8U:
        dr = ser.read(8)
        fr = bfp64(dr)
        print(fr, dr.hex())
    elif b == OP_FETCH1U:
        dr = ser.read(1)
        print(dr.hex(), int.from_bytes(dr, "little"))

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

ser.close()