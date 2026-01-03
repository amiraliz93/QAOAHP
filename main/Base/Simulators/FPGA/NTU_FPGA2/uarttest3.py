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
OP_NONE8       = ib8(0) # Send: 1, Res: 0.
OP_SEND1T      = ib1(1) # Send: 0, Res: 1.
OP_SEND8T      = ib1(2) # Send: 0, Res: 1.
OP_MOV_T2A     = ib1(3) # Send: 1, Res: 0.
OP_MOV_T2B     = ib1(4) # Send: 1, Res: 0.
OP_MOV_A2U     = ib1(5) # Send: 1, Res: 0.
OP_MOV_A2B     = ib1(6) # Send: 0, Res: 1.
OP_MOV_Info2U  = ib1(7) # Send: 0, Res: 1.
OP_MOV_S2U     = ib1(8) # Send: 0, Res: 1.OP_FETCH1U
OP_MOV_T2P     = ib1(9) # Send: 0, Res: 1.
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

qa_WAIT =  ib1(1)
qa_RUN =  ib1(2)
qa_MIXER =  ib1(4)
qa_COST =  ib1(8)
qa_INIT =  ib1(16)

v1 = 0.83134910
v2 = 0.13134750
v3 = 0.43248110
bv1 = fp64b(v1)
v1t = bfp64(bv1)
print(bv1.hex(), v1t, v1)

uart_port = 'COM4'
# uart_port = "/dev/ttyUSB0"
# uart_port = None
baud_rate = 115200

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
      OP_SEND8T, ib8(0x0800000000000000), # address to write BRAM, cos(beta)
      OP_MOV_T2A,
      OP_SEND8T, fp64b(10), # 10 write BRAM, cos(beta)
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(-0.1), # -0.1 write BRAM, sin(beta)
      OP_WRITE_T2RAM,
      OP_SEND8T, ib8(0x1000000000000000), # writing address to write state vector in BRAM
      OP_MOV_T2A,
      OP_SEND8T, fp64b(4.540779000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(4.751009000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(4.961239000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(5.171469000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_WRITE_T2RAM,

      OP_SEND8T, ib8(0x2000000000000000), # write address to write state vector in BRAM
      OP_MOV_T2A,
      OP_SEND8T, fp64b(5.802159000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(5.802159000000001),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(6.0323890000000015),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(6.222619000000002),
      OP_WRITE_T2RAM,
      OP_INC_A, # write to next address of BRAM
      OP_SEND8T, fp64b(6.432849000000002),
      OP_WRITE_T2RAM,

      OP_SEND1T, qa_INIT,
      OP_SEND_CMD,
      OP_SEND1T, qa_RUN,
      OP_SEND_CMD,
      OP_NONE, # need to wait for the pipeline end the process
      OP_SEND1T, qa_WAIT,
      OP_SEND_CMD,
      OP_SEND8T, ib8(0x1000000000000000), # read address of BRAM, real part of state vector
      OP_MOV_T2A,
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_SEND8T, ib8(0x2000000000000000), # read address of BRAM, imaginary part of state vector
      OP_MOV_T2A,
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U,
      OP_INC_A, # write to next address of BRAM
      OP_READ_RAM2U,
      OP_FETCH8U
]
f = open("uarttest_veri.sv", "w") # generate the same byte sequence above in verilog format so that we can run the testbench simulation with the same input to be supplied here by this python code.

AC = [b"".join(data_array)]
f.write("{\n")
for i, b in enumerate(data_array):
    print(type(b), b, len(b))
    for j in range(len(b)):
        f.write(f"8'h{ib1(b[j]).hex()}")
        if j != len(b)-1:
            f.write(", ")

    if i != len(data_array) -1:
        f.write(",\n")
f.write("}\n")
f.close()

ser = serial.Serial(port = uart_port, baudrate = baud_rate, timeout=None)
for b in data_array:
    ser.write(b); time.sleep(1e-4)  # recommend to wait tiny period of time, to prevent UART buffer overflow.
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
ser.close()