# example code to communicate NTU state machine.
# this code demonstrate how to send data to State machine, BRAM, and register in execution unit.
import serial
import struct
import time
from serial.tools import list_ports
import struct
import math
import numpy as np 

def fp64b(f):
    return struct.pack('<d', f)
def bfp64(b):
    return struct.unpack('<d', b)[0]
def ib8(i):
    return i.to_bytes(8, "little")
def ib1(i):
    return i.to_bytes(1, "little")

dop = {}

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
HOST_WAIT      = ib1(254) # wait until the qaoa system become the state of qa_wait. Not sent to FPGA. Protocol how to wait must be defined by software productor.

dop["OP_NONE"]        =  OP_NONE          
dop["OP_NONE8"]       =  OP_NONE8     
dop["OP_SEND1T"]      =  OP_SEND1T    
dop["OP_SEND8T"]      =  OP_SEND8T    
dop["OP_MOV_T2A"]     =  OP_MOV_T2A   
dop["OP_MOV_T2B"]     =  OP_MOV_T2B   
dop["OP_MOV_A2U"]     =  OP_MOV_A2U   
dop["OP_MOV_A2B"]     =  OP_MOV_A2B   
dop["OP_MOV_Info2U"]  =  OP_MOV_Info2U
dop["OP_MOV_S2U"]     =  OP_MOV_S2U   
dop["OP_MOV_T2P"]     =  OP_MOV_T2P   
dop["OP_FETCH1U"]     =  OP_FETCH1U   
dop["OP_FETCH8U"]     =  OP_FETCH8U   
dop["OP_ADD_B2A"]     =  OP_ADD_B2A   
dop["OP_MUL_B2A"]     =  OP_MUL_B2A   
dop["OP_ADDFP_B2A"]   =  OP_ADDFP_B2A
dop["OP_MULFP_B2A"]   =  OP_MULFP_B2A
dop["OP_INC_A"]       =  OP_INC_A
dop["OP_WRITE_T2RAM"] =  OP_WRITE_T2RAM
dop["OP_READ_RAM2U"]  =  OP_READ_RAM2U
dop["OP_SEND_CMD"]    =  OP_SEND_CMD
dop["HOST_WAIT"]      =  HOST_WAIT 

idop = {}
for k in dop:
    idop[dop[k].hex()] = k

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

beta = 0.1
gamma = 0.2
sinb = np.sin(beta)
cosb = np.cos(beta)
NQ = 3
NS = 2**NQ
Np = 8 # number of p layers.

# generate ideal result
sv = []

# initialize state vector 
for i in range(NS):
    sv.append(complex(0, 0))

sv[0] = complex(1, 0)
sv0 = sv # back up the initizal state.
def swap_bits(i, a, b):
    # 1. Extract the values of the bits at position a and b
    bit_a = (i >> a) & 1
    bit_b = (i >> b) & 1

    # 2. If the bits are different, we need to swap them
    if bit_a != bit_b:
        # Create a bitmask with 1s at position a and b
        # Example: (1 << 2) | (1 << 5) results in 00100100
        mask = (1 << a) | (1 << b)
        
        # XOR the original integer with the mask
        # This flips the bits at those two positions
        i ^= mask
        
    return i
import random
lcq = list(range(NQ))
random.shuffle(lcq)
print(lcq)
for p in range(Np):
    # output the current state vector
    f = open(f"sim_mixer_{p}-th.txt", "w")
    for i in range(NS):
        f.write(f"{sv[i]}\n")
    f.close()

    # apply mixer operator
    for cq in lcq: #counter of qbit.
        for id2 in range(NS//2):
            sa = id2*2
            sb = id2*2 + 1

            # swap bits, so that a is an index only flipped cq-th bit of b. 
            # in other words, a is a neighbor index of b in terms of cq-th bit.
            a = swap_bits(sa, cq, 0)
            b = swap_bits(sb, cq, 0)

            # apply rotation
            
            tsa = cosb * sv[a] + 1j * sinb * sv[b]
            tsb = 1j*sinb * sv[a] + cosb * sv[b]
            sv[a] = tsa
            sv[b] = tsb

            # p'_a = cos p_a + i sin p_b
            # p'_b = i sin p_a + cos p_b

data_array = [
      OP_SEND1T,
      ib1(12), OP_MOV_T2A, OP_MOV_A2U, OP_FETCH1U,
      OP_SEND1T, qa_WAIT,
      OP_SEND_CMD,
      
      OP_SEND8T, ib8(0x4000_0000_0000_0000),  # address of number of qbit's register
      OP_MOV_T2A,
      OP_SEND8T, ib8(0x0100_0000_0000_0000),  # address of number of qbit's register
      OP_MOV_T2B,
      OP_SEND8T, ib8(NQ-1),
      OP_WRITE_T2RAM,
      OP_ADD_B2A, # set address to next, 0x4100_0000_0000_0000
      OP_SEND8T, ib8(NQ-2),
      OP_WRITE_T2RAM,
      OP_ADD_B2A,  # set address to next, 0x4200_0000_0000_0000
      OP_SEND8T, ib8(NS-1),
      OP_WRITE_T2RAM,
      OP_ADD_B2A,  # set address to next, 0x4300_0000_0000_0000
      OP_SEND8T, ib8(NS-2),
      OP_WRITE_T2RAM,
      OP_ADD_B2A,  # set address to next, 0x4300_0000_0000_0000
      OP_SEND8T, ib8(Np),
      OP_WRITE_T2RAM]


data_array += [OP_SEND8T, ib8(0x0800_0000_0000_0000),  # write BRAM for cosb, sinb, gamma
      OP_MOV_T2A]

for p in range(Np):
    data_array += [
        OP_SEND8T, fp64b(cosb),
        OP_WRITE_T2RAM,
        OP_INC_A,
        OP_SEND8T, fp64b(sinb),
        OP_WRITE_T2RAM,
        OP_INC_A,
        OP_SEND8T, fp64b(gamma),
        OP_WRITE_T2RAM,
        OP_INC_A
      ]

# send initial value of state vector
# write real part.
# set the address
data_array += [OP_SEND8T, ib8(0x1000_0000_0000_0000), # set the address
    OP_MOV_T2A]
for i in range(NS): 
    value = sv0[i]
    data_array += [OP_SEND8T, fp64b(value.real), OP_WRITE_T2RAM, OP_INC_A]

# write imaginary part.
# set the address
data_array += [OP_SEND8T, ib8(0x2000_0000_0000_0000), # set the address
      OP_MOV_T2A]
for i in range(NS): 
    value = sv0[i]
    data_array += [OP_SEND8T, fp64b(value.imag), OP_WRITE_T2RAM, OP_INC_A]

data_array += [
      OP_SEND1T, qa_INIT,
      OP_SEND_CMD,
      OP_SEND1T, qa_RUN,
      OP_SEND_CMD,
      OP_NONE, # need to wait for the pipeline end the process
      OP_SEND1T, qa_WAIT,
      OP_SEND_CMD]

data_array += [HOST_WAIT]

data_array += [OP_SEND8T, ib8(0x1000000000000000), # read address of BRAM, real part of state vector
      OP_MOV_T2A]
for i in range(NS):
    data_array += [
    OP_READ_RAM2U,
    OP_FETCH8U,
    OP_INC_A # move to the next address.
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
        f.write(f",")
    skey = b.hex()
    if skey in idop:
        f.write(f" // {idop[skey]}\n")
    elif len(b) == 8:
        f.write(f" // {bfp64(b)}\n")
f.write("}\n")
f.close()
quit()

# send to serial interface, for physical test of the implementation.
ser = serial.Serial(port = uart_port, baudrate = baud_rate, timeout=None)
for b in data_array:
    if b == HOST_WAIT:
        # must wait here.
        for i in range(1e3):
            time.sleep(1)
            ser.write(OP_SEND8T)
            ser.write(ib8(0x0100_0000_0000_0000))
            ser.write(OP_READ_RAM2U)
            ser.write(OP_FETCH8U)
            dr = ser.read(8)
            ir = int.from_bytes(dr, "little")
            if dr[0] == qa_WAIT: # check the state is qa_WAIT
                break 

        continue
    ser.write(b); time.sleep(1e-4)  # recommend to wait tiny period of time, to prevent UART buffer overflow.
    if b == OP_FETCH8U:
        dr = ser.read(8)
        fr = bfp64(dr)
        print(fr, dr.hex())
    elif b == OP_FETCH1U:
        dr = ser.read(1)
        print(dr.hex(), int.from_bytes(dr, "little"))
    

ser.close()