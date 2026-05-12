# example code to communicate NTU state machine.
# this code demonstrate how to send data to State machine, BRAM, and register in execution unit.
import serial
import struct
import time
from serial.tools import list_ports
import struct
import math
import numpy as np 
import random
import sys
import datetime

import time


"""
Main configurable parameter blocks. 
"""
seed = 0x22a2039#random.getrandbits(31)
random.seed(seed)
uart_port = 'COM3'
# uart_port = "/dev/ttyUSB0"
# uart_port = None
baud_rate = 115200
l_cosb = []
l_sinb = []
l_gamma = []
NQ = 9
NS = 2**NQ # number of layers
Np = 2 # number of p layers.
LP_BRAM_A = 2
LP_BRAM_D = 1
LP_GEN_COST = 2
LP_MIXER_IN = 1
LP_MIXER_OUT = 1
L_BRAM_R = LP_BRAM_A + LP_BRAM_D + 2
L_BRAM_W = LP_BRAM_A + LP_BRAM_D + 2

Lc = 223 + 1 + L_BRAM_R + 1 + LP_GEN_COST + LP_GEN_COST# cost gen latency, output H latency, memory and register latency, address output latency.
Lm = 52 + 1 + L_BRAM_R + 1 + L_BRAM_W + 1 + LP_MIXER_IN + LP_MIXER_OUT  # mixer latency 52, output latency to mixer + memory read, output of address + write latency + write address latency.
LInit = 24
output_command_sv = "all_test_cmd.sv"
"""
Parameter modification blocks. The following block will modify the parameter if it does not meet the constraint. 
"""
LPipe = NS
tl = Lm + NS//2 + NS%2 
if tl >= NS:
    LPipe = tl
    print(f"Lm + NS//2 + NS%2  = {tl} >= NS = {NS}. LPipe become {LPipe}.")
else:
    print(f"LPipe = NS = {NS}")


DVTc = Lc // LPipe + 1; # make sure, pipe*DVTc-Tc-2 > LInit.

tGenCost = DVTc*LPipe - Lc
if tGenCost < LInit:
    tGenCost += LPipe

    print(f"tGenCost = {tGenCost} greater than LInit = {LInit}.")
tbGenCost = LPipe*(NQ+1)-Lc
t_Mixer = LPipe
if tbGenCost < LInit:
    t_Mixer = LPipe + LInit - tbGenCost
    print(f"tbGenCost = {tbGenCost} < {LInit} = LInit. set tbGenCost={LInit}, t_Mixer ={t_Mixer}")
     # need to prepare additional time for gen cost
    tbGenCost = LInit

t_Compute = t_Mixer + LPipe*NQ + Lm

lineend = "" # lineend. It can be configured through command line argument. if not blanck string, this script will add comments into the output file of output_command_sv.

if len(sys.argv) != 1:
    lineend = sys.argv[1]
    if lineend == "\\n":
        lineend = "\n"

print("DVTc:", DVTc)
print("NQ", NQ)
print("NS:", NS)
print("LPipe:", LPipe)
print("Np:", Np)
print("Lc:", Lc)
print("Lm:", Lm)
print("LInit:", LInit)
print("tbGenCost:", tbGenCost)
print("tGenCost:", tGenCost)
print("t_Mixer:", t_Mixer)
print("t_Compute:", t_Compute)


def fp64b(f):
    return struct.pack('<d', f)
def bfp64(b):
    return struct.unpack('<d', b)[0]
def ib8(i):
    return i.to_bytes(8, "little")
def ib1(i):
    return i.to_bytes(1, "little")

"""
Parameter blocks. These parameters must be consistent with the parameters in the logic implementation.
"""

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
OP_WRITE_T2_AG = ib1(119) 
HOST_WAIT      = ib1(254) # wait until the qaoa system become the state of qa_wait. Not sent to FPGA. Protocol how to wait must be defined by software productor.

AG_SET_t_L2Addr   = ib1(0)
AG_SET_t_L2PipeGC = ib1(1)
AG_SET_tb_B2GenCost= ib1(2)
AG_SET_t_L2Pipe  = ib1(3)
AG_SET_nPLayer   = ib1(4)
AG_SET_L1Qbit    = ib1(5)
AG_SET_AddrMask  = ib1(6)
AG_SET_t_B2GenCost  = ib1(7)
AG_SET_tb_B2Mixer  = ib1(8)
AG_SET_t_L2Compute  = ib1(9)

qa_WAIT =  ib1(1)
qa_RUN =  ib1(2)
qa_MIXER =  ib1(4)
qa_COST =  ib1(8)
qa_INIT =  ib1(16)

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
dop["OP_WRITE_T2_AG"] =  OP_WRITE_T2_AG 
dop["HOST_WAIT"]      =  HOST_WAIT 
dop["qa_WAIT"] =  qa_WAIT
dop["qa_RUN"]  =  qa_RUN
dop["qa_MIXER"] =  qa_MIXER
dop["qa_COST"]  =  qa_COST 
dop["qa_INIT"]  =  qa_INIT 

dop["AG_SET_t_L2Addr"]      =  AG_SET_t_L2Addr 
dop["AG_SET_t_L2PipeGC"]    =  AG_SET_t_L2PipeGC 
dop["AG_SET_tb_B2GenCost"]  =  AG_SET_tb_B2GenCost 
dop["AG_SET_t_L2Pipe"]    =  AG_SET_t_L2Pipe 
dop["AG_SET_nPLayer"]     =  AG_SET_nPLayer 
dop["AG_SET_L1Qbit"]      =  AG_SET_L1Qbit 
dop["AG_SET_AddrMask"]    =  AG_SET_AddrMask 
dop["AG_SET_t_B2GenCost"] =  AG_SET_t_B2GenCost 
dop["AG_SET_tb_B2Mixer"] =  AG_SET_tb_B2Mixer 
dop["AG_SET_t_L2Compute"] =  AG_SET_t_L2Compute


idop = {}
for k in dop:
    if dop[k].hex() not in idop:
        idop[dop[k].hex()] = []
    idop[dop[k].hex()].append(k)

# generate ideal result
sv = []
H = []
costFOP = []

# initialize state vector 
Lm = 0
for i in range(NS):
    if i == -1:
        sr = 1
        si = 0
    else:
        sr = random.uniform(1, -1)
        si = random.uniform(1, -1)
    com = complex(sr, si)
    Lm = Lm + sr*sr + si*si
    sv.append(com)

for i in range(NS):
    sv[i] = sv[i]/Lm

sv0 = sv.copy() # back up the initizal state.
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

for i in range(NS):
    Ht = random.uniform(-1, 1)
    H.append(Ht)

lcq = list(range(NQ))

"""
Generate simulated result in this script.
"""
print("a")
simpath = f"simulation.txt"
f = open(simpath, "w")
f.write(f"-------------------------------------------------------\n")
f.write(f"Version {random.random()}, {datetime.datetime.now()}\n")
f.write(f"-------------------------------------------------------\n")
for i in range(NS):
    f.write(f"H_{i}, {H[i]}\n")
for i in range(NS):
    f.write(f"p_{i}, {sv[i]}\n")

start1 = time.perf_counter()

for p in range(Np):
    # output the current state vector
    gamma = random.uniform(-np.pi, np.pi)
    beta = random.uniform(-np.pi, np.pi)
    sinb = np.sin(beta)
    cosb = np.cos(beta)
    l_sinb.append(sinb)
    l_cosb.append(cosb)
    l_gamma.append(gamma)
    f.write(f"-------------------------------------------------------\n")
    f.write(f"Starting {p}-th layer. Current params: gamma={gamma}, cosb={cosb}, sinb={sinb}\n")
    f.write(f"-------------------------------------------------------\n")
    
    for i in range(NS):
        gHt = gamma*H[i]
        costFt = math.cos(gHt) + 1j*math.sin(gHt)
        f.write(f"F_{i}: {costFt}\n")
        sv[i] = costFt*sv[i]
    for i in range(NS):
        f.write(f"F_{i}p_{i}: {sv[i]}\n")

    # apply mixer operator
    for cq in lcq: # counter of qbit.
        f.write(f"\n---{p}-th layer {cq}-th qbit----------------------\n\n")
        for id2 in range(NS//2):
            sa = id2*2
            sb = id2*2 + 1

            # swap bits, so that a is an index only flipped cq-th bit of b. 
            # in other words, a is a neighbor index of b in terms of cq-th bit.
            a = swap_bits(sa, cq, 0)
            b = swap_bits(sb, cq, 0)

            # apply rotation
            
            tsa = cosb * sv[a] - 1j * sinb * sv[b]
            tsb = -1j*sinb * sv[a] + cosb * sv[b]
            sv[a] = tsa
            sv[b] = tsb
            f.write(f"p_{a}: {tsa}\n")
            f.write(f"p_{b}: {tsb}\n")

            # p'_a = cos p_a + i sin p_b
            # p'_b = i sin p_a + cos p_b
end1 = time.perf_counter()

f.write(f"\n---Results----------------------\n\n")
for i in range(NS):
    f.write(f"Re(p_{i}), {sv[i].real}\n")
for i in range(NS):
    f.write(f"Im(p_{i}), {sv[i].imag}\n")
f.close()

path = f"result.txt"
f = open(path, "w")
for i in range(NS):
    f.write(f"{sv[i].real}\n")
for i in range(NS):
    f.write(f"{sv[i].imag}\n")

f.close()

"""
Generates command sequence can be used in top1cmd_tb.sv
"""
def mask64(a: int) -> int:
    if a <= 0:
        return 0
    if a >= 64:
        return (1 << 64) - 1
    return (1 << a) - 1
    

t_L2Addr   = NS-2
t_L2PipeGC = Lc-2
tb_B2GenCost= tbGenCost-2
t_L2Pipe  = LPipe-2
nPLayer = Np
L1Qbit    = NQ-1
AddrMask  = mask64(NQ-1)
t_B2GenCost  = tGenCost-2
tb_B2Mixer = t_Mixer -2
t_L2Compute = t_Compute

# configuration of addr_gen.sv
data_array = [
      OP_SEND1T,
      ib1(12), OP_MOV_T2A, OP_MOV_A2U, OP_FETCH1U,
      OP_SEND1T, qa_WAIT,
      OP_SEND_CMD,
      OP_SEND1T, AG_SET_t_L2Addr, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(t_L2Addr), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_t_L2Pipe, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(t_L2Pipe), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_t_L2PipeGC, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(t_L2PipeGC), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_tb_B2GenCost, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(tb_B2GenCost), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_t_B2GenCost, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(t_B2GenCost), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_nPLayer, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(nPLayer), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_L1Qbit, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(L1Qbit), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_AddrMask, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(AddrMask), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_tb_B2Mixer, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(tb_B2Mixer), # set the data
      OP_WRITE_T2_AG,
      OP_SEND1T, AG_SET_t_L2Compute, # set address to rA
      OP_MOV_T2A,
      OP_SEND8T, ib8(t_L2Compute), # set the data
      OP_WRITE_T2_AG
      ]

data_array += [OP_SEND8T, ib8(0x0800_0000_0000_0000),  # write BRAM for cosb, sinb, gamma
      OP_MOV_T2A]

l_cosb = [l_cosb[0]] + l_cosb
l_sinb = [l_sinb[0]] + l_sinb
l_gamma.append(-1)
for p in range(Np+1):
    cosb = l_cosb[p]
    sinb = l_sinb[p]
    gamma = l_gamma[p]
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
    
# set the address
data_array += [OP_SEND8T, ib8(0x0400_0000_0000_0000), # set the address
      OP_MOV_T2A]
# send cost function
for i in range(NS): 
    value = H[i]
    data_array += [OP_SEND8T, fp64b(value), OP_WRITE_T2RAM, OP_INC_A]

# run the simulation.
data_array += [
      OP_SEND1T, qa_RUN,
      OP_SEND_CMD
      ]

data_array += [HOST_WAIT]  # need to wait for the pipeline end the process
data_array += [OP_SEND1T, qa_WAIT,
      OP_SEND_CMD]

data_array += [OP_SEND8T, ib8(0x1000000000000000), # read address of BRAM, real part of state vector
      OP_MOV_T2A]

for i in range(NS):
    data_array += [
    OP_READ_RAM2U,
    OP_FETCH8U,
    OP_INC_A # move to the next address.
    ]  

data_array += [OP_SEND8T, ib8(0x2000000000000000), # read address of BRAM, imaginary part of state vector
      OP_MOV_T2A]
for i in range(NS):
    data_array += [
    OP_READ_RAM2U,
    OP_FETCH8U,
    OP_INC_A # move to the next address.
    ]  

# data_array += [OP_SEND8T, ib8(0x0400_0000_0000_0000), # read address of BRAM, imaginary part of state vector
#       OP_MOV_T2A]
# for i in range(NS):
#     data_array += [
#     OP_READ_RAM2U,
#     OP_FETCH8U,
#     OP_INC_A # move to the next address.
#     ]  

# data_array += [OP_SEND8T, ib8(0x0800_0000_0000_0000), # read address of BRAM, imaginary part of state vector
#       OP_MOV_T2A]
# for i in range(Np*3):
#     data_array += [
#     OP_READ_RAM2U,
#     OP_FETCH8U,
#     OP_INC_A # move to the next address.
#     ]  
ND = 0
for i, b in enumerate(data_array):
    ND += len(b)
print("outputting to", output_command_sv, "...")
f = open(output_command_sv, "w") # generate the same byte sequence above in verilog format so that we can run the testbench simulation with the same input to be supplied here by this python code.
f.write(f"integer t_L2Addr   = {t_L2Addr};\n")
f.write(f"integer t_L2PipeGC = {t_L2PipeGC};\n")
f.write(f"integer tb_B2GenCost= {tb_B2GenCost};\n")
f.write(f"integer t_L2Pipe  = {t_L2Pipe};\n")
f.write(f"integer nPLayer   = {nPLayer};\n")
f.write(f"integer L1Qbit    = {L1Qbit};\n")
f.write(f"integer AddrMask  = {AddrMask};\n")
f.write(f"integer t_B2GenCost  = {t_B2GenCost};\n")
f.write(f"integer tb_B2Mixer  = {tb_B2Mixer};\n")
f.write(f"integer t_L2Compute  = {t_L2Compute};\n")
f.write(f"integer seed = {seed};")
AC = [b"".join(data_array)]
f.write(f"// Version {random.random()}, {datetime.datetime.now()}\n")
f.write(f"localparam ND={ND};\n")
f.write(f"logic [7: 0] data_array [{ND}] = {{{ lineend}")
for i, b in enumerate(data_array):
    # print(type(b), b, len(b))
    for j in range(len(b)):
        f.write(f"8'h{ib1(b[j]).hex()}")
        if j != len(b)-1:
            f.write(", ")

    if i != len(data_array) -1:
        f.write(f",")
    skey = b.hex()
    if skey in idop:
        if lineend != "":
            f.write(f" // {idop[skey]}{lineend}")
    elif len(b) == 8:
        if lineend != "":
            f.write(f" // {bfp64(b)}{lineend}")
f.write("};" + lineend)
f.close()
f = open("resultpy.txt", "w")
# send to serial interface, for physical test of the implementation.
start2 = time.perf_counter()
end2 = time.perf_counter()
ser = serial.Serial(port = uart_port, baudrate = baud_rate, timeout=None)
for b in data_array:
    if b == HOST_WAIT:
        # must wait here.
        start2 = time.perf_counter()

        for i in range(1024):
            print("waiting...:")
            ser.write(OP_MOV_S2U)
            ser.write(OP_FETCH1U)
            dr = ser.read(1)
            ir = int.from_bytes(dr, "little")
            opecode = ""
            if dr.hex() in idop:
                opecode = idop[dr.hex()]
            print("Status:", opecode, ir)
            if dr == qa_WAIT: # check the state is qa_WAIT
                break 
            time.sleep(0.01)
        end2 = time.perf_counter()

        continue

    fr = float("inf")
    opecode = ""
    if len(b) == 8:
        fr = bfp64(b)
    if b.hex() in idop:
        opecode = idop[b.hex()]

    print("writing:", b.hex(), fr, opecode)
    ser.write(b); time.sleep(2e-4)  # recommend to wait tiny period of time, to prevent UART buffer overflow.
    if b == OP_FETCH8U:
        dr = ser.read(8)
        fr = bfp64(dr)
        print(f"Received {8} bytes: hex={dr.hex()} fp64={fr}, int64={int.from_bytes(dr, "little")}\n")
        f.write(f"{fr}\n")
    elif b == OP_FETCH1U:
        dr = ser.read(1)
        print(f"Received {1} bytes: hex={dr.hex()} int={int.from_bytes(dr, "little")}\n")
ser.close()
f.close()
f = open(simpath, "a")

f.write(f"\n---------------------------------------\n")    
f.write(f"  summary of the computation \n")    
f.write(f"---------------------------------------\n\n")   
dt1 =  end1 - start1
dt2 =  end2 - start2
f.write(f"Python time: {dt1:.6f} s\n")    
f.write(f"PFGA time: {dt2:.6f} s\n")    
f.write(f"python/PFGA: {dt1/dt2:.6f}\n")    
f.close()

