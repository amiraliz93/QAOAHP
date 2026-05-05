`timescale 1ns / 1ns
// program, split uart functionality to reduce the simulation time
// employment of block design.
// implement all brams.

// todo
// test transmission
// todo 2026, 1, 1.
// set maxAddr, using 64
// set NQbits, using 65
// test mixer, we need theoretical value for this test bench. I need to write some python code

module addr_gen_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;
parameter CLOCKWIDTH = 10;
parameter CLOCKWIDTH_HALF = 5;
parameter CMD_WAIT = CLOCKWIDTH*20;

parameter NM = 16; // address width for state vector's and cost function's BRAM. Thus, the number of maximum qubits the system can deal with.
parameter P = 64; // data width for numerical number
localparam N_BIT_SWAP_POINTER = $clog2(NM);

reg [7:0] ag_addr_Param;
reg [NM-1:0] ag_Param;
reg ag_wen;

wire ag_enPipe;
wire ag_enCostF;
reg f_run_Computation;
wire f_L1Computation; // last 1 clock before the computation ends.
wire [15:0] ag_en_Inits;
wire [1:0] ag_mixSwitch;

wire [NM-1:0] cAddr;       // Next address counter
wire [NM-1:0] cAddrCF;     // Next address counter
wire [N_BIT_SWAP_POINTER-1:0] bsp1;
wire [N_BIT_SWAP_POINTER-1:0] bsp2;
wire [31:0] version;
// we need transmitter and receiver to tset state machine (ntu_smachine)
addr_gen #(.N_BIT_SWAP_POINTER(N_BIT_SWAP_POINTER), .NM(NM)) addr_gen_inst(
   .CLK(CLK),
   .RST(RST), 
   // connected to control interface
   .addr_Param_in(ag_addr_Param),
   .Param_in(ag_Param),
   .wen_in(ag_wen),
   // connected to qaoa_system
   .en_Pipe_out(ag_enPipe),
   .en_CostF_out(ag_enCostF),
   .f_run_Computation_in(f_run_Computation),
   .f_L1Computation_out(f_L1Computation),
   .en_Inits_out(ag_en_Inits),
   .mixSwitch_out(ag_mixSwitch),
   .cAddrCF_out(cAddrCF),
   .cAddr_out(cAddr),
   // connected to bit_swap
   .bsp1_out(bsp1),
   .bsp2_out(bsp2),
   .version(version)
)
;
localparam AG_SET_t_L2Addr   = 0;
localparam AG_SET_t_L2PipeGC = 1;
localparam AG_SET_tb_B2GenCost= 2;
localparam AG_SET_t_L2Pipe  = 3;
localparam AG_SET_nPLayer   = 4;
localparam AG_SET_L1Qbit    = 5;
localparam AG_SET_AddrMask  = 6;
localparam AG_SET_t_B2GenCost  = 7;
localparam AG_SET_t_B2Mixer  = 8;
localparam AG_SET_t_L2Compute  = 9;

integer t_L2Addr   = 62;
integer t_L2PipeGC = 225;
integer tb_B2GenCost= 394;
integer t_L2Pipe  = 87;
integer nPLayer   = 8;
integer L1Qbit    = 5;
integer AddrMask  = 31;
integer t_B2GenCost  = 38;
integer tb_B2Mixer  = 87;

integer i;

// test for transmitter
initial begin 
    RST <= 0;
    CLK <= 0;
    f_run_Computation <= 0;
    ag_wen <= 0;
 
    // Apply reset
    #CMD_WAIT;
    RST <= 1; // Reset active-high
    #CMD_WAIT;
    RST <= 0; 
    ag_addr_Param <= AG_SET_t_L2Addr;
    ag_Param <= t_L2Addr; // assuming 6 qbits
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_L2PipeGC;
    ag_Param <= t_L2PipeGC;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_tb_B2GenCost;
    ag_Param <= tb_B2GenCost;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_L2Pipe;
    ag_Param <= t_L2Pipe; 
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_nPLayer;
    ag_Param <= nPLayer;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_L1Qbit;
    ag_Param <= L1Qbit;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_AddrMask;
    ag_Param <= AddrMask;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_B2GenCost;
    ag_Param <= t_B2GenCost; // need to wait first 16 cyclea, so that qaoa_system2.sv can load cos, sin, gamma.
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_B2Mixer;
    ag_Param <= tb_B2Mixer; // need to wait first 16 cyclea, so that qaoa_system2.sv can load cos, sin, gamma.
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_L2Compute;
    ag_Param <= tb_B2Mixer; // need to wait first 16 cyclea, so that qaoa_system2.sv can load cos, sin, gamma.
    ag_wen <= 1;
    #CLOCKWIDTH
    f_run_Computation <= 1;
    ag_wen <= 0;
    #CLOCKWIDTH

    for(i=0;i<((t_L2Pipe + 2)*(L1Qbit + 2) + tb_B2Mixer)*(nPLayer+2);i = i+1)  begin 
      #CLOCKWIDTH;
    end
    #CMD_WAIT;
    #CMD_WAIT;
    $stop;
      
end
always
begin
      #CLOCKWIDTH_HALF;
      CLK <= ~CLK; // clock generation, half period
end

always @(posedge CLK) begin 
    if(f_L1Computation) begin 
        f_run_Computation <= 0;
    end
end
endmodule
