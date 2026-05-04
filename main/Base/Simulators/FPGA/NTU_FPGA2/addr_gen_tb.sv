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
parameter CLOCKWIDTH = 2;
parameter CLOCKWIDTH_HALF = 1;
parameter CLOCKWIDTH500 = CLOCKWIDTH*50;
parameter CMD_WAIT = CLOCKWIDTH*20;
parameter TIME_WAIT_TB = CLOCKWIDTH*50;

parameter NM = 13; // address width for state vector's and cost function's BRAM. Thus, the number of maximum qubits the system can deal with.
parameter P = 64; // data width for numerical number
parameter Ni = 32;// data width of auxiary information on pipeline.
parameter NBRAM = 6; // number of block RAMs connected to qaoa system.
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
localparam AG_SET_t_L2PipeCF = 1;
localparam AG_SET_tb_B2GenCost= 2;
localparam AG_SET_t_L2Pipe  = 3;
localparam AG_SET_nL1PLayer = 4;
localparam AG_SET_L1Qbit    = 5;
localparam AG_SET_AddrMask  = 6;
localparam AG_SET_t_B2GenCost  = 7;
localparam NP = 4;
localparam D = 64;
localparam N = 6;
localparam Tc = 131;
localparam Lpipe = 64; //D/2 + T; // if LPipe < D, then LPipe = D;
localparam DVTc = 2; // make sure, pipe*DVTc-Tc-2 > 16.

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
    ag_Param <= D-2; // assuming 6 qbits
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_L2PipeCF;
    ag_Param <= Tc-2;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_tb_B2GenCost;
    ag_Param <= Lpipe*(N+1)-Tc-2;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_L2Pipe;
    ag_Param <= Lpipe-2; 
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_nL1PLayer;
    ag_Param <= NP;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_L1Qbit;
    ag_Param <= N-1;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_AddrMask;
    ag_Param <= 64'b0000_0011_1111;
    ag_wen <= 1;
    #CLOCKWIDTH
    ag_addr_Param <= AG_SET_t_B2GenCost;
    ag_Param <= Lpipe*DVTc-Tc-2; // need to wait first 16 cyclea, so that qaoa_system2.sv can load cos, sin, gamma.
    ag_wen <= 1;
    #CLOCKWIDTH
    f_run_Computation <= 1;

    for(i=0;i<Lpipe*(N+1)*(NP+1);i = i+1)  begin 
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
