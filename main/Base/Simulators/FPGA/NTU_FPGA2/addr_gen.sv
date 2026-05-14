
//============================================================================
// Documentation
//============================================================================
// Author: Amir Alizadeh (NTU), Hiroki Shibata (Tokyo Metropolitan University)
// Date: November 2026

// Address controling unit. This unit generate all signals and addresses
//  required for computation.
// Controling parallel implementation is now under the consideration.
// F_max = 539 MHz, on 20260429.
// 
// Futher pipeline for logics can be possible, by introducing t_L3Addr, t_L3Pipe.
// If we have t_L3Addr, t_L3Pipe, we can have a switch to hide the case t_L2Addr = t_L2Pipe. The case t_L2Addr = t_L2Pipe is problematic because generally we off the flag at t_L1Pipe, and at t_L1Pipe. If we use t_L2Pipe, no enough time to prepare this problematic case. But having t_L3Pipe, there is possible way. 
// This introduction require D > 4, but it does not contradict the non stopping pipeline. Consider if F_max is constrained by this module.
//===========================================================================
// ------------------------------------
// Control parameters of computation.
// Need to be set timing parameters by host. Usually, thoes parameters 
// begin with t_* for thier name
// -------------------------------------

// Parallel implementations is in a way having indexes,  i_0 = i , i_1 = (i+1)%NM,  i_2 = (i+2)%NM, ... so on. cAddr > maxAddr, is OK, but take reminder of it. 
// !! with L lantency, we can read L data successively from the same BRAM, in parallel mode.
// In QA implementation, I think we do not need busrt transfer mode, because the source block does not need memory address of required data. it can be calculated at source side as well because it is determined primialriry.
module addr_gen#(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter NM = 32,   // Max address counter, can be greater than BRAM address width (2^13 = 8192 elements)
    parameter N_BIT_SWAP_POINTER=5)    // Info/ control signal width
  (
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  CLK,     // system clock
    input  RST,     // Active-high reset
    //-------------------------------------------
    input [7:0] addr_Param_in,
    input [NM-1:0] Param_in,
    input wen_in,

    input f_run_Computation_in,
    output [NM-1:0] cAddrGC_out,
    output [NM-1:0] cAddr_out,
    output [N_BIT_SWAP_POINTER-1:0] bsp1_out, // bit swap pointer 1, always 0.
    output [N_BIT_SWAP_POINTER-1:0] bsp2_out, // bit swap pointer 2
    output en_Pipe_out,
    output en_CostF_out,

    output reg [31:0] version,
    output [1:0] mixSwitch_out,
    output [23:0] en_Inits_out, // notify the system to prepare variables.
    output f_L1Computation_out // last 1 clock before the computation ends.
);
localparam AG_SET_t_L2Addr   = 0;
localparam AG_SET_t_L2PipeGC = 1;
localparam AG_SET_tb_B2GenCost= 2;
localparam AG_SET_t_L2Pipe  = 3;
localparam AG_SET_nPLayer   = 4;
localparam AG_SET_L1Qbit    = 5;
localparam AG_SET_AddrMask  = 6;
localparam AG_SET_t_B2GenCost  = 7;
localparam AG_SET_tb_B2Mixer  = 8;
localparam AG_SET_t_L2Compute  = 9;

reg wen;
reg [7:0] addr_Param;
reg [NM-1:0] Param;
reg [N_BIT_SWAP_POINTER-1:0] bsp1; // next bit swap pointer 1
reg [N_BIT_SWAP_POINTER-1:0] bsp2; // next bit swap pointer 2
assign bsp1_out = bsp1;
assign bsp2_out = bsp2;
logic [N_BIT_SWAP_POINTER-1:0] n_bsp1; // next bit swap pointer 1
logic [N_BIT_SWAP_POINTER-1:0] n_bsp2; // next bit swap pointer 2
reg [23:0] en_Inits; // notify the system to prepare variables.
assign en_Inits_out = en_Inits;

//----------------------------------------------------------------------------
// Address Counters
//----------------------------------------------------------------------------
reg [NM-1:0] cAddr;         // Current address counter (mixer loop)
reg [NM-1:0] cAddrGC;         // Current address counter (mixer loop)
reg [NM-1:0] AddrMask;    // Mask to get reminder of valid address.
logic [NM-1:0] n_cAddr;     // Next address counter
logic [NM-1:0] n_cAddrGC;     // Next address counter
logic [NM-1:0] n_AddrMask;    // Mask to get reminder of valid address.
assign cAddrGC_out = cAddrGC;
assign cAddr_out = cAddr;

reg [31:0] cPLayer; logic [31:0] n_cPLayer;
//-----------------------------------------------------------

logic n_en_Pipe;
logic [1:0] n_mixSwitch;
assign en_Pipe_out = n_en_Pipe;
assign mixSwitch_out = n_mixSwitch;
// ------------------------------------
// Control parameters of computation.
// Need to be programed by host.
// -------------------------------------
reg f_run_Computation2;
reg f_run_Computation1;
reg f_run_Computation;
reg [NM-1:0] c_Compute; logic [NM-1:0] n_c_Compute; // Counts after starting p layer.
reg [NM-1:0] t_B2GenCost;  logic [NM-1:0] n_t_B2GenCost; 
reg [NM-1:0] to_B2GenCost;  logic [NM-1:0] n_to_B2GenCost; 
reg [NM-1:0] t_L2Compute;  logic [NM-1:0] n_t_L2Compute; 
reg [NM-1:0] t_L2Addr;  logic [NM-1:0] n_t_L2Addr; // 2^{N}-2
reg [NM-1:0] t_L2PipeGC;  logic [NM-1:0] n_t_L2PipeGC;  // Tc -2, where Tc is the length of gen_cost pipeline, not 2^{N-1} nor T. 
reg [NM-1:0] t_B2Mixer;  logic [NM-1:0] n_t_B2Mixer;  // Tc -2, where Tc is the length of gen_cost pipeline, not 2^{N-1} nor T. 
reg [NM-1:0] tb_B2Mixer;  logic [NM-1:0] n_tb_B2Mixer;  // Tc -2, where Tc is the length of gen_cost pipeline, not 2^{N-1} nor T. 
reg [NM-1:0] tb_B2GenCost;  logic [NM-1:0] n_tb_B2GenCost; 
reg [NM-1:0] t_L2Pipe; logic [NM-1: 0] n_t_L2Pipe;  // time the pipeline get valid, it is T -3 == 2^{N-1}-3. If T <= 2^{N-1}, then set T = 2^{N-1},
reg [NM-1:0] nPLayer; logic [NM-1:0] n_nPLayer;
reg [NM-1:0] L1Qbit; logic [NM-1:0] n_L1Qbit;

reg en_CostF; logic n_en_CostF;
assign en_CostF_out = en_CostF;
assign en_Pipe_out = n_en_Pipe;
reg en_mixer; logic n_en_mixer;
reg f_L1Mixer; logic lf_L2Mixer;  // last 1 mixer operation. Need to determine to start the next cost function, or finish the computation.
reg f_B1Mixer; logic lf_B2Mixer;  // last 1 mixer operation. Need to determine to start the next cost function, or finish the computation.
reg f_B1CostF; logic lf_B2CostF;
reg f_L1Pipe; logic lf_L2Pipe;
reg f_L1Addr; logic lf_L2Addr;
reg f_L1CostF; logic lf_L2CostF;
reg f_L1All; logic lf_L2All; 
reg f_L1Compute; logic lf_L2Compute; 
assign f_L1Computation_out = f_L1Compute;
reg f_B1GenCost; logic lf_B2GenCost;
reg v_mixer; logic n_v_mixer;
reg v_GenCost; logic n_v_GenCost;
reg v_Flushing; logic n_v_Flushing;

integer i;
always_comb begin: computingBlock
    // -----------------------------
    // Block to load cost function simulteneously at any time.
    // initialize variables is defined. f_run_Computation control initilization and execution.
    // -----------------------------
   
    n_t_L2Addr     = t_L2Addr;
    n_t_L2PipeGC   = t_L2PipeGC;
    n_t_B2GenCost = t_B2GenCost; // the inital value. After second P, tb_B2GenCost is used and tb_B2GenCost can be programmable.
    n_to_B2GenCost = to_B2GenCost; // the inital value. After second P, tb_B2GenCost is used and tb_B2GenCost can be programmable.
    n_tb_B2GenCost = tb_B2GenCost;
    n_t_L2Pipe  = t_L2Pipe;
    n_nPLayer = nPLayer;
    n_L1Qbit    = L1Qbit;
    n_AddrMask  = AddrMask;
    n_t_B2Mixer  = t_B2Mixer;
    n_tb_B2Mixer  = tb_B2Mixer;
	n_t_L2Compute = t_L2Compute;
    if(wen) begin
        case(addr_Param) 
            AG_SET_t_L2Addr: begin
                n_t_L2Addr   = Param;
            end
            AG_SET_t_L2PipeGC: begin
                n_t_L2PipeGC = Param;
            end
            AG_SET_tb_B2GenCost: begin
                n_tb_B2GenCost = Param;
            end
            AG_SET_t_L2Pipe: begin
                n_t_L2Pipe  = Param;
            end
            AG_SET_nPLayer: begin
                n_nPLayer = Param;
            end
            AG_SET_L1Qbit: begin
                n_L1Qbit    = Param;
            end
            AG_SET_AddrMask: begin
                n_AddrMask  = Param;
            end
            AG_SET_t_B2GenCost: begin 
                n_t_B2GenCost = Param;
                n_to_B2GenCost = Param;
            end
            AG_SET_tb_B2Mixer: begin 
                n_tb_B2Mixer = Param;
            end
            AG_SET_t_L2Compute: begin 
                n_t_L2Compute = Param;
            end
				default: begin end
        endcase
    end
    // ---------
    // f_mixOK and t_B2GenCost, tb_B2GenCost, is used for a trick to skip the first mixer and start from the cost function.
    n_cPLayer = 0;
    n_bsp2 = 0;
    n_bsp1 = 0;
    n_v_Flushing = 0;

    n_en_CostF = 0;
    n_cAddrGC = 0;
    n_cAddr = 0;
    n_c_Compute = 0;
    n_mixSwitch = 'b00; 
    n_en_Pipe = '0; // with en_Pipe = 0, result of the compute will not be written back to the memory.
    n_en_mixer = 0;
    n_v_mixer = 0;
    n_v_GenCost = 0;
    // -------------------------------------------------------
    // Block to generate flags.
    // -------------------------------------------------------

    lf_L2Addr = (cAddr == t_L2Addr);
    lf_L2Pipe = (cAddr == t_L2Pipe);
    lf_L2All = (cPLayer == nPLayer) && lf_L2Pipe && (bsp2 == L1Qbit);
    lf_B2GenCost = (t_B2GenCost == c_Compute);
    lf_L2Mixer = lf_L2Pipe && (bsp2 == L1Qbit);
    lf_B2Mixer = (t_B2Mixer == c_Compute);
    lf_L2CostF = lf_L2Addr && en_CostF;
    lf_B2CostF = (cAddrGC == t_L2PipeGC) && v_GenCost;
    lf_L2Compute = (cPLayer == nPLayer) && (t_L2Compute == c_Compute);
    
    // --------- Checking list -----------
    // - set overwritten function for L1QBit,s like that, . ok 20260428
    // - make sure the block of qa_RUNC, with regard to bs_info, . ok 20260428
    // - initialize registers. . ok 20260428
    // - timing of v_mixer is susupicious,
    // - timing document must be created, 
    // - prepare software side to set registers like L1QBit. 
    // - check default values in comb blocks. . ok 20260428
    // - consider to move this block to independent module file. We can test only addressing function of this block. . ok 20260428

    if(f_run_Computation) begin 
        n_c_Compute = c_Compute + 1'b1;
        n_en_CostF = en_CostF;
        n_en_mixer = en_mixer;
        n_cAddr = cAddr + 1'b1; 
        n_cAddrGC = cAddrGC + 1'b1;
        n_bsp1 = bsp1;
        n_en_Pipe       = (en_mixer || en_CostF) && (~v_Flushing); // enable write back the result
        n_mixSwitch[0] = ~cAddr[0] && ~en_CostF; // 0 for cost
        n_mixSwitch[1] = cAddr[0]  && ~en_CostF; // 0 for cost
        n_v_mixer = v_mixer;
        n_v_GenCost = v_GenCost;
        n_v_Flushing = v_Flushing;
        
        n_bsp2 = bsp2;
        n_t_B2GenCost = t_B2GenCost;
        n_cPLayer = cPLayer;                // Keep layer counter
        // -------------------------------------------------------
        // Computational controling block. 
        // f_run_Computation is used to define initial state
        // initial state consits of default values in this block, computingBlock.
        // EX1: A Block is an exclusive switching block. Because cost operator shares some of control registers with mixer operator, it needs exclusive conrol block as following
        // -------------------------------------------------------
        if(f_L1Pipe && v_mixer) begin 
            n_bsp2 = bsp2 + 1'b1;
        end
        else if(f_B1Mixer || f_B1CostF) begin
            n_bsp2 = '0;
        end

        case({f_L1CostF, f_B1CostF && (~f_L1All)})
            2'b01:begin 
                n_bsp2 = '0;
                n_c_Compute = '0;
                n_t_B2GenCost = tb_B2GenCost; // trick to skip the first mixer.
                n_cPLayer = cPLayer + 1;
                n_en_CostF = 1;
                n_t_B2Mixer = tb_B2Mixer;
            end
            2'b10: begin 
                n_en_CostF = '0;
                n_v_GenCost = 0;
            end
				default: begin end
        endcase
        if(f_B1Mixer) begin 
            n_v_mixer = 1;
        end
        else if(f_B1CostF) begin 
            n_v_mixer = 0;
        end
        if(f_L1Addr) begin
            n_en_mixer = 0;
        end
        if(f_L1Pipe || f_B1Mixer) begin // note: over write block if(f_L1Addr), in the case none-stopping condition
            n_en_mixer = n_v_mixer;
        end
        if(f_L1All) begin 
            n_v_Flushing = 1;
        end
        
        if(f_L1Pipe || f_B1Mixer) begin
            n_cAddr = '0; // start main pipeline.
        end
        if(f_B1GenCost) begin
            n_cAddrGC = '0; // Start cost function generation.
            n_v_GenCost = 1;
        end

        if(f_L1Compute) begin 
            // restore the all controle values.
            n_t_B2Mixer = '1;
            n_t_B2GenCost = to_B2GenCost;
        end

    end 

end

always_ff @(posedge CLK) begin 
    if(RST) begin
        t_B2GenCost <= '1;
        to_B2GenCost <= '1; 
        t_L2Addr   <= '1;
        t_B2Mixer <= '1;
        tb_B2Mixer <= '1;
        t_L2PipeGC <= '1;
        tb_B2GenCost <= '1;
        t_L2Pipe  <= '1;
        nPLayer <= '1;
        L1Qbit <= '1;
        AddrMask <= '1;
        t_L2Compute <= '1;
        wen <= 0;
        f_L1Compute <= 0;
        f_run_Computation2 <= 0;
        f_run_Computation1 <= 0;
        f_run_Computation <= 0;
        en_mixer <= 0;
        en_CostF <= 0;
    end
    else begin 
        t_B2GenCost <= n_t_B2GenCost; 
        to_B2GenCost <= n_to_B2GenCost; 
        t_B2Mixer <= n_t_B2Mixer;
        tb_B2Mixer <= n_tb_B2Mixer;
        t_L2Compute   <= n_t_L2Compute;
        t_L2Addr   <= n_t_L2Addr;
        t_L2PipeGC <= n_t_L2PipeGC;
        tb_B2GenCost <= n_tb_B2GenCost;
        t_L2Pipe  <= n_t_L2Pipe;
        nPLayer <= n_nPLayer;
        L1Qbit <= n_L1Qbit;
        AddrMask <= n_AddrMask;
        f_L1Compute <= lf_L2Compute;
        wen <= wen_in;
        f_run_Computation2 <= f_run_Computation_in;
        f_run_Computation1 <= f_run_Computation2;
        f_run_Computation <= f_run_Computation1;    
        en_mixer <= n_en_mixer;
        en_CostF <= n_en_CostF;
    end
    v_Flushing <= n_v_Flushing;
    f_L1All <= lf_L2All;
    v_mixer <= n_v_mixer;
    v_GenCost <= n_v_GenCost;
    f_B1Mixer <= lf_B2Mixer;
    version <= 'hfa920a2d;
    cAddr     <= n_cAddr;
    cAddrGC   <= n_cAddrGC;
    c_Compute  <= n_c_Compute;
    bsp1   <= n_bsp1;
    bsp2   <= n_bsp2;
    f_L1CostF <= lf_L2CostF;
    f_L1Addr  <= lf_L2Addr;
    f_L1Pipe  <= lf_L2Pipe;
    f_B1CostF <= lf_B2CostF;
    f_L1Mixer <= lf_L2Mixer;  
    f_B1GenCost <= lf_B2GenCost;
    cPLayer <= n_cPLayer;

    addr_Param <= addr_Param_in;
    Param <= Param_in;

    for(i = 0;i<24;i=i+1) begin
        en_Inits[i] <= (c_Compute == i) && f_run_Computation;
    end
end
endmodule