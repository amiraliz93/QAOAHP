
//============================================================================
// Documentation
//============================================================================
// Author: Amir Alizadeh (NTU), Hiroki Shibata (Tokyo Metropolitan University)
// Date: November 2026

// Address controling unit. This unit generate all signals and addresses
//  required for computation.
// Controling parallel implementation is now under the consideration.
//===========================================================================

// ------------------------------------
// Control parameters of computation.
// Need to be set timing parameters by host. Usually, thoes parameters 
// begin with t_* for thier name
// -------------------------------------

module addr_gen#(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter NM = 13,   // BRAM address width (2^13 = 8192 elements)
    parameter BCVW=32,     // must be greater than or equal to 32.
    parameter N_BIT_SWAP_POINTER = $clog2(NM))    // Info/ control signal width
  (
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  CLK,     // system clock
    input  RST,     // Active-high reset
    //-------------------------------------------
    input [4:0] addr_Param_in;
    input [BCVW-1:0] Param_in;
    input wen_in;

    input f_run_Computation_in,
    output [NM-1:0] cAddrCF_out;
    output [NM-1:0] cAddr_out;
    output [N_BIT_SWAP_POINTER-1:0] bsp1_out, // bit swap pointer 1, always 0.
    output [N_BIT_SWAP_POINTER-1:0] bsp2_out, // bit swap pointer 2
    output en_Pipe_out,
    output en_CostF_out,
    output [1:0] mixSwitch_out,
    output [15:0] en_Inits_out, // notify the system to prepare variables.
    output f_L1Computation_out // last 1 clock before the computation ends.
);
localparam AG_SET_n_t_L2Addr = 6'b00_0001;
localparam AG_SET_t_L2PipeCF = 6'b00_0010;
localparam AG_SET_tb_B2GenCost= 6'b00_0100;
localparam AG_SET_t_L2Pipe  = 6'b00_1000;
localparam AG_SET_nL1PLayer = 6'b01_0000;
localparam AG_SET_n_L1Qbit  = 6'b10_0000;

reg wen;
reg [4:0] addr_Param;
reg [BCVW-1:0] Param;
reg [N_BIT_SWAP_POINTER-1:0] bsp1; // next bit swap pointer 1
reg [N_BIT_SWAP_POINTER-1:0] bsp2; // next bit swap pointer 2
assign bsp1_out = bsp1;
assign bsp2_out = bsp2;
logic [N_BIT_SWAP_POINTER-1:0] n_bsp1; // next bit swap pointer 1
logic [N_BIT_SWAP_POINTER-1:0] n_bsp2; // next bit swap pointer 2
reg [15:0] en_Inits; // notify the system to prepare variables.
assign en_Inits_out = en_Inits;

//----------------------------------------------------------------------------
// Address Counters
//----------------------------------------------------------------------------
reg [NM-1:0] cAddr;         // Current address counter (mixer loop)
reg [NM-1:0] cAddrCF;         // Current address counter (mixer loop)
logic [NM-1:0] n_cAddr;     // Next address counter
logic [NM-1:0] n_cAddrCF;     // Next address counter
assign cAddrCF_out = cAddrCF;
assign cAddr_out = cAddr;

//-----------------------------------------------------------

logic n_en_Pipe;
logic n_mixSwitch;
assign en_Pipe_out = n_en_Pipe;
assign mixSwitch_out = n_mixSwitch;
// ------------------------------------
// Control parameters of computation.
// Need to be programed by host.
// -------------------------------------
reg [BCVW-1:0] c_Compute; logic [BCVW-1:0] n_c_Compute; // Counts after starting p layer.
reg [BCVW-1:0] t_B2GenCost;  logic [BCVW-1:0] n_t_B2GenCost; 
reg [BCVW-1:0] t_L2Addr;  logic [BCVW-1:0] n_t_L2Addr; // 2^{N}-2
reg [BCVW-1:0] t_L2PipeCF;  logic [BCVW-1:0] n_t_L2PipeCF;  // Tc -2, where Tc is the length of gen_cost pipeline, not 2^{N-1} nor T. 
reg [BCVW-1:0] tb_B2GenCost;  logic [BCVW-1:0] n_tb_B2GenCost; 
reg [BCVW-1:0] t_L2Pipe; logic [BCVW-1: 0] n_t_L2Pipe;  // time the pipeline get valid, it is T -3 == 2^{N-1}-3. If T <= 2^{N-1}, then set T = 2^{N-1},
reg [BCVW-1:0] nL1PLayer; logic [BCVW-1:0] n_nL1PLayer;
reg [NM-1:0] L1Qbit; logic [NM-1:0] n_L1Qbit;

reg en_CostF; logic n_en_CostF;
assign en_CostF_out = en_CostF;
assign en_Pipe_out = n_en_Pipe;
reg en_mixer; logic n_en_mixer;
reg f_L1Mixer; logic lf_L2Mixer;  // last 1 mixer operation. Need to determine to start the next cost function, or finish the computation.
reg f_B1CostF; logic lf_B2CostF;
reg f_L1Pipe; logic lf_L2Pipe;
reg f_L1Addr; logic lf_L2Addr;
reg f_L1CostF; logic lf_L2CostF;
reg f_L1Compute; logic lf_L2Compute; assign f_L1Computation_out = f_L1Compute;
reg v_mixOK; logic n_v_mixOK;

always_comb begin: computingBlock
    // -----------------------------
    // Block to load cost function simulteneously at any time.
    // initialize variables is defined. f_run_Computation control initilization and execution.
    // -----------------------------
    n_cPLayer = 0;
    n_P5pointer = 0;
    n_bsp2 = 0;
    n_bsp1 = 0;
    n_tb_B1GenCost = tb_B1GenCost;
    n_t_L2Addr = t_L2Addr;
    n_t_endCostF = t_endCostF;
    n_t_endCostFM1 = t_StartCostF;
    n_t_endMixPipe = t_endMixPipe;
    n_L1Qbit = 3;

    // ---------
    // f_mixOK and t_B2GenCost, tb_B2GenCost, is used for a trick to skip the first mixer and start from the cost function.
    n_v_mixOK = 0; // set to zero for initial time, so that the system can skip the first mixer.
    n_t_B2GenCost = 16; // the inital value. After second P, tb_B2GenCost is used and tb_B2GenCost can be programmable.
    
    n_en_CostF = 0;
    n_cAddrCF = 0;
    n_cAddr = 0;
    n_c_Compute = 0;
    n_mixSwitch = 'b00; 
    n_en_Pipe = '0; // with en_Pipe = 0, result of the compute will not be written back to the memory.
    n_en_mixer = 0;
    // -------------------------------------------------------
    // Block to generate flags.
    // -------------------------------------------------------

    lf_L2Addr = (cAddr == t_L2Addr);
    lf_L2Pipe = (cAddr == t_L2Pipe);
    lf_L2Compute = (cPLayer == nL1PLayer) && lf_L2Pipe && en_CostF;
    lf_B2GenCost = (t_B2GenCost == c_Compute);
    lf_L2Mixer = f_L2Addr && (bsp2 == L1Qbit);
    lf_L2CostF = f_L2Addr && en_CostF;
    lf_L2CostGen = (cAddrCF == t_L2Addr);
    lf_B2CostF = (cAddrCF == t_L2PipeCF);
   
    // -------------------------------------------------------
    // Computational controling block. 
    // f_run_Computation is used to define initial state
    // initial state consits of default values in this block, computingBlock.
    // EX1: A Block is an exclusive switching block. Because cost operator shares some of control registers with mixer operator, it needs exclusive conrol block as following
    // -------------------------------------------------------
    case({f_L1CostF,f_L1Addr, f_L1Mixer}) // Block EX1
        2'b001: begin
            n_bps2 = 0;
            n_v_mixOK = 0;
            n_t_B2GenCost = t_B2GenCost; 
            n_cPLayer = cPLayer;                // Keep layer counter
        end
        2'b010: begin 
            n_bps2 = bps2 + 1;
            n_v_mixOK = v_mixOK;
            n_t_B2GenCost = t_B2GenCost; 
            n_cPLayer = cPLayer;                // Keep layer counter
        end
        2'b100: begin
            n_bps2 = 0;
            n_v_mixOK = 1;
            n_t_B2GenCost = tb_B2GenCost; // trick to skip the first mixer.
            n_cPLayer = cPLayer + 1;
        end 
        default: begin 
            n_bsp2 = bsp2;
            n_v_mixOK = v_mixOK;
            n_t_B2GenCost = t_B2GenCost;
            n_cPLayer = cPLayer;                // Keep layer counter
        end
    endcase
    // --------- Checking list -----------
    // - set overwritten function for L1QBit,s like that, . ok 20260428
    // - make sure the block of qa_RUNC, with regard to bs_info, . ok 20260428
    // - initialize registers. . ok 20260428
    // - timing of v_mixOK is susupicious,
    // - timing document must be created, 
    // - prepare software side to set registers like L1QBit. 
    // - check default values in comb blocks. . ok 20260428
    // - consider to move this block to independent module file. We can test only addressing function of this block. . ok 20260428

    if(f_run_Computation) begin 
        n_c_Compute = c_Compute + 1;
        n_v_Addr = v_Addr;
        n_en_CostF = en_CostF;
        n_en_mixer = en_mixer;
        n_cAddr = cAddr + 1; 
        n_cAddrCF = cAddrCF + 1;
        n_P5pointer = P5pointer;  // Keep parameter pointer
        n_v_CompPipe = v_CompPipe;
        n_bsp2 = bsp2;
        n_bsp1 = bsp1;
        n_en_Pipe       = en_mixer || en_CostF; // enable write back the result
        n_mixSwitch[0] = ~c_addr[0] && ~en_CostF; // 0 for cost
        n_mixSwitch[1] = c_addr[0]  && ~en_CostF; // 0 for cost

        if(f_B1CostF) begin 
            n_en_CostF = 1;
        end
        //---- exclusive block.
        case({f_L1Pipe, f_L1Addr})
            2'b01:  begin 
                n_en_mixer = 0;
            end
            2'b10: begin
                n_en_mixer = v_mixOK;
            end
        endcase
        
        if(f_L1Pipe) begin
            n_cAddr = 0; // start main pipeline.
        end
        if(f_B1GenCost) begin
            n_cAddrCF = 0; // Start cost function generation.
        end

    end 

    n_t_L2Addr     = t_L2Addr;
    n_t_L2PipeCF   = t_L2PipeCF;
    n_tb_B2GenCost = tb_B2GenCost;
    n_t_L2Pipe  = t_L2Pipe;
    n_nL1PLayer = nL1PLayer;
    n_L1Qbit    = L1Qbit;
    
    if(ag_wen) begin
        case(addr_Param_in) 
            AG_SET_n_t_L2Addr: begin
                n_t_L2Addr   = Param_in;
            end
            AG_SET_t_L2PipeCF: begin
                n_t_L2PipeCF = Param_in;
            end
            AG_SET_tb_B2GenCost: begin
                n_tb_B2GenCost = Param_in;
            end
            AG_SET_t_L2Pipe: begin
                n_t_L2Pipe  = Param_in;
            end
            AG_SET_nL1PLayer: begin
                n_nL1PLayer = Param_in;
            end
            AG_SET_n_L1Qbit: begin
                n_L1Qbit    = Param_in;
            end
        endcase
    end
end

always_ff @(posedge CLK) begin 
    if(RST) begin
        t_B2GenCost <= 'heffffff; 
        t_L2Addr   <= 'heffffff;
        t_L2PipeCF <= 'heffffff;
        tb_B2GenCost <= 'heffffff;
        t_L2Pipe  <= 'heffffff;
        nL1PLayer <= 'heffffff;
        L1Qbit <= 'heffffff;
        f_L1Compute <= 0;
        wen <= 0;
    end
    else begin 
        t_B2GenCost <= n_t_B2GenCost; 
        t_L2Addr   <= t_L2Addr;
        t_L2PipeCF <= n_t_L2PipeCF;
        tb_B2GenCost <= n_tb_B2GenCost;
        t_L2Pipe  <= n_t_L2Pipe;
        nL1PLayer <= n_nL1PLayer;
        L1Qbit <= n_L1Qbit;
        f_L1Compute <= lf_L2Compute;
        wen <= wen_in;
    end

    cAddr     <= n_cAddr;
    cAddrCF   <= n_cAddrCF;

    c_Compute  <= n_c_Compute;
    bsp1   <= n_bsp1;
    bsp2   <= n_bsp2;
    f_L1CostF <= lf_L2CostF
    f_L1Addr  <= lf_L2Addr;
    f_L1Pipe  <= lf_L2Pipe;
    f_B1CostF <= lf_B2CostF;
    f_L1Mixer <= lf_L2Mixer;  
    addr_Param <= addr_Param_in;
    Param <= Param_in;

    en_mixer <= n_en_mixer;
    en_CostF <= n_en_CostF;

    for(i = 0;i<16;i=i+1) begin
        en_Inits[i] <= (c_Compute == i) && f_run_Computation;
    end

end