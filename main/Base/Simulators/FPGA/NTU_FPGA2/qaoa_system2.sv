//============================================================================
// Documentation
//============================================================================
// Author: Amir Alizadeh (NTU), Hiroki Shibata (Tokyo Metropolitan University)
// Date: November 2025

// Description:
//   State machine controller Mixer_operation simulation. Handles:
//  Implement mixer operation of QAOA 
// BRAM nterface for quantum state storage (real.imaginary parts) 
// status reporting to control systm (ntu_smachine)

// Key Features:
//   - Supports up to 2^13 = 8192 state vector elements
//   - 6 independent BRAM banks for parallel access
//   - Pipelined mixer and cost operations
//   - Command-driven state machine (qa_WAIT, qa_RUN, qa_MIXER, qa_COST)

// Interface Protocol:
//   1. PC sends OP_SEND_CMD with qa_INIT → Initialize system
//   2. PC sends OP_SEND_CMD with qa_RUN → Start QAOA layer
//   3. System executes: qa_RUN → qa_MIXER → qa_COST → qa_WAIT
//   4. Repeat for nPLayer iterations

// BRAM Organization:
//   BRAM[0]: State vector real part (A_r)
//   BRAM[1]: State vector imaginary part (A_i)
//   BRAM[2]: Cost function coefficients
//   BRAM[5]: Parameters (cos(β), sin(β), γ)
//   BRAM[3-4]: Reserved for future use
//===========================================================================

module qaoa_system2#(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter NM = 32,   // BRAM address width (2^13 = 8192 elements)
    parameter P = 64,    // Data width (64bit FP)
    parameter NBRAM=4,   // number of Bram banks
    parameter L_BRAM_R = 3
	 )  
  (
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  CLK,     // system clock
    input  RST,     // Active-high reset
    //------------------------------------------------------------------------
    // COMMAND INTERFACE (among Control Interface and addr_gen)
    //------------------------------------------------------------------------
    input [23:0]  r_CMD, // 
    output f_run_Computation_out, 
    input f_L1Computation_in, 
    input [1:0] mixSwitch_in, 
    input enPipe_in,
    input enCostF_in,
    input [15:0] en_Inits_in,
    //------------------------------------------------------------------------
    // MEMORY INTERFACE from Control Interface
    //------------------------------------------------------------------------
    // read Interface
    input [63:0] n_r_addr,         // Read address
    input n_r_req,                 // Read request
    output reg [63:0] r_data,      // Read data output
    output reg r_vd,               // Read data valid
    output reg [63:0] Status,      // Read data valid
    
    // Write Interface
    input [63:0] n_w_addr,   // Write address
    input [63:0] n_w_data,   // Write data
    input n_w_req,           // Write request
    //------------------------------------------------------------------------
    // MIXER OPERATION INTERFACE
    //------------------------------------------------------------------------
    // Input to Mixer
    output reg [P-1:0] cosb,        // cos(β) parameter, for all pipelines
    output reg [P-1:0] sinb,        // sin(β) parameter, for all pipelines
    output reg [P-1:0] mix_ar,        // State real part input
    output reg [P-1:0] mix_ai,        // State imaginary part input
    output reg [NM:0] mix_info,     // Control info (address, enable, etc.)
    output reg [1:0] mix_switch,     // Control info (address, enable, etc.)
    

    // Output from Mixer
    input [P-1:0] mix_ar_res,
    input [P-1:0] mix_ai_res,
    input [NM:0] mix_info_res,

    //------------------------------------------------------------------------
    // COST HAMILTONIAN INTERFACE
    //------------------------------------------------------------------------
    output reg [P-1:0] gamma, // cos gamma - γ parameter
    output reg [P-1:0] HGC,    // Cost Hamiltonian coefficient

    input [P-1:0] Hr_res,
    input [P-1:0] Hi_res,
    
    //------------------------------------------------------------------------
    // Address INTERFACE (FOR MIXER, and COST FUNCTION GEN)
    //------------------------------------------------------------------------
    input wire  [NM-1:0] swapped_cAddr_in,
    input wire  [NM-1:0] cAddrCF_in,
    //------------------------------------------------------------------------
    //BRAM ARRAY INTERFACE (3 banks + 1 general bank)
    //------------------------------------------------------------------------
    output reg [NBRAM-1:0] bram_wen,
    input [P-1:0] bram_data_r [NBRAM],
    output reg [P-1:0] bram_data_w [NBRAM],
    output reg [NM-1:0] bram_addr_r [NBRAM],
    output reg [NM-1:0] bram_addr_w [NBRAM]
);
//============================================================================
// LOCALPARAMS - Internal Constants
//============================================================================
//----------------------------------------------------------------------------
// QAOA State Machine States
//----------------------------------------------------------------------------
localparam qa_WAIT  = 8'h1;   // Waiting for command from ntu_smachine
localparam qa_RUN   = 8'h2;   // Running QAOA layer initialization
localparam qa_RUNB1 = 8'h4;   // Executing mixer operation
localparam qa_RUNC  = 8'h8;   // Executing mixer operation

localparam W_REQ_BASE = NM + 8;
localparam LATENCY_BRAM = L_BRAM_R + 1; // need to add 1, for output register inside this block
//============================================================================
// INTERNAL REGISTERS & SIGNALS
//============================================================================

//----------------------------------------------------------------------------
// Main State Machine
//----------------------------------------------------------------------------
reg [7:0] cmd;                // Current command state
logic [7:0] n_cmd;            // Next command state

reg f_run_Computation;
logic n_f_run_Computation;
assign f_run_Computation_out = f_run_Computation;
//----------------------------------------------------------------------------
// BRAM Access Registers (from ntu_smachine)
//----------------------------------------------------------------------------
reg [P-1:0] r_addr;           // Buffered read address
reg r_req;                    // Buffered read request
reg [P-1:0] w_addr;           // Buffered write address
reg [P-1:0] w_data;           // Buffered write data
reg w_req;                    // Buffered write request

logic [P-1:0] n_r_data;       // Next read data
logic n_r_vd;                 // Next read valid, for control interface
//-----------------
// Parameter Storage Pointer
//----------------------------------------------------------------------------
reg [31:0] P5pointer;         // Pointer into BRAM[3] for parameter reads
logic [31:0] n_P5pointer;     // Next pointer value

//----------------------------------------------------------------------------
// QAOA Parameters (cos(β), sin(β), γ)
//----------------------------------------------------------------------------
logic [P-1:0] nb_cosb;         // Next cos(β), buffer to align the timining 
logic [P-1:0] nb_sinb;         // Next sin(β), buffer to align the timining 
logic [P-1:0] n_cosb;         // Next cos(β), buffer to align the timining 
logic [P-1:0] n_sinb;         // Next sin(β), buffer to align the timining 
reg [P-1:0] b_cosb;         // Next cos(β), buffer to align the timining 
reg [P-1:0] b_sinb;         // Next sin(β), buffer to align the timining 
logic [P-1:0] n_gamma;        // Next γ
logic [P-1:0] nb_gamma;        // Next γ
reg [P-1:0] b_gamma; // cos gamma - γ parameter

//----------------------------------------------------------------------------
// Mixer Operation Registers
//----------------------------------------------------------------------------
logic [P-1:0] n_mix_ar;         // Next mixer real input
logic [P-1:0] n_mix_ai;         // Next mixer imaginary input
logic [NM:0] n_mix_info;      // Next mixer control info
logic [1:0]  n_mix_switch;
//----------------------------------------------------------------------------
// Cost function Operation Registers
//----------------------------------------------------------------------------

logic [P-1:0] n_HGC;            // next Cost Hamiltonian coefficient

//----------------------------------------------------------------------------
// BRAM Request Pipeline (3-stage delay)
//----------------------------------------------------------------------------
reg [W_REQ_BASE + 3*P+4-1:0] bram_reqP[LATENCY_BRAM];     // 2-stage pipeline
logic [W_REQ_BASE + 3*P+4-1:0] n_bram_reqQ;   // Next request (input to pipeline)

logic [P-1:0] n_bram_gammaQ;
logic [P-1:0] n_bram_cosbQ;
logic [P-1:0] n_bram_sinbQ;
logic [3:0] n_bram_infoAQ;
assign n_bram_reqQ[W_REQ_BASE+:3*P+4] = {n_bram_infoAQ, n_bram_sinbQ, n_bram_cosbQ, n_bram_gammaQ};

wire [P-1:0] bram_gammaO;
wire [P-1:0] bram_cosbO;
wire [P-1:0] bram_sinbO;
wire [3:0] bram_infoAO;
wire [W_REQ_BASE-1:0] bram_reqR;   
assign {bram_infoAO, bram_sinbO, bram_cosbO, bram_gammaO, bram_reqR} = bram_reqP[LATENCY_BRAM-1];
//----------------------------------------------------------------------------
// BRAM Control Signals (Next Values)
//----------------------------------------------------------------------------
logic [NM-1:0] n_bram_addr_r [NBRAM];  // Next read addresses
logic [NM-1:0] n_bram_addr_w [NBRAM];  // Next write addresses
logic [P-1:0]  n_bram_data_w [NBRAM];  // Next write data
logic [NBRAM-1:0] n_bram_wen;          // Next write enables

//----------------------------------------------------------------------------
// Test/Debug Register
//----------------------------------------------------------------------------
reg [P-1:0] testReg;          // Test register for debug reads/writes
logic [P-1:0] n_testReg;      // Next test register value

always_comb begin: memorySwitchingBlock

    //------------------------------------------------------------------------
    // This block defines Memory interconnect. 
    // All memory addressing querying part must be in this block
    // A block only taking the read result can be in another block.  
    // DEFAULT ASSIGNMENTS - Prevent Latches
    //------------------------------------------------------------------------
    n_r_vd = 0;                         // No read valid by default
    n_testReg = testReg;                // Keep test register
    n_r_data = 'd0;                       // Clear read data
    n_bram_wen = 'd0;                     // No BRAM writes by default

    n_bram_addr_w = bram_addr_w;        // Keep write addresses
    n_bram_addr_r = bram_addr_r;        // Keep read addresses
    n_bram_data_w = bram_data_w;        // Keep write data
    n_bram_reqQ[W_REQ_BASE-1:0] = 'd0;                  // No new BRAM request
    n_cmd = cmd;                        // Keep current command

    n_mix_info = 'd0;                       // Clear mixer info
    n_mix_ar = 'd0;                         // Clear mixer real input
    n_mix_ai = 'd0;                         // Clear mixer imag input
    n_mix_switch = '0;
    n_HGC = 'd0;

    n_f_run_Computation = f_run_Computation;
    n_gamma = 64'h0; // 0; default value 
    n_cosb = 64'h3fc00aeb5da15be0; // 0.12533323356430426; default value  
    n_sinb = 64'h3fefbf675480d903; // 0.9921147013144779; default value
    n_bram_gammaQ = 64'h3eafbf675480d903;
    n_bram_cosbQ =  64'h3eafbf675480d903;
    n_bram_sinbQ = '0;
    n_bram_infoAQ = '0;

    nb_cosb = b_cosb; // keep
    nb_sinb = b_sinb; // keep
    nb_gamma = b_gamma;  // Keep γ

    n_P5pointer = P5pointer;
    // read cosb, sinb, gamma, at the beginning of the first 
    // request cos(beta)
    if(en_Inits_in[0] || en_Inits_in[1] || en_Inits_in[2]) begin 
        n_P5pointer = P5pointer + 1;
    end

    if(en_Inits_in[3]) begin
        nb_cosb = bram_data_r[3]; // store to temporal buffer of cosb, later written to cosb itself.
    end
    if(en_Inits_in[4]) begin
        // write to sinb
        nb_sinb = bram_data_r[3]; // store to temporal buffer
    end
    if(en_Inits_in[5]) begin 
        nb_gamma = bram_data_r[3];
    end
    //------------------------------------------------------------------------
    // MAIN STATE MACHINE
    //------------------------------------------------------------------------
    case(cmd)

    // qa_WAIT: Idle state - Handle read/write requests from ntu_smachine
    //========================================================================
    qa_WAIT: begin // accept operation from ntu_smachine.

        // READ OPERATIONS (based on r_addr[63:56])
        case(r_addr[63:56])
        1: begin //read the state
            n_r_data[7:0] = cmd;
            n_r_data[63:8] = '0;
            n_r_vd = r_req;
        end
        2: begin 
            n_r_data = testReg;
            n_r_vd = r_req;
        end
        4: begin 
            n_bram_addr_r[2] = r_addr[NM-1:0];
            n_bram_reqQ[NM] = r_req;
            n_r_vd = bram_reqR[NM];
            n_r_data = bram_data_r[2];
        end
        8: begin  // read sin(b), cos(b), gamma, I think we will use this function just for debugging.
            n_bram_addr_r[3] =  r_addr[NM-1:0];
            n_bram_reqQ[NM+1] = r_req;
            n_r_vd = bram_reqR[NM+1];
            n_r_data = bram_data_r[3];
        end
        16: begin // read real part of the state vector
            n_bram_addr_r[0] =  r_addr[NM-1:0]; 
            n_bram_reqQ[NM+2] = r_req;
            n_r_vd = bram_reqR[NM+2];
            n_r_data = bram_data_r[0];
        end 
        32: begin // read imag part of the state vector
            n_bram_addr_r[1] =  r_addr[NM-1:0];
            n_bram_reqQ[NM+3] = r_req;
            n_r_vd = bram_reqR[NM+3];
            n_r_data = bram_data_r[1];
        end 
        default: begin 
            n_r_vd = r_req;
            n_r_data = testReg;
        end
        endcase

        // WRITE OPERATIONS (based on w_addr[63:56])
        //--------------------------------------------------------------------
        
        if(w_req) begin
            case(w_addr[63:56])
            1: begin //write register
                n_testReg = w_data;
            end
            2: begin 
                n_testReg = testReg + 1;
            end
            4: begin // write to cost function
                n_bram_addr_w[2] = w_addr[NM-1:0];
                n_bram_data_w[2] = w_data;
                n_bram_wen[2] = 1;
            end
            8: begin // write to sin(b) cos(b), gamma, 
                n_bram_addr_w[3] = w_addr[NM-1:0];
                n_bram_data_w[3] = w_data;
                n_bram_wen[3] = 1;
            end
            16: begin // write to state vector, we need to implement a switching function of writing to real and imaginary part.
                n_bram_addr_w[0] = w_addr[NM-1:0];
                n_bram_data_w[0] = w_data;
                n_bram_wen[0] = 1;
            end 
            32: begin // write to state vector, we need to implement a switching function of writing to real and imaginary part.
                n_bram_addr_w[1] = w_addr[NM-1:0];
                n_bram_data_w[1] = w_data;
                n_bram_wen[1] = 1;
            end 
            default: begin 
                n_testReg = w_data;
            end
            endcase
        end
    end
    // initialize. Start at the next clock.
    qa_RUN: begin
        n_f_run_Computation = 1;
        n_cmd = qa_RUNC;
        n_P5pointer = 0;
    end
    qa_RUNB1: begin // wait 1 clock, for register update. Addressing needs 1 clock after getting f_run_Computation = 1.
        n_cmd = qa_RUNC;
    end
    // qa_RUN: Apply cost Hamiltonian and check layer completion
    //========================================================================
    qa_RUNC: begin
         // --------------------------------------------
        // Memory access part.
        // Index generator to BRAM. Index generator input is defined in n_bs_info_in.
        // --------------------------------------------
        n_bram_addr_r[2] = cAddrCF_in; // addressing for cost function generation
        n_bram_addr_r[0] = swapped_cAddr_in;
        n_bram_addr_r[1] = swapped_cAddr_in;
        n_bram_reqQ[NM-1:0] = swapped_cAddr_in; // pass address, for storation after mixer.
        n_bram_addr_r[3] = P5pointer[NM-1:0];

        // --------------------------------------------
        // BRAM data to each arithmetric block
        // --------------------------------------------
        n_HGC = bram_data_r[2]; // supply hamiltonian to generator.
        n_mix_switch = bram_infoAO[2:1];
        n_mix_ar = bram_data_r[0]; 
        n_mix_ai = bram_data_r[1];
        n_sinb = bram_sinbO;
        n_cosb = bram_cosbO;
        n_gamma = bram_gammaO;
        n_mix_info[NM-1:0] = bram_reqR[NM-1:0];
        n_mix_info[NM] = bram_infoAO[0]; // contains write enable.

        // --------------------------------------------
        // Write back output of arithmetric block to BRAM
        // Output from cost function gen is directly supplied to mixer block, see n_bs_info_in part in always_comb computingBlock.
        // --------------------------------------------
        n_bram_addr_w[0] = mix_info_res[NM-1:0];
        n_bram_addr_w[1] = mix_info_res[NM-1:0];
        n_bram_data_w[0] = mix_ar_res;
        n_bram_data_w[1] = mix_ai_res;
        n_bram_wen[0] = mix_info_res[NM];
        n_bram_wen[1] = mix_info_res[NM];
        // !! remove bswap chain from pipeline. No need to include addressing latency. ok 20260429
        // !! any number of pipeline for addressing is possible!.
        // !! include enCostF in bs information chain.
        if(enCostF_in) begin  
            n_bram_cosbQ = Hr_res; // provided to mixer, pipe line of n_bram_reqQcosb consists of  bit swap  and memory access latency.
            n_bram_sinbQ = Hi_res; // provided to mixer
            n_bram_gammaQ = gamma; // keep the old value
        end
        else begin 
            n_bram_cosbQ = b_cosb; // provided to mixer
            n_bram_sinbQ = b_sinb; // provided to mixer
            n_bram_gammaQ = b_gamma; // provide the next value
        end
        n_bram_infoAQ[0] = enPipe_in; // enable write back the result
        n_bram_infoAQ[1] = mixSwitch_in[0]; // 0 for cost
        n_bram_infoAQ[2] = mixSwitch_in[1]; // 0 for cost

        if(f_L1Computation_in) begin 
            n_cmd = qa_WAIT;
            n_f_run_Computation = 0;
        end
    end
	 default: begin
	 end
    endcase
    
end

//============================================================================
// SEQUENTIAL LOGIC - Register Updates
//============================================================================
integer i;
always@(posedge CLK) begin

    //------------------------------------------------------------------------
    // RESET: Initialize all registers
    //------------------------------------------------------------------------
    
    if(RST)begin
        // BRAM interface
        r_data <= '0;
        r_addr <= '0;
        r_req <= '0;
        w_addr <= '0;
        w_data <= '0;
        w_req <= '0;
        r_vd <= '0;

        // BRAM control
        bram_wen <= '0;
        bram_addr_w <= '{default: 0}; // Initialize all array elements to 0
        bram_addr_r <= '{default: 0}; // Initialize all array elements to 0
        bram_data_w <= '{default: 0}; // Initialize all array elements to 0

        // State machine
        cmd <= qa_WAIT;
        // Counters
        P5pointer <= '0;

        // QAOA parameters (default values for testing)
        cosb <= 64'h3fb999999999999a;  // cos(0.1) in FP64
        sinb <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64
        gamma <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64
        b_gamma <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64
		  
        b_cosb <= 64'h3fb999999999999a;  // cos(0.1) in FP64
        b_sinb <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64

        // Mixer interface
        mix_ar <= 'd0;
        mix_ai <= 'd0;
        mix_info <= 'd0;

        HGC <= 'd0;

        // Pipeline
        for(i=0;i<LATENCY_BRAM;i=i+1) begin 
            bram_reqP[i] <= '0;
        end 
        // Debug
        testReg <= 'd0;

        // Configuration
        mix_switch <= 'd0;
        bram_wen <= 'd0;
        Status <= '0;

        f_run_Computation <= 'd0;
        // initialize with invalid values.

    end

    // NORMAL OPERATION: Update registers
    //------------------------------------------------------------------------
    else begin
        
        //--------------------------------------------------------------------
        // Command Override (from ntu_smachine via r_CMD)
        //--------------------------------------------------------------------
        if(r_CMD[23]) begin
            cmd <= r_CMD[7:0]; // FIXED: Non-blocking assignment
        end
        else begin
            cmd <= n_cmd;
        end
        
        //--------------------------------------------------------------------
        // Update All Registers
        //--------------------------------------------------------------------
        P5pointer <= n_P5pointer;

        // BRAM control
        bram_wen <= n_bram_wen;
        bram_addr_w <= n_bram_addr_w;
        bram_addr_r <= n_bram_addr_r;
        bram_data_w <= n_bram_data_w;

        // BRAM read interface
        r_data <= n_r_data;
        r_addr <= n_r_addr;
        r_req <= n_r_req;
        r_vd <= n_r_vd;

        // BRAM write interface
        w_addr <= n_w_addr;
        w_data <= n_w_data;
        w_req <= n_w_req;

        // BRAM request pipeline (3-stage delay for timing)
        bram_reqP[0] <= n_bram_reqQ;
        for(i=0;i<LATENCY_BRAM-1;i=i+1) begin 
            bram_reqP[i+1] <= bram_reqP[i];
        end 
        
        cosb <= n_cosb;
        sinb <= n_sinb;
        b_cosb <= nb_cosb;
        b_sinb <= nb_sinb;
        gamma <= n_gamma; 
        b_gamma <= nb_gamma;

        // QAOA parameters
        // Mixer interface
        mix_ar <= n_mix_ar;
        mix_ai <= n_mix_ai;
        mix_info <= n_mix_info;
        mix_switch <= n_mix_switch;
        HGC <= n_HGC;
        // Debug
        Status[7:0] <= n_cmd[7:0];
        Status[32] <= f_run_Computation;
        Status[8+:NM] <= P5pointer;
        testReg <= n_testReg;
        // en_CostF = 1, after finish cost generation. Do not use global timing, because
        // need to be disscussed more in detail.
        f_run_Computation <= n_f_run_Computation;
        
    end
end

endmodule