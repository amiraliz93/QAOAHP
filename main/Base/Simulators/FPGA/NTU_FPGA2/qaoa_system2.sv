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
    parameter NM = 13,   // BRAM address width (2^13 = 8192 elements)
    parameter P = 64,    // Data orecision (64bit FP)
    parameter NBRAM=6,   // number of Bram banks
    parameter Ni=32,     // must be greater than or equal to 32.
    parameter N_BIT_SWAP_POINTER = $clog2(NM))    // Info/ control signal width
  (
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  CLK,     // system clock
    input  RST,     // Active-high reset
    //------------------------------------------------------------------------
    // COMMAND INTERFACE (from ntu_smachine)
    //------------------------------------------------------------------------
    input [23:0]  r_CMD, // 
                   
     //------------------------------------------------------------------------
    // BRAM READ/WRITE INTERFACE (from ntu_smachine)
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

    // Input  to Mixer
    output reg [P-1:0] cosb,        // cos(β) parameter
    output reg [P-1:0] sinb,        // sin(β) parameter
    output reg [P-1:0] mix_ar,        // State real part input
    output reg [P-1:0] mix_ai,        // State imaginary part input
    output reg [Ni-1:0] mix_info,     // Control info (address, enable, etc.)
    output reg [1:0] mix_switch,     // Control info (address, enable, etc.)

    // Output from Mixer
    input [P-1:0] mix_ar_res,
    input [P-1:0] mix_ai_res,
    input [Ni-1:0] mix_info_res,

    //------------------------------------------------------------------------
    // COST HAMILTONIAN INTERFACE
    //------------------------------------------------------------------------
    output reg [P-1:0] gamma, // cos gamma - γ parameter
    output reg [P-1:0] HGC,         // Cost Hamiltonian coefficient
    output reg [Ni-1:0]  info_inGC, // information, like addresses, enabled signal, and so on.

    input [P-1:0] Hr_res,
    input [P-1:0] Hi_res,
    input  [Ni-1:0]  info_outGC, // information, like addresses, enabled signal, and so on.
    
    //------------------------------------------------------------------------
    // BIT SWAPING UNIT INTERFACE (FOR MIXER)
    //------------------------------------------------------------------------
    input wire [Ni-1:0] bs_info_out,
    output reg [Ni-1:0] bs_info_in,
    output reg [NM-1:0] bswap_in,  // bit swap in swap pointer 1
    input wire  [NM-1:0] bswap_out,
    output reg [N_BIT_SWAP_POINTER-1:0] bsp1, // next bit swap pointer 1
    output reg [N_BIT_SWAP_POINTER-1:0] bsp2, // next bit swap pointer 2

    
    //------------------------------------------------------------------------
    //BRAM ARRAY INTERFACE (6 banks)
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
localparam qa_MIXER = 8'h4;   // Executing mixer operation
localparam qa_COST  = 8'h8;   // Executing cost Hamiltonian
localparam qa_INIT  = 8'h10;  // Initialization state
localparam qa_MIXER_WAIT_PIPE = 8'h20;
localparam qa_MIXER_PREPARE = 8'h40;

localparam mixer_PIPLINE_NUM = 21 + 27 + 4 + 5 + 2 + 3; // mixer pipline + bram + bitswap + write + registers, latencies
localparam costGen_PIPLINE_NUM = 218 + 2 + 5 + 2 + 3; // cost function generation + bram + bitswap + write + registers, latencies
localparam cost_PIPLINE_NUM = mixer_PIPLINE_NUM; 

//============================================================================
// INTERNAL REGISTERS & SIGNALS
//============================================================================

//----------------------------------------------------------------------------
// Main State Machine
//----------------------------------------------------------------------------
reg [7:0] cmd;                // Current command state
logic [7:0] n_cmd;            // Next command state

//----------------------------------------------------------------------------
// Run Sub-State Machine (for parameter loading)
//----------------------------------------------------------------------------

reg [12:0] runState;          // Current run state
logic [12:0] n_runState;      // Next run state

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

//----------------------------------------------------------------------------
// Address Counters
//----------------------------------------------------------------------------
reg [NM-1:0] maxAddr;         // Maximum address for mixer (e.g., 8 for 3 qubits)
reg [NM-1:0] maxAddrM1;         // Maximum address for mixer (e.g., 8 for 3 qubits)
reg [NM-1:0] addr_c0;         // Current address counter (mixer loop)
logic [NM-1:0] n_maxAddr;         // Maximum address for mixer (e.g., 8 for 3 qubits)
logic [NM-1:0] n_maxAddrM1;         // Maximum address for mixer (e.g., 8 for 3 qubits)
logic [NM-1:0] n_addr_c0;     // Next address counter

reg [9:0] r_costGen_PIPLINE_NUM;
wire [9:0] nr_costGen_PIPLINE_NUM = costGen_PIPLINE_NUM;

//----------------------------------------------------------------------------
// Parameter Storage Pointer
//----------------------------------------------------------------------------
reg [31:0] P5pointer;         // Pointer into BRAM[5] for parameter reads
logic [31:0] n_P5pointer;     // Next pointer value

//----------------------------------------------------------------------------
// Layer Counter
//----------------------------------------------------------------------------
reg [31:0] cPLayer;           // Current QAOA layer count
logic [31:0] n_cPLayer;       // Next layer count

//----------------------------------------------------------------------------
// Pipeline Wait Counter
//----------------------------------------------------------------------------
reg [7:0] waitPipeline;       // Wait counter for mixer pipeline
logic [7:0] n_waitPipeline;   // Next wait count
reg [7:0] waitCGPipeline;       // Wait counter for mixer pipeline
logic [7:0] n_waitCGPipeline;   // Next wait count

//----------------------------------------------------------------------------
// QAOA Parameters (cos(β), sin(β), γ)
//----------------------------------------------------------------------------
logic [P-1:0] n_cosb;         // Next cos(β)
logic [P-1:0] n_sinb;         // Next sin(β)
logic [P-1:0] n_gamma;        // Next γ
logic [NM-1:0] n_NQbits;
logic [NM-1:0] n_NQbitsM1;
reg [NM-1:0] NQbits;
reg [NM-1:0] NQbitsM1;
//----------------------------------------------------------------------------
// Mixer Operation Registers
//----------------------------------------------------------------------------
logic [P-1:0] n_mix_ar;         // Next mixer real input
logic [P-1:0] n_mix_ai;         // Next mixer imaginary input
logic [Ni-1:0] n_mix_info;      // Next mixer control info
logic [1:0]  n_mix_switch;
//----------------------------------------------------------------------------
// Cost function Operation Registers
//----------------------------------------------------------------------------

logic [P-1:0] n_HGC;            // next Cost Hamiltonian coefficient
logic [Ni-1:0] n_info_inGC; // information, like addresses, enabled signal, and so on.

//----------------------------------------------------------------------------
// BRAM Request Pipeline (3-stage delay)
//----------------------------------------------------------------------------
reg [40:0] bram_reqP[2];     // 2-stage pipeline
reg [40:0] bram_reqR;       // Final delayed request
logic [40:0] n_bram_reqQ;   // Next request (input to pipeline)

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

// input wire  [NM-1:0] bswap_out;
// input wire [Ni-1:0] bs_info_out;
logic [NM-1:0] n_bswap_in;  // bit swap in swap pointer 1
logic [NM-1:0] n_bs_info_in;
logic [N_BIT_SWAP_POINTER-1:0] n_bsp1; // next bit swap pointer 1
logic [N_BIT_SWAP_POINTER-1:0] n_bsp2; // next bit swap pointer 2

reg [31:0] nPLayer;
logic [31:0] n_nPLayer;
logic [3:0] n_mixer_flag;        // pre-compute conditional flag to speed up the logic and keep logics flexible
reg [3:0] mixer_flag;
reg [2:0] costGen_flag;
logic [2:0] n_costGen_flag;
always_comb begin: mainCombBlock

    //------------------------------------------------------------------------
    // DEFAULT ASSIGNMENTS - Prevent Latches
    //------------------------------------------------------------------------
    
    n_r_vd = 0;                         // No read valid by default
    n_testReg = testReg;                // Keep test register
    n_r_data = 'd0;                       // Clear read data
    n_bram_wen = 'd0;                     // No BRAM writes by default
    n_addr_c0 = addr_c0; 
    n_bram_addr_w = bram_addr_w;        // Keep write addresses
    n_bram_addr_r = bram_addr_r;        // Keep read addresses
    n_bram_data_w = bram_data_w;        // Keep write data
    n_bram_reqQ = 'd0;                  // No new BRAM request
    n_cmd = cmd;                        // Keep current command
    n_costGen_flag = costGen_flag;
    n_waitPipeline = waitPipeline;                 // Reset wait counter
    n_waitCGPipeline = waitCGPipeline;                 // Reset wait counter
    n_cosb = cosb;                      // Keep cos(β)
    n_sinb = sinb;                      // Keep sin(β)
    n_gamma = gamma;                    // Keep γ
    n_mix_info = 'd0;                       // Clear mixer info
    n_mix_ar = 'd0;                         // Clear mixer real input
    n_mix_ai = 'd0;                         // Clear mixer imag input
    n_cPLayer = cPLayer;                // Keep layer counter
    n_runState = runState;              // Keep run state
    n_P5pointer = P5pointer;            // Keep parameter pointer
    n_bsp1 = bsp1;
    n_bsp2 = bsp2;
    n_bswap_in = bswap_in;
    n_bs_info_in = 'b0000;
    n_mixer_flag = mixer_flag;
    n_mix_switch = '0;
    n_HGC = 'd0;
    n_info_inGC = 'd0;
    n_nPLayer = nPLayer;
    n_maxAddr = maxAddr;
    n_maxAddrM1 = maxAddrM1;
    n_NQbits = NQbits;
    n_NQbitsM1 = NQbitsM1;
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
            n_r_data = 64'h01efef80aa80aaaa;
            n_r_vd = r_req;
        end
        4: begin // read from cost function, just for debugging
            n_bram_addr_r[2] = r_addr[NM-1:0];
            n_bram_reqQ[32] = r_req;
            n_r_vd = bram_reqR[32];
            n_r_data = bram_data_r[2];
        end
        8: begin  // read sin(b), cos(b), gamma, I think we will use this function just for debugging.
            n_bram_addr_r[5] =  r_addr[NM-1:0];
            n_bram_reqQ[33] = r_req;
            n_r_vd = bram_reqR[33];
            n_r_data = bram_data_r[5];
        end
        16: begin // read real part of the state vector
            n_bram_addr_r[0] =  r_addr[NM-1:0]; 
            n_bram_reqQ[34] = r_req;
            n_r_vd = bram_reqR[34];
            n_r_data = bram_data_r[0];
        end 
        32: begin // read imag part of the state vector
            n_bram_addr_r[1] =  r_addr[NM-1:0];
            n_bram_reqQ[35] = r_req;
            n_r_vd = bram_reqR[35];
            n_r_data = bram_data_r[1];
        end 
        33: begin // read imag part of the state vector
            n_bram_addr_r[3] =  r_addr[NM-1:0];
            n_bram_reqQ[36] = r_req;
            n_r_vd = bram_reqR[36];
            n_r_data = bram_data_r[3];
        end 
        34: begin // read imag part of the state vector
            n_bram_addr_r[4] =  r_addr[NM-1:0];
            n_bram_reqQ[37] = r_req;
            n_r_vd = bram_reqR[37];
            n_r_data = bram_data_r[4];
        end 
        endcase

        // WRITE OPERATIONS (based on w_addr[63:56])
        //--------------------------------------------------------------------
        
        case(w_addr[63:56])
        1: begin //write register
            if(w_req) begin
                n_testReg = w_data;
            end
        end
        2: begin 
            if(w_req) begin
                n_testReg = w_data + 8;
            end
        end
        4: begin // write to cost function
            n_bram_addr_w[2] = w_addr[NM-1:0];
            n_bram_data_w[2] = w_data;
            n_bram_wen[2] = w_req;
        end
        8: begin // write to sin(b) cos(b), gamma, 
            n_bram_addr_w[5] = w_addr[NM-1:0];
            n_bram_data_w[5] = w_data;
            n_bram_wen[5] = w_req;
        end
        16: begin // write to state vector, we need to implement a switching function of writing to real and imaginary part.
            n_bram_addr_w[0] = w_addr[NM-1:0];
            n_bram_data_w[0] = w_data;
            n_bram_wen[0] = w_req;
        end 
        32: begin // write to state vector, we need to implement a switching function of writing to real and imaginary part.
            n_bram_addr_w[1] = w_addr[NM-1:0];
            n_bram_data_w[1] = w_data;
            n_bram_wen[1] = w_req;
        end 
        'h40: begin
            if(w_req) begin
                n_NQbits = w_data;   // set maximum bit swapping pointer.
            end
        end
        'h41: begin
            if(w_req) begin
                n_NQbitsM1 = w_data;   // set maximum bit swapping pointer - 1. 
            end
        end
        'h42: begin 
            if(w_req) begin
                n_maxAddr = w_data;   // set maximum address
            end
        end
        'h43: begin 
            if(w_req) begin
                n_maxAddrM1 = w_data;   // set maximum address - 1
            end
        end
        'h44: begin 
            if(w_req) begin
                n_nPLayer = w_data;   // set maximum address - 1
            end
        end
        endcase
    end

    // qa_INIT: Initialize system for new QAOA execution
    //========================================================================
    
    qa_INIT: begin
        n_cPLayer = 0;
        n_P5pointer = 0;
        n_runState = 1;
    end
    
    // qa_RUN: Load parameters (cos(β), sin(β), γ) from BRAM[5]
    //========================================================================
    qa_RUN: begin
        // necessary to set this state, before going to Run.
        // need to wait multiple clocks here because cos, sin, gamma must be read from the same BRAM sequentially.
        case(runState)
        1: begin
            n_addr_c0 = 0;
            n_waitPipeline = 0;
            n_waitCGPipeline = 0;
            n_bsp2 = 0;
            n_bsp1 = 0;
            n_cPLayer = cPLayer + 1;
            n_mixer_flag = 3'b001;
            n_costGen_flag = 3'b001;
        end
        2: begin
            // request cos(beta)
            n_bram_addr_r[5] = P5pointer[NM-1:0];
            n_P5pointer = P5pointer + 1;
        end
        4: begin
            // request sin(beta)
            n_bram_addr_r[5] = P5pointer[NM-1:0];
            n_P5pointer = P5pointer + 1;
        end
        8: begin
            // request gamma
            n_bram_addr_r[5] = P5pointer[NM-1:0];
            n_P5pointer = P5pointer + 1;
        end
        16: begin
            // write to cosb
            n_cosb = bram_data_r[5];
        end
        32: begin
            // write to sinb
            n_sinb = bram_data_r[5];
        end
        64: begin
            // write to gamma
            n_gamma = bram_data_r[5];
        end
        128: begin
            n_cmd = qa_MIXER;
        end
        endcase
        n_runState = runState << 1;
    end

    // qa_MIXER: Execute mixer operation on state vector
    //========================================================================
    qa_MIXER: begin
        // mixer operation, and cost function generation.
        // assuming qa_INIT is called before this state.
        n_bram_addr_r[0] = bswap_out;
        n_bram_addr_r[1] = bswap_out;
        n_bram_addr_r[2] =  bswap_out; // addressing for cost function generation
        n_bram_reqQ[31] = bs_info_out[0];
        n_bram_reqQ[32] = bs_info_out[1]; // information for cost function generation
        n_bram_reqQ[33] = bs_info_out[2]; // information for cost function generation
        n_bram_reqQ[34] = bs_info_out[3];
        n_bram_reqQ[NM-1:0] = bswap_out;
        
        // need to mix address bit.
        n_bswap_in = addr_c0;
        case(costGen_flag)
            3'b001: begin
                if(addr_c0 == maxAddr) begin
                    n_costGen_flag = 3'b010; // go to cost function operator
                end
            end
            3'b010: begin
                n_waitCGPipeline = waitCGPipeline + 1;
                if(waitCGPipeline == r_costGen_PIPLINE_NUM) begin
                    n_costGen_flag = 3'b100; // go to cost function operator
                end
            end
            3'b100: begin
            end
        endcase

        case(mixer_flag)
            4'b0001: begin
                n_addr_c0 = addr_c0 + 1; 
                n_bs_info_in[0] = 1;
                n_bs_info_in[1] = ~addr_c0[0];
                n_bs_info_in[2] = addr_c0[0];
                n_bs_info_in[3] = costGen_flag[0];
                if(addr_c0 == maxAddr) begin
                    n_mixer_flag = 4'b0010;
                end
            end
            4'b0010: begin
                n_bs_info_in = 'b0000;
                n_waitPipeline = waitPipeline + 1;
                if(waitPipeline == mixer_PIPLINE_NUM) begin
                    n_mixer_flag = 4'b0100;
                end
            end
            4'b0100: begin // pipeline latency + memory latency + bit swaping latency.
                n_bs_info_in = 'b0000;
                n_bsp1 = 0;
                n_addr_c0 = 0; 
                n_waitPipeline = 0;
                if(bsp2 == NQbitsM1) begin
                    n_mixer_flag = 4'b1000;
                    n_bsp2 = 0;
                end
                else begin
                    n_bsp2 = bsp2 + 1; 
                    n_mixer_flag = 4'b0001;
                end
            end
            4'b1000: begin
                n_bs_info_in = 'b0000;
                if(costGen_flag == 3'b100) begin
                    n_cmd = qa_COST; // go to cost function operator
                    n_mixer_flag = 4'b0001;
                end
            end
        endcase

        // Feed Data to Mixer (when BRAM read completes)
        n_mix_ar = bram_data_r[0]; 
        n_mix_ai = bram_data_r[1]; 
        n_mix_info = bram_reqR[31:0];
        n_mix_switch = bram_reqR[33:32];
        n_HGC = bram_data_r[2]; 
        n_info_inGC = {bram_reqR[34], bram_reqR[30:0]};
        
        // Addressing part for cost function generatoin.
        // **** list of interfaces to cost function generation units.
        // .gamma(gamma), //  gamma
        // .HGC(H),
        // .Hr_res(Hr_o),
        // .Hi_res(Hi_o),
        // .info_inGC(info_inGC), // information, like addresses, enabled signal, and so on.
        // .info_outGC(info_outGC) // information, like addresses, enabled signal, and so on.

        // Write Mixer Results Back to BRAM
        n_bram_addr_w[0] = mix_info_res[NM-1:0];
        n_bram_addr_w[1] = mix_info_res[NM-1:0];
        n_bram_addr_w[3] = info_outGC[NM-1:0]; // cost function's real part address
        n_bram_addr_w[4] = info_outGC[NM-1:0]; // cost function's imag part address
        n_bram_data_w[0] = mix_ar_res;
        n_bram_data_w[1] = mix_ai_res;
        n_bram_data_w[3] = Hr_res; // cost function's real part 
        n_bram_data_w[4] = Hi_res; // cost function's imag part 
        n_bram_wen[0] = mix_info_res[31];
        n_bram_wen[1] = mix_info_res[31];
        n_bram_wen[3] = info_outGC[31];  // cost function's real part write enable
        n_bram_wen[4] = info_outGC[31];  // cost function's real part write enable

    end
    // qa_COST: Apply cost Hamiltonian and check layer completion
    //========================================================================
    qa_COST: begin
        // mixer operation, and cost function generation.
        // assuming qa_INIT is called before this state.
        n_bram_addr_r[0] = addr_c0;
        n_bram_addr_r[1] = addr_c0;
        n_bram_addr_r[3] = addr_c0; // addressing for cost function real part
        n_bram_addr_r[4] = addr_c0; // addressing for cost function imag part
        n_bram_reqQ[31] = mixer_flag[0];
        n_bram_reqQ[NM-1:0] = addr_c0;
        n_mix_switch = 2'b00;

        case(mixer_flag)
            4'b0001: begin // addressing part
                n_addr_c0 = addr_c0 + 1; 
                if(addr_c0 == maxAddr) begin
                    n_mixer_flag = 4'b0010;
                end
            end
            4'b0010: begin // pipeline flushing part
                n_waitPipeline = waitPipeline + 1;
                if(waitPipeline == cost_PIPLINE_NUM) begin
                    n_mixer_flag = 4'b0100;
                end
            end
            4'b0100: begin // pipeline latency + memory latency + bit swaping latency.
                // Check if all QAOA layers completed
                if(cPLayer == nPLayer) begin
                    n_cmd = qa_WAIT;
                end
                // Else: start next layer
                else begin
                    n_cmd = qa_RUN;
                    n_runState = 1;
                end
            end
        endcase
        n_mix_ar = bram_data_r[0]; 
        n_mix_ai = bram_data_r[1]; 
        n_cosb = bram_data_r[3];
        n_sinb = bram_data_r[4];
        // Feed Data to Mixer operation unit. Note that mixer unit can be used for cost function operation as well.
        n_mix_info = bram_reqR[Ni-1:0];
        
        // Write Mixer Results Back to BRAM
        n_bram_addr_w[0] = mix_info_res[NM-1:0];
        n_bram_addr_w[1] = mix_info_res[NM-1:0];
        n_bram_data_w[0] = mix_ar_res;
        n_bram_data_w[1] = mix_ai_res;
        n_bram_wen[0] = mix_info_res[31];
        n_bram_wen[1] = mix_info_res[31];
    end
    endcase
end

//============================================================================
// SEQUENTIAL LOGIC - Register Updates
//============================================================================


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
        cmd <= qa_INIT;
        runState <= '0;

        // Counters
        addr_c0 <= '0;
        waitPipeline <= '0;
        cPLayer <= '0;
        P5pointer <= '0;

        // QAOA parameters (default values for testing)
        cosb <= 64'h3fb999999999999a;  // cos(0.1) in FP64
        sinb <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64
        gamma <= 64'hbfeccccccccccccd;  // -sin(0.1) in FP64
        bsp1 <= 'd0;
        bsp2 <= 'd0;
        info_inGC <= 'h0;
        costGen_flag <= '0;
        waitCGPipeline <= 'd0;
        bswap_in <= 'h0;
        // Mixer interface
        mix_ar <= 'd0;
        mix_ai <= 'd0;
        mix_info <= 'd0;
        mixer_flag <= 'b001;

        HGC <= 'd0;

        // Pipeline
        bram_reqR <= 'd0;
        bram_reqP[0] <= 'd0;
        bram_reqP[1] <= 'd0;

        // Debug
        testReg <= 'd0;

        // Configuration
        maxAddr <= 'h0008;  // Default: 8 states (3 qubits)
        maxAddrM1 <= 'h0008 -1;
        nPLayer <= 'd1;
        NQbits <= 'd1;
        NQbitsM1 <= 'd0;
        mix_switch <= 'd0;
        bram_wen <= 'd0;
        r_costGen_PIPLINE_NUM <= costGen_PIPLINE_NUM;
        Status <= '0;
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
        runState <= n_runState;
        cPLayer <= n_cPLayer;
        waitPipeline <= n_waitPipeline;
        maxAddr <= n_maxAddr;
        maxAddrM1 <= n_maxAddrM1;
        
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
        bram_reqP[1] <= bram_reqP[0]; 
        bram_reqR <= bram_reqP[1]; // Final delayed request

        // QAOA parameters
        gamma <= n_gamma;
        cosb <= n_cosb;
        sinb <= n_sinb;
        NQbits <= n_NQbits;
        NQbitsM1 <= n_NQbitsM1;
        nPLayer <= n_nPLayer;
        bsp1 <= n_bsp1;
        bsp2 <= n_bsp2;
        info_inGC <= n_info_inGC;
        costGen_flag <= n_costGen_flag;
        waitCGPipeline <= n_waitCGPipeline;
        // Mixer interface
        mix_ar <= n_mix_ar;
        mix_ai <= n_mix_ai;
        mix_info <= n_mix_info;
        mix_switch <= n_mix_switch;
        mixer_flag <= n_mixer_flag;
        bs_info_in <= n_bs_info_in;
        bswap_in <= n_bswap_in;
        HGC <= n_HGC;
        // Address counter
        addr_c0 <= n_addr_c0;
        // Debug
        Status[7:0] <= n_cmd[7:0];
        Status[32+:32] <= n_cPLayer[31:0];
        Status[8+:12] <= n_bsp2[11:0];
        Status[31:20] <= 'd0;
        testReg <= n_testReg;
        r_costGen_PIPLINE_NUM <= nr_costGen_PIPLINE_NUM;


    end
end

endmodule