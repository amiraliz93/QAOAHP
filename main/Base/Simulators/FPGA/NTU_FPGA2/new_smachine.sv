//============================================================================
// Documentation
//============================================================================
// Author: Amir Alizadeh (NTU), Hiroki Shibata (Tokyo Metropolitan University)
// Date: November 2025

// Description:
//   State machine controller for FPGA-based QAOA simulation. Handles:
//   - UART communication (command/data exchange with PC)
//   - Register file management (rA, rB, rT, rU - 64-bit each)
//   - Arithmetic operations (fixed-point and FP64)
//   - BRAM interface for quantum state storage
//   - QAOA-specific operations (mixer, cost phases)

//    - interface description (port)
//    Inputs: CLK, RSTlv1A, RSTlv1B, i_Rx_Serial, rbram_vd, r_data
//    Output: o_Tx_Serial, w_addr, r_addr, w_data, w_req, r_req
//===========================================================================

module ntu_smachine #(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter UART_CLKS_PER_BIT = 868,      // Baud rate divider
    parameter MAX_QUBITS = 20,              // Maximum qubits
    parameter MAX_EDGES = 190,              // Maximum graph edges
    parameter BRAM_ADDR_WIDTH = 11,         // BRAM address width
    parameter BRAM_DATA_WIDTH = 64,         // BRAM data width
    parameter FP_PRECISION = 64,            // FP precision (32 or 64)
    parameter FIXED_INT_BITS = 40,          // Fixed-point integer bits
    parameter FIXED_FRAC_BITS = 24,         // Fixed-point fractional bits
    parameter FP64_ADD_LATENCY = 27,        // FP64 add latency
    parameter FP64_MUL_LATENCY = 24,        // FP64 mul latency
    parameter FIX64_ADD_LATENCY = 2,        // Fixed add latency
    parameter FIX24_MUL_LATENCY = 8         // Fixed mul latency
)(
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  wire        CLK,
    input  wire        RSTlv1A,             // Reset for state machine
    input  wire        RSTlv1B,             // Reset for UART/FIFOs
    
    //------------------------------------------------------------------------
    // UART INTERFACE
    //------------------------------------------------------------------------
    input  wire        i_Rx_Serial,         // UART RX input
    output wire        o_Tx_Serial,         // UART TX output
    
    //------------------------------------------------------------------------
    // BRAM INTERFACE
    //------------------------------------------------------------------------
    output reg  [BRAM_ADDR_WIDTH-1:0] w_addr,   // Write address
    output reg  [BRAM_ADDR_WIDTH-1:0] r_addr,   // Read address
    output reg  [BRAM_DATA_WIDTH-1:0] w_data,   // Write data
    input  wire [BRAM_DATA_WIDTH-1:0] r_data,   // Read data
    output reg         w_req,               // Write request
    output reg         r_req,               // Read request
    input  wire        rbram_vd             // Read data valid
);

//============================================================================
//---------------------- Localparam / Parameters 
//============================================================================

//---------------------------  MAIN STATE MACHINE STATES -----------------------
//---------------------------------------------------------------------------
// 8 state so need max 3 bit width  - formula = ceil(log2(max value +1))
// 3 Width - d(base) - 0 (value)
localparam s_IDLE        = 3'd0;  // Idle: waiting for command
localparam s_Fetch       = 3'd1;  // Fetching opcode/data from RX FIFO
localparam s_Operation   = 3'd2;  // Decode and dispatch operation
localparam s_WAIT_COMP   = 3'd3;  // Wait for arithmetic operation to complete
localparam s_WRITE_REG   = 3'd4;  // Write result to register
localparam s_WRITE_BRAM  = 3'd5;  // Write data to Block RAM
localparam s_READ_BRAM   = 3'd6;  // Read data from Block RAM
localparam s_TXData      = 3'd7;  // Transmit data to PC via UART


//---------------------- Operation Code (Data Transfer) --------------------
//---------------------------------------------------------------------------
// 255 value - max width 8
localparam OP_NONE       = 8'd0;   // No operation
localparam OP_SEND1T     = 8'd1;   // Request 1 byte from PC → rT
localparam OP_SEND8T     = 8'd2;   // Request 8 bytes from PC → rT
localparam OP_MOV_T2A    = 8'd3;   // Move rT → rA
localparam OP_MOV_T2B    = 8'd4;   // Move rT → rB
localparam OP_MOV_A2U    = 8'd5;   // Move rA → rU (address register)
localparam OP_MOV_A2B    = 8'd6;   // Move rA → rB

//------------------- Data Retrieval Operations (60-70)
localparam OP_FETCH1U    = 8'd60;  // Send 1 byte from rU to PC
localparam OP_FETCH8U    = 8'd61;  // Send 8 bytes from rU to PC


//--------------- Arithmetic Operations (Fixed-Point) (80-85)
localparam OP_ADD_B2A    = 8'd80;  // rA = rA + rB (64-bit fixed, 2 cycles)
localparam OP_MUL_B2A    = 8'd81;  // rA = rA * rB (24-bit fixed, 8 cycles)
localparam OP_INC_A     = 8'd84;    // rA = rA +1

// --------------- Arithmetic Operations (Floating-Point)
localparam OP_ADDFP_B2A  = 8'd82;  // rA = rA + rB (FP64, 27 cycles)
localparam OP_MULFP_B2A  = 8'd83;  // rA = rA * rB (FP64, 24 cycles)

//------------------- Command Operations (for QAOA system) (100-115)
localparam OP_RUN_MIXER        = 8'd100;  // Execute QAOA mixer step
localparam OP_RUN_COST         = 8'd101;  // Execute QAOA cost step
localparam OP_RUN_CONTINUOUS   = 8'd103;  // Continuous QAOA execution
localparam OP_SEND_CMD         = 8'd115;  // Send command to external module
localparam OP_ENABLE_INTRUPTION = 8'd121; // Enable interrupt on status change

//-------------------- Memory Operations 
localparam OP_WRITE_T2RAM = 8'd111;  // Write rT to BRAM[rA]
localparam OP_READ_RAM2U  = 8'd112;  // Read BRAM[rA] → rU

//---------------------SUB-STATE DEFINITIONS (Fetch, Store) -----------------
//---------------------------------------------------------------------------
//--------------------- Fetch parameters
localparam FETCH_IDLE    = 3'd0;  // Idle: not fetching
localparam FETCH_DATA    = 3'd1;  // Fetching data bytes into rT
localparam FETCH_WAIT    = 3'd2;  // Waiting for FIFO data valid
localparam FETCH_GETOP   = 3'd4;  // Getting operation code
localparam FETCH_w_BRAM  = 3'd6;  // (Reserved for BRAM streaming - not implemented)

//--------------------- Store parameters
localparam STORE_IDLE    = 3'd0;  // Idle: not transmitting
localparam STORE_LEN     = 3'd1;  // Transmitting bytes from rU
localparam STORE_BRAM    = 3'd2;  // (Reserved for BRAM streaming - not implemented)
localparam STORE_WAIT    = 3'd4;  // Waiting for TX FIFO space

//--------------------------- Write Operation parameters (Codes) --------------------
//---------------------------------------------------------------------------
localparam WRITE_NONE        = 5'd0;   // No write operation
localparam WRITE_T2A         = 5'd1;   // rA = rT
localparam WRITE_T2B         = 5'd2;   // rB = rT
localparam WRITE_A2B         = 5'd3;   // rB = rA
localparam WRITE_A2U         = 5'd4;   // rU = rA
localparam WRITE_B2A         = 5'd5;   // rA = rB
localparam WRITE_mulFP64_rA  = 5'd6;   // rA = res_mulFP64 (FP multiply result)
localparam WRITE_addFP64_rA  = 5'd7;   // rA = res_addFP64 (FP add result)
localparam WRITE_mul_rA      = 5'd8;   // rA = res_mulAB (fixed multiply result)
localparam WRITE_add_rA      = 5'd9;   // rA = res_addAB (fixed add result)
localparam WRITE_rA1         = 5'd10;  // rA = rA + 1
localparam WRITE_rB1         = 5'd11;  // rB = rB + 1
localparam WRITE_BRAM_U      = 5'd12;  // rU = r_data (BRAM read result)

//======================================================================================



//============================================================================
//---------------------- INTERNAL REGISTERS & SIGNALS
//============================================================================
// 2.1 Inputs (Clock, Reset, External signals)
// 2.2 Outputs (UART, Control signals)
// 2.3 Bidirectional (if any)

//------------ Main State Mchine Registers
reg [2:0] state;              // Current main state
logic [2:0] n_state;          // Next state

reg [7:0] ope_state;          // Current operation code
logic [7:0] n_ope_state;      // Next operation code

//--------------- Fetch Sub-State Machine
reg [2:0] fetchReg;           //current fetch state
logic [2:0] n_fetchReg;        // Next fetch state

reg [3:0] rPos;         //Current byte position in rT(0-7)
logic [3:0] n_rPos;      // next byte position

reg [3:0] fetchMaxPos;   //Max bytes to fetch (0=1 byte, 7=8 bytes)
logic [3:0] n_fetchMaxPos;    // Next max position

reg [10:0] txBRPos; // position to store the byte 
logic [10:0] n_txBRPos; // position to store the byte  // BRAM transmit position

reg [10:0] rBRPos; // position to store the byte 
logic [10:0] n_rBRPos; // position to store the byte  // Next BRAM transmit position

//--------------------- Store/TX Sub-state Mchine
reg [2:0] storeReg;           //Current strore state
logic [2:0] n_storeReg;      // Next store state

reg [3:0] txPos;         // current tx byte position (0-7)
logic [3:0] n_txPos;     // Next TX position

logic [3:0] n_txMaxPos;  // Next max tx position

//------------------- Write Backend Control 
reg [4:0] writeReg;           //current write operation
reg [4:0] bwriteReg;          // buffered write (for arthemetic operation)
logic [4:0] n_writeReg;       // Next write operation
logic [4:0] nb_writeReg;      // Next buffered write

// -------------Data Path Registers (64bit)
reg [63:0] rA;                // Register A (general purpose)
reg [63:0] rB;                // Register B
reg [63:0] rT;                // Temporary Register (RX data)
reg [63:0] rU;                // Address register (TX data)

logic [63:0] n_rA, n_rB, n_rT, n_rU;      //Next values

//-----------------------  pipline Register (for FP arithmetic timing)
reg [63:0] rA2, rA3, rA4;     // rA pipeline (stages 1-3)
reg [63:0] rB2, rB3, rB4;     // rB pipeline (stages 1-3)

//------------------------ Arithmetic Unit Inputs/Outputs

// Fixed-point units
// These perform integer/fixed-point operations with shorter latency
reg [23:0] mulA, mulB;        // Multiplier inputs
reg [63:0] addA, addB;        // Adder inputs
reg [47:0] res_mulAB;         // Multiply result
reg [63:0] res_addAB;         // Add result

//---------------------- Floating-point units
// Performs: res_addFP64 = rA2 + rB2
// Latency: 27 clock cycles (pipelined)
reg [63:0] res_addFP64;       // FP64 add result
reg [63:0] res_mulFP64;       // FP64 multiply result

//------------------  Wait Counter (for arithmetic latency)
reg [7:0] c_wait;             // Current wait cycles
reg [7:0] opa_c_wait;         // Target wait cycles
logic [7:0] n_c_wait;         // Next wait count
logic [7:0] n_opa_c_wait;     // Next target

//============================================================================

//============================================================================
//---------------------- FIFO Interface Signals & Registers
//===========================================================================

//------------------------  RX FIFO (incoming data from UART)
wire [7:0] rf_data;           // Data from RX FIFO
wire rf_empty;                // RX FIFO empty flag
wire rf_full;                 // RX FIFO full flag
reg rf_req;                   // Read request
reg [1:0] rf_dv;              // Data valid (delayed)

//---------------------   TX FIFO (outgoing data to UART)
reg [7:0] tf_data;            // Data to TX FIFO
wire [7:0] tx_data_in;        // TX FIFO output
wire tx_active;               // UART transmitter active lag
wire rx_dv;                   // UART receiver data valid 
wire [7:0] rx_data_out;   // UART receiver data output
wire tf_empty;                // TX FIFO empty flag
wire tf_full;                 // TX FIFO full flag
reg tf_write;                 // Write enable
reg [1:0] tx_dv;              // TX data valid


// -----------------------  BRAM Interface Signals
reg [10:0] w_addr;            // Write address
reg [10:0] r_addr;            // Read address
reg [63:0] w_data;            // Write data
wire [63:0] r_data;           // Read data
reg w_req;                    // Write request
reg r_req;                    // Read request
wire rbram_vd;                // Read data valid

logic [10:0] n_w_addr, n_r_addr;
logic [63:0] n_w_data;
logic n_w_req, n_r_req;

//---------------------- Helper Wires
wire [63:0] rAinc = rA3 + 1;  // Increment operations
wire [63:0] rBinc = rB3 + 1;

//---------- Conditional states (wait if FIFO full/empty)
wire [2:0] wstoreReg = tf_full ? STORE_WAIT : storeReg;
wire [2:0] wfetchReg = rf_dv[0] ? fetchReg : FETCH_WAIT;
//============================================================================


//============================================================================
// MAIN STATE MACHINE - COMBINATIONAL LOGIC
//============================================================================
// This block determines next state and outputs based on current state
always_comb begin: main_StateBlock
      
      //------------------------------------------------------------------------
      // DEFAULT VALUES - Prevent latches by assigning all signals
      //------------------------------------------------------------------------
      n_state = state;                    // Stay in current state by default
      n_opa_c_wait = opa_c_wait;          // Keep wait counter target
      n_fetchReg = fetchReg;              // Keep fetch state
      n_storeReg = storeReg;              // Keep store state
      n_c_wait = c_wait;                  // Keep current wait count
      n_fetchMaxPos = fetchMaxPos;        // Keep fetch byte limit
      n_writeReg = writeReg;              // Keep write operation
      nb_writeReg = bwriteReg;            // Keep buffered write op
      n_txMaxPos = txMaxPos;              // Keep TX byte limit
      n_w_req = 0;                        // Default: no BRAM write request
      n_w_addr = rA;                      // BRAM write address = rA
      n_r_addr = rA;                      // BRAM read address = rA
      n_w_data = rT;                      // BRAM write data = rT
      n_rA = rA;                          // Keep register A value
      n_rB = rB;                          // Keep register B value
      n_rT = rT;                          // Keep temporary register
      n_rU = rU;                          // Keep address register
      n_txPos = 0;                        // Reset TX position
      n_txBRPos = 0;                      // Reset BRAM TX position
      tf_data = 0;                        // No TX FIFO write by default
      tf_write = 0;                       // TX FIFO write disabled
      n_rPos = rPos;                      // Keep receive position
      n_rBRPos = rBRPos;                  // Keep BRAM receive position
      
      // RX FIFO read request: read if (1) FIFO not empty AND (2) not in IDLE
      rf_req = (!rf_empty) & (fetchReg != FETCH_IDLE);
      
      n_ope_state = ope_state;            // Keep operation code
      n_r_req = 0;                        // Default: no BRAM read request
      
      //------------------------------------------------------------------------
      // MAIN STATE MACHINE LOGIC
      //------------------------------------------------------------------------
      case(state)
      
      //------------------------------------------------------------------------
      // s_Fetch: Receive opcode and optional data from UART/FIFO
      //------------------------------------------------------------------------
      s_Fetch: begin
            case(wfetchReg)  // Check fetch sub-state (with FIFO check)
            
            // FETCH_DATA: Receiving data bytes into rT register
            FETCH_DATA: begin
                // Extract byte from FIFO and place in rT at current position
                  n_rT[rPos*8+:8] = rf_data;  // [rPos*8+:8] = 8 bits starting at rPos*8
                  n_rPos = rPos + 1;          // Move to next byte position
                  
                  // If all bytes received, get next opcode
                  if(rPos == fetchMaxPos) begin
                        n_fetchReg = FETCH_GETOP;
                  end
            end
            
            // FETCH_WAIT: Waiting for FIFO to have data
            FETCH_WAIT: begin
                  n_rPos = rPos;              // Hold position
                  n_rBRPos = rBRPos;          // Hold BRAM position
            end
            
            // FETCH_GETOP: Getting operation code from FIFO
            FETCH_GETOP: begin
                  n_ope_state = rf_data;      // Store opcode
                  n_state = s_Operation;      // Move to operation state
                  n_rPos = 0;                 // Reset byte position
            end
            
            // Default case
            default: begin
                  n_rPos = 0;                 // Reset positions
                  n_rBRPos = 0;
            end
            endcase
      end
      
      //------------------------------------------------------------------------
      // s_Operation: Decode opcode and dispatch to appropriate action
      //------------------------------------------------------------------------
      s_Operation: begin
            case (ope_state)
            
            //--------------------------------------------------------------------
            // DATA TRANSFER OPERATIONS
            //--------------------------------------------------------------------
            
            // OP_SEND1T: Request 1 byte from PC
            OP_SEND1T: begin
                  n_state = s_Fetch;          // Return to fetch state
                  n_fetchReg = FETCH_DATA;    // Set to receive data
                  n_fetchMaxPos = 0;          // Receive 1 byte (position 0)
            end
            
            // OP_SEND8T: Request 8 bytes from PC
            OP_SEND8T: begin
                  n_state = s_Fetch;          // Return to fetch state
                  n_fetchMaxPos = 7;          // Receive 8 bytes (positions 0-7)
                  n_fetchReg = FETCH_DATA;    // Set to receive data
            end
            
            // OP_MOV_T2A: Move temporary register to register A
            OP_MOV_T2A: begin
                  n_state = s_WRITE_REG;      // Go to write state
                  n_writeReg = WRITE_T2A;     // Set write operation
            end
            
            // OP_MOV_A2U: Move register A to address register U
            OP_MOV_A2U: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_A2U;
            end
            
            // OP_MOV_A2B: Move register A to register B
            OP_MOV_A2B: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_A2B;
            end
            
            //--------------------------------------------------------------------
            // ARITHMETIC OPERATIONS (with wait cycles)
            //--------------------------------------------------------------------
            
            // OP_INC_A: Increment register A
            OP_INC_A: begin
                  n_opa_c_wait = 1;           // Wait 1 cycle
                  n_state = s_WAIT_COMP;      // Go to wait state
                  nb_writeReg = WRITE_rA1;    // Buffer write operation
            end
            
            // OP_ADD_B2A: Fixed-point addition (rA = rA + rB)
            OP_ADD_B2A: begin
                  n_opa_c_wait = 2;           // Wait 2 cycles (addfix8 latency)
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_add_rA;
            end
            
            // OP_MUL_B2A: Fixed-point multiplication (rA = rA * rB)
            OP_MUL_B2A: begin
                  n_opa_c_wait = 8;           // Wait 8 cycles (mulfix8 latency)
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_mul_rA;
            end
            
            // OP_ADDFP_B2A: Floating-point addition (rA = rA + rB)
            OP_ADDFP_B2A: begin
                  n_opa_c_wait = 27;          // Wait 27 cycles (addFPF64 latency)
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_addFP64_rA;
            end
            
            // OP_MULFP_B2A: Floating-point multiplication (rA = rA * rB)
            OP_MULFP_B2A: begin
                  n_opa_c_wait = 24;          // Wait 24 cycles (mulFPF64 latency)
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_mulFP64_rA;
            end
            
            //--------------------------------------------------------------------
            // DATA RETRIEVAL OPERATIONS
            //--------------------------------------------------------------------
            
            // OP_FETCH1U: Send 1 byte from rU to PC
            OP_FETCH1U: begin
                  n_storeReg = STORE_LEN;     // Set to transmit mode
                  n_txMaxPos = 0;             // Send 1 byte
                  n_state = s_TXData;         // Go to transmit state
            end
            
            // OP_FETCH8U: Send 8 bytes from rU to PC
            OP_FETCH8U: begin
                  n_storeReg = STORE_LEN;
                  n_txMaxPos = 7;             // Send 8 bytes (positions 0-7)
                  n_state = s_TXData;
            end
            
            //--------------------------------------------------------------------
            // MEMORY OPERATIONS
            //--------------------------------------------------------------------
            
            // OP_WRITE_T2RAM: Write rT to BRAM[rA]
            OP_WRITE_T2RAM: begin
                  n_w_req = 1;                // Assert write request
                  n_state = s_WRITE_BRAM;     // Go to BRAM write state
            end
            
            // OP_READ_RAM2U: Read BRAM[rA] into rU
            OP_READ_RAM2U: begin
                  // Note: No streaming support as of Sept 2025
                  n_r_req = 1;                // Assert read request
                  n_state = s_READ_BRAM;      // Go to BRAM read state
            end
            
            //--------------------------------------------------------------------
            // DEFAULT: Unknown opcode
            //--------------------------------------------------------------------
            default: begin
                  n_state = s_IDLE;           // Return to idle on error
            end
            endcase
      end
      
      //------------------------------------------------------------------------
      // s_READ_BRAM: Wait for BRAM read data to be valid
      //------------------------------------------------------------------------
      s_READ_BRAM: begin
            n_r_req = 0;  // De-assert read request (only 1 cycle needed)
            
            // When BRAM data is valid, store it in rU
            if(rbram_vd) begin
                  n_writeReg = WRITE_BRAM_U;  // Set write operation (rU = r_data)
                  n_state = s_WRITE_REG;      // Go to register write state
            end
            // else: stay in this state until rbram_vd = 1
      end
      
      //------------------------------------------------------------------------
      // s_WRITE_BRAM: Write data to BRAM (single cycle)
      //------------------------------------------------------------------------
      s_WRITE_BRAM: begin
            n_w_req = 0;                      // De-assert write request
            n_state = s_IDLE;                 // Write completes immediately
      end
      
      //------------------------------------------------------------------------
      // s_WAIT_COMP: Wait for arithmetic operation to complete
      //------------------------------------------------------------------------
      s_WAIT_COMP: begin
            n_c_wait = c_wait + 1;            // Increment wait counter
            
            // When counter reaches target latency, write result
            if(c_wait == opa_c_wait) begin
                  n_writeReg = bwriteReg;     // Use buffered write operation
                  n_state = s_WRITE_REG;      // Go to register write state
            end
            // else: stay in this state
      end
      
      //------------------------------------------------------------------------
      // s_WRITE_REG: Write result to destination register
      //------------------------------------------------------------------------
      s_WRITE_REG: begin
            case (writeReg)
                  
                  // WRITE_T2A: rA = rT
                  WRITE_T2A: begin
                        n_rA = rT;
                  end
                  
                  // WRITE_T2B: rB = rT
                  WRITE_T2B: begin
                        n_rB = rT;
                  end
                  
                  // WRITE_A2B: rB = rA
                  WRITE_A2B: begin
                        n_rB = rA;
                  end
                  
                  // WRITE_A2U: rU = rA (address register)
                  WRITE_A2U: begin
                        n_rU = rA;
                  end
                  
                  // WRITE_mulFP64_rA: rA = FP64 multiply result
                  WRITE_mulFP64_rA: begin
                        n_rA = res_mulFP64;
                  end
                  
                  // WRITE_addFP64_rA: rA = FP64 add result
                  WRITE_addFP64_rA: begin
                        n_rA = res_addFP64;
                  end
                  
                  // WRITE_mul_rA: rA = fixed multiply result
                  WRITE_mul_rA: begin
                        n_rA = res_mulAB;
                  end
                  
                  // WRITE_add_rA: rA = fixed add result
                  WRITE_add_rA: begin
                        n_rA = res_addAB;
                  end
                  
                  // WRITE_rA1: rA = rA + 1
                  WRITE_rA1: begin
                        n_rA = rAinc;
                  end
                  
                  // WRITE_rB1: rB = rB + 1
                  WRITE_rB1: begin
                        n_rB = rBinc;
                  end
                  
                  // WRITE_BRAM_U: rU = BRAM read data
                  WRITE_BRAM_U: begin
                        n_rU = r_data;
                  end
                  
                  default: begin
                        // No operation
                  end
            endcase
            
            n_state = s_IDLE;  // Always return to idle after write
      end
      
      //------------------------------------------------------------------------
      // s_TXData: Transmit data bytes to PC via UART
      //------------------------------------------------------------------------
      s_TXData: begin
            case(wstoreReg)  // Check store sub-state (with FIFO check)
            
            // STORE_LEN: Transmitting bytes from rU
            STORE_LEN: begin
                  // Extract byte from rU at current position
                  tf_data = rU[txPos*8+:8];   // Get byte from rU
                  n_txPos = txPos + 1;        // Move to next byte
                  
                  // If all bytes transmitted, return to idle
                  if(txPos == txMaxPos) begin
                        n_state = s_IDLE;
                  end
                  
                  tf_write = 1;  // Write byte to TX FIFO
            end
            
            // Default/STORE_WAIT: Hold position
            default: begin
                  n_txPos = txPos;
                  n_txBRPos = txBRPos;
            end
            endcase
      end
      
      //------------------------------------------------------------------------
      // DEFAULT (s_IDLE): Initialize for next operation
      //------------------------------------------------------------------------
      default: begin
            // Reset all counters and prepare for next command
            n_txBRPos = 0;              // Reset BRAM TX position
            n_txPos = 0;                // Reset TX byte position
            n_c_wait = 0;               // Reset wait counter
            n_state = s_Fetch;          // Go to fetch state
            n_storeReg = STORE_IDLE;    // Set store state to idle
            n_writeReg = WRITE_NONE;    // No write operation
            n_fetchReg = FETCH_GETOP;   // Ready to get opcode
            n_fetchMaxPos = 0;          // No data bytes expected yet
      end
      
      endcase
end

//============================================================================
// SEQUENTIAL LOGIC - Register Updates on Clock Edge
//============================================================================
// All register updates happen synchronously on rising edge of CLK
always @(posedge CLK) begin
      
      //------------------------------------------------------------------------
      // RESET: Initialize all registers to known state
      //------------------------------------------------------------------------
      if (RSTlv1A) begin
            rf_dv <= 0;              // Clear FIFO data valid
            CP <= 4123;              // Program counter (unused in current design)
            state <= s_IDLE;         // Start in idle state
            fetchReg <= FETCH_IDLE;  // Fetch state = idle
            ope_state <= 0;          // No operation
            rPos <= 0;               // Reset byte positions
            
            // Clear all data registers
            rA <= 0;
            rA2 <= 0;  // Pipeline registers
            rA3 <= 0;
            rA4 <= 0;
            rB <= 0;
            rB2 <= 0;
            rB3 <= 0;
            rB4 <= 0;
            rT <= 0;   // Temporary register
            rU <= 0;   // Address register
            
            // Clear control registers
            c_wait <= 0;
            txPos <= 0;
            txMaxPos <= 0;
            fetchMaxPos <= 0;
            storeReg <= 0;
            opa_c_wait <= 0;
      end
      
      //------------------------------------------------------------------------
      // NORMAL OPERATION: Update registers with next values
      //------------------------------------------------------------------------
      else begin
            // FIFO handshake delay
            rf_dv[0] <= rf_req;       // Delay 1: rf_req
            rf_dv[1] <= rf_dv[0];     // Delay 2: used for timing
            
            // Control registers
            opa_c_wait <= n_opa_c_wait;  // Update wait target
            c_wait <= n_c_wait;          // Update wait counter
            
            // Data registers (with pipeline for FP operations)
            rA <= n_rA;              // Update register A
            rA2 <= n_rA;             // Pipeline stage 1 (for FP add)
            rA3 <= n_rA;             // Pipeline stage 2
            rA4 <= n_rA;             // Pipeline stage 3 (for FP mul)
            
            rB <= n_rB;              // Update register B
            rB2 <= n_rB;             // Pipeline stage 1
            rB3 <= n_rB;             // Pipeline stage 2
            rB4 <= n_rB;             // Pipeline stage 3
            
            rT <= n_rT;              // Update temporary register
            rU <= n_rU;              // Update address register
            
            // State machine registers
            state <= n_state;        // Update main state
            ope_state <= n_ope_state;// Update operation code
            fetchReg <= n_fetchReg;  // Update fetch sub-state
            storeReg <= n_storeReg;  // Update store sub-state
            
            // Position counters
            rBRPos <= n_rBRPos;
            fetchMaxPos <= n_fetchMaxPos;
            rPos <= n_rPos;
            txPos <= n_txPos;
            txMaxPos <= n_txMaxPos;
            
            // Write backend registers
            bwriteReg <= nb_writeReg;  // Update buffered write op
            writeReg <= n_writeReg;    // Update current write op
            
            // BRAM interface
            w_data <= n_w_data;      // Update BRAM write data
            w_addr <= n_w_addr;      // Update BRAM write address
            r_addr <= n_r_addr;      // Update BRAM read address
            r_req <= n_r_req;        // Update BRAM read request
            w_req <= n_w_req;        // Update BRAM write request
      end

end
//============================================================================


//============================================================================
// MODULE INSTANTIATIONS
//============================================================================
// 4.1 UART Transmitter/Receiver
// 4.2 FIFOs (RX, TX)
// 4.3 Floating-Point Units
// 4.4 Fixed-Point Units

//----------------------------------------------------------------------------
// TX FIFO CONTROLLER - Manages transmission of data from FIFO to UART
//----------------------------------------------------------------------------
// This block controls when data is read from TX FIFO and sent to UART transmitter
// It implements a 2-stage handshake to ensure proper timing

// sending block from fifo to uart unit. This looks so waste. Do I really need such this block intrinsically?
// I guess that fifoblock should be incorpolated into transmitter.
always @(posedge CLK) begin 
      if(RSTlv1B) begin
            tx_dv <= 0;
      end
      else begin
            if(tx_dv[0] | tx_dv[1]) begin
                  tx_dv[0] <= 0;
            end
            else if(~tx_active & ~tf_empty) begin
                  tx_dv[0] <= 1;
            end
            tx_dv[1] <= tx_dv[0];
      end
end


//----------------------------------------------------------------------------
// UART TRANSMITTER INSTANCE
//----------------------------------------------------------------------------
// Converts parallel bytes to serial UART format (115200 baud)

transmitter #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) t0(
      .i_Clock(CLK),
      .RST(RSTlv1B),
      .i_Tx_DV(tx_dv[1]),        // Data valid (delayed by 1 cycle)
      .i_Tx_Byte(tx_data_in),    // Byte to transmit (from TX FIFO)
      .o_Tx_Active(tx_active),   // High when transmitting
      .o_Tx_Serial(o_Tx_Serial), // Serial output to PC
      .o_Tx_Done()               // Transmission complete (unused)
);

//----------------------------------------------------------------------------
// UART RECEIVER INSTANCE
//----------------------------------------------------------------------------
receiver #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) uart1 
(
      .i_Clock       (CLK),
      .RST       (RSTlv1B),
      .i_Rx_Serial   (i_Rx_Serial),
	.o_Rx_DV   (rx_dv),
      .o_Rx_Byte       (rx_data_out)    
);





//----------------------------------------------------------------------------
// RX FIFO - Buffers incoming UART data
//----------------------------------------------------------------------------

// Stores received bytes from UART until state machine reads them
fifo1 fifo1_inst (
      .clock(CLK),
      .data(rx_data_out),   // Input: byte from UART receiver
      .rdreq(rf_req),       // Read request from state machine
      .wrreq(rx_dv),        // Write request from UART (when byte received)
      .empty(rf_empty),     // High when FIFO is empty
      .full(rf_full),       // High when FIFO is full
      .q(rf_data)           // Output: byte to state machine
);

//----------------------------------------------------------------------------
// TX FIFO - Buffers outgoing data to UART
//----------------------------------------------------------------------------
// Stores bytes to be transmitted until UART is ready
fifo1 fifoW_inst (
      .clock(CLK),
      .data(tf_data),       // Input: byte from state machine
      .rdreq(tx_dv[0]),     // Read request (when UART ready)
      .wrreq(tf_write),     // Write request from state machine
      .empty(tf_empty),     // High when FIFO is empty
      .full(tf_full),       // High when FIFO is full (causes STORE_WAIT)
      .q(tx_data_in)        // Output: byte to UART transmitter
);



addFPF64 addFPF64(
      .clk(CLK),
      .areset(RSTlv1B),      // Active-high reset
      .a(rA2),               // Input A (delayed by 2 cycles from rA)
      .b(rB2),               // Input B (delayed by 2 cycles from rB)
      .q(res_addFP64)        // Output: A + B (available after 27 cycles)
);

//----------------------------------------------------------------------------
// FLOATING-POINT MULTIPLIER (64-bit IEEE 754)
//----------------------------------------------------------------------------
// Performs: res_mulFP64 = rA4 * rB4
// Latency: 24 clock cycles (pipelined)
mulFPF64 mf64i(
      .clk(CLK),
      .areset(RSTlv1B),      // Active-high reset
      .a(rA4),               // Input A (delayed by 4 cycles from rA)
      .b(rB4),               // Input B (delayed by 4 cycles from rB)
      .q(res_mulFP64)        // Output: A * B (available after 24 cycles)
);

//----------------------------------------------------------------------------
// FIXED-POINT ARITHMETIC UNITS
//----------------------------------------------------------------------------
// These perform integer/fixed-point operations with shorter latency

// Pipeline inputs to match ALU timing
always @(posedge CLK) begin
      mulA <= rA[23:0];    // Extract lower 24 bits of rA
      mulB <= rB[23:0];    // Extract lower 24 bits of rB
      addA <= rA;          // Full 64-bit value
      addB <= rB;          // Full 64-bit value
end

//----------------------------------------------------------------------------
// FIXED-POINT MULTIPLIER
//----------------------------------------------------------------------------
// Performs: res_mulAB = mulA * mulB (24-bit x 24-bit = 48-bit)
// Latency: 8 clock cycles
mulfix8 mulfix8_inst (
      .clock(CLK),
      .dataa(mulA),
      .datab(mulB),
      .result(res_mulAB)   // 48-bit result
);

//----------------------------------------------------------------------------
// FIXED-POINT ADDER
//----------------------------------------------------------------------------
// Performs: res_addAB = addA + addB (64-bit + 64-bit = 64-bit)
// Latency: 2 clock cycles
addfix8 addix8_inst (
      .clock(CLK),
      .dataa(addA),
      .datab(addB),
      .result(res_addAB)   // 64-bit result
);


//============================================================================
// STATUS CODES (for rS64 register - used by QAOA)
//============================================================================
// sR64 = 100: Mixer operation in progress
// sR64 = 102: Mixer operation complete
// sR64 = 101: Cost operation in progress
// sR64 = 103: Cost operation complete
// rS64 = 77:  Data received and written to BRAM successfully
// rS64 = 76:  Data sent to PC successfully
// rS64 = 79:  Data received with overflow (partial write)
// rS64 = 78:  Data sent with overflow (partial read)
// rS64 = 0:   Idle state


endmodule

//============================================================================
// END OF MODULE
//============================================================================

/============================================================================
// SECTION 7: ASSERTIONS & DEBUG (Optional)
//============================================================================

