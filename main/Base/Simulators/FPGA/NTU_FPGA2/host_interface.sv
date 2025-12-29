// This is a simple module act on state machine controlled via serial communcation.
// Default implementation uses UART module as an background, but it can be replaced with another module with the speed less 
// than the processing capacity of this statemachine. The processing capacity is a quarter of the clock.
// tested by ntu_smachine_tb.v, on 2025 Sep. 2nd.
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
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
//    Inputs: CLK, RST, rx_data_in, rbram_vd, r_data, rx_dv, tx_OK
//    Output: tx_data_out, w_addr, r_addr, w_data, w_req, r_req, tx_en
//===========================================================================

module ntu_smachine2#(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter MAX_QUBITS = 20,              // Maximum qubits
    parameter MAX_EDGES = 190,              // Maximum graph edges
    parameter BRAM_ADDR_WIDTH = 64,         // BRAM address width
    parameter BRAM_DATA_WIDTH = 64,         // BRAM data width
    parameter FP_PRECISION = 64,            // FP precision (32 or 64)
    parameter FIXED_INT_BITS = 40,          // Fixed-point integer bits
    parameter FIXED_FRAC_BITS = 24,         // Fixed-point fractional bits
    parameter FP64_ADD_LATENCY = 27,        // FP64 add latency
    parameter FP64_MUL_LATENCY = 24,        // FP64 mul latency
    parameter FIX64_ADD_LATENCY = 2,        // Fixed add latency
    parameter FIX24_MUL_LATENCY = 8,        // Fixed mul latency
    parameter HOST_DATA_WIDTH = 8           // DATA width between Host
   )(
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  wire CLK,
    input  wire RST,
    //------------------------------------------------------------------------
    // HOST INTERFACE
    //------------------------------------------------------------------------
    input  wire tx_OK,
    output wire tx_en,
    output [HOST_DATA_WIDTH-1:0] tx_data_out,
    input [HOST_DATA_WIDTH-1:0] rx_data_in,
    input rx_dv,
    
    //------------------------------------------------------------------------
    // BRAM INTERFACE
    //------------------------------------------------------------------------
    output reg  [BRAM_ADDR_WIDTH-1:0] w_addr,   // Write address
    output reg  [BRAM_ADDR_WIDTH-1:0] r_addr,   // Read address
    output reg  [BRAM_DATA_WIDTH-1:0] w_data,   // Write data
    input  wire [BRAM_DATA_WIDTH-1:0] r_data,   // Read data
    output wire        w_req,                   // Write request
    output reg         r_req,                   // Read request
    input  wire        rbram_vd,                // Read data valid
    
    //------------------------------------------------------------------------
    // Interface for testing, not core 
    //------------------------------------------------------------------------
    input [63:0] rS, // status of qaoa system.
    output reg [23:0] CMD
);

//============================================================================
//---------------------- Localparam / Parameters 
//============================================================================

//---------------------------  MAIN STATE MACHINE STATES -----------------------
//---------------------------------------------------------------------------
// 9 state so need max 9 bit for optimal performance with one hot encoding.
// state representation with one hot encoding enables fast logic circuit synthesis.
// But this is within the range of optimmization at the compilation stage.
localparam s_IDLE        = 10'h1;   // Idle: waiting for command
localparam s_Fetch       = 10'h2;   // Fetching opcode/data from RX FIFO
localparam s_Operation   = 10'h4;   // Decode and dispatch operation
localparam s_WAIT_COMP   = 10'h8;   // Wait for arithmetic operation to complete
localparam s_WRITE_REG   = 10'h10;  // Write result to register
localparam s_WRITE_BRAM  = 10'h20;  // Write data to Block RAM
localparam s_READ_BRAM   = 10'h40;  // Read data from Block RAM
localparam s_TXData      = 10'h80;  // Transmit data to PC via HOST interface
localparam s_FetchWait   = 10'h100; // Wait for a data from FIFO

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
localparam OP_MOV_Info2U = 8'd7;   // send firmware version info
localparam OP_MOV_S2U    = 8'd8;      // Move rS → rU
localparam OP_MOV_T2P    = 8'd9;      // Move rT → rP
//------------------- Data Retrieval Operations (60-70)
localparam OP_FETCH1U    = 8'd60;  // Send 1 byte from rU to PC
localparam OP_FETCH8U    = 8'd61;  // Send 8 bytes from rU to PC

//---------------  Arithmetic Operations (Fixed-Point) (80-85)
localparam OP_ADD_B2A    = 8'd80;  // rA = rA + rB (64-bit fixed, 2 cycles)
localparam OP_MUL_B2A    = 8'd81;  // rA = rA * rB (24-bit fixed, 8 cycles)
localparam OP_INC_A     = 8'd84;    // rA = rA +1

// ---------------  Arithmetic Operations (Floating-Point)
localparam OP_ADDFP_B2A  = 8'd82;  // rA = rA + rB (FP64, 27 cycles)
localparam OP_MULFP_B2A  = 8'd83;  // rA = rA * rB (FP64, 24 cycles)
//-------------------- Memory Operations 
localparam OP_WRITE_T2RAM = 8'd111;  // Write rT to BRAM[rA]
localparam OP_READ_RAM2U  = 8'd112;  // Read BRAM[rA] → rU
localparam OP_SEND_CMD    = 8'd118;  // send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv

//---------------------  SUB-STATE DEFINITIONS (Fetch, Store) --------------
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
localparam WRITE_Info2U      = 5'd13;  // rU = firmware version string 

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

reg [10:0] txBRPos;     // position to store the byte 
logic [10:0] n_txBRPos; // position to store the byte  // BRAM transmit position

reg [3:0] txMaxPos;     // max position index to store the byte 
logic [3:0] n_txMaxPos; // next max position index to store the byte 

reg [10:0] rBRPos; // position to store the byte 
logic [10:0] n_rBRPos; // position to store the byte  // Next BRAM transmit position

//--------------------- Store/TX Sub-state Mchine
reg [2:0] storeReg;           //Current strore state
logic [2:0] n_storeReg;      // Next store state

reg [3:0] txPos;         // current tx byte position (0-7)
logic [3:0] n_txPos;     // Next TX position

//------------------- Write Backend Control 
reg [4:0] writeReg;           //current write operation
reg [4:0] bwriteReg;          // buffered write (for arthemetic operation)
logic [4:0] n_writeReg;       // Next write operation
logic [4:0] n_bwriteReg;      // Next buffered write

// ------------- Data Path Registers (64bit)
reg [63:0] rA;                // Register A (general purpose)
reg [63:0] rB;                // Register B
reg [63:0] rT;                // Temporary Register (RX data)
reg [63:0] rU;                // Address register (TX data)
reg [63:0] rC;
reg [63:0] rD;
reg [63:0] rV;

logic [63:0] n_rA, n_rB, n_rT, n_rU, n_rC, n_rD, n_rV;      //Next values

logic [7:0] n_o_CMD;
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
logic rf_req;                   // Read request
reg [1:0] rf_dv;              // Data valid (delayed)

//---------------------   TX FIFO (outgoing data to UART)
logic [7:0] tf_data;            // Data to TX FIFO
wire [7:0] tx_data_in;        // TX FIFO output
wire tx_active;               // UART transmitter active lag
wire [7:0] rx_data_out;   // UART receiver data output
wire tf_empty;                // TX FIFO empty flag
wire tf_full;                 // TX FIFO full flag
logic tf_write;                 // Write enable
reg [1:0] tx_dv;              // TX data valid


// -----------------------  BRAM Interface Signals

logic  [23:0] n_CMD;          // next command
logic [10:0] n_w_addr, n_r_addr;
logic [63:0] n_w_data;
logic l_w_req, n_r_req;


//---------------------- Helper Wires
wire [63:0] rAinc = rA3 + 1;  // Increment operations
wire [63:0] rBinc = rB3 + 1;

//---------- Conditional states (wait if FIFO full/empty)
wire [2:0] wstoreReg = tf_full ? STORE_WAIT : storeReg;
wire [2:0] wfetchReg = rf_dv[0] ? fetchReg : FETCH_WAIT;
//============================================================================

assign w_req = l_w_req;
assign tx_en = tx_dv[1];

//---------- Helper registers ------------------------------
reg RSTlv1A;  // Buffer reset wire to reduce performance decrease
reg [31:0] CP; // program counter, just for debugging
//============================================================================

always_comb begin: main_StateBlock
      n_state = state;
      n_opa_c_wait = opa_c_wait;
      n_fetchReg = fetchReg;
      n_storeReg = storeReg;
      n_c_wait = c_wait;
      n_fetchMaxPos = fetchMaxPos;
      n_writeReg = writeReg;
      n_bwriteReg = bwriteReg;
      n_txMaxPos = txMaxPos;
      l_w_req = 0;
      n_w_addr = rA;
      n_r_addr = rA;
      n_w_data = rT;
      n_CMD = 0;
      n_rA = rA; 
      n_rB = rB; 
      n_rT = rT;
      n_rU = rU;
      n_txPos = 0;
      n_txBRPos = 0;
      tf_data = 0;
      tf_write = 0;
      n_rPos = rPos;
      n_rBRPos = rBRPos;
      rf_req = (!rf_empty) & (fetchReg != FETCH_IDLE);
      n_ope_state = ope_state;
      n_r_req = 0;
      case(state)
      s_Fetch: begin
            case(wfetchReg)
            FETCH_DATA: begin
                  n_rT[rPos*8+:8] = rf_data;
                  n_rPos = rPos + 1;
                  if(rPos == fetchMaxPos) begin
                        n_fetchReg = FETCH_GETOP;
                  end
            end
            FETCH_WAIT: begin
                  n_rPos = rPos;
                  n_rBRPos = rBRPos;
            end
            FETCH_GETOP:begin
                  n_ope_state = rf_data;
                  n_state = s_Operation;
                  n_rPos = 0;
            end
            default: begin
                  n_rPos = 0;
                  n_rBRPos = 0;
            end
            endcase
      end
      s_Operation: begin
            // writing backend of register
            case (ope_state)
            OP_SEND1T: begin
                  n_state = s_Fetch;
                  n_fetchReg = FETCH_DATA;
                  n_fetchMaxPos = 0;
            end
            OP_SEND8T: begin
                  n_state = s_Fetch;
                  n_fetchMaxPos = 7;
                  n_fetchReg = FETCH_DATA;
            end
            OP_MOV_T2A: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_T2A;
            end
            OP_MOV_A2U: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_A2U;
            end
            OP_MOV_A2B: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_A2B;
            end
		OP_MOV_Info2U: begin
                  n_state = s_WRITE_REG;
                  n_writeReg = WRITE_Info2U;
            end
            OP_INC_A: begin
                  n_opa_c_wait = 1;
                  n_state = s_WAIT_COMP;
                  n_bwriteReg = WRITE_rA1;
            end
            OP_ADD_B2A: begin
                  n_opa_c_wait = 2;
                  n_state = s_WAIT_COMP;
                  n_bwriteReg = WRITE_add_rA;
            end
            OP_MUL_B2A: begin
                  n_opa_c_wait = 8;
                  n_state = s_WAIT_COMP;
                  n_bwriteReg = WRITE_mul_rA;
            end
            OP_ADDFP_B2A: begin
                  n_opa_c_wait = 27;
                  n_state = s_WAIT_COMP;
                  n_bwriteReg = WRITE_addFP64_rA;
            end
            OP_MULFP_B2A: begin
                  n_opa_c_wait = 24;
                  n_state = s_WAIT_COMP;
                  n_bwriteReg = WRITE_mulFP64_rA;
            end
            OP_FETCH1U: begin
                  n_storeReg = STORE_LEN;
                  n_txMaxPos = 0;
                  n_state = s_TXData;
            end
            OP_FETCH8U: begin
                  n_storeReg = STORE_LEN;
                  n_txMaxPos = 7;
                  n_state = s_TXData;
            end
            OP_WRITE_T2RAM: begin
                  l_w_req = 1;
                  n_state = s_WRITE_BRAM;
            end
            OP_READ_RAM2U: begin
                  // need to wait the latency.
                  n_r_req = 1;
                  n_state = s_READ_BRAM;
            end
            OP_SEND_CMD: begin 
                  n_CMD[23] = 1;
                  n_CMD[7:0] = rT[7:0];
                  n_state = s_IDLE;
            end
            default: begin // idle
                  n_state = s_IDLE;
            end
            endcase
      end
      s_READ_BRAM:begin
            n_r_req = 0;
            if(rbram_vd) begin
                  n_writeReg = WRITE_BRAM_U;
                  n_state = s_WRITE_REG;
            end
      end
      s_WRITE_BRAM: begin
            l_w_req = 0;
            n_state = s_IDLE;
      end
      s_WAIT_COMP: begin 
            n_c_wait = c_wait + 1;
            if(c_wait == opa_c_wait)begin 
                  n_writeReg = bwriteReg;
                  n_state = s_WRITE_REG;
            end 
      end
      s_WRITE_REG: begin // writing to register.
            
            case (writeReg)
                  WRITE_T2A: begin
                        n_rA = rT;
                  end
                  WRITE_T2B: begin
                        n_rB = rT;
                  end
                  WRITE_A2B: begin
                        n_rB = rA;
                  end
                  WRITE_A2U: begin
                        n_rU = rA;
                  end
                  WRITE_mulFP64_rA: begin
                        n_rA = res_mulFP64;
                  end
                  WRITE_addFP64_rA: begin
                        n_rA = res_addFP64;
                  end
                  WRITE_mul_rA: begin
                        n_rA = res_mulAB;
                  end
                  WRITE_add_rA: begin
                        n_rA = res_addAB;
                  end
                  WRITE_rA1: begin
                        n_rA = rAinc;
                  end
                  WRITE_rB1: begin
                        n_rB = rBinc;
                  end
                  WRITE_BRAM_U: begin
                        n_rU = r_data;
                  end
                  WRITE_Info2U: begin
                        n_rU = 64'h3233761d5355544e; // "NTUSMv01"
                  end
                  default: begin
                  end
            endcase
            n_state =  s_IDLE;
      end
      s_TXData: begin
            case(wstoreReg)
            STORE_LEN: begin
                  tf_data = rU[txPos*8+:8];
                  n_txPos = txPos + 1;
                  if(txPos == txMaxPos) begin
                        n_state = s_IDLE;
                  end
                  tf_write = 1;
            end
            default: begin
                  n_txPos = txPos;
                  n_txBRPos = txBRPos;
            end
            endcase
      end
      default: begin
            // get a new operation.
            n_txBRPos = 0;
            n_txPos = 0;
            n_c_wait = 0;
            n_state = s_Fetch;
            n_storeReg = STORE_IDLE;
            n_writeReg = WRITE_NONE;
            n_fetchReg = FETCH_GETOP;
            n_fetchMaxPos = 0;
      end
      
endcase
end

always @(posedge CLK) begin
	if (RSTlv1A) begin
            rf_dv <= 0;
		CP <= 4123;
		state <= s_IDLE;
            fetchReg <= FETCH_IDLE;
		ope_state <= 0;
            rPos <= 0;
            rA <= 0;
            rB <= 0;
            rT <= 0;
            rU <= 0;
            c_wait <= 0;
            txPos <= 0;
            txMaxPos <= 0;
            fetchMaxPos <= 0;
            rPos <= 0;
            fetchReg <= 0;
            storeReg <= 0;
            opa_c_wait <= 0;
            CMD <= '0;
	end
      else begin
            rf_dv[0] <= rf_req;
            rf_dv[1] <= rf_dv[0];
            opa_c_wait <= n_opa_c_wait;
            c_wait <= n_c_wait;
            rA <= n_rA;
            rB <= n_rB;
            rT <= n_rT;
            rU <= n_rU;
            CMD <= n_CMD;
            state <= n_state;
            ope_state <= n_ope_state;
            fetchReg <= n_fetchReg;
            storeReg <= n_storeReg;
            rBRPos <= n_rBRPos;
            fetchMaxPos <= n_fetchMaxPos;
            rPos <= n_rPos;
            txPos <= n_txPos;
            txMaxPos <= n_txMaxPos;

            bwriteReg <= n_bwriteReg;
            writeReg <= n_writeReg;

            w_data <= n_w_data;
            w_addr <= n_w_addr;
            r_addr <= n_r_addr;
            r_req <= n_r_req;
      end
end


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

always_ff @( CLK ) begin : local_rest
      RSTlv1A <= RST;
end

always @(posedge CLK) begin 
      if(RSTlv1A) begin
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
      .areset(RSTlv1A),      // Active-high reset
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
      .areset(RSTlv1A),      // Active-high reset
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
// STATUS REGISTER USAGE GUIDE (o_Status)
//============================================================================
// The o_Status register is a 32-bit register used for bidirectional
// communication between the FPGA and PC control software.
//
// FORMAT:
//   [31:8] - Reserved for future use (currently always 0)
//   [7:0]  - Command/Status byte (updated via OP_SEND_CMD or internally)

// STATUS CODES (o_Status[7:0])
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

