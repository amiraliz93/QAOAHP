// This is a simple module act on state machine controlled via serial communcation.
// Default implementation uses UART module as an background, but it can be replaced with another module with the speed less 
// than the processing capacity of this statemachine. The processing capacity is a quarter of the clock.
// tested by ntu_smachine_tb.v, on 2025 Sep. 2nd.
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
module ntu_smachine
  #(parameter UART_CLKS_PER_BIT = 434) // 50 MHz with 115200 baud rate
  (
   input       CLK,
   input       RST,
   input       i_Rx_Serial,
   output      o_Tx_Serial,
   output [31:0]  o_Status, // counter to wait read FIFO latency. 
   input  [63:0] r_data,
   output reg  [63:0] r_addr,
   input  r_req,
   input  rbram_vd,
   output reg [63:0] w_addr,
   output reg [63:0] w_data,
   output reg w_req
);


logic  [63:0] n_w_addr;
logic  [63:0] n_w_data;
logic  [63:0] n_r_addr;
logic n_w_req;
logic n_r_req;

wire [7:0] tx_data_in;
wire tx_active;
wire rx_dv;

reg [31:0] CP; // program counter

reg [7:0] state;
logic [7:0] n_state;
reg [7:0] state_after_fetch;
logic [7:0] n_state_after_fetch;
logic [63:0] n_rA;
logic [63:0] n_rB;
logic [63:0] n_rC;
logic [63:0] n_rD;
logic [63:0] n_rU; // address to memory space
logic [63:0] n_rV; // address to memory space
logic [63:0] n_rT;
reg [63:0] rA;
reg [63:0] rB;
reg [63:0] rC;
reg [63:0] rD;
reg [63:0] rU;
reg [63:0] rV;
reg [63:0] rT;
reg [7:0] ope_state;
logic [7:0] n_ope_state;
wire [7:0] rfifo_data;
wire [7:0] rx_data_out;
logic rx_fifio_req;
wire rx_empty;
wire rx_full;
wire tx_empty;
wire tx_full;
logic tx_fifo_write;
logic [7:0] tx_fifo_data;
reg [1:0] tx_dv;
reg [10:0] c_wait; // remaining count to wait.
logic [10:0] n_c_wait; // remaining count to wait.
reg [10:0] opa_c_wait; // remaining count to wait.
logic [10:0] n_opa_c_wait; // remaining count to wait.

reg [1:0] rf_dv;

reg RSTlv1A;
reg RSTlv1B;
always_ff @( CLK ) begin : local_rest
      RSTlv1A <= RST;
      RSTlv1B <= RST;
end

assign o_Status = {state, ope_state, rB, rA};
receiver #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) uart1 
(
      .i_Clock       (CLK),
      .RST       (RSTlv1B),
      .i_Rx_Serial   (i_Rx_Serial),
	.o_Rx_DV   (rx_dv),
      .o_Rx_Byte       (o_Rx_Byte)    
);
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
            else if(~tx_active & ~tx_empty) begin
                  tx_dv[0] <= 1;
            end
            tx_dv[1] <= tx_dv[0];
      end
end
	
transmitter #(.CLKS_PER_BIT(UART_CLKS_PER_BIT) // 50 MHz
) t0(
      .i_Clock(CLK),
      .RST(RSTlv1B),
      .i_Tx_DV(tx_dv[1]),
      .i_Tx_Byte(tx_data_in), 
      .o_Tx_Active(tx_active),
      .o_Tx_Serial(o_Tx_Serial),
      .o_Tx_Done()
);

fifo1	fifo1_inst (
.clock ( CLK ),
.data ( o_Rx_Byte ), //input
.rdreq ( rx_fifio_req ),
.wrreq ( rx_dv ),
.empty ( rx_empty ),
.full ( rx_full ),
.q ( rfifo_data ) // output
);

fifo1	fifoW_inst (
.clock ( CLK ),
.data ( tx_fifo_data ), //input
.rdreq ( tx_dv[0] ),
.wrreq ( tx_fifo_write ),
.empty ( tx_empty ),
.full ( tx_full ),
.q ( tx_data_in ) // output
);


wire [63:0] res_addFP64;
wire [63:0] res_mulFP64;
wire [15:0] res_mulAB;
wire [7:0] res_addAB;
// addFPF64 addFPF64(
//       .clk(CLK),    //    clk.clk
//       .areset(RSTlv1B), // areset.reset
//       .a(rA64),      //      a.a
//       .b(rB64),      //      b.b
//       .q(res_addFP64)       //      q.q
// );

// mulFPF64 mf64i(
//       .clk(CLK),    //    clk.clk
//       .areset(RSTlv1B), // areset.reset
//       .a(rA64),      //      a.a
//       .b(rB64),      //      b.b
//       .q(res_mulFP64)       //      q.q
// );
mulfix8 mulfix8_inst (
	.clock ( CLK ),
	.dataa ( rA ),
	.datab ( rB ),
	.result ( res_mulAB )
);

addfix8 addix8_inst (
	.clock ( CLK ),
	.dataa ( rA ),
	.datab ( rB ),
	.result ( res_addAB )
);

wire rAinc = rA + 1;
wire rBinc = rB + 1;
parameter s_IDLE = 0;
parameter s_Fetch = 1;
parameter s_Operation = 2;
parameter s_WAIT_COMP = 3;
parameter s_WRITE_REG = 4;
parameter s_WRITE_BRAM = 5;
parameter s_READ_BRAM = 6;
parameter s_TXData = 7;
parameter OP_NONE = 0; // Send: 1, Res: 0.
parameter OP_SEND1T = 31; // Send: 0, Res: 1.
parameter OP_SEND8T = 31; // Send: 0, Res: 1.
parameter OP_MOV_T2A = 1; // Send: 1, Res: 0.
parameter OP_MOV_T2B = 2; // Send: 1, Res: 0.
parameter OP_INC_A = 5; // Send: 0, Res: 0.
parameter OP_MOV_A2U = 31; // Send: 0, Res: 1.
parameter OP_MOV_A2U = 31; // Send: 0, Res: 1.
parameter OP_FETCH1U = 60; // Send: 0, Res: 8.
parameter OP_FETCH8U = 60; // Send: 0, Res: 8.
parameter OP_ADD_B2A = 61; // Send: 0, Res: 0.
parameter OP_MUL_B2A = 61; // Send: 0, Res: 0.
parameter OP_ADDFP_B2A = 61; // Send: 0, Res: 0.
parameter OP_MULFP_B2A = 62; // Send: 0, Res: 0.
parameter OP_WRITE_T2RAM = 81; // Send: 0, Res: 0.
parameter OP_READ_RAM2U = 82; // Send: 0, Res: 0.

parameter OP_RUN_MIXER = 100; // Send: 0, Res: 0. Run the mixer operation 1 step.
parameter OP_RUN_COST = 101; // Send: 0, Res: 0. Run the cost operation 1 step. 
parameter OP_RUN_CONTINUOUS = 103; // Send: 0, Res: 0. Run the cost operation 1 step. 
parameter OP_ENABLE_INTRUPTION = 121; // Make the state machine send an 1 byte message to PC (always 43), if rS64 was changed. Note that rS64 cannot be changed from PC, so this event occurs from FPGA side only. This intruption may minimize the waiting time of the culculation. You can implement a function so that it will be invoked with this signal, by checking reading the UART port always, and immediately take an action if the state of FPGA changed spontaneously. Such function is called "callback function" or "event handler". event handler minimize the wainting time of external process without wasting computing resouces. Default of this mode is off.

// operation status of rS64.
// sR64=100: during Mixer operation.
// sR64=102: Mixer operation done.
// sR64=101: during COST operation.
// sR64=103: COST operation done.
// rS64=77: data recieved and set to the BRAM successfully.
// rS64=76: data sent to PC from the BRAM successfully.
// rS64=79: data recieved and set to the BRAM, part of data was desposed because the block overflow the boundary of BRAM. Usually, not problem
// rS64=78: data sent to PC from the BRAM, part of data was desposed because the block overflow the boundary of BRAM. Usually, not problem
// rS64=0:  idle state, nothing is in operation.
reg [3:0] fetchReg;
reg [3:0] storeReg;
logic [3:0] n_storeReg;
logic [3:0] n_fetchReg;
logic endStore;
logic endFetch;
reg [10:0] txBRPos; // position to store the byte 
reg [10:0] rBRPos; // position to store the byte 
reg [3:0] txPos; // position to store the byte 
reg [3:0] txMaxPos;
reg [3:0] rPos; // position to store the byte c
reg [3:0] rMaxPos; // position to store the byte c
logic [10:0] n_txBRPos; // position to store the byte 
logic [10:0] n_rBRPos; // position to store the byte 
logic [3:0] n_txPos; // position to store the byte  
logic [3:0] n_txMaxPos;
logic [3:0] n_rPos; // position to store the byte 
logic [3:0] n_rMaxPos; // position to store the byte c
parameter STORE_IDLE = 0;
parameter STORE_LEN = 1;
parameter STORE_BRAM = 4;
parameter STORE_WAIT = 5;

wire wstoreReg = tx_full? STORE_WAIT: storeReg;
always_comb begin : STORE_COMb
      endStore = 0;
      n_txPos = 0;
      n_txBRPos = 0;
      tx_fifo_write = 0;
      tx_fifo_data = 0;
      n_r_addr = rR64 + txBRPos;
      case(wstoreReg)
      STORE_LEN: begin
            tx_fifo_data = rU64[txPos*8+:8];
            if(txPos == txMaxPos) begin
                  endStore = 1;
                  n_txPos = 0;
            end
            else begin
                  n_txPos = txPos + 1;
            end
            tx_fifo_write = 1;
      end
      STORE_WAIT: begin
            n_txPos = txPos;
            n_txBRPos = txBRPos;
      end
      default: begin
            n_txBRPos = 0;
            n_txPos = 0;
      end
      endcase

end

parameter FETCH_IDLE = 0;
parameter FETCH_LEN = 3;
parameter FETCH_w_BRAM = 4;
always_comb begin: FETCH_BLOCK 
      n_rPos = rPos;
      n_rBRPos = rBRPos;
      endFetch = 0;
      n_rT64 = rT64;
      n_w_addr = rR64 + rBRPos;
      n_w_data = w_data;
      rx_fifio_req = (!rx_empty) & (fetchReg != FETCH_IDLE);
      case(fetchReg)
      FETCH_LEN: begin
            n_rT64[rPos*8+:8] = rfifo_data;
            if(rPos == rMaxPos) begin
                  n_rPos = 0;
                  endFetch = 1;
            end
            else begin
                  n_rPos = rPos+1;
            end 
      end
      default: begin
            n_rPos = 0;
            n_rBRPos = 0;
      end
      endcase
end
reg [3:0] writeReg;
reg [3:0] bwriteReg;
logic [3:0] n_writeReg;
logic [3:0] nb_writeReg;

parameter WRITE_NONE = 0;
parameter WRITE_rT_rA = 1;
parameter WRITE_rT_rB = 2;
parameter WRITE_rA_rB = 3;
parameter WRITE_rT_rA64 = 4;
parameter WRITE_rB64_rA64 = 5;
parameter WRITE_mulFP64_rA64 = 6;
parameter WRITE_addFP64_rA64 = 7;
parameter WRITE_mul8_rA = 8;
parameter WRITE_add8_rA = 9;
parameter WRITE_rA1 = 10;
parameter WRITE_rB1 = 11;
parameter WRITE_rV64rU64 = 12;

always_comb begin : WRITING_TO_REGISTER_BLOCK
    
      n_rA = rA; 
      n_rB = rB; 
      n_rU = rU;
      n_rT = rT;
      n_rA = rA; 
      n_rB = rB; 
      case (writeReg)
            WRITE_rT_rA: begin
                  n_rA = rT64[7:0];
            end
            WRITE_rT_rB: begin
                  n_rA = rT64[7:0];
            end
            WRITE_rA_rB: begin
                  n_rA = rB;
            end
            WRITE_rT_rA64: begin
                  n_rA64 = rT64;
            end
            WRITE_rB64_rA64: begin
                  n_rB64 = rA64;
            end
            WRITE_mulFP64_rA64: begin
                  n_rA64 = res_mulFP64;
            end
            WRITE_addFP64_rA64: begin
                  n_rA64 = res_addFP64;
            end
            WRITE_mul8_rA: begin
                  n_rA = res_mulAB;
            end
            WRITE_add8_rA: begin
                  n_rA = res_addAB;
            end
            WRITE_rA1: begin
                  n_rA = rAinc;
            end
            WRITE_rB1: begin
                  n_rB = rBinc;
            end
            WRITE_rVrU: begin
                  n_rU64 = rV64;
            end
            default: begin
            end
      endcase
end
always_comb begin: main_StateBlock
      n_state = state;
      n_opa_c_wait = 0;
      n_fetchReg = fetchReg;
      n_storeReg = storeReg;
      n_ope_state = ope_state;
      n_c_wait = c_wait;
      n_rMaxPos = 0;
      n_state_after_fetch = s_Operation;
      case(state)
      s_Fetch: begin
            if(endFetch) begin
                  n_state = state_after_fetch;
                  n_fetchReg = FETCH_IDLE;
                  n_ope_state = rT64[7:0];
            end
      end
      s_Operation: begin
            // writing backend of register
            case (ope_state)
            OP_SEND1T: begin
                  n_state = s_Fetch;
                  n_state_after_fetch = s_Fetch;
                  n_fetchReg = FETCH_LEN;
            end
            OP_SEND8T: begin
                  n_state = s_Fetch;
                  n_rMaxPos = 7;
                  n_fetchReg = FETCH_LEN;
                  n_state_after_fetch = s_Fetch;
            end
            OP_MOV_T2A: begin
                  n_state = s_Fetch;
                  n_writeReg = WRITE_T2A;
                  n_fetchReg = FETCH_LEN;
            end
            OP_MOV_A2U: begin
                  n_state = s_Fetch;
                  n_writeReg = WRITE_A2U;
                  n_fetchReg = FETCH_LEN;
            end
            OP_MOV_A2B: begin
                  n_state = s_Fetch;
                  n_writeReg = WRITE_A2B;
                  n_fetchReg = FETCH_LEN;
            end
            OP_INC_rA: begin
                  n_opa_c_wait = 1;
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_rA1;
            end
            OP_ADD_B2A: begin
                  n_opa_c_wait = 2;
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_add_rA;
            end
            OP_MUL_B2A: begin
                  n_opa_c_wait = 8;
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_mul_rA;
            end
            OP_FETCH1U: begin
                  n_storeReg = STORE_LEN;
                  n_storeMaxPos = 0;
                  n_state = s_TXData;
            end
            OP_FETCH8U: begin
                  n_storeReg = STORE_LEN;
                  n_storeMaxPos = 7;
                  n_state = s_TXData;
            end
            default: begin // idle
                  n_fetchReg = FETCH_LEN;
                  n_state = s_Fetch;
            end
            endcase
      end
      s_WAIT_COMP: begin 
            if(c_wait == opa_c_wait)begin 
                  n_c_wait = 0;
                  n_writeReg = bwriteReg;
                  n_state = s_WRITE_REG;
            end 
            else begin
                  n_c_wait = c_wait + 1;
            end
      end
      s_WRITE_REG: begin // writing to register.
            n_state = s_IDLE;
            n_writeReg = WRITE_NONE;
      end
      s_TXData: begin
            if(endStore) begin
                  n_storeReg = STORE_IDLE;
                  n_fetchReg = FETCH_LEN;
                  n_state = s_Fetch;
                  n_ope_state = OP_NONE;
            end
      end
      default: begin
            // get a new operation.
            n_state = s_Fetch;
            n_writeReg = WRITE_NONE;
            n_rMaxPos = 0;
            n_fetchReg = FETCH_LEN;
      end
      
endcase
end

always @(posedge CLK) begin
	if (RSTlv1A) begin
            rf_dv <= 0;
		CP <= 4123;
		state <= s_IDLE;
		state_after_fetch <= s_IDLE;
            fetchReg <= FETCH_IDLE;
		ope_state <= 0;
            r8Pos <= 0;
            rA <= 0;
            rB <= 0;
            c_wait <= 0;
            state_after <= 0;
            tx8Pos <= 0;
            storeMaxPos <= 0;
            storeReg <= 0;
            opa_c_wait <= 0;
	end
      else begin
            rf_dv[0] <= rx_fifio_req;
            rf_dv[1] <= rf_dv[0];
            opa_c_wait <= n_opa_c_wait;
            c_wait <= n_c_wait;
            rA <= n_rA;
            rB <= n_rB;
            rA64 <= n_rA64;
            rB64 <= n_rB64;
            state <= n_state;
            state_after_fetch <= n_state_after_fetch;
            ope_state <= n_ope_state;
            fetchReg <= nr_fetchReg;
            storeReg <= nr_storeReg;
            tx8Pos <= n_tx8Pos;
            r8Pos <= n_r8Pos;
            rBRPos <= n_rBRPos;
            storeMaxPos <= n_storeMaxPos;

            bwriteReg <= nb_writeReg;
            writeReg <= n_writeReg;

            w_data <= n_w_data;
      end
end
endmodule

