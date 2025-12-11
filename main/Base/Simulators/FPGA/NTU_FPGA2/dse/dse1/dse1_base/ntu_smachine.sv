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
   output reg  r_req,
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
logic [63:0] n_rA;
logic [63:0] n_rB;
logic [63:0] n_rC;
logic [63:0] n_rD;
logic [63:0] n_rU; // address to memory space
logic [63:0] n_rV; // address to memory space
logic [63:0] n_rT;
reg [63:0] rA;
reg [63:0] rA2;
reg [63:0] rA3;
reg [63:0] rA4;
reg [63:0] rB;
reg [63:0] rB2;
reg [63:0] rB3;
reg [63:0] rB4;
reg [63:0] rC;
reg [63:0] rD;
reg [63:0] rU;
reg [63:0] rV;
reg [63:0] rT;
reg [7:0] ope_state;
logic [7:0] n_ope_state;
wire [7:0] rf_data;
wire [7:0] rx_data_out;
logic rf_req;
wire rf_empty;
wire rf_full;
wire tf_empty;
wire tf_full;
logic tf_write;
logic [7:0] tf_data;
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
      .o_Rx_Byte       (rx_data_out)    
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
            else if(~tx_active & ~tf_empty) begin
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
.data ( rx_data_out ), //input
.rdreq ( rf_req ),
.wrreq ( rx_dv ),
.empty ( rf_empty ),
.full ( rf_full ),
.q ( rf_data ) // output
);

fifo1	fifoW_inst (
.clock ( CLK ),
.data ( tf_write ), //input
.rdreq ( tx_dv[0] ),
.wrreq ( tf_write ),
.empty ( tf_empty ),
.full ( tf_full ),
.q ( tx_data_in ) // output
);


wire [63:0] res_addFP64;
wire [63:0] res_mulFP64;
wire [47:0] n_res_mulAB;
addFPF64 addFPF64(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA2),      //      a.a
      .b(rB2),      //      b.b
      .q(res_addFP64)       //      q.q
);

mulFPF64 mf64i(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA2),      //      a.a
      .b(rB2),      //      b.b
      .q(res_mulFP64)       //      q.q
);
reg [23:0] mulA;
reg [23:0] mulB;
reg [63:0] addA;
reg [63:0] addB;
wire [63:0] res_addAB;
reg [63:0] n_res_addAB;
reg [23:0] res_mulAB;
always @(posedge CLK)
begin
      res_mulAB <= n_res_mulAB[23:0];
      res_addAB <= n_res_addAB;
      mulA <= rA[23:0];
      mulB <= rB[23:0];
      addA <= rA;
      addB <= rB;
end
mulfix8 mulfix8_inst (
	.clock ( CLK ),
	.dataa ( mulA ),
	.datab ( mulB ),
	.result ( n_res_mulAB )
);

addfix8 addix8_inst (
	.clock ( CLK ),
	.dataa ( addA ),
	.datab ( addB),
	.result ( n_res_addAB )
);

wire rAinc = rA3 + 1;
wire rBinc = rB3 + 1;
parameter s_IDLE = 0;
parameter s_Fetch = 1;
parameter s_Operation = 2;
parameter s_WAIT_COMP = 4;
parameter s_WRITE_REG = 8;
parameter s_WRITE_BRAM = 16;
parameter s_READ_BRAM = 32;
parameter s_TXData = 64;
parameter s_FetchWait = 128;
parameter OP_NONE = 0; // Send: 1, Res: 0.
parameter OP_SEND1T = 1; // Send: 0, Res: 1.
parameter OP_SEND8T = 2; // Send: 0, Res: 1.
parameter OP_MOV_T2A = 3; // Send: 1, Res: 0.
parameter OP_MOV_T2B = 4; // Send: 1, Res: 0.
parameter OP_MOV_A2U = 5; // Send: 1, Res: 0.
parameter OP_MOV_A2B = 6; // Send: 0, Res: 1.
parameter OP_FETCH1U = 60; // Send: 0, Res: 8.
parameter OP_FETCH8U = 61; // Send: 0, Res: 8.
parameter OP_ADD_B2A = 80; // Send: 0, Res: 0.
parameter OP_MUL_B2A = 81; // Send: 0, Res: 0.
parameter OP_ADDFP_B2A = 82; // Send: 0, Res: 0.
parameter OP_MULFP_B2A = 83; // Send: 0, Res: 0.
parameter OP_INC_A = 84; // Send: 0, Res: 0.
parameter OP_WRITE_T2RAM = 111; // Send: 0, Res: 0.
parameter OP_READ_RAM2U = 112; // Send: 0, Res: 0.

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
reg [10:0] txBRPos; // position to store the byte 
reg [10:0] rBRPos; // position to store the byte 
reg [3:0] txPos; // position to store the byte 
reg [3:0] txMaxPos;
reg [3:0] rPos; // position to store the byte c
reg [3:0] fetchMaxPos; // position to store the byte c
logic [10:0] n_txBRPos; // position to store the byte 
logic [10:0] n_rBRPos; // position to store the byte 
logic [3:0] n_txPos; // position to store the byte  
logic [3:0] n_txMaxPos;
logic [3:0] n_rPos; // position to store the byte 
logic [3:0] n_fetchMaxPos; // position to store the byte c
parameter STORE_IDLE = 0;
parameter STORE_LEN = 1;
parameter STORE_BRAM = 2;
parameter STORE_WAIT = 4;

wire [3:0] wstoreReg = tf_full? STORE_WAIT: storeReg;
wire [3:0] wfetchReg = rf_dv[0]? FETCH_WAIT: fetchReg;

always_comb begin : STORE_COMb
      

end

parameter FETCH_IDLE = 0;
parameter FETCH_DATA = 1;
parameter FETCH_WAIT = 2;
parameter FETCH_GETOP = 4;
parameter FETCH_w_BRAM = 8;
always_comb begin: FETCH_BLOCK 
      
end
reg [14:0] writeReg;
reg [14:0] bwriteReg;
logic [14:0] n_writeReg;
logic [14:0] nb_writeReg;

parameter WRITE_NONE = 0;
parameter WRITE_T2A = 1;
parameter WRITE_T2B = 2;
parameter WRITE_A2B = 4;
parameter WRITE_A2U = 8;
parameter WRITE_B2A = 16;
parameter WRITE_mulFP64_rA = 32;
parameter WRITE_addFP64_rA = 64;
parameter WRITE_mul_rA = 128;
parameter WRITE_add_rA = 256;
parameter WRITE_rA1 = 512;
parameter WRITE_rB1 = 1024;
parameter WRITE_BRAM_U = 2048;


always_comb begin : WRITING_TO_REGISTER_BLOCK
    
end
always_comb begin: main_StateBlock
      n_state = state;
      n_opa_c_wait = opa_c_wait;
      n_fetchReg = fetchReg;
      n_storeReg = storeReg;
      n_c_wait = c_wait;
      n_fetchMaxPos = fetchMaxPos;
      n_writeReg = writeReg;
      nb_writeReg = bwriteReg;
      n_txMaxPos = txMaxPos;
      n_w_req = 0;
      n_w_addr = rA;
      n_r_addr = rA;
      n_w_data = rT;
      n_rA = rA; 
      n_rB = rB; 
      n_rU = rU;
      n_rA = rA; 
      n_rB = rB; 

      n_txPos = 0;
      n_txBRPos = 0;
      tf_write = 0;
      tf_write = 0;
      n_rPos = rPos;
      n_rBRPos = rBRPos;
      n_rT = rT;
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
            OP_INC_A: begin
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
            OP_ADDFP_B2A: begin
                  n_opa_c_wait = 27;
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_addFP64_rA;
            end
            OP_MULFP_B2A: begin
                  n_opa_c_wait = 24;
                  n_state = s_WAIT_COMP;
                  nb_writeReg = WRITE_mulFP64_rA;
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
                  n_w_req = 1;
                  n_state = s_IDLE;
            end
            OP_READ_RAM2U: begin
                  // need to wait the latency.
                  n_r_req = 1;
            end
            default: begin // idle
                  n_state = s_IDLE;
            end
            endcase
      end
      s_READ_BRAM:begin
            if(rbram_vd) begin
                  n_writeReg = WRITE_BRAM_U;
                  n_state = s_WRITE_REG;
            end
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
            rA2 <= 0;
            rA3 <= 0;
            rB <= 0;
            rB2 <= 0;
            rB3 <= 0;
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
	end
      else begin
            rf_dv[0] <= rf_req;
            rf_dv[1] <= rf_dv[0];
            opa_c_wait <= n_opa_c_wait;
            c_wait <= n_c_wait;
            rA <= n_rA;
            rA2 <= n_rA;
            rA3 <= n_rA;
            rA4 <= n_rA;
            rB <= n_rB;
            rB2 <= n_rB;
            rB3 <= n_rB;
            rB4 <= n_rB;
            rT <= n_rT;
            rU <= n_rU;
            state <= n_state;
            ope_state <= n_ope_state;
            fetchReg <= n_fetchReg;
            storeReg <= n_storeReg;
            rBRPos <= n_rBRPos;
            fetchMaxPos <= n_fetchMaxPos;
            rPos <= n_rPos;
            txPos <= n_txPos;
            txMaxPos <= n_txMaxPos;

            bwriteReg <= nb_writeReg;
            writeReg <= n_writeReg;

            w_data <= n_w_data;
            w_addr <= n_w_addr;
            r_addr <= n_r_addr;
            r_req <= n_r_req;
      end
end
endmodule

