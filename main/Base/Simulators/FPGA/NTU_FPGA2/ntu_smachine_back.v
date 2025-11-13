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
	input  [127:0] i_Status,
   output      o_Tx_Serial,
   output [31:0]  o_Status, // counter to wait read FIFO latency. 
	output [63:0]  o_waddr, // address of the data writtetn
	output [63:0]  o_wdata, // data to be written  
	output  o_wvd,
	input  o_rvd,
	output [63:0]  o_raddr, // address of the data read
	output [63:0]  o_rdata // data to be read  
);

wire [7:0] tx_data_in;
wire tx_active;
wire rx_dv;

reg [31:0] CP; // program counter

reg [7:0] state;
reg [7:0] rA;
reg [7:0] rB;
reg [31:0] rA32;
reg [31:0] rB32;
reg [63:0] rA64;
reg [63:0] rB64;
reg [63:0] rX64;
reg [63:0] rS64;
reg [7:0] rT;
reg [7:0] ope_state;
wire [7:0] rfifo_data;
wire [7:0] o_Rx_Byte;
reg rdreq;
wire empty;
wire full;
wire tx_empty;
wire tx_full;
reg tx_fifo_write;
reg [1:0] tx_dv;
reg [7:0] tx_fifo_data;
reg [10:0] c_wait; // remaining count to wait.
reg [3:0] r8Pos; // position to store the byte 

assign o_Status = {state, ope_state, rB, rA};
receiver #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) uart1 
(
      .i_Clock       (CLK),
      .RST       (RST),
      .i_Rx_Serial   (i_Rx_Serial),
	.o_Rx_DV   (rx_dv),
      .o_Rx_Byte       (o_Rx_Byte)    
);
// sending block from fifo to uart unit. This looks so waste. Do I really need such this block intrinsically?
// I guess that fifoblock should be incorpolated into transmitter.
always @(posedge CLK) begin 
      if(RST) begin
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
      .RST(RST),
      .i_Tx_DV(tx_dv[1]),
      .i_Tx_Byte(tx_data_in), 
      .o_Tx_Active(tx_active),
      .o_Tx_Serial(o_Tx_Serial),
      .o_Tx_Done()
);
  
fifo1	fifo1_inst (
.clock ( CLK ),
.data ( o_Rx_Byte ), //input
.rdreq ( rdreq ),
.wrreq ( rx_dv ),
.empty ( empty ),
.full ( full ),
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
wire [7:0] res_mul8;
wire [7:0] res_add8;
addFPF64 addFPF64(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA64),      //      a.a
      .b(rB64),      //      b.b
      .q(res_addFP64)       //      q.q
);

mulFPF64 mf64i(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA64),      //      a.a
      .b(rB64),      //      b.b
      .q(res_mulFP64)       //      q.q
);
mulfix8 mulfix8_inst (
	.clock ( CLK ),
	.dataa ( rA ),
	.datab ( rB ),
	.result ( res_mul8 )
);

addfix8 addix8_inst (
	.clock ( CLK ),
	.dataa ( rA ),
	.datab ( rB ),
	.result ( res_add8 )
);


parameter s_IDLE = 0;
parameter s_TXData = 208;
parameter s_Fetch = 209;
parameter s_FetchData = 210;
parameter s_WAIT = 211;
parameter s_Operation = 212;
parameter s_Operand = 213;
parameter OP_MOV_rA = 1; // Send: 1, Res: 0.
parameter OP_MOV_rB = 2; // Send: 1, Res: 0.
parameter OP_ADD_rB2rA = 3; // Send: 0, Res: 0.
parameter OP_MUL_rB2rA = 4; // Send: 0, Res: 0.
parameter OP_INC_rA = 5; // Send: 0, Res: 0.
parameter OP_INC_rB = 6; // Send: 0, Res: 0.
parameter OP_READ_rA = 31; // Send: 0, Res: 1.
parameter OP_READ_rB = 32; // Send: 0, Res: 1.
parameter OP_MOV_rA64 = 51; // Send: 0, Res: 1.
parameter OP_MOV_rA64rB = 52; // Send: 0, Res: 0.
parameter OP_MOV_rA64rX = 53; // Send: 0, Res: 0.
parameter OP_MOV_rX64rA = 54; // Send: 0, Res: 0.
parameter OP_MOV_rB64rA = 56; // Send: 0, Res: 0.
parameter OP_MOV_rS64rA = 57; // Send: 0, Res: 0.
parameter OP_READ_rA64 = 55; // Send: 0, Res: 8.
parameter OP_ADD_rBrA64FP = 61; // Send: 0, Res: 0.
parameter OP_MUL_rBrA64FP = 62; // Send: 0, Res: 0.

parameter OP_SEND_STATES = 82; // send: 1024, Res: 0. this operation send data to the space beginning with the address with the value in rX64. Need to send 1024 bytes after sending this operation. Response value will be stored in rS64. 77 means success, the other means failed.
parameter OP_SEND_COST = 83; // Send: 512, Res: 0. send 512 bytes data to the space beginning with the address with the value in rX64.  Response value will be stored in rS64.  77 means success, the other means failed.
parameter OP_SEND_COST_COMPLEX = 84; // Send: 1024, Res: 0. send bytes data to the space beginning with the address with the value in rX64.  Response value will be stored in rS64.  77 means success, the other means failed.
parameter OP_READ_STATES = 86; // Send: 0. Res: 1024. Read the data of state vectors from FPGA.
parameter OP_READ_COST = 87; // Send: 0. Res: 512. Read the data of state vectors from FPGA.
parameter OP_READ_CSOT_COMPLEX = 88;  // Send: 0. Res: 1024. Read the data of state vectors from FPGA. Current status will be stored in rS64.
parameter OP_SEND_SC_BETA = 88;  // Send: 16. Res: 0. sine cosine beta. Number of operation you call this function before OP_RUN will determine the 
parameter OP_SEND_COSSINEBETA = 89;  // Send: 16. Res: 0. Send a pair of cosine and sine beta. Number of paris you sent determine the number of steps when OP_RUN_CONTINUOUS sent.
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

always @(posedge CLK) begin
	if (RST) begin
		CP <= 4123;
		state <= 0;
		rdreq <= 0;
		ope_state <= 0;
		tx_fifo_write <= 0;
		tx_fifo_data <= 0;
            rA <= 0;
            rB <= 0;
            c_wait <= 0;
	end
      else begin
		
            // so at the next clock, the data will be read on the input of tx of UART.
            if(c_wait != 0)begin
                  c_wait <= c_wait -1;
            end
            else if(rdreq) begin
                  if(~empty) begin
                        rdreq <= 0;
                  end
            end
            else if(state == s_IDLE) begin
                  rdreq <= 1; // get operation anyway. 
                  state <= s_Fetch;
            end
            else if(state == s_Fetch) begin
                  ope_state <= rfifo_data; // fetch operation. 
                  state <= s_Operation;
            end
            else if(state == s_FetchData) begin
                  rT <= rfifo_data;
                  state <= s_Operand;
            end
            else if(state == s_Operation) begin
                  if(ope_state == OP_MOV_rA) begin
                        rdreq <= 1;
                        state <= s_FetchData;
                  end
                  else if(ope_state == OP_MOV_rA64) begin
                        rdreq <= 1;
                        state <= s_FetchData;
                        r8Pos <= 0;
                  end
                  else if(ope_state == OP_MOV_rA64rB) begin
                        rB64 <= rA64;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MOV_rB) begin
                        rdreq <= 1;
                        state <= s_FetchData;
                  end
                  else if(ope_state == OP_INC_rA) begin
                        rA <= rA + 1;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_INC_rB) begin
                        rB <= rB + 1;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_ADD_rB2rA) begin
                        rA <= rA + rB;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MUL_rB2rA) begin
                        c_wait <= 13;
                        state <= s_Operand;
                  end
                  else if(ope_state == OP_ADD_rBrA64FP) begin
                        c_wait <= 27;
                        state <= s_Operand;
                  end
                  else if(ope_state == OP_MUL_rBrA64FP) begin
                        c_wait <= 23;
                        state <= s_Operand;
                  end
                  else if(ope_state == OP_READ_rA) begin
                        tx_fifo_data <= rA;
                        tx_fifo_write <= 1;
                        state <= s_TXData;
                  end
                  else if(ope_state == OP_READ_rA64) begin
                        r8Pos <= 0;
                        state <= s_TXData;
                  end
                  else if(ope_state == OP_READ_rB) begin
                        tx_fifo_data <= rB;
                        tx_fifo_write <= 1;
                        state <= s_TXData;
                  end
            end
            else if(state == s_Operand) begin
                  if(ope_state == OP_MOV_rA) begin
                        rA <= rT;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MOV_rA64) begin
                        rA64[r8Pos*8+:8] <= rT;
                        if(r8Pos != 7) begin
                              r8Pos <= r8Pos + 1;
                              rdreq <= 1;
                              state <= s_FetchData;
                        end
                        else begin 
                              state <= s_IDLE;
                        end 
                  end
                  else if(ope_state == OP_ADD_rBrA64FP) begin
                        rA64 <= res_addFP64;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MUL_rBrA64FP) begin
                        rA64 <= res_mulFP64;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MUL_rB2rA) begin
                        rA <= res_mul8;
                        state <= s_IDLE;
                  end
                  else if(ope_state == OP_MOV_rB) begin
                        rB <= rT;
                        state <= s_IDLE;
                  end
            end 
            else if(state == s_TXData) begin
                  if(tx_full) begin
                        // nothing, just wait.
                  end
                  else if(ope_state == OP_READ_rA64) begin
                        if(r8Pos != 8) begin
                              tx_fifo_data <= rA64[r8Pos*8+:8];
                              tx_fifo_write <= 1;
                              r8Pos <= r8Pos + 1;
                        end
                        else begin
                              state <= s_IDLE;
                              tx_fifo_write <= 0;
                        end
                  end
                  else begin
                        tx_fifo_write <= 0;
                        state <= s_IDLE;
                  end
            end
      end
end
endmodule

