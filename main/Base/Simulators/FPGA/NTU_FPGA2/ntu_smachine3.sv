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
   output [31:0]  o_Status // counter to wait read FIFO latency. 
);

wire [7:0] tx_data_in;
wire tx_active;
wire rx_dv;

reg [31:0] CP; // program counter

reg [7:0] state;
reg [7:0] state_after;
reg [7:0] rA;
reg [7:0] rB;
reg [63:0] rA64;
reg [63:0] rB64;
reg [7:0] rT;
reg [63:0] rT64;
reg [63:0] sT64;
reg [7:0] ope_state;
wire [7:0] rfifo_data;
wire [7:0] o_Rx_Byte;
wire rdreq;
wire empty;
wire full;
wire tx_empty;
wire tx_full;
logic tx_fifo_write;
logic [7:0] tx_fifo_data;
reg [1:0] tx_dv;
reg [10:0] c_wait; // remaining count to wait.
reg [3:0] r8Pos; // position to store the byte 
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
wire [15:0] res_mul8i16;
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
	.result ( res_mul8i16 )
);

addfix8 addix8_inst (
	.clock ( CLK ),
	.dataa ( rA ),
	.datab ( rB ),
	.result ( res_add8 )
);

parameter [7:0] s_IDLE = 0;
parameter s_TXData = 208;
parameter s_Fetch = 209;
parameter s_FetchData = 210;
parameter s_WAIT_COMP = 211;
parameter s_WAIT_READ = 212;
parameter s_WAIT_READ8 = 213;
parameter s_Operation = 214;
parameter s_Operand = 215;
parameter OP_MOV_rA = 1;
parameter OP_MOV_rB = 2;
parameter OP_ADD_rB2rA = 3;
parameter OP_MUL_rB2rA = 4;
parameter OP_INC_rA = 5;
parameter OP_INC_rB = 6;
parameter OP_READ_rA = 31;
parameter OP_READ_rB = 32;
parameter OP_MOV_rA64 = 51;
parameter OP_MOV_rA64rB = 52;
parameter OP_READ_rA64 = 55;
parameter OP_ADD_rBrA64FP = 61;
parameter OP_MUL_rBrA64FP = 62;

assign rdreq = (state == s_WAIT_READ) | (state == s_WAIT_READ8);

reg [3:0] storeReg;
logic [3:0] n_storeReg;
reg [3:0] tx8Pos; // position to store the byte 
logic [3:0] n_tx8Pos; // position to store the byte 
reg [3:0] storeMaxPos;
parameter STORE_IDLE = 0;
parameter STORE_sTLEN = 1;
always_comb begin : STORE_COMb
      
      case(storeReg)
      STORE_sTLEN: begin
            tx_fifo_data = sT64[tx8Pos*8+:8];
            if(!tx_full) begin
                  if(tx8Pos == storeMaxPos) begin
                        n_storeReg = STORE_IDLE;
                        n_tx8Pos = 0;
                  end
                  else begin
                        n_storeReg = STORE_sTLEN;
                        n_tx8Pos = tx8Pos + 1;
                  end
            end else begin 
                  n_storeReg = STORE_sTLEN;
                  n_tx8Pos = tx8Pos;
            end
            tx_fifo_write = 1;
      end
      default: begin
            n_tx8Pos = 0;
            n_storeReg = STORE_IDLE;
            tx_fifo_write = 0;
            tx_fifo_data = 0;
      end
      endcase

end
always @(posedge CLK) begin // flip flop data valid to state machine block
      if(RSTlv1B)
      begin
            rf_dv <= 0;
      end
      else begin
            rf_dv[0] <= rdreq & ~empty;
            rf_dv[1] <= rf_dv[0];
      end
end
always @(posedge CLK) begin
	if (RSTlv1A) begin
		CP <= 4123;
		state <= 0;
		ope_state <= 0;
            r8Pos <= 0;
            rA <= 0;
            rB <= 0;
            c_wait <= 0;
            state_after <= 0;
            tx8Pos <= 0;
            storeMaxPos <= 0;
            storeReg <= 0;
	end
      else begin
            tx8Pos <= n_tx8Pos;
            storeReg <= n_storeReg;
            case(state)
                  s_IDLE: begin
                        state <= s_WAIT_READ; // get operation anyway.
                        state_after <= s_Operation;
                  end
                  s_WAIT_READ: begin
                        if(~empty) begin
                              state <= s_Fetch; // nead to wait fifo pipeline.
                        end
                  end
                  s_WAIT_READ8: begin
                        if(r8Pos != 8) begin
                              rT64[r8Pos*8+:8] <= rfifo_data;
                              r8Pos <= r8Pos + rf_dv[0];
                        end
                        else begin
                              r8Pos <= 0;
                              state <= state_after;
                        end 
                  end
                  s_WAIT_COMP: begin // waiting computing pipeline
                        if(c_wait != 0)begin
                              c_wait <= c_wait -1;
                        end
                        else begin
                              state <= state_after;
                        end
                  end
                  s_Fetch: begin
                        case(state_after)
                        s_Operation: begin
                              ope_state <= rfifo_data;
                        end
                        s_Operand: begin
                              rT <= rfifo_data;
                        end
                        endcase
                        state <= state_after;
                  end
                  s_Operation: begin
                        case (ope_state)
                        OP_MOV_rA: begin
                              state <= s_WAIT_READ;
                              state_after <= s_Operand;
                        end
                        OP_MOV_rA64: begin
                              state <= s_WAIT_READ8;
                              state_after <= s_Operand;
                        end
                        OP_MOV_rA64rB: begin
                              rB64 <= rA64;
                              state <= s_IDLE;
                        end
                        OP_MOV_rB: begin
                              state <= s_WAIT_READ;
                              state_after <= s_Operand;
                        end
                        OP_INC_rA: begin
                              rA <= rA + 1;
                              state <= s_IDLE;
                        end
                        OP_INC_rB: begin
                              rB <= rB + 1;
                              state <= s_IDLE;
                        end
                        OP_ADD_rB2rA: begin
                              rA <= rA + rB;
                              state <= s_IDLE;
                        end
                        OP_MUL_rB2rA: begin
                              c_wait <= 8;
                              state <= s_WAIT_COMP;
                              state_after <= s_Operand;
                        end
                        OP_ADD_rBrA64FP: begin
                              c_wait <= 27;
                              state <= s_WAIT_COMP;
                              state_after <= s_Operand;
                        end
                        OP_MUL_rBrA64FP: begin
                              c_wait <= 26;
                              state <= s_WAIT_COMP;
                              state_after <= s_Operand;
                        end
                        OP_READ_rA: begin
                              storeReg <= STORE_sTLEN;
                              storeMaxPos <= 0;
                              sT64[7:0] <= rA;
                              state <= s_IDLE;
                        end
                        OP_READ_rA64: begin
                              storeReg <= STORE_sTLEN;
                              storeMaxPos <= 7;
                              sT64 <= rA64;
                              state <= s_TXData;
                        end
                        OP_READ_rB: begin
                              storeReg <= STORE_sTLEN;
                              sT64[7:0] <= rB;
                              storeMaxPos <= 0;
                              state <= s_IDLE;
                        end
                        default: begin
                              state <= s_IDLE;
                        end
                        endcase
                  end
                  s_Operand: begin
                        case (ope_state)
                        OP_MOV_rA: begin
                              rA <= rT;
                              state <= s_IDLE;
                        end
                        OP_MOV_rA64: begin
                              rA64 <= rT64;
                              state <= s_IDLE;
                              
                        end
                        OP_ADD_rBrA64FP: begin
                              rA64 <= res_addFP64;
                              state <= s_IDLE;
                        end
                        OP_MUL_rBrA64FP: begin
                              rA64 <= res_mulFP64;
                              state <= s_IDLE;
                        end
                        OP_MUL_rB2rA: begin
                              rA <= res_mul8i16[7:0];
                              state <= s_IDLE;
                        end
                        OP_MOV_rB: begin
                              rB <= rT;
                              state <= s_IDLE;
                        end
                        default: begin
                              state <= s_IDLE;
                        end
                        endcase
                  end 
                  s_TXData: begin
                        if(storeReg == STORE_IDLE) begin
                              state <= s_IDLE;
                        end
                  end
            endcase
      end
end
endmodule

