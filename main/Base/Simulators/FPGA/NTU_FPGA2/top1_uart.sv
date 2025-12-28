module top1_uart 
  #(parameter UART_CLKS_PER_BIT = 434)(
   input  CLK,
   input  RST,
   input   i_Rx_Serial,
   output  o_Tx_Serial,
   output [31:0]  o_Status // counter to wait read FIFO latency. 
);

// Declare signals to connect to the UART module
wire r_req;
wire rbram_vd;
wire w_req;
wire tx_OK;
wire tx_active;
wire tx_en;
assign tx_OK = ~tx_active;

wire rx_dv;
wire [7:0] rx_data;
wire [7:0] tx_data;
receiver #(.CLKS_PER_BIT(UART_CLKS_PER_BIT)) uart1 
(
      .i_Clock       (CLK),
      .RST       (RST),
      .i_Rx_Serial   (i_Rx_Serial),
	   .o_Rx_DV   (rx_dv),
      .o_Rx_Byte       (rx_data)    
);
transmitter #(.CLKS_PER_BIT(UART_CLKS_PER_BIT) // 50 MHz
) t0(
      .i_Clock(CLK),
      .RST(RST),
      .i_Tx_DV(tx_en),
      .i_Tx_Byte(tx_data), 
      .o_Tx_Active(tx_active),
      .o_Tx_Serial(o_Tx_Serial),
      .o_Tx_Done()
);
// we need transmitter and receiver to tset state machine (ntu_smachine)
top1 top 
(
   .CLK(CLK),        // Connect to your system clock wire
   .RST(RST),        // Connect to your system reset wire
   .tx_OK(tx_OK),
   .tx_en(tx_en),
   .tx_data_out(tx_data),
   .rx_data_in(rx_data),
   .rx_dv(rx_dv),
   .o_Status(o_Status)         // Connect to the wire for the received byte
);
endmodule
