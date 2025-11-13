// NTU_High Speed Communication Interface
// Only support 512 bits packet transmmission.
// Background is UART.

module ntu_HSCI(
  #(parameter CLOCK_REAL = 50000000 // 50 MHz
  )(
   input       CLK,
   input       RST,
   input       i_Tx_DV,
   input [511:0] i_Tx_Byte, 
   output      o_Tx_Active,
   output reg  o_Tx_Serial,
   output      o_Tx_Done
   );

receiver #(.CLKS_PER_BIT(CLOCK_REAL/115200)) uart1 
(
      .i_Clock       (clk),
      .RST       (RST),
      .i_Rx_Serial   (UART_RX),
	.o_Rx_DV   (rx_DV),
      .o_Rx_Byte       (o_Rx_Byte)     // Connect to the LED output wire
);
	
transmitter #(.CLKS_PER_BIT(CLOCK_REAL/115200) // 50 MHz
) t0(
      .i_Clock(clk),
      .RST(RST),
      .i_Tx_DV(tx_dv),
      .i_Tx_Byte(tx_data_in), 
      .o_Tx_Active(tx_active),
      .o_Tx_Serial(tx_out),
      .o_Tx_Done()
);
  
reg [7:0] data;
reg [3:0] t;
reg [31:0] cnt2;
reg [31:0] cnt3;
always @(posedge clk) begin
	if (RST) begin
		t <= 0;
		cnt2 <= 0;
		cnt3 <= 0;
		data <= 0;
	end
	else if(rx_DV) begin
		data <= o_Rx_Byte;
	end
end
assign LED[0] = ~data[0];
assign LED[1] = ~data[1];
assign LED[2] = ~data[2];
assign LED[3] = ~cnt2[25];
endmodule

