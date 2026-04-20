`timescale 1ns / 1ns

// todo
// test transmission
module test_mul_tb ();

parameter P=64;// number of word width

// Declare signals to connect to the UART module
reg RST2;
reg CLK2;

reg [P-1:0] b_0;
reg [P-1:0] a_0;
wire [P-1:0] q;


test_mul CI 
(
   .CLK(CLK2),        // Connect to your system clock wire
   .RST(RST2),        // Connect to your system reset wire
   .a(a_0),
   .b(b_0),
   .q(q)
);


initial begin 
     a_0 = 64'd0;
	  b_0 = 64'd0;
	  CLK2 = 0;
	  RST2 = 0;
	  # 100
	  RST2 = 1;
	  # 1000
	  
     a_0 = 64'd2;
	  b_0 = 64'd2;
	  # 2
     a_0 = 64'd3;
	  b_0 = 64'd4;
	  $stop;
	  
end

always
begin
      # 1;
      CLK2 <= ~CLK2; // clock generation, half period
end
endmodule
