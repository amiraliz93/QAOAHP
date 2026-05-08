
`timescale 1ns / 1ns

// todo
// test transmission
module test_mul_tb ();

	parameter P=64;// number of word width

	// Declare signals to connect to the UART module
	reg RST2;
	reg CLK2;

	reg signed  [P-1:0] b_0;
	reg signed  [P-1:0] a_0;
	wire [2*P-1:0] q;


Mul_64_FixedP CI 
(
   .CLK(CLK2),        // Connect to your system clock wire
   .RST(RST2),        // Connect to your system reset wire
   .a(a_0),
   .b(b_0),
   .q(q)
);

//data = {64'h16c6b9b4f03aee00, 64'h1391cee3c778fa00, 64'h020b57a2ccc02400, 64'hea96fe27a7f2f200, 64'hecd056a657be2800};

initial begin 
     a_0 = 64'h020b57a2ccc02400;
	  b_0 = 64'hea96fe27a7f2f200;
	  CLK2 = 0;
	  RST2 = 0;
	  # 10
	  RST2 = 1;
	  # 100
	  RST2 = 0;
	  # 4
     a_0 =  64'h16c6b9b4f03aee00;
	  b_0 = 64'h1391cee3c778fa00;

     # 100
	  $stop;
	  
end

always
begin
      # 1;
      CLK2 <= ~CLK2; // clock generation, half period
end
endmodule


