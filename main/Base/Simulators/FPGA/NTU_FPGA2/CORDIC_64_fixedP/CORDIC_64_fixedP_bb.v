
module CORDIC_64_fixedP (
	clk,
	areset,
	a,
	c,
	s);	

	input		clk;
	input		areset;
	input	[63:0]	a;
	output	[62:0]	c;
	output	[62:0]	s;
endmodule
