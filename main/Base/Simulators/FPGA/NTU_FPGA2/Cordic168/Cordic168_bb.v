
module Cordic168 (
	clk,
	areset,
	a,
	c,
	s);	

	input		clk;
	input		areset;
	input	[55:0]	a;
	output	[54:0]	c;
	output	[54:0]	s;
endmodule
