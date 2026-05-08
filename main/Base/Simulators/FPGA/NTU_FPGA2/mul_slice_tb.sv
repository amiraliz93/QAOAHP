
`timescale 1ns / 1ns

module mul_slice_tb ();

    parameter P = 64;
    parameter IA = 4;
    parameter IB = 4;
    parameter IOUT = 4;

    reg RST2;
    reg CLK2;

    reg signed [2*P-1:0] a_0;
    wire signed [P-1:0] q;

mul_slice MS
(
    .CLK(CLK2),
    .RST(RST2),
    .a(a_0),
    .q(q)
);

initial begin 
    a_0 = 128'h0000000000000000_0000000000000000; //
    CLK2 = 0;
    RST2 = 0;
    # 10
    RST2 = 1;
    # 50 
    RST2 = 0;
    a_0 = 128'h16c6b9b4f03aee00_1391cee3c778fa00; // 0.1 * 0.2 in Q4.60
    # 50
	 $stop;

end
always begin
 #1;
 CLK2 <= ~CLK2;

end
endmodule
