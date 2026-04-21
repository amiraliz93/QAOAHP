// pipelined ALU test code for 2 floating points.
// 2025 0903, tested by alutest_tb.v
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
// Supply p_a, p_b altanatively clock by clock.
// p_info[m_Bit] == 1 means the input is of p_a.
module test_mul
  #(
  parameter P=64, // number of word width
  parameter Ni=32 // width of additional information. Biggest 3 bits are reserved.
  ) 
  (
   input       CLK,
   input       RST,
   input  [P-1:0]  a, 
   input  [P-1:0]  b, // all value here are wire
   output [P-1:0]   q
);
parameter N1 = 1 + 20 + 1;
parameter N3 = 1 + 20 + 1 + 2 + 27 + 1;
parameter NPip = N3;

reg [P-1:0] b_0;
reg [P-1:0] a_0;
logic [P-1:0] n_r0;
logic [P-1:0] n_r1;
logic [P-1:0] n_r2;
reg [P-1:0] r0;
reg [P-1:0] r1;
reg [P-1:0] r2;

assign q = r2;  // left value of assign must be wire - we must connect reg to wire

always_comb begin: mainCombBlock

// left value must be logic in this block

	n_r0 = b_0 * a_0;
	n_r1 =  r0;
	n_r2 =  r1;

end

always@(posedge CLK) begin

    //------------------------------------------------------------------------
    // we define timing
    //------------------------------------------------------------------------
    
    if(RST)begin
		b_0 <=  '0;
		a_0 <= '0;
		r0 <= '0;
		r1 <= '0;
		r2 <= '0;
		
    end
	 
	 else begin
	 b_0 <= b;
	 a_0 <= a;
	 r0 <= n_r0;
	 r1 <= n_r1;
	 r2 <= n_r2;
	 end
end

endmodule
