`timescale 1ns / 1ns
// Amir Alizadeh - Hiroki Shibata, NTU and Tokyo Metropollitan University, created at Nottingham Trent University.

// gamma should be in [-pi, pi]. 
module gen_cost
  #(
  parameter P=64, // number of word width
  parameter Ni=32) // width of additional information
  (
   input  CLK, // all ports
   input  RST,
   input  [P-1:0]  gamma, // cos gamma
   input  [P-1:0]   H,
   output signed [P-1:0]  Hr_o, // real part of e^(i gamma H)
   output signed [P-1:0]  Hi_o // Img part of e^(i gamma H)
);

localparam N0 = 10; // latency new test_mul 10 cycles
localparam N1 = 170; // latency of CORDIC

reg signed [P-1:0] mul_in1;
reg signed [P-1:0] mul_in2;
reg signed [2*P-1:0] slicer_in; 
reg signed [P-1:0] cordic_in;


reg signed [P-1:0]  cordic_c_out;
reg signed [P-1:0]  cordic_s_out;

// output of the IPs
wire signed [P-2:0] cordic_cos_out;
wire signed [P-2:0] cordic_sin_out;
wire signed [2*P-1:0] mul_out; // output of multiplier = gamma*H
wire signed [P-1:0]  slicer_out; // slicer output to cordic input



Mul_64_FixedP new_mul(
      .CLK(CLK),
      .RST(RST),
      .a(mul_in1), // H input of multiplier
      .b(mul_in2),
      .q(mul_out)
);

mul_slice #(
	.P(P),
      .IA(3),
      .IB(3),
      .IOUT(3)
) slicer (
      .CLK(CLK),
      .RST(RST),
      .a(mul_out),   // output of multiplier
      .q(slicer_out)   // input of cordic
      );
		

CORDIC_64_fixedP inst_cordic(
      	.a(cordic_in),  //  input (it is H*gamma) wire width = [63:0] 
		.areset(RST),    // areset.reset        
		.c(cordic_cos_out),         // output is 63 bits - [62:0] , cos - 61 frac and 2 int
		.clk(CLK),       // clk.clk         
		.s(cordic_sin_out)          //  output is 63 bits [62:0] , sin 61 frac and 2 int
	);



always @(posedge CLK) begin
      
      integer i;
	if (RST) begin
            mul_in1 <= '0; // input of mutiplier
            mul_in2 <= '0; // second input of multiplier
            slicer_in <= '0; // input of slicer
            cordic_in <= '0; // input of cordic
				end
      else begin
            mul_in1 <= gamma;
            mul_in2 <= H;	
            end
            slicer_in <= mul_out; // input of slicer is the output 
		cordic_in <= slicer_out; // input of cordic is the output of multiplier after N0 cycles.
		cordic_c_out <= {{1{cordic_cos_out[P-2]}}, cordic_cos_out}; // sign extend to 64 bit
		cordic_s_out <= {{1{cordic_sin_out[P-2]}}, cordic_sin_out};

end

assign Hr_o = cordic_c_out; 
assign Hi_o = cordic_s_out; 
endmodule

