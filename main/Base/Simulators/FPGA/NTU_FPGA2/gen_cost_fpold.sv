`timescale 1ns / 1ns
// pipelined ALU test code for 2 floating points.
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.

// gamma should be in [-pi, pi]. 
module gen_cost
  #(
  parameter P=64 // number of word width
  )
  (
   input  CLK,
   input  RST,
   input  [P-1:0]  gamma, // cos gamma
   input  [P-1:0]   H,
   output  [P-1:0]   Hr_o,
   output  [P-1:0]   Hi_o
);
localparam N0 = 20; // latency mulFPF64
localparam N1 = 11; // latency FP64 to Fix 56.53(sign)
localparam N2 = 172; // latency of CORDIC. 56.53(sign) to 55.53(sign)
localparam N3 = 15; // latency Fix 55.53(sign) to FP64
localparam NPip = 1 + N0 + 1 + N1 + 1 + N2 + 1 + N3 + 1; // = 218, number of pipeline. Depends on IP core like addFPF64 used in this module.

wire [P-1:0] n0_out;
reg [63:0] n1_in;
wire [55:0] n1_out;
reg [55:0] n2_in;
wire [54:0] n2_sin_out;
wire [54:0] n2_cos_out;
reg [54:0] n3_sin_in;
reg [54:0] n3_cos_in;
wire [63:0] n3_sin_out;
wire [63:0] n3_cos_out;

reg [P-1:0] Hi_NPip;
reg [P-1:0] Hr_NPip;
reg [P-1:0] H_0Pip;
reg [P-1:0] gamma_0Pip;
assign Hr_o = Hr_NPip;
assign Hi_o = Hi_NPip;

mulFPF64 mulFPF_rc(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(H_0Pip),      //    a.a, FP64
      .b(gamma_0Pip),  //    b.b, FP64
      .q(n0_out)       //    q.q, FP64
);
FP2Fix53 inst_FP2Fix53(
		.clk(CLK),    // clk.clk input  wire        
		.areset(RST), // areset.reset input  wire        
		.a(n1_in),    // a.a input  wire [63:0] 
		.q(n1_out)    // q.q output wire [55:0] 
	);

cordicR inst_cordic(
		.a(n2_in),  // a.a input  wire [55:0] 
		.areset(RST),    // areset.reset input  wire        
		.c(n2_cos_out),         // c.c output wire [54:0] , cosine
		.clk(CLK),       // clk.clk   input  wire        
		.s(n2_sin_out)          // s.s     output wire [54:0] , sine
	);

Fix53toFP64 inst_Fix53toFP64(
		.clk(CLK),    // clk.clk input  wire       
		.areset(RST), // areset.reset input  wire        
		.a(n3_sin_in),    //      a.a   input  wire [54:0] 
		.q(n3_sin_out)    //      q.q output wire [63:0] 
	);
Fix53toFP64 inst2_Fix53toFP64(
		.clk(CLK),    // clk.clk input  wire       
		.areset(RST), // areset.reset input  wire        
		.a(n3_cos_in),    //      a.a   input  wire [54:0] 
		.q(n3_cos_out)    //      q.q output wire [63:0] 
	);

reg [31:0] CP; // program counter
integer i;
always @(posedge CLK) begin
      n3_cos_in <= n2_cos_out;
      n3_sin_in <= n2_sin_out;
      n1_in <= n0_out;
      n2_in <= n1_out;
      gamma_0Pip <= gamma;
      H_0Pip <= H;
      Hi_NPip <= n3_sin_out;
      Hr_NPip <= n3_cos_out;
end
endmodule

