// pipelined ALU test code for 2 floating points.
// 2025 0903, tested by alutest_tb.v
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
module gen_cost
  #(
  parameter P=64, // number of word width
  parameter Ni=32) // width of additional information
  (
   input  CLK,
   input  RST,
   input  [P-1:0]  gamma, // cos gamma
   input  [P-1:0]   H,
   output  [P-1:0]   Hr_o,
   output  [P-1:0]   Hi_o,
   input  [Ni-1:0]    info_in, // information, like addresses, enabled signal, and so on.
   output  [Ni-1:0]    info_out // information, like addresses, enabled signal, and so on.
);
parameter NPip = 58 + 1; // number of pipeline. Depends on IP core like addFPF64 used in this module.

reg [Ni-1:0] p_info [NPip-1:0];
wire [Ni-1:0] pp_ar_out;

logic [P-1:0] n_Hr_NPip;
logic [P-1:0] n_Hi_NPip;
reg [P-1:0] Hi_NPip;
reg [P-1:0] Hr_NPip;
reg [P-1:0] H_0Pip;
reg [P-1:0] gamma_0Pip;
assign Hr_o = Hr_NPip;
assign Hi_o = Hi_NPip;
assign info_out = p_info[NPip-1];
// cordic (
//       input  wire [55:0] a,      //      a.a
//       input  wire        areset, // areset.reset
//       output wire [54:0] c,      //      c.c
//       input  wire        clk,    //    clk.clk
//       output wire [54:0] s       //      s.s
// );
// fp2fix64 (
// 		input  wire        clk,    //    clk.clk
// 		input  wire        areset, // areset.reset
// 		input  wire [63:0] a,      //      a.a
// 		output wire [54:0] q       //      q.q
// 	);
// cordic (
// 		input  wire [55:0] a,      //      a.a
// 		input  wire        areset, // areset.reset
// 		output wire [54:0] c,      //      c.c
// 		input  wire        clk,    //    clk.clk
// 		output wire [54:0] s       //      s.s
// 	);
// Instantiate the module
mulFPF64 mulFPF_rc(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(H_0Pip),      //      a.a
      .b(gamma_0Pip),      //      b.b
      .q(n_Hr_NPip)       //      q.q
);
mulFPF64 mulFPF_is(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(H_0Pip),      //      a.a
      .b(gamma_0Pip),      //      b.b
      .q(n_Hi_NPip)       //      q.q
);

reg [31:0] CP; // program counter
integer i;
always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
            end
            gamma_0Pip <= '0;
            Hi_NPip <= '0;
            Hr_NPip <= '0;
            H_0Pip <= '0;
	end
      else begin
            gamma_0Pip <= gamma;
            H_0Pip <= H;
            Hi_NPip <= n_Hi_NPip;
            Hr_NPip <= n_Hr_NPip;
            p_info[0] <= info_in;
            for (i = 0; i <= NPip-2; i = i + 1) begin
                  p_info[i+1] <= p_info[i];
            end
      end
end
endmodule


