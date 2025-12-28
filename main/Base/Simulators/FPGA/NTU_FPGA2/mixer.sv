// pipelined ALU test code for 2 floating points.
// 2025 0903, tested by alutest_tb.v
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
module mixer_test
  #(
  parameter P=64, // number of word width
  parameter Ni=32) // width of additional information
  (
   input       CLK,
   input       RST,
   input  [P-1:0]  cb, // cos beta
   input  [P-1:0]  sb, // cos beta
   input  [P-1:0]   p_ar,
   input  [P-1:0]   p_ai,
   output  [P-1:0]   p_ar_o,
   output  [P-1:0]   p_ai_o,
   input  [Ni-1:0]    info_in, // information, like addresses, enabled signal, and so on.
   output  [Ni-1:0]    info_out // information, like addresses, enabled signal, and so on.
);
parameter NPip = 21 + 1; // number of pipeline. Depends on IP core like addFPF64 used in this module.

reg [Ni-1:0] p_info [NPip-1:0];
wire [Ni-1:0] pp_ar_out;

logic [P-1:0] n_pr_NPip;
logic [P-1:0] n_pi_NPip;
reg [P-1:0] pi_NPip;
reg [P-1:0] pr_NPip;
reg [P-1:0] pr_0Pip;
reg [P-1:0] pi_0Pip;
reg [P-1:0] cb_0Pip;
reg [P-1:0] sb_0Pip;
assign p_ar_o = pr_NPip;
assign p_ai_o = pi_NPip;
assign info_out = p_info[NPip-1];

// Instantiate the module
mulFPF64 mulFPF_rc(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pr_0Pip),      //      a.a
      .b(cb_0Pip),      //      b.b
      .q(n_pr_NPip)       //      q.q
);
mulFPF64 mulFPF_is(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pi_0Pip),      //      a.a
      .b(sb_0Pip),      //      b.b
      .q(n_pi_NPip)       //      q.q
);

reg [31:0] CP; // program counter
integer i;
always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
            end
            pi_NPip <= '0;
            pr_NPip <= '0;
            pr_0Pip <= '0;
            pi_0Pip <= '0;
            cb_0Pip <= 1;
	end
      else begin
            cb_0Pip <= cb;
            sb_0Pip <= sb;
            pi_NPip <= n_pi_NPip;
            pr_NPip <= n_pr_NPip;
            pr_0Pip <= p_ar;
            pi_0Pip <= p_ai;
            p_info[0] <= info_in;
            for (i = 0; i <= NPip-2; i = i + 1) begin
                  p_info[i+1] <= p_info[i];
            end
      end
end
endmodule


