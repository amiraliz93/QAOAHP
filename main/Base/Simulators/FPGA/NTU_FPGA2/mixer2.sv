// pipelined ALU test code for 2 floating points.
// 2025 0903, tested by alutest_tb.v
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
// Supply p_a, p_b altanatively clock by clock.
// p_info[m_Bit] == 1 means the input is of p_a.
module mixer2
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
parameter N1 = 21 + 1;
parameter N3 = 21 + 1 + 2 + 27 + 1;
parameter NPip = N3; // number of pipeline. Depends on IP core like addFPF64 used in this module.
parameter mBit = 30;
reg [Ni-1:0] p_info [NPip-1:0];
wire [P-1:0] n_prc_N1Pip;
wire [P-1:0] n_prs_N1Pip;
wire [P-1:0] n_pic_N1Pip;
wire [P-1:0] n_pis_N1Pip;
reg [P-1:0] prc_N1Pip [3];
reg [P-1:0] prs_N1Pip [3];
reg [P-1:0] pic_N1Pip [3];
reg [P-1:0] pis_N1Pip [3];
reg [P-1:0] pr_N3Pip;
reg [P-1:0] pi_N3Pip;
reg [P-1:0] pr_0Pip;
reg [P-1:0] pi_0Pip;
reg [P-1:0] cb_0Pip;
reg [P-1:0] sb_0Pip;
reg [P-1:0] add1_a;
reg [P-1:0] add1_b;
reg [P-1:0] add2_a;
reg [P-1:0] add2_b;
wire [P-1:0] add1_res;
wire [P-1:0] add2_res;
assign p_ar_o = pr_N3Pip;
assign p_ai_o = pi_N3Pip;
assign info_out = p_info[NPip-1];

// Instantiate the module
mulFPF64 mul1(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pr_0Pip),      //      a.a
      .b(cb_0Pip),      //      b.b
      .q(n_prc_N1Pip)       //      q.q
);
mulFPF64 mul2(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pr_0Pip),      //      a.a
      .b(sb_0Pip),      //      b.b
      .q(n_prs_N1Pip)       //      q.q
);
mulFPF64 mul3(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pi_0Pip),      //      a.a
      .b(cb_0Pip),      //      b.b
      .q(n_pic_N1Pip)       //      q.q
);
mulFPF64 mul4(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(pi_0Pip),      //      a.a
      .b(sb_0Pip),      //      b.b
      .q(n_pis_N1Pip)       //      q.q
);


// need to wait one clock.
addFPF64 add1(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(add1_a),      //      a.a
      .b(add1_b),      //      b.b
      .q(add1_res)       //      q.q
);
addFPF64 add2(
      .clk(CLK),    //    clk.clk
      .areset(RST), // areset.reset
      .a(add2_a),      //      a.a
      .b(add2_b),      //      b.b
      .q(add2_res)       //      q.q
);

// p'_a = cos p_a + i sin p_b
// p'_b = i sin p_a + cos p_b

// pp_ar = cosb *p_ar - sinb *p_bi
// pp_ai = cosb *p_ai + sinb *p_br
// pp_br = - sinb* p_ai + cosb *p_br
// pp_bi = sinb *p_ar + cosb *p_bi
reg [31:0] CP; // program counter
integer i;

localparam [63:0] SIGN_MASK = 64'h8000_0000_0000_0000;
always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
            end
            cb_0Pip <= 1; // 0.5000000000000001
            sb_0Pip <= 0; // 0.8660254037844386
            for (i = 0; i < 3; i = i + 1) begin
                  prc_N1Pip[i] <= '0;
                  prs_N1Pip[i] <= '0;
                  pic_N1Pip[i] <= '0;
                  pis_N1Pip[i] <= '0;
            end
	end
      else begin
            cb_0Pip <= cb;
            sb_0Pip <= sb;
            pr_0Pip <= p_ar;
            pi_0Pip <= p_ai;
            p_info[0] <= info_in;
            prc_N1Pip[0] <= n_prc_N1Pip;
            prs_N1Pip[0] <= n_prs_N1Pip;
            pic_N1Pip[0] <= n_pic_N1Pip;
            pis_N1Pip[0] <= n_pis_N1Pip;
            for (i = 0; i < 2; i = i + 1) begin
                  prc_N1Pip[i+1] <= prc_N1Pip[i];
                  prs_N1Pip[i+1] <= prs_N1Pip[i];
                  pic_N1Pip[i+1] <= pic_N1Pip[i];
                  pis_N1Pip[i+1] <= pis_N1Pip[i];
            end
            for (i = 0; i <= NPip-2; i = i + 1) begin
                  p_info[i+1] <= p_info[i];
            end
            if(p_info[N1][mBit]) begin
                  // it is assumed in this case, N1Pip[1] has p_a, and N1Pip[0] has p_b. Generate p'_a.
                  // adder 1 will performe, pp_ar = cosb * p_ar - sinb *p_bi
                  // adder 2 will performe, pp_ai = cosb * p_ai + sinb *p_br
                  add1_a <= prc_N1Pip[1];
                  add1_b <= pis_N1Pip[0] ^ SIGN_MASK;
                  add2_a <= pic_N1Pip[1];
                  add2_b <= prs_N1Pip[0];
            end
            else begin
                   // it is assumed in this case, N1Pip[2] has p_a, and N1Pip[1] has p_b. Generate p'_b.
                  // adder 1 will performe, pp_br = - sinb* p_ai + cosb *p_br
                  // adder 2 will performe, pp_bi = sinb *p_ar + cosb *p_bi
                  add1_a <= prc_N1Pip[1];
                  add1_b <= pis_N1Pip[2] ^ SIGN_MASK;
                  add2_a <= pic_N1Pip[1];
                  add2_b <= prs_N1Pip[2];
            end
            pr_N3Pip <= add1_res;
            pi_N3Pip <= add2_res;
      end
end
endmodule

