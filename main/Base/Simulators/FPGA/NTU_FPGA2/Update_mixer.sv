
//Amir ALizadeh & Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
// Supply p_a, p_b altanatively clock by clock.
// formatation: the input and output of multiplier is better have same format 
// all data is better to be in signed 64bit Q3.61 (1 bit sign, 3 bit int, 61 frac) - [-4, 4) range. This is for the consideration of the range of cost function and the angle of mixer. But it can be changed by adjusting the format of multiplier and slicer.
// so we have: Q3.61 × Q3.61 → Q6.122 → slice → Q3.61

module Update_mixer
#(
    parameter P = 64, // number of word width
    parameter Ni = 32 // width  info -Biggest 3 bits are reserved.
)
(
    input CLK,
    input RST,
    input [P-1:0] cos_beta,
    input [P-1:0] sin_beta,
    input [P-1:0] p_ar,
    input [P-1:0] p_ai,
    input [1:0] switch_in, // operation switch. 2'b01 is mixer for p_a, 2'b10 is mixer for p_b, else for cost function. p_a and p_b must be supplied sequencially and altanatively.
    input [Ni -1:0] info_in,


    output [P-1:0] p_ar_o,
    output [P-1:0] p_ai_o,
    output [Ni-1:0] info_out
);

parameter N1 = 1+ 20 + 1;
parameter N3 = 1 + 20 + 1 + 2 + 27 + 1;
parameter NPip = N3; // number of pipline

reg [Ni-1:0] p_info [NPip-1:0];
reg [Ni-1:0] p_switch [NPip-1:0];
reg [P-1:0] prc_N1Pip [3];
reg [P-1:0] prs_N1Pip [3];
reg [P-1:0] pic_N1Pip [3];
reg [P-1:0] pis_N1Pip [3];
reg [P-1:0] pr_N3Pip;
reg [P-1:0] pi_N3Pip;
reg [P-1:0] pr_0Pip;
reg [P-1:0] pi_0Pip;
reg [P-1:0] cos_beta_0Pip, sin_beta_0Pip;
reg [P-1:0] add1_a, add1_b, add2_a, add2_b;



wire [2*P-1:0] mul1_raw;
wire [2*P-1:0] mul2_raw;
wire [2*P-1:0] mul3_raw;
wire [2*P-1:0] mul4_raw;
wire [P-1:0] n_prc_N1Pip;
wire [P-1:0] n_prs_N1Pip;
wire [P-1:0] n_pic_N1Pip;
wire [P-1:0] n_pis_N1Pip;
wire [P-1:0] add1_res;
wire [P-1:0] add2_res;

assign p_ar_o = pr_N3Pip; // connect output to the last pipeline register
assign p_ai_o = pi_N3Pip; // 
assign info_out = p_info[NPip-1]; // connect output info to the last pipeline register

// Instantiate the module
// input of the mul is pr_0Pip & pi_0Pip = format is signed Q3.61 (1 bit sign, 2 bit int, 61 frac) [-4, 4)
// 


Mul_64_FixedP mul1(
    .CLK(CLK),    //    clk.clk
    .RST(RST), // areset.reset
    .a(pr_0Pip),      //      a.a format is signed Q2.61 (1 bit sign, 2 bit int, 61 frac)
    .b(cos_beta_0Pip),      //      b.b format is Qm.n
    .q(mul1_raw)   // n_prc_N1Pip    //      q.q
);
mul_slice slicer1 (
      .CLK(CLK),
      >RST(RST),
      .a(mul1_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_prc_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
)

Mul_64_FixedP mul2(
    .CLK(CLK),    //    clk.clk
    .RST(RST), // areset.reset
    .a(pr_0Pip),      //      a.a Q3.61
    .b(sin_beta_0Pip),      //      b.b format is Qm.n
    .q(mul2_raw)       //      q.q
);
mul_slice slicer2 (
      .CLK(CLK),
      .RST(RST),
      .a(mul2_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_prs_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);

Mul_64_FixedP mul3(
    .CLK(CLK),    //    clk.clk
    .RST(RST), // areset.reset
    .a(pi_0Pip),      //      Q3.61
    .b(cos_beta_0Pip),      //      b.b
    .q(mul3_raw)       //      q.q
);
mul_slice slicer3 (
      .CLK(CLK),
      .RST(RST),
      .a(mul3_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_pic_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);
Mul_64_FixedP mul4(
    .CLK(CLK),    //    clk.clk
    .RST(RST), // areset.reset
    .a(pi_0Pip),      //      Q3.61
    .b(sin_beta_0Pip),      //      b.b
    .q(mul4_raw)       //      q.q
);
mul_slice slicer4 (
      .CLK(CLK),
      >RST(RST),
      .a(mul4_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_pis_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
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

// update amplitute (a) -> p'_a = cos_beta*p_a + i(sin_beta*p_b) 
// update amplitute (b) -> p'_b = i (sin_beta*p_a) + cos_beta*p_b
// by considering p_a = (p_ar + i p_ai) & p_b = (p_br + i p_bi)
// -->  p'_a =  cos_beta*p_ar + i(cos_beta*p_ai) + i(sin_beta*p_br) - sin_beta*p_bi
// --> p'_b = - sin_beta*p_ai + cos_beta*p_br + i(sin_beta*p_ar) + i(cos_beta*p_bi)
// pp_ar = cos_beta*p_ar - sin_beta*p_bi
// pp_ai = cosb *p_ai + sinb *
// pp_br = - sinb* p_ai + cosb *p_br
// pp_bi = sinb *p_ar + cosb *p_bi
reg [31:0] CP; // program counter
integer i;
localparam [63:0] SIGN_MASK = 64'h8000_0000_0000_0000;

always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
                  p_switch[i] <= 0;
            end
            cos_beta_0Pip <= 1;
            sin_beta_0Pip <= 0; 
            for (i = 0; i < 3; i = i + 1) begin
                  prc_N1Pip[i] <= '0;
                  prs_N1Pip[i] <= '0;
                  pic_N1Pip[i] <= '0;
                  pis_N1Pip[i] <= '0;
            end
	end
      else begin
            cos_beta_0Pip <= cos_beta;
            sin_beta_0Pip <= sin_beta;
            pr_0Pip <= p_ar;
            pi_0Pip <= p_ai;
            p_info[0] <= info_in;
            p_switch[0] <= switch_in;
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
                  p_info[i+1]   <= p_info[i];
                  p_switch[i+1] <= p_switch[i];
            end

            case(p_switch[N1])
                  2'b01: begin
                        // it is assumed in this case, N1Pip[1] has p_a, and N1Pip[0] has p_b. Generate p'_a.
                        // adder 1 will performe, pp_ar = cosb * p_ar - sinb * p_bi
                        // adder 2 will performe, pp_ai = cosb * p_ai + sinb * p_br
                        add1_a <= prc_N1Pip[1];
                        add1_b <= -pis_N1Pip[0] // this is for float ^ SIGN_MASK;
                        add2_a <= pic_N1Pip[1];
                        add2_b <= prs_N1Pip[0];
                  end
                  2'b10: begin
                        // it is assumed in this case, N1Pip[2] has p_a, and N1Pip[1] has p_b. Generate p'_b.
                        // adder 1 will performe, pp_br = - sinb * p_ai + cosb * p_br
                        // adder 2 will performe, pp_bi = sinb * p_ar + cosb * p_bi
                        add1_a <= prc_N1Pip[1];
                        add1_b <= -pis_N1Pip[2] //^ SIGN_MASK;
                        add2_a <= pic_N1Pip[1];
                        add2_b <= prs_N1Pip[2];
                  end
                  default: begin
                        // this is for cost function operator
                        // adder 1 will performe, pp_ar = - sinb* p_ai + cosb *p_ar
                        // adder 2 will performe, pp_ai = sinb *p_ar + cosb *p_ai
                        add1_a <= prc_N1Pip[1];
                        add1_b <= -pis_N1Pip[1] //^ SIGN_MASK;
                        add2_a <= pic_N1Pip[1];
                        add2_b <= prs_N1Pip[1];
                  end
            endcase
            pr_N3Pip <= add1_res;
            pi_N3Pip <= add2_res;
      end
end
endmodule