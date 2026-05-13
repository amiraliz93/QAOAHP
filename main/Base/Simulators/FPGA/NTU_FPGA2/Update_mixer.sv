
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
    input [P-1:0] p_r, // we get two time for a and b amplitute
    input [P-1:0] p_i, // we get two time for a and b amplitute
    input [1:0] switch_in, // operation switch. 2'b01 is mixer for p_a, 2'b10 is mixer for p_b, else for cost function. p_a and p_b must be supplied sequencially and altanatively.
    input [Ni -1:0] info_in,


    output [P-1:0] p_r_o,
    output [P-1:0] p_i_o,
    output [Ni-1:0] info_out
);
// latency of multiplier itself is 8 cycle - muls-slice is 1 cycle and adder 1 cycle

//adder 1 cycle
parameter N1 = 1 + 11 ; // data arrive at prc_N1Pip[1]
parameter N3 = 1 + 11 + 1 + 2; // suppose  add1_a visible at cycle 13 and add1_res on cycle 14


reg [Ni-1:0] p_info [N3-1:0];
reg [Ni-1:0] p_switch [N3-1:0];
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

assign p_r_o = pr_N3Pip; // connect output to the last pipeline register
assign p_i_o = pi_N3Pip; // 
assign info_out = p_info[N3-1]; // connect output info to the last pipeline register

// Instantiate the module
// input of the mul is pr_0Pip & pi_0Pip = format is signed Q3.61 (1 bit sign, 2 bit int, 61 frac) [-4, 4)
// 


Mul_64_FixedP mul1(
    .CLK(CLK),    //    clk.clk

    .a(pr_0Pip),      //      a.a format is signed Q2.61 (1 bit sign, 2 bit int, 61 frac)
    .b(cos_beta_0Pip),      //      b.b format is Qm.n
    .q(mul1_raw)   // n_prc_N1Pip    //      q.q
);
mul_slice #(.P(P), .IA(3), .IB(3), .IOUT(3)
) slicer1 (
      .CLK(CLK),
      .RST(RST),
      .a(mul1_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_prc_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);

Mul_64_FixedP mul2(
    .CLK(CLK),    //    clk.clk
    .a(pr_0Pip),      //      a.a Q3.61
    .b(sin_beta_0Pip),      //      b.b format is Qm.n
    .q(mul2_raw)       //      q.q
);
mul_slice #(.P(P), .IA(3), .IB(3), .IOUT(3)
) slicer2 (
      .CLK(CLK),
      .RST(RST),
      .a(mul2_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_prs_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);

Mul_64_FixedP mul3(
    .CLK(CLK),    //    clk.clk
    .a(pi_0Pip),      //      Q3.61
    .b(cos_beta_0Pip),      //      b.b
    .q(mul3_raw)       //      q.q
);
mul_slice #(.P(P), .IA(3), .IB(3), .IOUT(3)
) slicer3 (
      .CLK(CLK),
      .RST(RST),
      .a(mul3_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_pic_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);
Mul_64_FixedP mul4( // sinB * p_i  (second input)(p_bi)
    .CLK(CLK),    //    clk.clk
    .a(pi_0Pip),      //      Q3.61
    .b(sin_beta_0Pip),      //      b.b
    .q(mul4_raw)       //      q.q
);
mul_slice #(.P(P), .IA(3), .IB(3), .IOUT(3)
) slicer4 (
      .CLK(CLK),
      .RST(RST),
      .a(mul4_raw), // output of multiplier, format is Q(IA+IB).(FRAC_A + FRAC_B)
      .q(n_pis_N1Pip) // output of slicer, format is Q(IOUT).(FRAC_OUT)
);


adder_64_fixedP add1(
    .CLK(CLK),    //    clk.clk
    .a(add1_a),      //      a.a
    .b(add1_b),      //      b.b
    .q(add1_res)       //      q.q
);
adder_64_fixedP add2(

    .CLK(CLK),    //    clk.clk
    .a(add2_a),      //      a.a
    .b(add2_b),      //      b.b
    .q(add2_res)       //      q.q
);

// update amplitute (a) -> p'_a = cos_beta*p_a + i(sin_beta*p_b) 
// update amplitute (b) -> p'_b = i (sin_beta*p_a) + cos_beta*p_b
// by considering p_a = (p_r + i p_i) & p_b = (p_br + i p_bi)
// -->  p'_a =  cos_beta*p_r + i(cos_beta*p_i) + i(sin_beta*p_br) - sin_beta*p_bi
// --> p'_b = - sin_beta*p_i + cos_beta*p_br + i(sin_beta*p_r) + i(cos_beta*p_bi)
// pp_r = cos_beta*p_r - sin_beta*p_bi
// pp_i = cosb *p_i + sinb *
// pp_br = - sinb* p_i + cosb *p_br
// pp_bi = sinb *p_r + cosb *p_bi
integer i;
localparam [63:0] SIGN_MASK = 64'h8000_0000_0000_0000;

always @(posedge CLK) begin

            cos_beta_0Pip <= cos_beta;
            sin_beta_0Pip <= sin_beta;
            pr_0Pip <= p_r;
            pi_0Pip <= p_i;
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
            for (i = 0; i <= N3-2; i = i + 1) begin
                  p_info[i+1]   <= p_info[i];
                  p_switch[i+1] <= p_switch[i];
            end

            case(p_switch[N1])
                  2'b01: begin
                        // it is assumed in this case, N1Pip[1] has p_a, and N1Pip[0] has p_b. Generate p'_a.
                        // adder 1 will performe, pp_r = cosb * p_r - sinb * p_bi
                        // adder 2 will performe, pp_i = cosb * p_i + sinb * p_br

 // 0 means input second  time for amplitute b and 1 measn input first for amplitute a - 
 // prc = real part of amplitute (p_ar) * and cosb
 // prs = real part of amplitute (p_ar) * and sinb
 // pic = imag part of amplitute (p_ai) * and cosb    
// pis = imag part of amplitute (p_ai) * and sinb
                        add1_a <= prc_N1Pip[1]; // result mul1, which is cosb * p_r
                        add1_b <= pis_N1Pip[0]; 
                        add2_a <= pic_N1Pip[1];
                        add2_b <= -prs_N1Pip[0];
                  end
                  2'b10: begin
                        // it is assumed in this case, N1Pip[2] has p_a, and N1Pip[1] has p_b. Generate p'_b.
                        // adder 1 will performe, pp_br = - sinb * p_i + cosb * p_br
                        // adder 2 will performe, pp_bi = sinb * p_r + cosb * p_bi
                        add1_a <= prc_N1Pip[1];
                        add1_b <= pis_N1Pip[2]; 
                        add2_a <= pic_N1Pip[1];
                        add2_b <= -prs_N1Pip[2];
                  end
                  default: begin
                        // this is for cost function operator
                        // adder 1 will performe, pp_r = - sinb* p_i + cosb *p_r
                        // adder 2 will performe, pp_i = sinb *p_r + cosb *p_i
                        add1_a <= prc_N1Pip[1];
                        add1_b <= -pis_N1Pip[1]; //^ SIGN_MASK;
                        add2_a <= pic_N1Pip[1];
                        add2_b <= prs_N1Pip[1];
                  end
            endcase
            pr_N3Pip <= add1_res;
            pi_N3Pip <= add2_res;
      
end
endmodule