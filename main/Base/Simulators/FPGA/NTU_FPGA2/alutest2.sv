// pipelined ALU test code for 2 floating points.
// throuput is 1/2.
// we need to decrease throuput as low as possible. It can be decreased to 1/8, at least.
module alu_test
  #(
  parameter P=64, // number of significants
  parameter Ni=64) // width of additional information
  (
   input       CLK,
   input       RST,
   input  [P-1:0]  cb, // cos beta
   input  [P-1:0]  sb, // sine beta
   input  [P-1:0]     p_ar, // psi_a real part
   input  [P-1:0]     p_ai, // psi_a imaginary part
   input  [P-1:0]     p_br, // psi_b real part
   input  [P-1:0]     p_bi, // psi_b imaginary part
   output  [P-1:0]     p_ar_o,
   output  [P-1:0]     p_ai_o,
   output  [P-1:0]     p_br_o,
   output  [P-1:0]     p_bi_o,
   input  data_valid; // if busy = 0 and data_valid = 1, data will be incorpolated in a pipeline.
   input  busy;     // if busy = 0 and data_valid = 1, data will be incorpolated in a pipeline.
   input  [Ni-1:0]    info_in, // information, like addresses, enabled signal, and so on.
   output  [Ni-1:0]    info_out // information, like addresses, enabled signal, and so on.
);
parameter NPip = 11; // number of pipeline.

assign busy = (lst != 0);
wire             nan_out;
wire             overflow_out;
wire    [63:0]   result_out;
wire             underflow_out;
wire             zero_out;
reg              main_clock;
reg     [63:0]   data_a;
reg     [63:0]   data_b;

reg     [63:0]  rp_ar;
reg     [63:0]  rp_ai;
reg     [63:0]  rp_bi;
reg     [63:0]  rp_br;
reg     [63:0]  rcb; // registered cosine beta
reg     [63:0]  rsb; // registered sign beta
reg     [63:0]  rnsb; // registerd negative sine beta
reg [2:0] lst; // loading state

reg [Ni-1:0] p_info [NPip-1:0];
reg [Ni-1:0] p_state [NPip-1:0];
wire [Ni-1:0] pp_ar_out;

assign info_out = p_info[NPip-1];
parameter 

always_comb begin: pipeline_state
    case(1st)
    0: begin
        fp1_a = cb; // floating point unit 1, input a.
        fp2_a = cb;
        fp3_a = cb;
        fp4_a = cb;

        fp1_b = p_ar;
        fp2_b = p_ai;
        fp3_b = p_br;
        fp4_b = p_bi;
        p_state[0] = {0, };
    end
    1: begin
        fp1_a = rnsb; // floating point unit 1, input a.
        fp2_a = rsb;
        fp3_a = rnsb;
        fp4_a = rsb;

        fp1_b = rp_ar;
        fp2_b = rp_ai;
        fp3_b = rp_br;
        fp4_b = rp_bi;
        p_state[0] = {1, };
    end
    endcase
end 

wire p_fp1_o;
wire p_fp2_o;
wire p_fp3_o;
wire p_fp4_o;


addFPF64 addFPF64(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA2),      //      a.a
      .b(rB2),      //      b.b
      .q(res_addFP64)       //      q.q
);

mulFPF64 mf64i(
      .clk(CLK),    //    clk.clk
      .areset(RSTlv1B), // areset.reset
      .a(rA2),      //      a.a
      .b(rB2),      //      b.b
      .q(res_mulFP64)       //      q.q
);
// Instantiate the module
mulFPF64 fp64_1 (
    .clock        (CLK),     // input
    .dataa        (fp1_a),         // input [63:0]
    .datab        (fp1_b),         // input [63:0]
    .nan          (),        // output
    .overflow     (),   // output
    .result       (p_fp1_o),     // output [63:0]
    .underflow    (),  // output
    .zero         ()        // output
);

mulFPF64 fp64_2 (
    .clock        (CLK),     // input
    .dataa        (fp2_a),         // input [63:0]
    .datab        (fp2_b),         // input [63:0]
    .nan          (),        // output
    .overflow     (),   // output
    .result       (p_fp2_o),     // output [63:0]
    .underflow    (),  // output
    .zero         ()        // output
);

// Instantiate the module
mulFPF64 fp64_3 (
    .clock        (CLK),     // input
    .dataa        (fp3_a),         // input [63:0]
    .datab        (fp3_b),         // input [63:0]
    .nan          (),        // output
    .overflow     (),   // output
    .result       (p_fp3_o),     // output [63:0]
    .underflow    (),  // output
    .zero         ()        // output
);

mulFPF64 fp64_4 (
    .clock        (CLK),     // input
    .dataa        (fp4_a),         // input [63:0]
    .datab        (fp4_b),         // input [63:0]
    .nan          (),        // output
    .overflow     (),   // output
    .result       (p_fp4_o),     // output [63:0]
    .underflow    (),  // output
    .zero         ()        // output
);

reg rp_fp1_o;
reg rp_fp2_o;
reg rp_fp3_o;
reg rp_fp4_o;
reg [31:0] CP; // program counter
integer i;
always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
                  p_state[i] <= 0;
            end
	end
      else begin
            if(data_valid & ~busy) begin
                rnsb <= {sb[63], sb[62:0]};
                rcb <= cb;
                sb <= sb;
                rp_ar <= p_ar;
                rp_ai <= p_ai;
                rp_ar <= p_br;
                rp_bi <= p_bi;
                lst <= 0;
            end
            else begin 
                lst <= lst + 1;
            end
            if(p_state[12] == 0) begin  // depend on the latency of the FPU. Check in simulator for exact demand.
                rp_fp1_o <= p_fp1_o; // need buffering
                rp_fp2_o <= p_fp2_o;
                rp_fp3_o <= p_fp3_o;
                rp_fp4_o <= p_fp4_o;
            end
            p_info[0] <= info_in;
            p_state[0] <= state;
            for (i = 0; i <= NPip-2; i = i + 1) begin
                  p_info[i+1] <= p_info[i];
                  p_state[i+1] <= p_state[i];
            end
      end
end
endmodule

