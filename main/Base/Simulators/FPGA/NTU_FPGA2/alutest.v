// pipelined ALU test code for 2 floating points.
// 2025 0903, tested by alutest_tb.v
// Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.
module alu_test
  #(
  parameter P=64, // number of significants
  parameter Ni=32) // width of additional information
  (
   input       CLK,
   input       RST,
   input  [P-1:0]  cb, // cos beta
   input  [P-1:0]     p_ar,
   output  [P-1:0]     p_ar_o,
   input  [Ni-1:0]    info_in, // information, like addresses, enabled signal, and so on.
   output  [Ni-1:0]    info_out // information, like addresses, enabled signal, and so on.
);
parameter NPip = 11; // number of pipeline.

wire             nan_out;
wire             overflow_out;
wire    [63:0]   result_out;
wire             underflow_out;
wire             zero_out;
reg              main_clock;
reg     [63:0]   data_a;
reg     [63:0]   data_b;

reg [Ni-1:0] p_info [NPip-1:0];
wire [Ni-1:0] pp_ar_out;

assign info_out = p_info[NPip-1];

// Instantiate the module
altmul64_altfp_mult_heq fp64_1 (
    .clock        (CLK),     // input
    .dataa        (cb),         // input [63:0]
    .datab        (p_ar),         // input [63:0]
    .nan          (),        // output
    .overflow     (),   // output
    .result       (p_ar_o),     // output [63:0]
    .underflow    (),  // output
    .zero         ()        // output
);


reg [31:0] CP; // program counter
integer i;
always @(posedge CLK) begin
	if (RST) begin
            for (i = 0; i < NPip; i = i + 1) begin
                  p_info[i] <= 0;
            end
	end
      else begin
            p_info[0] <= info_in;
            for (i = 0; i <= NPip-2; i = i + 1) begin
                  p_info[i+1] <= p_info[i];
            end
      end
end
endmodule

