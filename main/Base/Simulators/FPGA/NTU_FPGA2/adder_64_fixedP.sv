
// Signed fixed-point 64-bit adder.
// Inputs and output share the same Q-format (default Q3.61).
// Pipelined: 1-cycle latency from a/b to q.
//
// Amir Alizadeh, Hiroki Shibata FPGA QAOA project, replacing addFPF64 in Update_mixer.sv.

module adder_64_fixedP #(
    parameter P = 64
)
(
    input CLK,
    input signed [P-1:0] a, 
    input signed [P-1:0] b, 
    output reg signed [P-1:0] q
);

reg signed [P-1:0] sum_r;

assign q = sum_r; // output is registered sum_r

always @(posedge CLK)   begin
        sum_r <= a + b; // // signed add, two's-complement wrap on overflow
end
    
endmodule
