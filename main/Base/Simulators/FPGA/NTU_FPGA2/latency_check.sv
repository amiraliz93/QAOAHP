module latency_check #(
    parameter P = 64 // number of word width
)
(
    input CLK,
    input RST,
    input signed [P-1:0] a,
    input signed [P-1:0] b,
    output signed [2*P-1:0] q_mul,
    output signed [P-1:0] q_slicer,
    output signed [P-1:0] q_add
);

Mul_64_FixedP u_mul (
    .CLK(CLK), 

    .a(a), 
    .b(b),
    .q(q_mul)
);

mul_slice #(
    .P(P),
    .IA(3),
    .IB(3),
    .IOUT(3)
) u_slicer (
    .CLK(CLK),
    .RST(RST),
    .a(q_mul),
    .q(q_slicer)
);

adder_64_fixedP adder (
    .CLK(CLK),
    .a(q_slicer),
    .b(64'h0000_0000_0000_0001), // add a small value to see the change in output
    .q(q_add)
);

endmodule

