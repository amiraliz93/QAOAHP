
// this module wrap the output of mulipier IP, Based on the target usage can be Q4.60 , Q3,61, ...


module mul_slice #(
    parameter P = 64,
    parameter int IA = 3, // integer bits of a (e.g. 4 for Q4.60)
    parameter int IB = 3, // integer bits of b
    parameter int IOUT = 3 // int of output Q-format
)
(
    input CLK,
    input RST,
    input signed [2*P-1:0] a,  // raw 128 bit of mtest_mul
    output signed[P-1:0] q
);

// Q-format math:
// in mul IP if a and b is Q_a and Q_b == (Ia + FRAC_A =64bit) & (Ib + FRAC_B=64bit)
// prod is Q(IA+IB).(FRAC_A + FRAC_B) --  where FRAC_A = P - IA, FRAC_B = P - IB
// Out is Q(IOUT).(FRAC_OUT)
// FRAC_OUT = P - IOUT
// slice =  [(FRAC_A + FRAC_B) + Iout-1: (FRAC_A + FRAC_B) - FRAC_OUT] 
localparam int FRAC_OUT = P - IOUT;
localparam int FRAC_A = P - IA; // fractional A
localparam int FRAC_B = P - IB;
localparam int LB = ((FRAC_A + FRAC_B) - FRAC_OUT); // first lowerst bit of slice
localparam int HB = LB + P -1; // highest bit of slice

initial begin
    if (LB <0) 
    $error(" mul_slice: FRACT_OUT is (%0d) > ((FRAC_A + FRAC_B)) %0d");
    if (HB > 2*P-1)
    $error(" mul_slice: IOUT is (%0d) too large for input formats, HB %0d > 2*P-1", IOUT, HB);
end

//reg signed [2*P-1:0] a_in; // input for pipline
reg signed [P-1:0] qreg; // output

always @(posedge CLK) begin
    if (RST) begin
        qreg <= '0;
    end else begin
        qreg <= a[HB : LB]; // slice the output of multiplier
    end
end
assign q = qreg;

endmodule
