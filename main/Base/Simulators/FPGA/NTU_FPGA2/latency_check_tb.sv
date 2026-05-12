// latency_probe_tb.sv — measure end-to-end latency of one Mul + slicer + adder chain
`timescale 1ns/1ns


// latency is 8  multiplier + 1 slicer + 1 adder = 10 cycles total from input to final output, with 1 cycle between each stage.
module latency_check_tb();
parameter P = 64;
reg CLK;
reg RST;
reg signed [P-1:0] a , b;
wire signed [2*P-1:0] q_mul;
wire signed [P-1:0] q_add;
wire signed [P-1:0] q_slicer;

always #1 CLK = ~CLK;

latency_check #(.P(P)) DUT (
    .CLK(CLK),
    .RST(RST),
    .a(a),
    .b(b),
    .q_mul(q_mul),
    .q_slicer(q_slicer),
    .q_add(q_add)
);

integer cycle;

initial begin 
    RST = 1;
    CLK = 0;
    a = 0;
    b = 0;
    cycle = 0;
    #10 RST =0; // release reset after 10ns
    @(posedge CLK); 
    a = 64'h19518bebead3c500;  // +1.0 in Q3.61
    b = 64'he2973c6b0f92a900;  // +1.0 in Q3.61  → product is +1.0
    @(posedge CLK);
    a = 0; b = 0;
    repeat (40) begin
        @(posedge CLK);
        cycle = cycle +1;
        $display("t=%0d  mul_q=%h  q_slicer=%h  q_add=%h", cycle, q_mul, q_slicer, q_add);
    end
    $stop;
end
endmodule

