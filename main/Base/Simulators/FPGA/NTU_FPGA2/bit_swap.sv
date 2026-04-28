//============================================================================
// Documentation
//============================================================================
// Author: Hiroki Shibata (Tokyo Metropolitan University)
// Date: November 2025

// Description:
//   This is a module to swap bits of 2 designated positions. Supply those position by input, q and p.
// FMAX was 794 MHz. This module should not be any bottleneck
//===========================================================================
module bit_swap#(
    //------------------------------------------------------------------------
    // CONFIGURABLE PARAMETERS
    //------------------------------------------------------------------------
    parameter N = 32, // register width.
    parameter Np = 4, // number of pipeline stage. 
    parameter M = 5  // bit width of pointer. M must be less than or equal to log_2(N)
        )    
  (
    //------------------------------------------------------------------------
    // CLOCK AND RESET
    //------------------------------------------------------------------------
    input  CLK,     // system clock
    input [N-1:0] a_in,
    input [M-1:0] p_in,
    input [M-1:0] q_in,
    output reg [N-1:0] a_out,
    );
reg [N-1:0] p_a [Np-1];
reg [M-1:0] p_p [Np-1];
reg [M-1:0] p_q [Np-1];

logic [N-1:0] n_a_out;
wire [M-1:0] pl = p_p[Np-2];  // p at the last of pipeline
wire [M-1:0] ql = p_q[Np-2];  // q at the last of pipeline
wire [N-1:0] al = p_a[Np-2];  // a at the last of pipeline

always_comb begin: mainCombBlock
    n_a_out = al;
    n_a_out[pl] = al[ql];
    n_a_out[ql] = al[pl];
end
always@(posedge CLK) begin
    p_a[0] <= a_in;
    p_p[0] <= p_in;
    p_q[0] <= q_in;
    a_out <= n_a_out;
end
genvar i;
generate
      for( i=1; i < Np-1; i++) begin   : stage_gen
            always@(posedge CLK) begin
                p_a[i] <= p_a[i-1];
                p_p[i] <= p_p[i-1];
                p_q[i] <= p_q[i-1];
            end
      end
endgenerate                    

endmodule


module bit_swapper_tb;
      localparam int N = 32;
      localparam int M = 5;

      wire [N-1:0] d_out;
      reg [N-1:0] d_in;
      reg [M-1:0] p;
      reg [M-1:0] q;
      reg CLK;

      bit_swap #(.M(M), .N(N), .Np(6)) b0(.CLK(CLK),
            .a_in(d_in),
            .a_out(d_out),
            .q_in(q),
            .p_in(p));
      
      initial begin
            d_in = 0;
            CLK <= 0;

            #50;
            p = 0;
            q = 1;
            d_in = 'b0101_0101_0000_0001;
            #10;
            p = 0;
            q = 2;
            #10;
            p = 0;
            q = 3;
            #10;
            p = 0;
            q = 4;
            #10;
            p = 0;
            q = 5;
            #10;
            p = 0;
            q = 6;
            #10;
            p = 0;
            q = 7;
            #10;
            p = 0;
            q = 8;
            #10;
            p = 0;
            q = 9;
            #10;
            p = 11;
            q = 3;
            #10;
            p = 12;
            q = 0;
            #10;
            p = 31;
            q = 0;

            #100;
            $stop;
      end
      always begin
            #5;
            CLK <= ~CLK;
      end
  
endmodule