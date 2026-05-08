// pipelined ALU test code for 2 fixed points.
// Amir Alizadeh & Hiroki Shibata, Tokyo Metropollitan University, created at Nottingham Trent University.

module Mul_64_FixedP
#(
	parameter P = 64
)
(
	input CLK,
	input RST,
	input signed [P-1:0] a, 
	input signed [P-1:0] b, 
	output signed [2*P-1:0] q
);

// Chunk widths for 18-bit DSP decomposition
//   chunk 0  : bits [17: 0]  → 18 bits
//   chunk 1  : bits [35:18]  → 18 bits
//   chunk 2  : bits [53:36]  → 18 bits
//   chunk 3  : bits [63:54]  → 10 bits  (zero-padded to 18)
localparam CW = 18;
localparam NC = 4;
localparam REDUN = 1; // redundant input register

reg signed [P-1:0] a_[0:REDUN];
reg signed [P-1:0] b_[0:REDUN];


// how to check --> ax_clock=0, ayscan_in_clock = 0, az_clock = 0
(* preserve, dont_merge *) reg [35:0] mul_p [0:NC-1][0:NC-1]; // pipline stage (in DSP)
(* preserve, dont_merge *) reg [35:0] prod_s1 [0:NC-1][0:NC-1]; // output stage (inside DSP)
// buffer reg 
(* preserve, dont_merge *) reg [35:0]   prod_s2 [0:NC-1][0:NC-1];
wire  [CW-1:0] ac [0: NC-1]; // ac[i] = is 1D array of 4 chunks
wire [CW-1:0] bc [0:NC-1]; // ac[0], ac[1], ac[2], ac[3] each is [17:0]



always @(posedge CLK) begin 
	a_[0] <= a;
	b_[0] <= b;
//	a_[1] <= a_[0];
//	b_[1] <= b_[0]; 

	for (int i=0; i<REDUN; i++) begin // // redundant input register 
		a_[i+1] <= a_[i];
		b_[i+1] <= b_[i];
	end
end

// extract chunk (combinationally from s0 regs — zero-extend to CW)
// They are connection wires, but they connect 
//from pipelined full words a_[REDUN], b_[REDUN] to chunk buses ac/bc
generate 
	genvar ci;
	for (ci=0; ci<NC; ci++) begin: hunk_extract
	
		localparam int LO = ci *CW;
		localparam int HI = (LO + CW  <= P) ? LO + CW:P;
		localparam int WID = HI - LO;
		
	if (ci == NC-1) begin
		// sign-extend the top chunk
		//assign ac[ci] = {{(CW-WID){a_[REDUN][HI-1]}}, a_[REDUN][HI-1:LO]}; //  slices a_[REDUN] into 4 chunk ac[0..3]
		//assign bc[ci] = {{(CW-WID){b_[REDUN][HI-1]}}, b_[REDUN][HI-1:LO]};
        assign ac[ci] = {{(CW-WID){1'b0}}, a_[REDUN][HI-1:LO]};
        assign bc[ci] = {{(CW-WID){1'b0}}, b_[REDUN][HI-1:LO]};
		end else begin
		// zero-extend lower chunks
		//assign ac[ci] = {{(CW-WID){1'b0}}, a_[REDUN][HI-1:LO]};
		//assign bc[ci] = {{(CW-WID){1'b0}}, b_[REDUN][HI-1:LO]};
        assign ac[ci] = a_[REDUN][HI-1:LO];
        assign bc[ci] = b_[REDUN][HI-1:LO];
		end
	end

endgenerate


/////////////////////////////////////////////////////
// -----stage 1: 16 parallel 18x18 multiplies in DSPs -----
// Per-DSP input registers, holding the chunk directly
(* preserve, dont_merge *) reg  [CW-1:0] ac_dsp [0:NC-1][0:NC-1];
(* preserve, dont_merge *) reg  [CW-1:0] bc_dsp [0:NC-1][0:NC-1];


// prod[i][j] = ac[i] * bc[j] - Each is an 18×18 multiply → 36-bit result.
//(* preserve, dont_merge *) - can be add before syntax of reg to prevent optimization and merging by synthesis
// Replicate each cunk into 16 registers. "ac_dssp" = 4x4 array of 18 bits
reg [35:0] prod_buf [0:NC-1][0:NC-1]; // buffer reg for prod_s1, to prevent long wires from DSP to next stage

always @(posedge CLK) begin
// Replicate to 16 copies (synthesis will place each near its DSP)
    for (int i=0; i<NC; i++) begin
        for (int j=0; j<NC; j++) begin
            ac_dsp[i][j] <= ac[i];
            bc_dsp[i][j] <= bc[j];
        end
    end

	for (int i= 0; i< NC; i++) begin
		for (int j=0; j< NC; j++) begin
			mul_p [i][j] <= ac_dsp[i][j] * bc_dsp[i][j]; // 18 *18 
			prod_s1[i][j] <= mul_p[i][j]; // copy / pack as az
		end
	end
end
// always @(posedge CLK) begin
//     for (int i = 0; i < NC; i++)
//         for (int j = 0; j < NC; j++)
//           prod_s2[i][j] <= prod_s1[i][j];  // fabric-side
// end
/////////////////////////////////////////////////////


// ---------stage 2 ----------
// we group the 16 products by their anti-diagonal (i+j = constant) to keep
localparam AW2 = 128;

reg signed [AW2-1:0] result_s4;
reg [37:0] diag [0:6];   // d = 0..6, 7 diagonals

always @(posedge CLK) begin
    diag[0] <= prod_s1[0][0];
    diag[1] <= prod_s1[0][1] + prod_s1[1][0];
    diag[2] <= prod_s1[0][2] + prod_s1[1][1] + prod_s1[2][0];
    diag[3] <= prod_s1[0][3] + prod_s1[1][2] + prod_s1[2][1] + prod_s1[3][0];
    diag[4] <= prod_s1[1][3] + prod_s1[2][2] + prod_s1[3][1];
    diag[5] <= prod_s1[2][3] + prod_s1[3][2];
    diag[6] <= prod_s1[3][3];
end
reg  [127:0] sum_e_lo, sum_e_hi;   // even diagonals split
reg [127:0] sum_o_lo, sum_o_hi;   // odd diagonals split

always @(posedge CLK) begin
    // Even diagonals (offsets 0, 36, 72, 108)
    sum_e_lo <= ((diag[0]) <<<   0)
              + ((diag[2]) <<<  36);
    sum_e_hi <= ((diag[4]) <<<  72)
              + ((diag[6]) <<< 108);

    // Odd diagonals (offsets 18, 54, 90)
    sum_o_lo <= ((diag[1]) <<<  18)
              + ((diag[3]) <<<  54);
    sum_o_hi <=  (diag[5]) <<<  90;   // only one term, just place it
end

// ---- Stage B2: combine even and odd halves ----
reg [127:0] sum_even, sum_odd;
reg signed [127:0] result_q;
// ------- Sign correction (Baugh-Wooley) -----
localparam CORR_LAT = 6;
reg [64:0] corr_pre [0: CORR_LAT-1];
reg [127:0] correction_pos;


// sign correction: 
//if a_[REDUN] is negative, add b_[REDUN] shifted by 64; if b_[REDUN] is negative, add a_[REDUN] shifted by 64
always @(posedge CLK) begin 
	corr_pre[0] <= (a_[REDUN][P-1] ? {1'b0, b_[REDUN]} : 65'd0)
				+ (b_[REDUN][P-1] ? {1'b0, a_[REDUN]} : 65'd0);
	for (int k=0; k< CORR_LAT-1; k++) begin
		corr_pre[k+1] <= corr_pre[k];
	end
	correction_pos <= {corr_pre[CORR_LAT-1][63:0], 64'b0};
end


always @(posedge CLK) begin
    sum_even <= sum_e_lo + sum_e_hi;
    sum_odd  <= sum_o_lo + sum_o_hi;
    result_q <= sum_even + sum_odd - correction_pos;
end

assign q = result_q;
endmodule	

////////////////////////////////////////////////////
////////----stage 3 and 4 -----//////
//localparam AW = 128; // raw 18 *18 product width
//localparam PAIRW = PW + CW +1;  // 55 (content-only)
// reg signed [AW-1:0] pair_lo [0:NC-1];  // (prod[i][0]<<i*18) + (prod[i][1]<<(i+1)*18)
// reg signed [AW-1:0] pair_hi [0: NC-1]; // (prod[i][2]<<(i+2)*18) + (prod[i][3]<<(i+3)*18)

// always @(posedge CLK) begin
// 	integer ii, jj;
	
// 	for (ii = 0; ii < NC; ii++) begin
// 		pair_lo[ii] <=
// 			({{(AW-36){prod_s1[ii][0][35]}}, prod_s1[ii][0]} <<< (ii*CW)) +
//     		({{(AW-36){prod_s1[ii][1][35]}}, prod_s1[ii][1]} <<< ((ii+1)*CW));
	
// 		pair_hi[ii] <= 
// 			    ({{(AW-36){prod_s1[ii][2][35]}}, prod_s1[ii][2]} <<< ((ii+2)*CW)) +
// 				({{(AW-36){prod_s1[ii][3][35]}}, prod_s1[ii][3]} <<< ((ii+3)*CW));

// 	end
// end

// reg signed [AW2-1:0] acc_s2 [0:NC-1];
// always @(posedge CLK) begin
//     integer ii; 
// 	 for (ii = 0; ii < NC; ii++) begin
// 		acc_s2[ii] <= pair_lo[ii] + pair_hi[ii];
// 	 end
// end

// reg signed [AW2-1:0] sum_s3_lo, sum_s3_hi;
// reg signed [AW2-1:0] sum_s4;
// always @(posedge CLK) begin 
// 	sum_s3_lo <= acc_s2[0] + acc_s2[1];
// 	sum_s3_hi <= acc_s2[2] + acc_s2[3];
// 	sum_s4 <= sum_s3_lo + sum_s3_hi;
// end

// STAGE 5  —  Output register
// always @(posedge CLK) begin
// result_s4 <= sum_s4; 
// end
// assign q = result_s4[P-1:0];
//endmodule

////////////////////////////////////////////////////

////////////////////////////////////////////////////
// New stage 3 and 4 combined //
/*
reg [AW2-1:0] sum_csa, carry_csa;

function automatic [2*AW2-1:0] csa3to2(
    input [AW2-1:0] a,
    input [AW2-1:0] b,
    input [AW2-1:0] c
);
    logic [AW2-1:0] sum, carry;
    sum   =  a ^ b ^ c;
    carry = (a & b) | (b & c) | (a & c);
    csa3to2 = {carry, sum};   // pack: upper half = carry, lower half = sum
endfunction

// Layer 1: 8 inputs → 6 wires
wire [AW2-1:0] s1a, c1a, s1b, c1b;
assign {c1a, s1a} = csa3to2(pair_lo[0], pair_hi[0], pair_lo[1]);
assign {c1b, s1b} = csa3to2(pair_hi[1], pair_lo[2], pair_hi[2]);
// Live: {s1a, c1a<<1, s1b, c1b<<1, pair_lo[3], pair_hi[3]} = 6

// Layer 2: 6 → 4
wire [AW2-1:0] s2a, c2a, s2b, c2b;
assign {c2a, s2a} = csa3to2(s1a,         c1a << 1,    s1b);
assign {c2b, s2b} = csa3to2(c1b << 1,    pair_lo[3],  pair_hi[3]);
// Live: {s2a, c2a<<1, s2b, c2b<<1} = 4

// Layer 3: 4 → 3
wire [AW2-1:0] s3, c3;
assign {c3, s3} = csa3to2(s2a, c2a << 1, s2b);
// Live: {s3, c3<<1, c2b<<1} = 3

// Layer 4: 3 → 2
wire [AW2-1:0] s4, c4;
assign {c4, s4} = csa3to2(s3, c3 << 1, c2b << 1);
// Live: {s4, c4<<1} = 2  ← done!

always @(posedge CLK) begin
    sum_csa   <= s4;
    carry_csa <= c4 << 1;
end
// STAGE 5  —  Output register: slice lower P bits
always @(posedge CLK) begin
result_s4 <= sum_csa + carry_csa;  
end
assign q = result_s4[P-1:0];
endmodule
*/
////////////////////////////////////////////////////

		

