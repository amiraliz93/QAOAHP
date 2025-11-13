`timescale 1ns / 1ps

// todo
// test transmission
module alutest_tb ();
// Set the parameters for the instance
localparam my_P = 64;
localparam my_Ni = 32;

// Declare signals for the parent module, with inputs as 'reg'
reg             CLK; // Capitalized signal name for the clock
reg             RST; // Capitalized signal name for the reset
reg  [my_P-1:0] cb_in;
reg  [my_P-1:0] sb_in;
reg  [my_P-1:0] p_ar_in;
reg  [my_P-1:0] p_ai_in;
reg  [my_P-1:0] p_br_in;
reg  [my_P-1:0] p_bi_in;
wire [my_P-1:0] p_ar_out;
wire [my_P-1:0] p_ai_out;
wire [my_P-1:0] p_br_out;
wire [my_P-1:0] p_bi_out;
reg  [my_Ni-1:0] info_in_signal;
wire [my_Ni-1:0] info_out_signal;


// Instantiate the module with named port association
alu_test
  #(.P(my_P), .Ni(my_Ni))
  alu_test_instance (
    .CLK      (CLK), // Connects to the capitalized register
    .RST      (RST), // Connects to the capitalized register
    .cb       (cb_in),
    .p_ar     (p_ar_in),
    .p_ar_o   (p_ar_out),
    .info_in  (info_in_signal),
    .info_out (info_out_signal)
);

always

begin
      #2.5 CLK <= ~CLK; // clock generation, half period
end
reg [my_P-1:0] get1;
reg [32-1:0] get2;
always @(posedge CLK) begin
	if(RST)begin
		get1 <= 0;
		get2 <= 0;
	end
	else begin
		get1 <= p_ar_out;
		get2 <= info_out_signal;
            info_in_signal <= info_in_signal + 1;
	end
end
// test for transmitter
initial begin 
	CLK <= 1;
      RST <= 0;
      #100.5
      RST <= 1;
      info_in_signal <= 0;
      cb_in <= 0;
		sb_in <= 0;
      p_ar_in <= 0;
      p_ai_in <= 0;
		p_br_in <= 0;
		p_bi_in <= 0;
      #100
      
      RST <= 0;
      #100
      cb_in <= 64'b0100000000001001001000011111101101010100010001000010110100011000; // 3.141592653589793
      p_ar_in <= 64'b0011111111110011011001001111011001110100011001100000001101101010; // 1.2121491
      p_ai_in <= 64'b0100000000001110011101101111000111111000001011100110101101100001; // 3.8080787076154796
      #5 
      cb_in <= 64'b0100000000100011001100110011001100110011001100110011001100110011; // 9.6
      p_ar_in <= 64'b0011111111100001111100111011011001000101101000011100101011000001; // 0.561
      p_ai_in <= 64'b0100000000010101100010101101101010111001111101010101100110110100; // 5.3856
      #5 
      cb_in <= 0;
      p_ar_in <= 0;
      p_ai_in <= 0;
      #5 
      cb_in <= 64'b0100000000010000100110001001001101110100101111000110101001111111; // 4.149
      p_ar_in <= 64'b0100000001010110010001111101111100111011011001000101101000011101; // 89.123
      p_ai_in <= 64'b0100000001110111000111000101011101011010111110101111100001011010; // 369.77132700000004
      #5 
      cb_in <= 0;
      p_ar_in <= 0;
      p_ai_in <= 0;
		#200
		$stop;
end

endmodule

