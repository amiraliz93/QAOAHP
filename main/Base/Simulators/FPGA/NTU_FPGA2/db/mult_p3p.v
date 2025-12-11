//lpm_mult CBX_DECLARE_ALL_CONNECTED_PORTS="OFF" DEVICE_FAMILY="Stratix V" DSP_BLOCK_BALANCING="Auto" LPM_PIPELINE=18 LPM_REPRESENTATION="SIGNED" LPM_WIDTHA=64 LPM_WIDTHB=64 LPM_WIDTHP=128 MAXIMIZE_SPEED=9 clock dataa datab result CARRY_CHAIN="MANUAL" CARRY_CHAIN_LENGTH=48
//VERSION_BEGIN 17.1 cbx_cycloneii 2017:10:25:18:06:53:SJ cbx_lpm_add_sub 2017:10:25:18:06:53:SJ cbx_lpm_mult 2017:10:25:18:06:53:SJ cbx_mgl 2017:10:25:18:08:29:SJ cbx_nadder 2017:10:25:18:06:53:SJ cbx_padd 2017:10:25:18:06:53:SJ cbx_stratix 2017:10:25:18:06:53:SJ cbx_stratixii 2017:10:25:18:06:53:SJ cbx_util_mgl 2017:10:25:18:06:53:SJ  VERSION_END
// synthesis VERILOG_INPUT_VERSION VERILOG_2001
// altera message_off 10463



// Copyright (C) 2017  Intel Corporation. All rights reserved.
//  Your use of Intel Corporation's design tools, logic functions 
//  and other software and tools, and its AMPP partner logic 
//  functions, and any output files from any of the foregoing 
//  (including device programming or simulation files), and any 
//  associated documentation or information are expressly subject 
//  to the terms and conditions of the Intel Program License 
//  Subscription Agreement, the Intel Quartus Prime License Agreement,
//  the Intel FPGA IP License Agreement, or other applicable license
//  agreement, including, without limitation, that your use is for
//  the sole purpose of programming logic devices manufactured by
//  Intel and sold by Intel or its authorized distributors.  Please
//  refer to the applicable agreement for further details.



//synthesis_resources = 
//synopsys translate_off
`timescale 1 ps / 1 ps
//synopsys translate_on
module  mult_p3p
	( 
	clock,
	dataa,
	datab,
	result) /* synthesis synthesis_clearbox=1 */;
	input   clock;
	input   [63:0]  dataa;
	input   [63:0]  datab;
	output   [127:0]  result;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_off
`endif
	tri0   clock;
`ifndef ALTERA_RESERVED_QIS
// synopsys translate_on
`endif

	reg  [63:0]  dataa_input_reg;
	reg  [63:0]  datab_input_reg;
	reg  [127:0]  result_output_reg;
	reg  [127:0]  result_extra0_reg;
	reg  [127:0]  result_extra1_reg;
	reg  [127:0]  result_extra2_reg;
	reg  [127:0]  result_extra3_reg;
	reg  [127:0]  result_extra4_reg;
	reg  [127:0]  result_extra5_reg;
	reg  [127:0]  result_extra6_reg;
	reg  [127:0]  result_extra7_reg;
	reg  [127:0]  result_extra8_reg;
	reg  [127:0]  result_extra9_reg;
	reg  [127:0]  result_extra10_reg;
	reg  [127:0]  result_extra11_reg;
	reg  [127:0]  result_extra12_reg;
	reg  [127:0]  result_extra13_reg;
	reg  [127:0]  result_extra14_reg;
	reg  [127:0]  result_extra15_reg;
	wire signed	[63:0]    dataa_wire;
	wire signed	[63:0]    datab_wire;
	wire signed	[127:0]    result_wire;


	// synopsys translate_off
	initial
		dataa_input_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		dataa_input_reg <= dataa;
	// synopsys translate_off
	initial
		datab_input_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		datab_input_reg <= datab;
	// synopsys translate_off
	initial
		result_output_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_output_reg <= result_extra15_reg;
	// synopsys translate_off
	initial
		result_extra0_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra0_reg <= result_wire[127:0];
	// synopsys translate_off
	initial
		result_extra1_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra1_reg <= result_extra0_reg;
	// synopsys translate_off
	initial
		result_extra2_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra2_reg <= result_extra1_reg;
	// synopsys translate_off
	initial
		result_extra3_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra3_reg <= result_extra2_reg;
	// synopsys translate_off
	initial
		result_extra4_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra4_reg <= result_extra3_reg;
	// synopsys translate_off
	initial
		result_extra5_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra5_reg <= result_extra4_reg;
	// synopsys translate_off
	initial
		result_extra6_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra6_reg <= result_extra5_reg;
	// synopsys translate_off
	initial
		result_extra7_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra7_reg <= result_extra6_reg;
	// synopsys translate_off
	initial
		result_extra8_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra8_reg <= result_extra7_reg;
	// synopsys translate_off
	initial
		result_extra9_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra9_reg <= result_extra8_reg;
	// synopsys translate_off
	initial
		result_extra10_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra10_reg <= result_extra9_reg;
	// synopsys translate_off
	initial
		result_extra11_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra11_reg <= result_extra10_reg;
	// synopsys translate_off
	initial
		result_extra12_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra12_reg <= result_extra11_reg;
	// synopsys translate_off
	initial
		result_extra13_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra13_reg <= result_extra12_reg;
	// synopsys translate_off
	initial
		result_extra14_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra14_reg <= result_extra13_reg;
	// synopsys translate_off
	initial
		result_extra15_reg = 0;
	// synopsys translate_on
	always @(posedge clock)
		result_extra15_reg <= result_extra14_reg;

	assign dataa_wire = dataa_input_reg;
	assign datab_wire = datab_input_reg;
	assign result_wire = dataa_wire * datab_wire;
	assign result = ({result_output_reg});

endmodule //mult_p3p
//VALID FILE
