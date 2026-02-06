module top1 (
   input  CLK,
   input  RST,
   input  tx_OK,
   output tx_en,
   output [7:0] tx_data_out,
   input [7:0] rx_data_in,
   input rx_dv,
   output [31:0]  o_Status // counter to wait read FIFO latency. 
);
parameter NM = 13; // address width for state vector's and cost function's BRAM. Thus, the number of maximum qubits the system can deal with.
parameter P = 64; // data width for numerical number
parameter Ni = 32;// data width of auxiary information on pipeline.
parameter NBRAM = 6; // number of block RAMs connected to qaoa system.
localparam N_BIT_SWAP_POINTER = $clog2(NM);
// Declare signals to connect to the UART module
wire  [63:0] r_data;
wire  [63:0] r_addr;
wire  [63:0] w_addr;
wire  [63:0] w_data;
wire r_req;
wire r_vd;
wire w_req;

wire [23:0] CMD;
wire [63:0] rS; // status of qaoa system.
// we need transmitter and receiver to tset state machine (ntu_smachine)
control_interface CI 
(
   .CLK(CLK),        // Connect to your system clock wire
   .RST(RST),        // Connect to your system reset wire
   .tx_OK(tx_OK),
   .tx_en(tx_en),
   .tx_data_out(tx_data_out),
   .rx_data_in(rx_data_in),
   .rx_dv(rx_dv),
   .r_data(r_data),
   .r_addr(r_addr),
   .r_req(r_req),
   .rbram_vd(r_vd),
   .w_addr(w_addr),
   .w_data(w_data),
   .w_req(w_req),
   .CMD(CMD),
   .rS(rS)
);


wire [NM-1:0] bram_addr_r [NBRAM];
wire [NM-1:0] bram_addr_w [NBRAM];
wire [P-1:0] bram_data_r [NBRAM];
wire [P-1:0] bram_data_w [NBRAM];
wire [NBRAM-1:0] bram_wen;

wire [P-1:0] cosb;
wire [P-1:0] sinb;
wire [P-1:0] p_ar;
wire [P-1:0] p_ai;
wire  [P-1:0]  p_ar_o;
wire  [P-1:0]  p_ai_o;
wire  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0]  info_out; // information, like addresses, enabled signal, and so on.
wire  [1:0]  mix_switch; // information, like addresses, enabled signal, and so on.

wire  [P-1:0]  gamma; // cos gamma
wire  [P-1:0]   HGC;
wire  [P-1:0]   Hr_o;
wire  [P-1:0]   Hi_o;
wire  [Ni-1:0]    info_inGC; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0]    info_outGC; // information, like addresses, enabled signal, and so on.

wire [Ni-1:0] bs_info_out;
wire [Ni-1:0] bs_info_in;
wire [NM-1:0] bswap_in;  // bit swap in swap pointer 1
wire [NM-1:0] bswap_out;
wire [N_BIT_SWAP_POINTER-1] bsp1; // next bit swap pointer 1
wire [N_BIT_SWAP_POINTER-1] bsp2; // next bit swap pointer 2
qaoa_system2 #(.NM(NM), .P(P), .NBRAM(NBRAM), .Ni(Ni) .N_BIT_SWAP_POINTER(N_BIT_SWAP_POINTER)) qs2
(
   .CLK(CLK),
   .RST(RST),
   .r_data(r_data),
   .n_r_addr(r_addr),
   .n_r_req(r_req),
   .r_vd(r_vd),
   .n_w_addr(w_addr),
   .n_w_data(w_data),
   .n_w_req(w_req),

   .bram_addr_r(bram_addr_r),
   .bram_addr_w(bram_addr_w),
   .bram_data_r(bram_data_r),
   .bram_data_w(bram_data_w),
   .bram_wen(bram_wen),
   .r_CMD(CMD),
   .cosb(cosb),
   .sinb(sinb),
   .mix_ar(p_ar),
   .mix_ai(p_ai),
   .mix_info(info_in),
   .mix_switch(mix_switch),
   
   .mix_ar_res(p_ar_o),
   .mix_ai_res(p_ai_o),
   .mix_info_res(info_out),
   
   .gamma(gamma), //  gamma
   .HGC(HGC),
   .Hr_res(Hr_o),
   .Hi_res(Hi_o),
   .info_inGC(info_inGC), // information, like addresses, enabled signal, and so on.
   .info_outGC(info_outGC) // information, like addresses, enabled signal, and so on.

   .bs_info_out(bs_info_out);
   .bs_info_in(bs_info_in);
   .bswap_in(bswap_in);  // bit swap in swap pointer 1
   .bswap_out(bswap_out);
   .bsp1(bsp1); // next bit swap pointer 1
   .bsp2(bsp2); // next bit swap pointer 2
);

bit_swap #(.M(M), .N(N), .Np(Ni), .Ni(N_BIT_SWAP_POINTER)) b0(.CLK(CLK),
    .a_in(bswap_in),
    .a_out(bswap_out),
    .q_in(bsp1),
    .p_in(bsp2),
    .info_in(bs_info_in),
    .info_out(bs_info_out)
);
gen_cost  #(.P(P),.Ni(Ni)) genCost
  (
   .CLK(CLK),
   .RST(RST),
   .gamma(gamma), //  gamma
   .H(H),
   .Hr_o(Hr_o),
   .Hi_o(Hi_o),
   .info_in(info_inGC), // information, like addresses, enabled signal, and so on.
   .info_out(info_outGC) // information, like addresses, enabled signal, and so on.
);
mixer_test #(.P(P),.Ni(Ni)) mix // width of additional information
(
   .CLK(CLK),
   .RST(RST),
   .cb(cosb), // cos beta
   .sb(sinb), // cos beta
   .p_ar(p_ar),
   .p_ai(p_ai),
   .p_ar_o(p_ar_o),
   .p_ai_o(p_ai_o),
   .info_in(info_in), // information, like addresses, enabled signal, and so on.
   .info_out(info_out) // information, like addresses, enabled signal, and so on.
);
// my_bram #(.ADDRESS_WIDTH(NM), .DEPTH(256), .DATA_WIDTH(P)) myRam1, does not work well, 2025 10 14.
// (
//    .clk(CLK),          // Clock signal
//    .wen(bram_data_w[5]), // Write enable signal
//    .addr_r(bram_addr_r[5]), // Address for read
//    .addr_w(bram_addr_w[5]), // Address for write
//    .data_in(bram_data_w[5]),   // Data to be written
//    .data_out(bram_data_r[5])  // Data to be read
// );

ram ramGen (.address_a(bram_addr_r[5]), // NM bit address
	.address_b(bram_addr_w[5]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[5]),
	.wren_a(),
	.wren_b(bram_wen[5]),
	.q_a(bram_data_r[5]),
	.q_b());
ram ramStateR (.address_a(bram_addr_r[0]), // NM bit address
	.address_b(bram_addr_w[0]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[0]),
	.wren_a(),
	.wren_b(bram_wen[0]),
	.q_a(bram_data_r[0]),
	.q_b());
ram ramStateI (.address_a(bram_addr_r[1]), // NM bit address
	.address_b(bram_addr_w[1]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[1]),
	.wren_a(),
	.wren_b(bram_wen[1]),
	.q_a(bram_data_r[1]),
	.q_b());
ram ramHR (.address_a(bram_addr_r[2]), // NM bit address
	.address_b(bram_addr_w[2]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[2]),
	.wren_a(),
	.wren_b(bram_wen[2]),
	.q_a(bram_data_r[2]),
	.q_b());
ram ramCR(.address_a(bram_addr_r[3]), // NM bit address
	.address_b(bram_addr_w[3]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[3]),
	.wren_a(),
	.wren_b(bram_wen[3]),
	.q_a(bram_data_r[3]),
	.q_b());
ram ramCI(.address_a(bram_addr_r[4]), // NM bit address
	.address_b(bram_addr_w[4]),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(bram_data_w[4]),
	.wren_a(),
	.wren_b(bram_wen[4]),
	.q_a(bram_data_r[4]),
	.q_b());
endmodule
