`default_nettype none
module top1 (
   input  wire CLK,
   input  wire RST,
   input  wire tx_OK,
   output wire tx_en,
   output wire [7:0] tx_data_out,
   input wire [7:0] rx_data_in,
   input wire rx_dv,
   output wire [31:0]  o_Status
);
localparam NM = 21; // address width for state vector's and cost function's BRAM. Thus, the number of maximum qubits the system can deal with.
localparam P = 64; // data width for numerical number
localparam Ni = 21 + NM;// data width of auxiary information on pipeline.
localparam NBRAM = 4; // number of block RAMs connected to qaoa system.
localparam N_BIT_SWAP_POINTER = $clog2(NM);
localparam L_BRAM_A = 2; // Address transmisshon pipeline latency
localparam L_BRAM_D = 2; // Data retrieving pipeline latency 
localparam L_BRAM_W = L_BRAM_A + L_BRAM_D + 1; // Write latency
localparam L_BRAM_R = L_BRAM_A + L_BRAM_D + 2; // Read latency


// Declare signals to connect to the UART module
wire  [P-1:0] r_data;
wire  [P-1:0] r_addr;
wire  [P-1:0] w_addr;
wire  [P-1:0] w_data;
wire r_req;
wire r_vd;
wire w_req;

wire [23:0] CMD;
wire [P-1:0] rS; // status of qaoa system.
// we need transmitter and receiver to tset state machine (ntu_smachine)
assign o_Status = rS[31:0];

wire [7:0] ag_addr_Param, ag_addr_ParamP;
wire [NM-1:0] ag_Param, ag_ParamP;
wire ag_wen, ag_wenP;

wire f_L1Computation; // last 1 clock before the computation ends.
wire ag_enPipe;
wire ag_enCostF;
wire [15:0] ag_en_Inits;
wire [1:0] ag_mixSwitch;
wire f_run_Computation;

wire [NM-1:0] ag_cAddr;       // Next address counter
wire [NM-1:0] ag_cAddrGC;     // Next address counter
wire [N_BIT_SWAP_POINTER-1:0] bsp1;
wire [N_BIT_SWAP_POINTER-1:0] bsp2;

wire [NM-1:0] bram_addr_r [NBRAM];
wire [NM-1:0] bram_addr_w [NBRAM];
wire [P-1:0] bram_data_r [NBRAM];
wire [P-1:0] bram_data_w [NBRAM];
wire [NM-1:0] bram_addr_rP [NBRAM];
wire [NM-1:0] bram_addr_wP [NBRAM];
wire [P-1:0] bram_data_rP [NBRAM];
wire [P-1:0] bram_data_wP [NBRAM];
wire [NBRAM-1:0] bram_wen;
wire [NBRAM-1:0] bram_wenP;

wire [P-1:0] cosb;
wire [P-1:0] sinb;
wire [P-1:0] p_ar;
wire [P-1:0] p_ai;
wire  [P-1:0]  p_ar_o;
wire  [P-1:0]  p_ai_o;
wire  [Ni-1:0]  ag_info; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0]  bs_info; // information, like addresses, enabled signal, and so on.
wire  [1:0]  mix_switch; // information, like addresses, enabled signal, and so on.
wire [NM:0] mix_info;  // NM-1:0 is for address, NM for enable signal.
wire [NM:0] mix_infoR;

wire  [P-1:0]  gamma; // cos gamma
wire  [P-1:0]   HGC;
wire  [P-1:0]   Hr_o;
wire  [P-1:0]   Hi_o;

wire [NM-1:0] bswap_out;


control_interface #(.NM(NM))CI 
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
   .rS(rS),
   .ag_addr_Param_out(ag_addr_Param),
   .ag_Param_out(ag_Param),
   .ag_wen_out(ag_wen)
);


regPipeline  #(.W_INFO($bits({ag_addr_Param, ag_Param, ag_wen})), .NPipe(2)) P_ci_addr (
   .CLK(CLK),
   .RST(RST),
   .info_in({ag_addr_Param, ag_Param, ag_wen}),
   .info_out({ag_addr_ParamP, ag_ParamP, ag_wenP})
);


addr_gen #(.N_BIT_SWAP_POINTER(N_BIT_SWAP_POINTER), .NM(NM)) addr_gen_inst(
   .CLK(CLK),
   .RST(RST), 
   // connected to control interface
   .addr_Param_in(ag_addr_ParamP),
   .Param_in(ag_ParamP),
   .wen_in(ag_wenP),
   // connected to qaoa_system
   .en_Pipe_out(ag_enPipe),
   .en_CostF_out(ag_enCostF),
   .f_run_Computation_in(f_run_Computation),
   .f_L1Computation_out(f_L1Computation),
   .en_Inits_out(ag_en_Inits),
   .mixSwitch_out(ag_mixSwitch),
   .cAddrGC_out(ag_cAddrGC),
   // connected to bit_swap
   .cAddr_out(ag_cAddr),
   .bsp1_out(bsp1),
   .bsp2_out(bsp2)
);

qaoa_system2 #(.NM(NM), .P(P), .NBRAM(NBRAM),  .L_BRAM_R(L_BRAM_R)) qs2
(
   .CLK(CLK),
   .RST(RST),
   
    //------------------------------------------------------------------------
    // COMMAND INTERFACE (among Control Interface and addr_gen)
    //------------------------------------------------------------------------
   .r_CMD(CMD),
   .enPipe_in(bs_info[0]),
   .enCostF_in(bs_info[1]),
   .f_L1Computation_in(bs_info[2]),
   .en_Inits_in(bs_info[20:5]),
   .mixSwitch_in(bs_info[4:3]),
   .f_run_Computation_out(f_run_Computation),
   
    //------------------------------------------------------------------------
    // Address INTERFACE (FOR MIXER, and COST FUNCTION GEN)
    //------------------------------------------------------------------------
   .swapped_cAddr_in(bswap_out),
   .cAddrCF_in(bs_info[21+:NM]),
    //------------------------------------------------------------------------
    // MEMORY INTERFACE from Control Interface
    //------------------------------------------------------------------------

   .r_data(r_data),
   .n_r_addr(r_addr),
   .n_r_req(r_req),
   .r_vd(r_vd),
   .n_w_addr(w_addr),
   .n_w_data(w_data),
   .n_w_req(w_req),

    //------------------------------------------------------------------------
    //BRAM ARRAY INTERFACE (3 banks + 1 general bank)
    //------------------------------------------------------------------------
   .bram_addr_r(bram_addr_r),
   .bram_addr_w(bram_addr_w),
   .bram_data_r(bram_data_r),
   .bram_data_w(bram_data_w),
   .bram_wen(bram_wen),

    //------------------------------------------------------------------------
    // MIXER OPERATION PIPELINE INTERFACE
    //------------------------------------------------------------------------

   .cosb(cosb),
   .sinb(sinb),
   .mix_ar(p_ar),
   .mix_ai(p_ai),
   .mix_info(mix_info),
   .mix_switch(mix_switch),
   
   .mix_ar_res(p_ar_o),
   .mix_ai_res(p_ai_o),
   .mix_info_res(mix_infoR),
   .Status(rS),

    //------------------------------------------------------------------------
    // COST HAMILTONIAN INTERFACE
    //------------------------------------------------------------------------
   .gamma(gamma), 
   .HGC(HGC),
   .Hr_res(Hr_o),
   .Hi_res(Hi_o)


);

bit_swap #(.M(N_BIT_SWAP_POINTER), .N(NM), .Np(5), .Ni(Ni)) bit_swap_inst(
    .CLK(CLK),
    .a_in(ag_cAddr),
    .a_out(bswap_out),
    .q_in(bsp1),
    .p_in(bsp2),
    .info_in(ag_info),
    .info_out(bs_info)
);

gen_cost  #(.P(P)) genCost
  (
   .CLK(CLK), // input
   .RST(RST), // input
   .gamma(gamma), //  gamma, input
   .H(HGC), // input
   .Hr_o(Hr_o),
   .Hi_o(Hi_o)
);

mixer2 #(.P(P),.Ni(Ni)) mix // width of additional information
(
   .CLK(CLK),
   .RST(RST),
   .cb(cosb), // cos beta
   .sb(sinb), // sin beta
   .p_ar(p_ar),
   .p_ai(p_ai),
   .p_ar_o(p_ar_o),
   .p_ai_o(p_ai_o),
   .switch_in(mix_switch),
   .info_in(mix_info), // information, like addresses, enabled signal, and so on.
   .info_out(mix_infoR) // information, like addresses, enabled signal, and so on.
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


genvar j;

generate
    for (j = 0; j < NBRAM; j = j + 1) begin : GEN_BRAM
         
         regPipeline  #(.W_INFO(P + P), .NPipe(L_BRAM_W)) P_BRAMD (
            .CLK(CLK),
            .RST(RST),
            .info_in({bram_data_r[j], bram_data_w[j]}),
            .info_out({bram_data_rP[j], bram_data_wP[j]})
         );
         regPipeline  #(.W_INFO(NM + NM + 1), .NPipe(L_BRAM_A)) P_BRAMR (
            .CLK(CLK),
            .RST(RST),
            .info_in({bram_addr_r[j], bram_addr_w[j], bram_wen[j]}),
            .info_out({bram_addr_rP[j], bram_addr_wP[j], bram_wenP[j]})
         );
         ram RAM (.address_a(bram_addr_rP[j]), // NM bit address
            .address_b(bram_addr_wP[j]),
            .clock(CLK),
            .data_a(), // 64 bit
            .data_b(bram_data_w[j]),
            .wren_a(),
            .wren_b(bram_wenP[j]),
            .q_a(bram_data_r[j]),
            .q_b()
            );
    end
endgenerate

// ram ramGen (.address_a(bram_addr_r[3]), // NM bit address
// 	.address_b(bram_addr_w[3]),
// 	.clock(CLK),
// 	.data_a(), // 64 bit
// 	.data_b(bram_data_w[3]),
// 	.wren_a(),
// 	.wren_b(bram_wen[3]),
// 	.q_a(bram_data_r[3]),
// 	.q_b());
// ram ramStateR (.address_a(bram_addr_r[0]), // NM bit address
// 	.address_b(bram_addr_w[0]),
// 	.clock(CLK),
// 	.data_a(), // 64 bit
// 	.data_b(bram_data_w[0]),
// 	.wren_a(),
// 	.wren_b(bram_wen[0]),
// 	.q_a(bram_data_r[0]),
// 	.q_b());
// ram ramStateI (.address_a(bram_addr_r[1]), // NM bit address
// 	.address_b(bram_addr_w[1]),
// 	.clock(CLK),
// 	.data_a(), // 64 bit
// 	.data_b(bram_data_w[1]),
// 	.wren_a(),
// 	.wren_b(bram_wen[1]),
// 	.q_a(bram_data_r[1]),
// 	.q_b());
// ram ramHR (.address_a(bram_addr_r[2]), // NM bit address
// 	.address_b(bram_addr_w[2]),
// 	.clock(CLK),
// 	.data_a(), // 64 bit
// 	.data_b(bram_data_w[2]),
// 	.wren_a(),
// 	.wren_b(bram_wen[2]),
// 	.q_a(bram_data_r[2]),
// 	.q_b());
endmodule
