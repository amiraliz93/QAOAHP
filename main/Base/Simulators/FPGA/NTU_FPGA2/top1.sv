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
localparam Ni = 5 + 24 + NM;// data width of auxiary information on pipeline.
localparam NBRAM = 4; // number of block RAMs connected to qaoa system. 
localparam N_BIT_SWAP_POINTER = $clog2(NM);
localparam LP_BRAM_A = 2; // Address transmisshon pipeline latency
localparam LP_BRAM_D = 1; // Data retrieving pipeline latency 
localparam LP_GEN_COST = 2;
localparam LP_MIXER_IN = 1;
localparam LP_MIXER_OUT = 1;
localparam L_BRAM_W = LP_BRAM_A + LP_BRAM_D + 2; // Write latency
localparam L_BRAM_R = LP_BRAM_A + LP_BRAM_D + 2; // Read latency


// Declare signals to connect to the UART module
wire  [P-1:0] ci_r_data;
wire  [P-1:0] ci_addr;
wire  [P-1:0] ci_w_data;
wire ci_r_req;
wire ci_r_vd;
wire ci_w_req;

wire [23:0] ci_CMD;
wire [15:0] rS; // status of qaoa system.
// we need transmitter and receiver to tset state machine (ntu_smachine)
assign o_Status = rS[15:0];

wire [1 + 8 + NM-1:0] ag_CI_wire, ag_CI_wireR;

wire f_run_Computation;

wire [NM-1:0] ag_cAddr;       // Next address counter
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

wire [P-1:0] cosb, cosbP;
wire [P-1:0] sinb, sinbP;
wire [P-1:0] p_ar, p_arP;
wire [P-1:0] p_ai, p_aiP;
wire  [P-1:0]  p_ar_o, p_ar_oP;
wire  [P-1:0]  p_ai_o, p_ai_oP;
wire  [Ni-1:0]  ag_info; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0]  bs_info, bs_infoR; // information, like addresses, enabled signal, and so on.
wire  [1:0]  mix_switch, mix_switchP; // information, like addresses, enabled signal, and so on.
wire [NM:0] mix_info, mix_infoP, mix_infoR, mix_infoRP;  // NM-1:0 is for address, NM for enable signal.

wire  [P-1:0]  gamma, gammaP; // cos gamma
wire  [P-1:0]  HGC, HGCP;
wire  [P-1:0]  Hr_o, Hr_oP;
wire  [P-1:0]  Hi_o, Hi_oP;

wire [NM-1:0] bswap_out, bswap_outR;

reg RSTci, RSTag, RSTqs; 
always @(posedge CLK) begin
   RSTci <= RST;
   RSTag <= RST;
   RSTqs <= RST;
end

control_interface #(.P(P), .NM(NM)) CI 
(
   .CLK(CLK),        // Connect to your system clock wire
   .RST(RSTci),        // Connect to your system reset wire
   .tx_OK(tx_OK),
   .tx_en(tx_en),
   .tx_data_out(tx_data_out),
   .rx_data_in(rx_data_in),
   .rx_dv(rx_dv),
   // --------- interfaces to qaoa system
   .r_data(ci_r_data),
   .addr_out(ci_addr),
   .r_req(ci_r_req),
   .rbram_vd(ci_r_vd),
   .w_data(ci_w_data),
   .w_req(ci_w_req),
   .CMD(ci_CMD),
   .rS(rS),
   // --------- interfaces to addr_gen

   .ag_addr_Param_out(ag_CI_wire[0+:8]),
   .ag_Param_out(ag_CI_wire[8+:NM]),
   .ag_wen_out(ag_CI_wire[8+NM])
   
);


regPipeline  #(.W_INFO($bits(ag_CI_wire)), .NPipe(4)) P_ci_addr (
   .CLK(CLK),
   .info_in(ag_CI_wire),
   .info_out(ag_CI_wireR)
);


addr_gen #(.N_BIT_SWAP_POINTER(N_BIT_SWAP_POINTER), .NM(NM)) addr_gen_inst(
   .CLK(CLK),
   .RST(RSTag), 
   // connected to control interface
   .addr_Param_in(ag_CI_wireR[0+:8]),
   .Param_in(ag_CI_wireR[8+:NM]),
   .wen_in(ag_CI_wireR[8+NM]),
   // connected to qaoa_system
   .en_Pipe_out(ag_info[0]),
   .en_CostF_out(ag_info[1]),
   .f_run_Computation_in(f_run_Computation),
   .f_L1Computation_out(ag_info[2]),
   .en_Inits_out(ag_info[5+:24]),
   .mixSwitch_out(ag_info[4:3]),
   .cAddrGC_out(ag_info[29+:NM]),
   // connected to bit_swap
   .cAddr_out(ag_cAddr),
   .bsp1_out(bsp1),
   .bsp2_out(bsp2)
);

qaoa_system2 #(.NM(NM), .P(P), .NBRAM(NBRAM),  .L_BRAM_R(L_BRAM_R)) qs2
(
   .CLK(CLK),
   .RST(RSTqs),
   
    //------------------------------------------------------------------------
    // COMMAND INTERFACE (among Control Interface and addr_gen)
    //------------------------------------------------------------------------
   .enPipe_in(bs_infoR[0]),
   .enCostF_in(bs_infoR[1]),
   .f_L1Computation_in(bs_infoR[2]),
   .en_Inits_in(bs_infoR[5+:24]),
   .mixSwitch_in(bs_infoR[4:3]),
   .f_run_Computation_out(f_run_Computation),
   
    //------------------------------------------------------------------------
    // Address INTERFACE (FOR MIXER, and COST FUNCTION GEN)
    //------------------------------------------------------------------------
   .swapped_cAddr_in(bswap_outR),
   .cAddrCF_in(bs_infoR[29+:NM]),
    //------------------------------------------------------------------------
    // MEMORY INTERFACE from Control Interface
    //------------------------------------------------------------------------

   .r_data(ci_r_data),
   .n_ci_addr_in(ci_addr),
   .n_r_req(ci_r_req),
   .r_vd(ci_r_vd),
   .n_w_data(ci_w_data),
   .n_w_req(ci_w_req),
   .r_CMD(ci_CMD),

    //------------------------------------------------------------------------
    //BRAM ARRAY INTERFACE (3 banks + 1 general bank)
    //------------------------------------------------------------------------
   .bram_addr_r(bram_addr_r),
   .bram_addr_w(bram_addr_w),
   .bram_data_r(bram_data_rP),
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
   
   .mix_ar_res(p_ar_oP),
   .mix_ai_res(p_ai_oP),
   .mix_info_res(mix_infoRP),
   .Status(rS),

    //------------------------------------------------------------------------
    // COST HAMILTONIAN INTERFACE
    //------------------------------------------------------------------------
   .gamma(gamma), 
   .HGC(HGC),
   .Hr_res(Hr_oP),
   .Hi_res(Hi_oP)


);

regPipeline  #(.W_INFO($bits({bs_info, bswap_out})), .NPipe(4)) P_bs_system (
   .CLK(CLK),
   .info_in({bs_info, bswap_out}),
   .info_out({bs_infoR, bswap_outR})
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

regPipeline  #(.W_INFO($bits({gamma, HGC, Hr_o, Hi_o})), .NPipe(LP_GEN_COST)) instp_gc_qs_inst (
   .CLK(CLK),
   .info_in({gamma, HGC, Hr_o, Hi_o}),
   .info_out({gammaP, HGCP, Hr_oP, Hi_oP})
);

gen_costFixedP  #(.P(P)) inst_gencostFixedP
  (
   .CLK(CLK), // input
   .gamma(gammaP), //  gamma, input
   .H(HGCP), // input
   .Hr_o(Hr_o),
   .Hi_o(Hi_o)
);

regPipeline  #(.W_INFO($bits({cosb, sinb, p_ar, p_ai, mix_switch, mix_info})), .NPipe(LP_MIXER_OUT)) inst_p_mix_in (
   .CLK(CLK),
   .info_in({cosb, sinb, p_ar, p_ai, mix_switch, mix_info}),
   .info_out({cosbP, sinbP, p_arP, p_aiP, mix_switchP, mix_infoP})
);
regPipeline  #(.W_INFO($bits({p_ar_o, p_ai_o, mix_infoR})), .NPipe(LP_MIXER_IN)) inst_p_mix_out (
   .CLK(CLK),
   .info_in({p_ar_o, p_ai_o, mix_infoR}),
   .info_out({p_ar_oP, p_ai_oP, mix_infoRP})
);

Update_mixer #(.P(P),.Ni(NM+1)) Umix // width of additional information
(
   .CLK(CLK),
   .cos_beta(cosbP), // cos beta
   .sin_beta(sinbP), // sin beta
   .p_r(p_arP),
   .p_i(p_aiP),
   .p_r_o(p_ar_o),
   .p_i_o(p_ai_o),
   .switch_in(mix_switchP),
   .info_in(mix_infoP), // information, like addresses, enabled signal, and so on.
   .info_out(mix_infoR) // information, like addresses, enabled signal, and so on.
);

genvar j;

generate
    for (j = 0; j < NBRAM; j = j + 1) begin : GEN_BRAM
         
         regPipeline  #(.W_INFO(P), .NPipe(LP_BRAM_D)) P_BRAMD (
            .CLK(CLK),
            .info_in({bram_data_r[j]}),
            .info_out({bram_data_rP[j]})
         );
         regPipeline  #(.W_INFO(NM + NM + 1+ P), .NPipe(LP_BRAM_A)) P_BRAMR (
            .CLK(CLK),
            .info_in({bram_addr_r[j], bram_addr_w[j], bram_wen[j], bram_data_w[j]}),
            .info_out({bram_addr_rP[j], bram_addr_wP[j], bram_wenP[j], bram_data_wP[j]})
         );
         ram RAM (.address_a(bram_addr_rP[j]), // NM bit address
            .address_b(bram_addr_wP[j]),
            .clock(CLK),
            .data_a(), // 64 bit
            .data_b(bram_data_wP[j]),
            .wren_a(),
            .wren_b(bram_wenP[j]),
            .q_a(bram_data_r[j]),
            .q_b()
            );
    end
endgenerate

endmodule
