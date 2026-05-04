`timescale 10ns / 10ns
// program, split uart functionality to reduce the simulation time
// employment of block design.
// implement all brams.

// todo
// test transmission
// todo 2026, 1, 1.
// set maxAddr, using 64
// set NQbits, using 65
// test mixer, we need theoretical value for this test bench. I need to write some python code

module top1cmd_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;
wire rx_dv;
wire [7:0] rx_data_out;
reg [7:0] tx_data_in;
reg tx_dv;
wire [31:0] o_Status;
reg tx_OK;
localparam CLOCKWIDTH      = 2;
localparam CLOCKWIDTH_HALF = 1;
localparam CLOCKWIDTH500   = CLOCKWIDTH*50;
localparam CMD_WAIT        = CLOCKWIDTH*20;
localparam TIME_WAIT_TB    = CLOCKWIDTH*50;
localparam TIME_WAIT_QAOA  = CLOCKWIDTH*100;
localparam N_MAX_WAIT      = 512;
// we need transmitter and receiver to tset state machine (ntu_smachine)
top1 top 
(
   .CLK(CLK),        // Connect to your system clock wire
   .RST(RST),        // Connect to your system reset wire
   .tx_OK(tx_OK),
   .tx_en(rx_dv),
   .tx_data_out(rx_data_out),
   .rx_data_in(tx_data_in),
   .rx_dv(tx_dv),
   .o_Status(o_Status)         // Connect to the wire for the received byte
);


always
begin
      #CLOCKWIDTH_HALF;
      CLK <= ~CLK; // clock generation, half period
end

localparam OP_NONE = 0; // Send: 1, Res: 0.
localparam OP_SEND1T = 1; // Send: 0, Res: 1.
localparam OP_SEND8T = 2; // Send: 0, Res: 1.
localparam OP_MOV_T2A = 3; // Send: 1, Res: 0.
localparam OP_MOV_T2B = 4; // Send: 1, Res: 0.
localparam OP_MOV_A2U = 5; // Send: 1, Res: 0.
localparam OP_MOV_A2B = 6; // Send: 0, Res: 1.
localparam OP_MOV_Info2U = 8'd7;   // send firmware version info
localparam OP_MOV_S2U    = 8'd8;   // status value to rU.
localparam OP_FETCH1U = 60; // Send: 0, Res: 8.
localparam OP_FETCH8U = 61; // Send: 0, Res: 8.
localparam OP_ADD_B2A = 80; // Send: 0, Res: 0.
localparam OP_MUL_B2A = 81; // Send: 0, Res: 0.
localparam OP_ADDFP_B2A = 82; // Send: 0, Res: 0.
localparam OP_MULFP_B2A = 83; // Send: 0, Res: 0.
localparam OP_INC_A = 84; // Send: 0, Res: 0.
localparam OP_WRITE_T2RAM = 111; // Send: 0, Res: 0.
localparam OP_READ_RAM2U = 112; // Send: 0, Res: 0.
localparam OP_SEND_CMD = 118; // Send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv
localparam HOST_WAIT  = 254;  // 

localparam qa_WAIT = 1;
localparam qa_RUN = 2;
localparam qa_MIXER = 4;
localparam qa_COST = 8;
localparam qa_INIT = 16;

integer fd;
reg [63:0] fp64;
reg [63:0] None8;
reg [63:0] fp64rx = 0;
reg [7:0] recState;
reg [7:0] recCount;
reg [7:0] txCount;
integer i;
integer j;
integer k;
always @(posedge CLK) begin
      if(recState == 1) begin
            if(recCount != 8)begin
                  if(rx_dv) begin
                        fp64rx[recCount*8+:8] <= rx_data_out;
                        recCount <= recCount + 1;
                  end
            end
            else begin
                  recCount <= 0;
                  recState <= 0;
            end
      end
end
// test for transmitter
localparam ND=1912; logic [7: 0] data_array [1912] = {8'h01,8'h0c,8'h03,8'h05,8'h3c,8'h01,8'h01,8'h76,8'h01,8'h00,8'h03,8'h02,8'h1e, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h03,8'h03,8'h02,8'h53, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h01,8'h03,8'h02,8'he1, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h02,8'h03,8'h02,8'h19, 8'h01, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h07,8'h03,8'h02,8'h1a, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h04,8'h03,8'h02,8'h08, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h05,8'h03,8'h02,8'h04, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h01,8'h06,8'h03,8'h02,8'h0f, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h77,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h08,8'h03,8'h02,8'hb8, 8'h35, 8'hc4, 8'h72, 8'hb7, 8'h8a, 8'hed, 8'hbf,8'h6f,8'h54,8'h02,8'hea, 8'h8d, 8'hea, 8'hd4, 8'h5b, 8'h99, 8'hd8, 8'hbf,8'h6f,8'h54,8'h02,8'h50, 8'h2b, 8'h00, 8'h0a, 8'hfe, 8'h32, 8'hd3, 8'h3f,8'h6f,8'h54,8'h02,8'hb8, 8'h35, 8'hc4, 8'h72, 8'hb7, 8'h8a, 8'hed, 8'hbf,8'h6f,8'h54,8'h02,8'hea, 8'h8d, 8'hea, 8'hd4, 8'h5b, 8'h99, 8'hd8, 8'hbf,8'h6f,8'h54,8'h02,8'ha3, 8'hc4, 8'h2a, 8'h3c, 8'h09, 8'h23, 8'h06, 8'hc0,8'h6f,8'h54,8'h02,8'hc1, 8'h10, 8'hfe, 8'h1a, 8'h22, 8'h7c, 8'hd1, 8'hbf,8'h6f,8'h54,8'h02,8'hd6, 8'h3d, 8'h5f, 8'hb1, 8'h59, 8'hc8, 8'hee, 8'hbf,8'h6f,8'h54,8'h02,8'hd0, 8'h2b, 8'h01, 8'hb3, 8'hc2, 8'h22, 8'hf2, 8'h3f,8'h6f,8'h54,8'h02,8'hdb, 8'hb5, 8'hca, 8'h1a, 8'hbb, 8'hbe, 8'hec, 8'h3f,8'h6f,8'h54,8'h02,8'h3d, 8'h31, 8'hd7, 8'hc7, 8'h6d, 8'h1f, 8'hdc, 8'hbf,8'h6f,8'h54,8'h02,8'h4c, 8'h23, 8'hb3, 8'h48, 8'h18, 8'haf, 8'hf2, 8'hbf,8'h6f,8'h54,8'h02,8'h0c, 8'h4d, 8'h71, 8'he7, 8'h6e, 8'h7c, 8'heb, 8'h3f,8'h6f,8'h54,8'h02,8'h8f, 8'h1b, 8'he3, 8'h1c, 8'hef, 8'h62, 8'he0, 8'h3f,8'h6f,8'h54,8'h02,8'h78, 8'h19, 8'h65, 8'hb7, 8'h5f, 8'hd3, 8'hd2, 8'hbf,8'h6f,8'h54,8'h02,8'h00, 8'hc9, 8'h4a, 8'he5, 8'h1d, 8'hb0, 8'hd3, 8'h3f,8'h6f,8'h54,8'h02,8'h28, 8'hba, 8'h2f, 8'hb1, 8'hc0, 8'h72, 8'hee, 8'hbf,8'h6f,8'h54,8'h02,8'h18, 8'hdc, 8'h73, 8'hfc, 8'h20, 8'h98, 8'hfd, 8'h3f,8'h6f,8'h54,8'h02,8'hff, 8'h41, 8'h4d, 8'h71, 8'h48, 8'h29, 8'hd4, 8'h3f,8'h6f,8'h54,8'h02,8'hc2, 8'hea, 8'h49, 8'hc6, 8'he7, 8'h5e, 8'hee, 8'h3f,8'h6f,8'h54,8'h02,8'h7d, 8'h6e, 8'hb1, 8'hdc, 8'he9, 8'hb9, 8'hf9, 8'hbf,8'h6f,8'h54,8'h02,8'hbd, 8'h56, 8'h81, 8'h4d, 8'h8b, 8'h90, 8'hec, 8'h3f,8'h6f,8'h54,8'h02,8'h15, 8'hb2, 8'hbc, 8'hba, 8'h45, 8'hd9, 8'hdc, 8'h3f,8'h6f,8'h54,8'h02,8'h50, 8'h36, 8'hf3, 8'h9f, 8'ha4, 8'h43, 8'hc4, 8'h3f,8'h6f,8'h54,8'h02,8'h8e, 8'h10, 8'hbe, 8'hf2, 8'h9e, 8'ha5, 8'he6, 8'hbf,8'h6f,8'h54,8'h02,8'h59, 8'h03, 8'hec, 8'hbe, 8'h9c, 8'h9b, 8'he6, 8'h3f,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'hf0, 8'hbf,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h10,8'h03,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'hf0, 8'h3f,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h20,8'h03,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00,8'h6f,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h04,8'h03,8'h02,8'h24, 8'h8b, 8'hd5, 8'he0, 8'ha5, 8'h8c, 8'hd6, 8'hbf,8'h6f,8'h54,8'h02,8'hc8, 8'h66, 8'hd7, 8'hb4, 8'h7c, 8'h58, 8'he6, 8'hbf,8'h6f,8'h54,8'h02,8'hfc, 8'hc7, 8'hc5, 8'h20, 8'hd2, 8'h51, 8'hd3, 8'h3f,8'h6f,8'h54,8'h02,8'hc0, 8'h07, 8'h6e, 8'h31, 8'h34, 8'h5d, 8'heb, 8'hbf,8'h6f,8'h54,8'h02,8'h00, 8'h3a, 8'h06, 8'h46, 8'h20, 8'h5f, 8'hb2, 8'h3f,8'h6f,8'h54,8'h02,8'h7c, 8'he6, 8'hac, 8'h06, 8'h1b, 8'h31, 8'hd1, 8'hbf,8'h6f,8'h54,8'h02,8'h14, 8'hed, 8'hb8, 8'he0, 8'hbe, 8'h49, 8'hec, 8'hbf,8'h6f,8'h54,8'h02,8'h80, 8'heb, 8'hec, 8'h6d, 8'hee, 8'h74, 8'h8e, 8'h3f,8'h6f,8'h54,8'h02,8'he6, 8'hfa, 8'h4f, 8'hcf, 8'hab, 8'h99, 8'hed, 8'hbf,8'h6f,8'h54,8'h02,8'h70, 8'h55, 8'h9e, 8'hb2, 8'h98, 8'hfc, 8'hc0, 8'hbf,8'h6f,8'h54,8'h02,8'h48, 8'h1f, 8'h13, 8'h1e, 8'h7d, 8'h87, 8'heb, 8'hbf,8'h6f,8'h54,8'h02,8'h8a, 8'h74, 8'h97, 8'h0b, 8'hc2, 8'h31, 8'hea, 8'hbf,8'h6f,8'h54,8'h02,8'h50, 8'hc4, 8'h1b, 8'hde, 8'hb5, 8'h52, 8'hc3, 8'hbf,8'h6f,8'h54,8'h02,8'h96, 8'h0c, 8'h86, 8'h2c, 8'h25, 8'heb, 8'he4, 8'h3f,8'h6f,8'h54,8'h02,8'h7e, 8'h9f, 8'h6b, 8'hf0, 8'ha0, 8'h13, 8'he8, 8'hbf,8'h6f,8'h54,8'h02,8'h68, 8'h7a, 8'hf4, 8'hea, 8'h73, 8'hb6, 8'he1, 8'hbf,8'h6f,8'h54,8'h02,8'h8c, 8'hf4, 8'h53, 8'h59, 8'hbb, 8'h4f, 8'hd0, 8'h3f,8'h6f,8'h54,8'h02,8'h86, 8'hb1, 8'h7e, 8'h68, 8'h43, 8'ha7, 8'hec, 8'h3f,8'h6f,8'h54,8'h02,8'h58, 8'hc1, 8'hbc, 8'hd2, 8'h04, 8'hbd, 8'hc3, 8'h3f,8'h6f,8'h54,8'h02,8'hb0, 8'hc3, 8'h69, 8'hfe, 8'h25, 8'h73, 8'hca, 8'hbf,8'h6f,8'h54,8'h02,8'h8c, 8'hc6, 8'hc4, 8'hb1, 8'hf6, 8'h7a, 8'hee, 8'h3f,8'h6f,8'h54,8'h02,8'h48, 8'hf3, 8'h8b, 8'h13, 8'hca, 8'h04, 8'hed, 8'hbf,8'h6f,8'h54,8'h02,8'hf4, 8'hbd, 8'h10, 8'hb1, 8'h25, 8'hf1, 8'he6, 8'h3f,8'h6f,8'h54,8'h02,8'hdc, 8'h34, 8'h4b, 8'h39, 8'h15, 8'hee, 8'hda, 8'hbf,8'h6f,8'h54,8'h02,8'he2, 8'hac, 8'hac, 8'h53, 8'h86, 8'hc4, 8'he6, 8'hbf,8'h6f,8'h54,8'h02,8'hce, 8'hc4, 8'h6e, 8'h8b, 8'h17, 8'h76, 8'he8, 8'hbf,8'h6f,8'h54,8'h02,8'hac, 8'h2f, 8'h09, 8'he7, 8'haa, 8'h83, 8'hd8, 8'hbf,8'h6f,8'h54,8'h02,8'hd4, 8'hbd, 8'h74, 8'h0d, 8'h6a, 8'h3b, 8'he4, 8'h3f,8'h6f,8'h54,8'h02,8'h22, 8'hf1, 8'h2c, 8'h9f, 8'hfa, 8'h6e, 8'he4, 8'hbf,8'h6f,8'h54,8'h02,8'h68, 8'h4e, 8'h47, 8'h92, 8'hbf, 8'he3, 8'hc4, 8'h3f,8'h6f,8'h54,8'h02,8'h0c, 8'h85, 8'h01, 8'ha3, 8'hea, 8'hc7, 8'hd1, 8'h3f,8'h6f,8'h54,8'h02,8'ha4, 8'hec, 8'h70, 8'hfe, 8'h46, 8'h55, 8'hd0, 8'hbf,8'h6f,8'h54,8'h01,8'h02,8'h76,8'hfe,8'h01,8'h01,8'h76,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h10,8'h03,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h20,8'h03,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h04,8'h03,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h02,8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h08,8'h03,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54,8'h70,8'h3d,8'h54};

initial begin 
      tx_dv <= 0;
      tx_OK <= 1;
      tx_data_in <= 0;
      recCount <= 0;
      recState <= 0;
      txCount <=  0;

      fp64 = 64'b0100000000010000100110001001001101110100101111000110101001111111; // 4.149
      None8 = 64'd0;

      #CMD_WAIT;
      #CLOCKWIDTH500;
      
      fd= $fopen("Sample.txt", "w");

      for (i = 0; i <ND ; i = i + 1) begin
            tx_data_in <= data_array[i];
            // $fdisplay(fd, "Processing %d -th, data= %d", i, data_array[i]);
            if(txCount != 0) begin 
                  // need to wait until the data flushed.
                  tx_dv <= 1;
                  txCount <= txCount -1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;
            end
            else if(data_array[i] == OP_SEND8T) begin 
                  txCount <= 8;
                  tx_dv <= 1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;
            end
            else if(data_array[i] == OP_SEND1T) begin 
                  txCount <= 1;
                  tx_dv <= 1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;

            end
            else if(data_array[i] == HOST_WAIT) begin
                  for(j = 0; j < N_MAX_WAIT; j = j + 1) begin
                        tx_data_in <= OP_MOV_S2U;
                        tx_dv <= 1;
                        # CLOCKWIDTH;
                        tx_dv <= 0;
                        # CMD_WAIT;
                        recCount <= 0;
                        recState <= 1;
                        tx_dv <= 1;
                        tx_data_in <= OP_FETCH8U;
                        # CLOCKWIDTH;
                        tx_dv <= 0;
                        for(k = 0; k < N_MAX_WAIT; k = k + 1) begin
                              #CMD_WAIT;
                              if(k == N_MAX_WAIT -1) begin 
                                    $fwrite(fd, "waiting time out %d\n", k);
                              end
                              if(recState == 0) begin 
                                    break;
                              end 
                        end

                        if(fp64rx[7:0] == qa_WAIT) begin
                              break;
                        end
                        # CMD_WAIT;
                  end
            end
            else if(data_array[i] == OP_FETCH8U) begin
                  recCount <= 0;
                  recState <= 1;
                  tx_dv <= 1;
                  # CLOCKWIDTH;
                  tx_dv <= 0;
                  for(k = 0; k < N_MAX_WAIT; k = k + 1) begin
                        #CMD_WAIT;
                        if(k == N_MAX_WAIT -1) begin 
                              $fwrite(fd, "waiting time out %d\n", k);
                        end
                        if(recState == 0) begin 
                              break;
                        end 
                  end
                  $fwrite(fd, "%f\n", $bitstoreal(fp64rx));
            end
            else begin
                  tx_dv <= 1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;
            end
      end


      $fclose(fd);

      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      $fwrite(fd, "finished the computation\n");
      $stop;
end
// Initialize signals
initial begin
      RST <= 1; // Reset active-high
      CLK <= 0;
      // Apply reset
      #CMD_WAIT;
      RST <= 0;


      // End simulation
end

endmodule
