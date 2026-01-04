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

module top1_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;
wire rx_dv;
wire [7:0] rx_data_out;
reg [7:0] tx_data_in;
reg tx_dv;
wire [31:0] o_Status;
reg tx_OK;
parameter CLOCKWIDTH = 2;
parameter CLOCKWIDTH_HALF = 1;
parameter CLOCKWIDTH500 = CLOCKWIDTH*50;
parameter CMD_WAIT = CLOCKWIDTH*20;
parameter TIME_WAIT_TB = CLOCKWIDTH*50;
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

parameter OP_NONE = 0; // Send: 1, Res: 0.
parameter OP_SEND1T = 1; // Send: 0, Res: 1.
parameter OP_SEND8T = 2; // Send: 0, Res: 1.
parameter OP_MOV_T2A = 3; // Send: 1, Res: 0.
parameter OP_MOV_T2B = 4; // Send: 1, Res: 0.
parameter OP_MOV_A2U = 5; // Send: 1, Res: 0.
parameter OP_MOV_A2B = 6; // Send: 0, Res: 1.
parameter OP_FETCH1U = 60; // Send: 0, Res: 8.
parameter OP_FETCH8U = 61; // Send: 0, Res: 8.
parameter OP_ADD_B2A = 80; // Send: 0, Res: 0.
parameter OP_MUL_B2A = 81; // Send: 0, Res: 0.
parameter OP_ADDFP_B2A = 82; // Send: 0, Res: 0.
parameter OP_MULFP_B2A = 83; // Send: 0, Res: 0.
parameter OP_INC_A = 84; // Send: 0, Res: 0.
parameter OP_WRITE_T2RAM = 111; // Send: 0, Res: 0.
parameter OP_READ_RAM2U = 112; // Send: 0, Res: 0.
parameter OP_SEND_CMD = 118; // Send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv
parameter OP_WAIT_TB = 212; // Send: 0, Res: 0. see qa_INIT, qa_WAIT, qa_RUN in qaoa_system.sv

parameter qa_WAIT = 1;
parameter qa_RUN = 2;
parameter qa_MIXER = 4;
parameter qa_COST = 8;
parameter qa_INIT = 16;

reg [63:0] fp64;
reg [63:0] None8;
reg [63:0] fp64rx = 0;
reg [7:0] recState;
reg [7:0] recCount;
integer i;
always @(posedge CLK) begin
      if(recState == 1) begin
            if(recCount != 8)begin
                  if(rx_dv) begin
                        fp64rx[recCount*8+:8] <=rx_data_out;
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
logic [7: 0] data_array [];
initial begin 
      tx_dv <= 0;
      tx_OK <= 1;
      tx_data_in <= 0;
      recCount <= 0;
      recState <= 0;

      fp64 = 64'b0100000000010000100110001001001101110100101111000110101001111111; // 4.149
      None8 = 64'd0;
      data_array = 
      {
8'h01,
8'h0c,
8'h03,
8'h05,
8'h3c,
8'h06,
8'h50,
8'h05,
8'h3c,
8'h51,
8'h05,
8'h3c,
8'h01,
8'h01,
8'h76,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h01,
8'h03,
8'h70,
8'h3d,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h02,
8'h03,
8'h70,
8'h3d,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h04,
8'h03,
8'h02,
8'h9a, 8'he3, 8'h81, 8'h6d, 8'h69, 8'h9a, 8'hea, 8'h3f,
8'h6f,
8'h70,
8'h3d,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h08,
8'h03,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h24, 8'h40,
8'h6f,
8'h54,
8'h02,
8'h9a, 8'h99, 8'h99, 8'h99, 8'h99, 8'h99, 8'hb9, 8'hbf,
8'h6f,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h10,
8'h03,
8'h02,
8'h50, 8'h74, 8'h5d, 8'hf8, 8'hc1, 8'h29, 8'h12, 8'h40,
8'h6f,
8'h54,
8'h02,
8'hb5, 8'h01, 8'hd8, 8'h80, 8'h08, 8'h01, 8'h13, 8'h40,
8'h6f,
8'h54,
8'h02,
8'h1a, 8'h8f, 8'h52, 8'h09, 8'h4f, 8'hd8, 8'h13, 8'h40,
8'h6f,
8'h54,
8'h02,
8'h7f, 8'h1c, 8'hcd, 8'h91, 8'h95, 8'haf, 8'h14, 8'h40,
8'h6f,
8'h54,
8'h6f,
8'h54,
8'h6f,
8'h54,
8'h6f,
8'h54,
8'h6f,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h20,
8'h03,
8'h02,
8'hae, 8'hc4, 8'h3c, 8'h2b, 8'h69, 8'h35, 8'h17, 8'h40,
8'h6f,
8'h54,
8'h02,
8'hae, 8'hc4, 8'h3c, 8'h2b, 8'h69, 8'h35, 8'h17, 8'h40,
8'h6f,
8'h54,
8'h02,
8'h27, 8'h00, 8'hff, 8'h94, 8'h2a, 8'h21, 8'h18, 8'h40,
8'h6f,
8'h54,
8'h02,
8'h78, 8'hdf, 8'h31, 8'h3c, 8'hf6, 8'he3, 8'h18, 8'h40,
8'h6f,
8'h54,
8'h02,
8'hdd, 8'h6c, 8'hac, 8'hc4, 8'h3c, 8'hbb, 8'h19, 8'h40,
8'h6f,
8'h01,
8'h10,
8'h76,
8'h01,
8'h02,
8'h76,
8'h00,
8'h01,
8'h01,
8'h76,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h10,
8'h03,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h02,
8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h00, 8'h20,
8'h03,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d,
8'h54,
8'h70,
8'h3d}

;

      #CMD_WAIT;
      #CLOCKWIDTH500;

      for (i = 0; i <data_array.size() ; i = i + 1) begin
            tx_data_in <= data_array[i];
            if(data_array[i] == OP_WAIT_TB) begin
                  #TIME_WAIT_TB;
            end
            else if(data_array[i] == OP_FETCH8U) begin
                  recCount <= 0;
                  recState <= 1;
                  tx_dv <= 1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
                  #CMD_WAIT;
            end
            else begin
                  tx_dv <= 1;
                  #CLOCKWIDTH;
                  tx_dv <= 0;
                  #CMD_WAIT;
            end
      end
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
      #CMD_WAIT;
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
