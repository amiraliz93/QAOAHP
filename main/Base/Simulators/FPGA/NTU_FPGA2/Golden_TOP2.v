// ============================================================================
// Copyright (c) 2016 by Terasic Technologies Inc.
// ============================================================================
//
// Permission:
//
//   Terasic grants permission to use and modify this code for use
//   in synthesis for all Terasic Development Boards and Altera Development 
//   Kits made by Terasic.  Other use of this code, including the selling 
//   ,duplication, or modification of any portion is strictly prohibited.
//
// Disclaimer:
//
//   This VHDL/Verilog or C/C++ source code is intended as a design reference
//   which illustrates how these types of functions can be implemented.
//   It is the user's responsibility to verify their design for
//   consistency and functionality through the use of formal
//   verification methods.  Terasic provides no warranty regarding the use 
//   or functionality of this code.
//
// ============================================================================
//           
//  Terasic Technologies Inc
//  9F., No.176, Sec.2, Gongdao 5th Rd, East Dist, Hsinchu City, 30070. Taiwan
//  
//  
//                     web: http://www.terasic.com/  
//                     email: support@terasic.com
//
// ============================================================================
//Date:  Thur Jul 07 14:19:17 2016
// ============================================================================

//`define ENABLE_DDR3
//`define ENABLE_PCIE
//`define ENABLE_SATA
//`define ENABLE_FMCA_XCVR
//`define ENABLE_FMCB_XCVR
//`define ENABLE_FMCC_XCVR
//`define ENABLE_FMCD_XCVR

module Golden_TOP3(

      ///////// CLOCK /////////
      input              OSC_MAIN,
      input              OSC_50_B4A,
      input              OSC_50_B4D,
      input              OSC_50_B7A,
      input              OSC_50_B7D,
      input              OSC_50_B8A,
      input              OSC_50_B8D,

      ///////// KEY /////////
      input              CPU_RESET_n,
      input    [ 3: 0]   BUTTON, //KEY is Low-Active

      ///////// SW /////////
      input    [ 3: 0]   SW,

      ///////// LED /////////
      output   [ 3: 0]   LED, //LED is Low-Active

      ///////// FAN /////////
      input              FAN_ALERT_n,

      ///////// SSRAM /////////
      output             SSRAM_CLK,
      output             SSRAM_CKE_n,
      output             SSRAM_CE_n,
      output             SSRAM_WE_n,
      output             SSRAM_OE_n,
      output             SSRAM_ADV,
      output             SSRAM_BWA_n,
      output             SSRAM_BWB_n,

      ///////// FLASH /////////
      output             FLASH_CLK,
      output             FLASH_CE_n,
      output             FLASH_WE_n,
      output             FLASH_OE_n,
      output             FLASH_ADV_n,
      output             FLASH_RESET_n,
      input              FLASH_RDY_BSY_n,

      ///////// FSM /////////
      output   [26: 1]   FSM_A,
      inout    [15: 0]   FSM_D,

      ///////// SD Card /////////
      output             SD_CLK,
      inout    [ 3: 0]   SD_DATA,
      inout              SD_CMD,

`ifdef ENABLE_DDR3
      ///////// DDR3 /////////
      input              DDR3_REFCLK_p,
      output   [15: 0]   DDR3_A,
      output   [ 2: 0]   DDR3_BA,
      output   [ 1: 0]   DDR3_CK,
      output   [ 1: 0]   DDR3_CK_n,
      output   [ 1: 0]   DDR3_CKE,
      inout    [ 7: 0]   DDR3_DQS,
      inout    [ 7: 0]   DDR3_DQS_n,
      inout    [63: 0]   DDR3_DQ,
      output   [ 7: 0]   DDR3_DM,
      output   [ 1: 0]   DDR3_CS_n,
      output             DDR3_WE_n,
      output             DDR3_CAS_n,
      output             DDR3_RAS_n,
      output             DDR3_RESET_n,
      output   [ 1: 0]   DDR3_ODT,
      input              DDR3_EVENT_n,
      output             DDR3_SCL,
      inout              DDR3_SDA,
`endif /*ENABLE_DDR3*/

      ///////// RZQ /////////
      input              RZQ_DDR3,
      input              RZQ_FMC,

      ///////// Uart to USB /////////
      output             UART_TX,
      input              UART_RX,

      ///////// TPS40422 /////////
      output             TPS40422_CLK,
      inout              TPS40422_DATA,
      input              TPS40422_ALERT,

      ///////// External PLL /////////
      output             LMK04906_CLK,
      output             LMK04906_DATAIN,
      input              LMK04906_DATAOUT,
      output             LMK04906_LE,

      ///////// I2C /////////
      output             CLOCK_SCL,
      inout              CLOCK_SDA,

      ///////// Shared I2C /////////
      inout              FPGA_I2C_SCL,
      inout              FPGA_I2C_SDA,

      ///////// Temperature /////////
      input              TEMP_INT_n,
      input              TEMP_OVERT_n,

      ///////// POWER Monitor /////////
      input              POWER_MONITOR_ALERT,

      ///////// GPIO /////////
      inout    [35: 0]   GPIO,

      ///////// FMCA /////////
      input    [ 1: 0]   FMCA_CLK_M2C_p,
      input    [ 1: 0]   FMCA_CLK_M2C_n,
      input              FMCA_HA_RX_CLK_p,
      input              FMCA_HA_RX_CLK_n,
      output             FMCA_HA_TX_CLK_p,
      output             FMCA_HA_TX_CLK_n,
      input              FMCA_HB_RX_CLK_p,
      input              FMCA_HB_RX_CLK_n,
      output             FMCA_HB_TX_CLK_p,
      output             FMCA_HB_TX_CLK_n,
      input              FMCA_LA_RX_CLK_p,
      input              FMCA_LA_RX_CLK_n,
      output             FMCA_LA_TX_CLK_p,
      output             FMCA_LA_TX_CLK_n,
      inout    [10: 0]   FMCA_HA_TX_p,
      inout    [10: 0]   FMCA_HA_TX_n,
      inout    [10: 0]   FMCA_HA_RX_p,
      inout    [10: 0]   FMCA_HA_RX_n,
      inout    [10: 0]   FMCA_HB_TX_p,
      inout    [10: 0]   FMCA_HB_TX_n,
      inout    [10: 0]   FMCA_HB_RX_p,
      inout    [10: 0]   FMCA_HB_RX_n,
      inout    [16: 0]   FMCA_LA_TX_p,
      inout    [16: 0]   FMCA_LA_TX_n,
      inout    [14: 0]   FMCA_LA_RX_p,
      inout    [14: 0]   FMCA_LA_RX_n,

`ifdef ENABLE_FMCA_XCVR	
      input    [ 1: 0]   FMCA_GBTCLK_M2C_p,
      input    [ 1: 0]   FMCA_ONBOARD_REFCLK_p,
      output   [ 9: 0]   FMCA_DP_C2M_p,
      input    [ 9: 0]   FMCA_DP_M2C_p,
`endif /*ENABLE_FMCA_XCVR*/

      output   [ 1: 0]   FMCA_GA,
      inout              FMCA_SCL,
      inout              FMCA_SDA,

      ///////// FMCB /////////
      input    [ 1: 0]   FMCB_CLK_M2C_p,
      input    [ 1: 0]   FMCB_CLK_M2C_n,
      input              FMCB_LA_RX_CLK_p,
      input              FMCB_LA_RX_CLK_n,
      output             FMCB_LA_TX_CLK_p,
      output             FMCB_LA_TX_CLK_n,
      inout    [16: 0]   FMCB_LA_TX_p,
      inout    [16: 0]   FMCB_LA_TX_n,
      inout    [14: 0]   FMCB_LA_RX_p,
      inout    [14: 0]   FMCB_LA_RX_n,

`ifdef ENABLE_FMCB_XCVR	
      input    [ 0: 0]   FMCB_GBTCLK_M2C_p,
      input    [ 0: 0]   FMCB_ONBOARD_REFCLK_p,
      output   [ 0: 0]   FMCB_DP_C2M_p,
      input    [ 0: 0]   FMCB_DP_M2C_p,
`endif /*ENABLE_FMCB_XCVR*/

      output   [ 1: 0]   FMCB_GA,
      inout              FMCB_SCL,
      inout              FMCB_SDA,

      ///////// FMCC /////////
      input    [ 1: 0]   FMCC_CLK_M2C_p,
      input    [ 1: 0]   FMCC_CLK_M2C_n,
      input              FMCC_LA_RX_CLK_p,
      input              FMCC_LA_RX_CLK_n,
      output             FMCC_LA_TX_CLK_p,
      output             FMCC_LA_TX_CLK_n,
      inout    [16: 0]   FMCC_LA_TX_p,
      inout    [16: 0]   FMCC_LA_TX_n,
      inout    [14: 0]   FMCC_LA_RX_p,
      inout    [14: 0]   FMCC_LA_RX_n,

`ifdef ENABLE_FMCC_XCVR	
      input    [ 0: 0]   FMCC_GBTCLK_M2C_p,
      input    [ 0: 0]   FMCC_ONBOARD_REFCLK_p,
      output   [ 0: 0]   FMCC_DP_C2M_p,
      input    [ 0: 0]   FMCC_DP_M2C_p,
`endif /*ENABLE_FMCC_XCVR*/

      output   [ 1: 0]   FMCC_GA,
      inout              FMCC_SCL,
      inout              FMCC_SDA,

      ///////// FMCD /////////
      input    [ 1: 0]   FMCD_CLK_M2C_p,
      input    [ 1: 0]   FMCD_CLK_M2C_n,
      input              FMCD_HA_RX_CLK_p,
      input              FMCD_HA_RX_CLK_n,
      output             FMCD_HA_TX_CLK_p,
      output             FMCD_HA_TX_CLK_n,
      input              FMCD_HB_RX_CLK_p,
      input              FMCD_HB_RX_CLK_n,
      output             FMCD_HB_TX_CLK_p,
      output             FMCD_HB_TX_CLK_n,
      input              FMCD_LA_RX_CLK_p,
      input              FMCD_LA_RX_CLK_n,
      output             FMCD_LA_TX_CLK_p,
      output             FMCD_LA_TX_CLK_n,
      inout    [10: 0]   FMCD_HA_TX_p,
      inout    [10: 0]   FMCD_HA_TX_n,
      inout    [10: 0]   FMCD_HA_RX_p,
      inout    [10: 0]   FMCD_HA_RX_n,
      inout    [10: 0]   FMCD_HB_TX_p,
      inout    [10: 0]   FMCD_HB_TX_n,
      inout    [10: 0]   FMCD_HB_RX_p,
      inout    [10: 0]   FMCD_HB_RX_n,
      inout    [16: 0]   FMCD_LA_TX_p,
      inout    [16: 0]   FMCD_LA_TX_n,
      inout    [14: 0]   FMCD_LA_RX_p,
      inout    [14: 0]   FMCD_LA_RX_n,
		
`ifdef  ENABLE_FMCD_XCVR
      input    [ 1: 0]   FMCD_GBTCLK_M2C_p,
      input    [ 1: 0]   FMCD_ONBOARD_REFCLK_p,
      output   [ 9: 0]   FMCD_DP_C2M_p,
      input    [ 9: 0]   FMCD_DP_M2C_p,
`endif /*ENABLE_FMCD_XCVR*/	

      output   [ 1: 0]   FMCD_GA,
      inout              FMCD_SCL,
      inout              FMCD_SDA,


`ifdef ENABLE_PCIE
      ///////// PCIE /////////
      input              PCIE_ONBOARD_REFCLK_p,
      input              PCIE_REFCLK_p,
      output   [ 3: 0]   PCIE_TX_p,
      input    [ 3: 0]   PCIE_RX_p,
      input              PCIE_PERST_n,
      output             PCIE_WAKE_n,
`endif /*ENABLE_PCIE*/

`ifdef ENABLE_SATA
      ///////// SATA /////////
      input              SATA_HOST_REFCLK_p,
      output             SATA_HOST_TX_p,
      input              SATA_HOST_RX_p,
      input              SATA_DEVICE_REFCLK_p,
      output             SATA_DEVICE_TX_p,
      input              SATA_DEVICE_RX_p, 
`endif /*ENABLE_SATA*/


      ///////// SMA /////////
      input              SMA_CLKIN_p,
      output             SMA_CLKOUT_p

);


//=======================================================
//  REG/WIRE declarations
//=======================================================


wire [12:0] Speed_Switch;

//=======================================================
//  Structural coding
//=======================================================
assign  Speed_Switch = (SW[0]==0)?13'd2000:13'd5000;

wire CLK;
wire lockedclk;
wire clk50 = OSC_50_B4D;
wire clk2M;
wire RSTorg = ~CPU_RESET_n;
wire RST;
pll2 (
      .refclk(OSC_MAIN),   //  refclk.CLK
      .rst(RSTorg),      //   reset.reset
      .outclk_0(CLK), // outclk0.CLK
		.outclk_1(clk2M),
      .locked(lockedclk)    //  locked.export
);

// Combine the reset and locked signals
wire reset_or_not_locked = RSTorg | ~lockedclk;

// Two-stage synchronizer
reg sync_stage1;
reg sync_stage2;

always @(posedge CLK) begin
  sync_stage1 <= reset_or_not_locked;
  sync_stage2 <= sync_stage1;
end

// The final synchronized reset signal
assign RST = sync_stage2;

Fan_Control u0
(
 	.CLK(OSC_50_B4D),
 	.RST_N(BUTTON[0]),
   .Speed_Set(Speed_Switch),//need set more than 1500rpm
	.Speed_Detected(),
	.Alert_Clear(BUTTON[1]),
	.Alert_Type(),
	.Alert(FAN_ALERT_n),
	.FAN_I2C_SCL(FPGA_I2C_SCL),
	.FAN_I2C_SDA(FPGA_I2C_SDA)
);



reg [31:0] cnt;
integer i;

reg [16:0] dt;
always @(posedge CLK) begin
	if(RST) begin
		cnt <= 32'b0;
	end	
	else begin 
		cnt <= cnt + 1;
	end	
	if(cnt[25]) begin
		dt[0] <= cnt[28];
		for(i=1;i<16;i=i+1) begin
			dt[i+1] <= dt[i];
		end
	end
end

wire [31:0] o_Status;

// we cannnot use baud rate higher than 11520 in TR5's IP core of USB to UART. 
// If you want that, consider to use GPIO. 

// we need transmitter and receiver to tset state machine (ntu_smachine)
qaoa_system #(.UART_CLKS_PER_BIT(200000000/115200)) ntuS 
(
.CLK     (CLK),        // Connect to your system clock wire
.RST         (RST),        // Connect to your system reset wire
.i_Rx_Serial (UART_RX),      // Connect to the incoming serial data wire
.o_Tx_Serial     (UART_TX),  // Connect to the wire for the data valid signal
.o_Status   (o_Status)         // Connect to the wire for the received byte
);

assign LED[0] = ~o_Status[0];
assign LED[1] = ~o_Status[1];
assign LED[2] = ~o_Status[2];
assign LED[3] = ~cnt[25];

assign GPIO[7:0] = o_Status[15:8];
assign GPIO[15:8] = dt[7:0];

endmodule

