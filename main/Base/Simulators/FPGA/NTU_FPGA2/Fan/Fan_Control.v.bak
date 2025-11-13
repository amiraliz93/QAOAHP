module Fan_Control(CLK,RST_N,Speed_Set,Alert_Clear,
						 Speed_Detected,Alert_Type,
						 FAN_I2C_SCL,FAN_I2C_SDA,Alert
						 );		 
						 
input 		       CLK;
input 		       RST_N;
input       [12:0] Speed_Set;
input					 Alert_Clear;
input 				 Alert;
output   	[3:0]  Alert_Type;
output      [13:0] Speed_Detected;
output   			 FAN_I2C_SCL;
inout					 FAN_I2C_SDA;
reg	      [3:0]  FAN_INIT_INDEX;
reg	      [15:0] FAN_REG_DATA;
reg         [12:0] FAN_SPEED_RPM;

						
wire 		  			 i2c_reg_control_start ;
wire 		  			 wr_cmd ;
wire 			[6:0]  slave_addr ;
wire 			[7:0]  reg_addr ;
wire 			[7:0]  reg_wdata ;
wire 					 i2c_rdata_rdy ;
wire 			[7:0]  i2c_rdata ;
wire 			       i2c_cmd_finish ;

reg  			[4:0]  Count;
reg 		  			 CLK_2M;
reg  		   [12:0] FAN_RPM;
reg  			[12:0] FAN_RPS;
reg  			[7:0]  KTACH;

`define KSCALE 4

always@(posedge CLK or negedge RST_N)
begin
  if(!RST_N)
  begin
    Count <= 0;
  end
  else
  begin
    if(Count == 5'd24)
	 begin
	  Count<=0;
	  CLK_2M <= ~CLK_2M;
	 end
	 else
	  Count <= Count + 1;
  end
end

always@(posedge CLK)
begin 
  FAN_RPM <= Speed_Set;
  FAN_RPS <= FAN_RPM/60;
  KTACH   <= (12'd992 *`KSCALE / FAN_RPS)-1;
end 

I2C_Config u0
(
	 .iClk(CLK_2M), 
	 .iRst_n(RST_N),
	 .oStart(i2c_reg_control_start),
	 .oSlave_Addr(slave_addr),
	 .oWord_Addr(reg_addr),
	 .owdata(reg_wdata),
	 .owcmd(wr_cmd),
	 .Speed_Set(KTACH),
	 .Speed_Detected(Speed_Detected),
	 .Alert_Type(Alert_Type),
	 .Alert(Alert),
	 .Alert_Clear(Alert_Clear),
    .iReadData(i2c_rdata),
	 .iReadData_rdy(i2c_rdata_rdy),
	 .iCONFIG_DONE(i2c_cmd_finish)
);
			 
I2C_Bus_Controller u1 (
    .iCLK(CLK_2M),
    .iRST_n(RST_N),
    .iStart(i2c_reg_control_start),
    .iSlave_addr(slave_addr),
    .iWord_addr(reg_addr),
    .iSequential_read(1'b0),
    .iRead_length(8'd1),
    .i2c_clk(FAN_I2C_SCL),
    .i2c_data(FAN_I2C_SDA),
    .i2c_read_data(i2c_rdata),
    .i2c_read_data_rdy(i2c_rdata_rdy),
    .wr_data(reg_wdata),
    .wr_cmd(wr_cmd),
    .oCONFIG_DONE(i2c_cmd_finish)
    ) ;						 
						 
endmodule 