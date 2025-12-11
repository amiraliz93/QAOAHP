module I2C_CDCM6208_Config (	//	Host Side
						iCLK,
						iRST_N,
						
						iFREQ_SELECT,
						iFREQ_DISABLE,
						
						//	I2C Side
						I2C_SCLK,
						I2C_SDAT,
						
					   I2C_DONE
					   	);
//	Host Side
input		iCLK;
input		iRST_N;


input [1:0] iFREQ_SELECT; 
//Frequecy select:
// 0:  FMCA(100),FMCD(100),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)
// 1:  FMCA(125),FMCD(125),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)
// 2:  FMCA(150),FMCD(150),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)

input [5:0] iFREQ_DISABLE;
//Disable Bit:  0 enable ,1 Disable
// Bit5:  DDR3
// Bit4:  SATA_DEVICE
// Bit3:  SATA_HOST
// Bit2:  PCIE
// Bit1:  FMCD
// Bit0:  FMCA



//	I2C Side
inout  	I2C_SCLK;
inout		I2C_SDAT;

// result : just show i2c program acks
output   I2C_DONE;


//	Internal Registers/Wires
reg	[15:0]	mI2C_CLK_DIV;
reg	[39:0]	mI2C_DATA;
reg 			mI2C_CTRL_CLK;//freq *4
reg			mI2C_GO;
wire		mI2C_END;
wire		mI2C_ACK;
reg	[31:0]	LUT_DATA;
reg	[5:0]	LUT_INDEX;
reg	[3:0]	mSetup_ST;
reg   [15:0] delay_cnt;
reg   [15:0] delay_time;

//	Clock Setting
parameter	CLK_Freq	=	50000000;	//	50	MHz
parameter	I2C_Freq	=	20000;		//	20	KHz
//	LUT Data Number
parameter	LUT_SIZE	=	27;

assign  I2C_DONE = (LUT_INDEX == LUT_SIZE) ?1'b1:1'b0;

/////////////////////	I2C Control Clock	freq *4 ////////////////////////
always@(posedge iCLK or negedge iRST_N)
begin
	if(!iRST_N)
	begin
		mI2C_CTRL_CLK	<=	0;
		mI2C_CLK_DIV	<=	0;
	end
	else
	begin
		if( mI2C_CLK_DIV	< (CLK_Freq/I2C_Freq/8) )
		mI2C_CLK_DIV	<=	mI2C_CLK_DIV+1;
		else
		begin
			mI2C_CLK_DIV	<=	0;
			mI2C_CTRL_CLK	<=	~mI2C_CTRL_CLK;
		end
	end
end

////////////////////////////////////////////////////////////////////
I2C_CDCM6208_Controller 	u0	(	
                  .CLOCK(mI2C_CTRL_CLK),		//	Controller Work Clock => freq *4
						.I2C_SCLK(wI2C_SCLK),		//	I2C CLOCK
 	 	 	 	 	 	.I2C_SDAT(I2C_SDAT),		//	I2C DATA
						.I2C_DATA(mI2C_DATA),		//	DATA:[SLAVE_ADDR,SUB_ADDR,DATA]
						.GO(mI2C_GO),      			//	GO transfor
						.END(mI2C_END),				//	END transfor 
						.ACK(mI2C_ACK),				//	ACK
						.RST_n(iRST_N)	);

wire wI2C_SCLK;
assign I2C_SCLK = wI2C_SCLK?1'bz:1'b0;

						
////////////////////////////////////////////////////////////////////
//////////////////////	Config Control	////////////////////////////
always@(posedge mI2C_CTRL_CLK or negedge iRST_N)
begin
	if(!iRST_N)
	begin
		LUT_INDEX	<=	0;
		mSetup_ST	<=	0;
		mI2C_GO		<=	0;
		delay_cnt <= 16'd0;
		delay_time <= 16'd10;
	end
	else
	begin
		if(LUT_INDEX<LUT_SIZE)
		begin
			case(mSetup_ST)
			0:	begin
			      if(LUT_DATA[31:16]== 16'hFFFF)
					begin
					   mI2C_GO		<=	0;
	            	delay_time  <= LUT_DATA[15:0];
					end
					else
					begin
					   mI2C_GO		<=	1;
	            	delay_time  <= 16'd10;
					   mI2C_DATA	<=	{8'hA8,LUT_DATA};
					end
					delay_cnt <= 16'd0;
					mSetup_ST <= 1;

				end
			1: begin // wait 
		         if(delay_cnt > delay_time)begin
					   delay_cnt <= 16'd0;
			         if(LUT_DATA[31:16]== 16'hFFFF)
						  mSetup_ST	<=	3;
						else
						  mSetup_ST	<=	2;
					end
					else
					begin
					   mSetup_ST	<=	1;
						delay_cnt   <= delay_cnt + 16'd1;
              end
	         end
			2:	begin
					if(mI2C_END)
					begin
						if(!mI2C_ACK) //not receive NACK
						mSetup_ST	<=	3;
						else
						mSetup_ST	<=	0;							
						mI2C_GO		<=	0;
					end
				end

			3:	begin
					LUT_INDEX	<=	LUT_INDEX+1;
					mSetup_ST	<=	0;
				end
			default: 	mSetup_ST	<=	0;
			endcase
		end
	end
end


always
begin
 case(iFREQ_SELECT)
	  0 :  LUT_DATA  <=  LUT_DATA0;
	  1 :  LUT_DATA  <=  LUT_DATA1;
	  2 :  LUT_DATA  <=  LUT_DATA2;
	 default  :  LUT_DATA  <=  LUT_DATA0;
   endcase
end


//input [5:0] iFREQ_DISABLE;
////Disable Bit:  0 enable ,1 Disable
//// Bit5:  DDR3
//// Bit4:  SATA_DEVICE
//// Bit3:  SATA_HOST
//// Bit2:  PCIE
//// Bit1:  FMCD
//// Bit0:  FMCA

// register  5: control for disable  Y0,Y1 (FMCA,FMCD)
// register  7: control for disable  Y2 (PCIE)
// register  9: control for disable  Y4 (SATA_DEVICE)
// register 12: control for disable  Y5 (SATA_HOST)
// register 15: control for disable  Y6 (DDR3)

wire [7:0] register5_mask;
wire [7:0] register7_mask;
wire [7:0] register9_mask;
wire [7:0] register12_mask;
wire [7:0] register15_mask;
assign register5_mask  = (iFREQ_DISABLE[0] ? 8'h00:8'h02)|(iFREQ_DISABLE[1] ? 8'h00:8'h20);
assign register7_mask  = iFREQ_DISABLE[2] ? 8'h00:8'h02;
assign register9_mask  = iFREQ_DISABLE[4] ? 8'h00:8'h02;
assign register12_mask = iFREQ_DISABLE[3] ? 8'h00:8'h02;
assign register15_mask = iFREQ_DISABLE[5] ? 8'h00:8'h02;




////////////////////////////////////////////////////////////////////
//FMCA(100),FMCD(100),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)
reg	[31:0]	LUT_DATA0;

always
begin
   case(LUT_INDEX)
	  0 :  LUT_DATA0  <=  32'hFFFF_0320;//For trigger delay (I2C cycle(1/20KHz))
	  1 :  LUT_DATA0  <=  32'h0003_0000;//reset
	  2 :  LUT_DATA0  <=  32'hFFFF_007D;
	  3 :  LUT_DATA0  <=  32'h0003_0040;
	  4 :  LUT_DATA0  <=  32'hFFFF_007D;
	  5 :  LUT_DATA0  <=  32'h0000_01B9;
	  6 :  LUT_DATA0  <=  32'h0001_0000;
	  7 :  LUT_DATA0  <=  32'h0002_0017;
	  8 :  LUT_DATA0  <=  32'h0003_00F5;
	  9 :  LUT_DATA0  <=  32'h0004_30EB;
	 10 :  LUT_DATA0  <=  32'h0005_0001|register5_mask;  
	 11 :  LUT_DATA0  <=  32'h0006_0005;
	 12 :  LUT_DATA0  <=  32'h0007_0001|register7_mask;
	 13 :  LUT_DATA0  <=  32'h0008_0005;
	 14 :  LUT_DATA0  <=  32'h0009_0001|register9_mask;
	 15 :  LUT_DATA0  <=  32'h000A_0030;
	 16 :  LUT_DATA0  <=  32'h000B_0000;
	 17 :  LUT_DATA0  <=  32'h000C_0001|register12_mask;
	 18 :  LUT_DATA0  <=  32'h000D_0030;
	 19 :  LUT_DATA0  <=  32'h000E_0000;
	 20 :  LUT_DATA0  <=  32'h000F_0201|register15_mask;
	 21 :  LUT_DATA0  <=  32'h0010_0014;
	 22 :  LUT_DATA0  <=  32'h0011_0000;
	 23 :  LUT_DATA0  <=  32'h0012_0001;
	 24 :  LUT_DATA0  <=  32'h0013_0000;
	 25 :  LUT_DATA0  <=  32'h0014_0000;
	 26 :  LUT_DATA0  <=  32'hFFFF_007D;
	 default  :  LUT_DATA0  <=  32'hFFFF_0000;
   endcase
end




////////////////////////////////////////////////////////////////////
//FMCA(125),FMCD(125),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)
reg	[31:0]	LUT_DATA1;
always
begin
   case(LUT_INDEX)
	  0 :  LUT_DATA1  <=  32'hFFFF_0320;//For trigger delay (I2C cycle(1/20KHz))
	  1 :  LUT_DATA1  <=  32'h0003_0000;//reset
	  2 :  LUT_DATA1  <=  32'hFFFF_007D;
	  3 :  LUT_DATA1  <=  32'h0003_0040;
	  4 :  LUT_DATA1  <=  32'hFFFF_007D;
	  5 :  LUT_DATA1  <=  32'h0000_01B9;
	  6 :  LUT_DATA1  <=  32'h0001_0000;
	  7 :  LUT_DATA1  <=  32'h0002_001D;
	  8 :  LUT_DATA1  <=  32'h0003_00F4;
	  9 :  LUT_DATA1  <=  32'h0004_30EB;
	 10 :  LUT_DATA1  <=  32'h0005_0001|register5_mask;
	 11 :  LUT_DATA1  <=  32'h0006_0005;
	 12 :  LUT_DATA1  <=  32'h0007_0001|register7_mask;
	 13 :  LUT_DATA1  <=  32'h0008_0005;
	 14 :  LUT_DATA1  <=  32'h0009_0001|register9_mask;
	 15 :  LUT_DATA1  <=  32'h000A_0040;
	 16 :  LUT_DATA1  <=  32'h000B_0000;
	 17 :  LUT_DATA1  <=  32'h000C_0001|register12_mask;
	 18 :  LUT_DATA1  <=  32'h000D_0040;
	 19 :  LUT_DATA1  <=  32'h000E_0000;
	 20 :  LUT_DATA1  <=  32'h000F_0201|register15_mask;
	 21 :  LUT_DATA1  <=  32'h0010_0014;
	 22 :  LUT_DATA1  <=  32'h0011_0000;
	 23 :  LUT_DATA1  <=  32'h0012_0201;
	 24 :  LUT_DATA1  <=  32'h0013_0014;
	 25 :  LUT_DATA1  <=  32'h0014_0000;
	 26 :  LUT_DATA1  <=  32'hFFFF_007D;
	 default  :  LUT_DATA1  <=  32'hFFFF_0000;
   endcase
end


////////////////////////////////////////////////////////////////////
//FMCA(150),FMCD(150),PCIE(100),SATA_HOST(150),SATA_DEVICE(150),DDR3(133.333)
reg	[31:0]	LUT_DATA2;

always
begin
   case(LUT_INDEX)
	  0 :  LUT_DATA2  <=  32'hFFFF_0320;//For trigger delay (I2C cycle(1/20KHz))
	  1 :  LUT_DATA2  <=  32'h0003_0000;//reset
	  2 :  LUT_DATA2  <=  32'hFFFF_007D;
	  3 :  LUT_DATA2  <=  32'h0003_0040;
	  4 :  LUT_DATA2  <=  32'hFFFF_007D;
	  5 :  LUT_DATA2  <=  32'h0000_01B9;
	  6 :  LUT_DATA2  <=  32'h0001_0000;
	  7 :  LUT_DATA2  <=  32'h0002_001D;
	  8 :  LUT_DATA2  <=  32'h0003_00F4;
	  9 :  LUT_DATA2  <=  32'h0004_30EB;
	 10 :  LUT_DATA2  <=  32'h0005_0001|register5_mask;
	 11 :  LUT_DATA2  <=  32'h0006_0004;
	 12 :  LUT_DATA2  <=  32'h0007_0001|register7_mask;
	 13 :  LUT_DATA2  <=  32'h0008_0005;
	 14 :  LUT_DATA2  <=  32'h0009_0001|register9_mask;
	 15 :  LUT_DATA2  <=  32'h000A_0040;
	 16 :  LUT_DATA2  <=  32'h000B_0000;
	 17 :  LUT_DATA2  <=  32'h000C_0001|register12_mask;
	 18 :  LUT_DATA2  <=  32'h000D_0040;
	 19 :  LUT_DATA2  <=  32'h000E_0000;
	 20 :  LUT_DATA2  <=  32'h000F_0201|register15_mask;
	 21 :  LUT_DATA2  <=  32'h0010_0014;
	 22 :  LUT_DATA2  <=  32'h0011_0000;
	 23 :  LUT_DATA2  <=  32'h0012_0201;
	 24 :  LUT_DATA2  <=  32'h0013_0014;
	 25 :  LUT_DATA2  <=  32'h0014_0000;
	 26 :  LUT_DATA2  <=  32'hFFFF_007D;
	 default  :  LUT_DATA2  <=  32'hFFFF_0000;
   endcase
end


////////////////////////////////////////////////////////////////////
endmodule
