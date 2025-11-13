// Author :  Zeng
module SPI_LMK04906_Config (	//	Host Side
						clk50, // max 40MHz  // for simplify  here input 50MHz then /4 in code
						rst_n,
						iFREQ_SELECT,

						// SPI Side
						LMK04906_CLK,
						LMK04906_DATAIN,
                  LMK04906_LE,

                  SPI_DONE						
					   	);


// Host Side
input		clk50;
input		rst_n;
input	[3:0]	iFREQ_SELECT;

// SPI Side
output  	LMK04906_CLK;
output	LMK04906_DATAIN;
output	LMK04906_LE;

output	SPI_DONE;

// parameter
localparam tLE_START   = 1,
           tCLK_START  = 2,
           tCLK_END    = 33,
           tLE_END    =  34,
           tCONFIG_END = 37;
			  
//	LUT Data Number
localparam	LUT_SIZE	=	30;


//	Internal Registers/Wires
wire clk;
reg  clk_enable; // must sync to clk in clk low
reg [5:0] tick;
wire config_init;
wire config_done;

reg [31:0] reg_data; 
reg [32:0] lut_data0; 
reg [32:0] lut_data1;
reg [32:0] lut_data2; 
reg [32:0] lut_data3;
reg [32:0] lut_data4; 
reg [32:0] lut_data5;
reg [32:0] lut_data6; 
reg [32:0] lut_data7;
reg [32:0] lut_data8; 
reg [32:0] lut_data9;
reg [32:0] lut_data10; 
reg [32:0] lut_data11;
reg [32:0] lut_data12; 
reg [32:0] lut_data13;
reg [32:0] lut_data14; 
reg [32:0] lut_data15;  
    
reg [32:0] lut_data; 

reg [5:0]  lut_index;
reg		   spi_go;
reg	[3:0]	mSetup_ST;
reg   [31:0] delay_cnt;

assign  SPI_DONE = (lut_index == LUT_SIZE) ?1'b1:1'b0;

reg [1:0] clk_cnt;
/////////////////////	12.5MHz clk ////////////////////////
always@(posedge clk50 or negedge rst_n)
	if(!rst_n) 	  clk_cnt <= 2'b00;
	else          clk_cnt <= clk_cnt + 1'b1;
	
assign clk = clk_cnt[1];
	
/////////////////////////////////////////////////////////////
// timer
always @ (negedge clk or negedge rst_n) 	 
	if (~rst_n)
		tick <= 6'd0;
	else begin
	   if (!spi_go )tick <= 6'd0;
		else         tick <= tick + 1'b1;
	end

assign config_init = (tick == (tCLK_START-1)) ? 1'b1: 1'b0;
assign config_done = (tick > tCONFIG_END)?1'b1:1'b0;	

assign LMK04906_CLK    = (tick >= tCLK_START && tick <= tCLK_END)?clk:1'b0;
assign LMK04906_LE     = (tick >= tLE_START && tick <= tLE_END)?1'b0:1'b1;	
assign LMK04906_DATAIN = reg_data[31];

/////////////////////////////////////////////////////////////
// MSB shift out
always @(negedge clk or negedge rst_n)	
   if (~rst_n)      reg_data <= 32'd0;
	else if (config_init) reg_data <= lut_data[31:0];
	else             reg_data <= {reg_data[30:0],1'b0};
						
////////////////////////////////////////////////////////////////////
//////////////////////	Config Control	////////////////////////////
always@(negedge clk or negedge rst_n)
begin
	if(!rst_n)
	begin
		lut_index	<=	0;
		mSetup_ST	<=	0;
		spi_go		<=	0;
		delay_cnt <= 32'd0;
	end
	else
	begin
		if(lut_index<LUT_SIZE)
		begin
			case(mSetup_ST)
			0:	begin
			      if(lut_data[32]) // if bit 32 high ,for delay
					begin
					   spi_go		<=	0;
						mSetup_ST   <= 1;
					end
					else
					begin
					   spi_go		<=	1;
						mSetup_ST   <= 2;
					end
					delay_cnt <= 32'd0;
				end
			1: begin // wait 
		         if(delay_cnt >= lut_data[31:0])begin
					   delay_cnt <= 32'd0;
						mSetup_ST <= 3;
					end
					else
					begin
					   mSetup_ST	<=	1;
						delay_cnt   <= delay_cnt + 32'd1;
               end
	         end
			2:	begin
					if(config_done)
					begin
						mSetup_ST	<=	3;							
						spi_go		<=	0;
					end
				end

			3:	begin
					lut_index	<=	lut_index+1;
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
	  0 :  lut_data  <=  lut_data0;
	  1 :  lut_data  <=  lut_data1;
          2 :  lut_data  <=  lut_data2;
	  3 :  lut_data  <=  lut_data3;
          4 :  lut_data  <=  lut_data4;
	  5 :  lut_data  <=  lut_data5;
          6 :  lut_data  <=  lut_data6;
	  7 :  lut_data  <=  lut_data7;
          8 :  lut_data  <=  lut_data8;
	  9 :  lut_data  <=  lut_data9;
          10 :  lut_data  <=  lut_data10;
	  11 :  lut_data  <=  lut_data11;
          12 :  lut_data  <=  lut_data12;
	  13 :  lut_data  <=  lut_data13;
          14 :  lut_data  <=  lut_data14;
	  15 :  lut_data  <=  lut_data15;
	 default  :  lut_data  <=  lut_data0;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/B/C/D 322.265625
always
begin
   case(lut_index)
	  0 :  lut_data0  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data0  <=  33'h0_80160200; 
	  2 :  lut_data0  <=  33'h1_000061A8;
	  3 :  lut_data0  <=  33'h0_20140100;
	  4 :  lut_data0  <=  33'h1_000061A8;
	  5 :  lut_data0  <=  33'h0_10140101;
	  6 :  lut_data0  <=  33'h0_A0140082;
	  7 :  lut_data0  <=  33'h0_00140103;
	  8 :  lut_data0  <=  33'h0_00140104;
	  9 :  lut_data0  <=  33'h0_80140C85;
	 10 :  lut_data0  <=  33'h0_01140006;
	 11 :  lut_data0  <=  33'h0_01110007;
	 12 :  lut_data0  <=  33'h0_00010008;
	 13 :  lut_data0  <=  33'h0_55555549;
	 14 :  lut_data0  <=  33'h0_9002410A;
	 15 :  lut_data0  <=  33'h0_3401100B;
	 16 :  lut_data0  <=  33'h0_130C006C;
	 17 :  lut_data0  <=  33'h0_3B02820D;
	 18 :  lut_data0  <=  33'h0_0200000E;
	 19 :  lut_data0  <=  33'h0_8000800F;
	 20 :  lut_data0  <=  33'h0_C1550410;
	 21 :  lut_data0  <=  33'h0_000000D8;
	 22 :  lut_data0  <=  33'h0_02C9C419;
	 23 :  lut_data0  <=  33'h0_8FA8001A;
	 24 :  lut_data0  <=  33'h0_10000F3B;
	 25 :  lut_data0  <=  33'h0_0080031C;
	 26 :  lut_data0  <=  33'h0_0080227D;
	 27 :  lut_data0  <=  33'h0_0300227E;
	 28 :  lut_data0  <=  33'h0_001F001F;
	 29 :  lut_data0  <=  33'h1_000061A8;
	 default  :  lut_data0  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
//  for FMCA/D/B 322.265625 FMCC 644.53125MHz
always
begin
   case(lut_index)
	  0 :  lut_data1  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data1  <=  33'h0_80160200; 
	  2 :  lut_data1  <=  33'h1_000061A8;
	  3 :  lut_data1  <=  33'h0_20140100;
	  4 :  lut_data1  <=  33'h1_000061A8;
	  5 :  lut_data1  <=  33'h0_10140101;
	  6 :  lut_data1  <=  33'h0_A0140082;
	  7 :  lut_data1  <=  33'h0_00140103;
	  8 :  lut_data1  <=  33'h0_00140084;
	  9 :  lut_data1  <=  33'h0_80140C85;
	 10 :  lut_data1  <=  33'h0_01140006;
	 11 :  lut_data1  <=  33'h0_01110007;
	 12 :  lut_data1  <=  33'h0_00010008;
	 13 :  lut_data1  <=  33'h0_55555549;
	 14 :  lut_data1  <=  33'h0_9002410A;
	 15 :  lut_data1  <=  33'h0_3401100B;
	 16 :  lut_data1  <=  33'h0_130C006C;
	 17 :  lut_data1  <=  33'h0_3B02820D;
	 18 :  lut_data1  <=  33'h0_0200000E;
	 19 :  lut_data1  <=  33'h0_8000800F;
	 20 :  lut_data1  <=  33'h0_C1550410;
	 21 :  lut_data1  <=  33'h0_000000D8;
	 22 :  lut_data1  <=  33'h0_02C9C419;
	 23 :  lut_data1  <=  33'h0_8FA8001A;
	 24 :  lut_data1  <=  33'h0_10000F3B;
	 25 :  lut_data1  <=  33'h0_0080031C;
	 26 :  lut_data1  <=  33'h0_0800227D;
	 27 :  lut_data1  <=  33'h0_0300227E;
	 28 :  lut_data1  <=  33'h0_001F001F;
	 29 :  lut_data1  <=  33'h1_000061A8;
	 default  :  lut_data1  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
//  for FMCA/D/C 322.265625 FMCB 644.53125MHz
always
begin
   case(lut_index)
	  0 :  lut_data2  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data2  <=  33'h0_80160200; 
	  2 :  lut_data2  <=  33'h1_000061A8;
	  3 :  lut_data2  <=  33'h0_20140100;
	  4 :  lut_data2  <=  33'h1_000061A8;
	  5 :  lut_data2  <=  33'h0_10140101;
	  6 :  lut_data2  <=  33'h0_A0140082;
	  7 :  lut_data2  <=  33'h0_00140083;
	  8 :  lut_data2  <=  33'h0_00140104;
	  9 :  lut_data2  <=  33'h0_80140C85;
	 10 :  lut_data2  <=  33'h0_01140006;
	 11 :  lut_data2  <=  33'h0_01110007;
	 12 :  lut_data2  <=  33'h0_00010008;
	 13 :  lut_data2  <=  33'h0_55555549;
	 14 :  lut_data2  <=  33'h0_9002410A;
	 15 :  lut_data2  <=  33'h0_3401100B;
	 16 :  lut_data2  <=  33'h0_130C006C;
	 17 :  lut_data2  <=  33'h0_3B02820D;
	 18 :  lut_data2  <=  33'h0_0200000E;
	 19 :  lut_data2  <=  33'h0_8000800F;
	 20 :  lut_data2  <=  33'h0_C1550410;
	 21 :  lut_data2  <=  33'h0_000000D8;
	 22 :  lut_data2  <=  33'h0_02C9C419;
	 23 :  lut_data2  <=  33'h0_8FA8001A;
	 24 :  lut_data2  <=  33'h0_10000F3B;
	 25 :  lut_data2  <=  33'h0_0080031C;
	 26 :  lut_data2  <=  33'h0_0800227D;
	 27 :  lut_data2  <=  33'h0_0300227E;
	 28 :  lut_data2  <=  33'h0_001F001F;
	 29 :  lut_data2  <=  33'h1_000061A8;
	 default  :  lut_data2  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/D 322.265625 FMCB/C 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data3  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data3  <=  33'h0_80160200; 
	  2 :  lut_data3  <=  33'h1_000061A8;
	  3 :  lut_data3  <=  33'h0_20140100;
	  4 :  lut_data3  <=  33'h1_000061A8;
	  5 :  lut_data3  <=  33'h0_10140101;
	  6 :  lut_data3  <=  33'h0_A0140082;
	  7 :  lut_data3  <=  33'h0_00140083;
	  8 :  lut_data3  <=  33'h0_00140084;
	  9 :  lut_data3  <=  33'h0_80140C85;
	 10 :  lut_data3  <=  33'h0_01140006;
	 11 :  lut_data3  <=  33'h0_01110007;
	 12 :  lut_data3  <=  33'h0_00010008;
	 13 :  lut_data3  <=  33'h0_55555549;
	 14 :  lut_data3  <=  33'h0_9002410A;
	 15 :  lut_data3  <=  33'h0_3401100B;
	 16 :  lut_data3  <=  33'h0_130C006C;
	 17 :  lut_data3  <=  33'h0_3B02820D;
	 18 :  lut_data3  <=  33'h0_0200000E;
	 19 :  lut_data3  <=  33'h0_8000800F;
	 20 :  lut_data3  <=  33'h0_C1550410;
	 21 :  lut_data3  <=  33'h0_000000D8;
	 22 :  lut_data3  <=  33'h0_02C9C419;
	 23 :  lut_data3  <=  33'h0_8FA8001A;
	 24 :  lut_data3  <=  33'h0_10000F3B;
	 25 :  lut_data3  <=  33'h0_0080031C;
	 26 :  lut_data3  <=  33'h0_0080227D;
	 27 :  lut_data3  <=  33'h0_0300227E;
	 28 :  lut_data3  <=  33'h0_001F001F;
	 29 :  lut_data3  <=  33'h1_000061A8;
	 default  :  lut_data3  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/B/C 322.265625 FMCD 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data4  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data4  <=  33'h0_80160200; 
	  2 :  lut_data4  <=  33'h1_000061A8;
	  3 :  lut_data4  <=  33'h0_20140100;
	  4 :  lut_data4  <=  33'h1_000061A8;
	  5 :  lut_data4  <=  33'h0_10140081;
	  6 :  lut_data4  <=  33'h0_A0140082;
	  7 :  lut_data4  <=  33'h0_00140103;
	  8 :  lut_data4  <=  33'h0_00140104;
	  9 :  lut_data4  <=  33'h0_80140C85;
	 10 :  lut_data4  <=  33'h0_01140006;
	 11 :  lut_data4  <=  33'h0_01110007;
	 12 :  lut_data4  <=  33'h0_00010008;
	 13 :  lut_data4  <=  33'h0_55555549;
	 14 :  lut_data4  <=  33'h0_9002410A;
	 15 :  lut_data4  <=  33'h0_3401100B;
	 16 :  lut_data4  <=  33'h0_130C006C;
	 17 :  lut_data4  <=  33'h0_3B02820D;
	 18 :  lut_data4  <=  33'h0_0200000E;
	 19 :  lut_data4  <=  33'h0_8000800F;
	 20 :  lut_data4  <=  33'h0_C1550410;
	 21 :  lut_data4  <=  33'h0_000000D8;
	 22 :  lut_data4  <=  33'h0_02C9C419;
	 23 :  lut_data4  <=  33'h0_8FA8001A;
	 24 :  lut_data4  <=  33'h0_10000F3B;
	 25 :  lut_data4  <=  33'h0_0080031C;
	 26 :  lut_data4  <=  33'h0_0080227D;
	 27 :  lut_data4  <=  33'h0_0300227E;
	 28 :  lut_data4  <=  33'h0_001F001F;
	 29 :  lut_data4  <=  33'h1_000061A8;
	 default  :  lut_data4  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/B 322.265625 FMCD/C 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data5  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data5  <=  33'h0_80160200; 
	  2 :  lut_data5  <=  33'h1_000061A8;
	  3 :  lut_data5  <=  33'h0_20140100;
	  4 :  lut_data5  <=  33'h1_000061A8;
	  5 :  lut_data5  <=  33'h0_10140081;
	  6 :  lut_data5  <=  33'h0_A0140082;
	  7 :  lut_data5  <=  33'h0_00140103;	//33'h0_00140083
	  8 :  lut_data5  <=  33'h0_00140084;	//33'h0_00140104
	  9 :  lut_data5  <=  33'h0_80140C85;
	 10 :  lut_data5  <=  33'h0_01140006;
	 11 :  lut_data5  <=  33'h0_01110007;
	 12 :  lut_data5  <=  33'h0_00010008;
	 13 :  lut_data5  <=  33'h0_55555549;
	 14 :  lut_data5  <=  33'h0_9002410A;
	 15 :  lut_data5  <=  33'h0_3401100B;
	 16 :  lut_data5  <=  33'h0_130C006C;
	 17 :  lut_data5  <=  33'h0_3B02820D;
	 18 :  lut_data5  <=  33'h0_0200000E;
	 19 :  lut_data5  <=  33'h0_8000800F;
	 20 :  lut_data5  <=  33'h0_C1550410;
	 21 :  lut_data5  <=  33'h0_000000D8;
	 22 :  lut_data5  <=  33'h0_02C9C419;
	 23 :  lut_data5  <=  33'h0_8FA8001A;
	 24 :  lut_data5  <=  33'h0_10000F3B;
	 25 :  lut_data5  <=  33'h0_0080031C;
	 26 :  lut_data5  <=  33'h0_0080227D;
	 27 :  lut_data5  <=  33'h0_0300227E;
	 28 :  lut_data5  <=  33'h0_001F001F;
	 29 :  lut_data5  <=  33'h1_000061A8;
	 default  :  lut_data5  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/C 322.265625 FMCD/B 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data6  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data6  <=  33'h0_80160200; 
	  2 :  lut_data6  <=  33'h1_000061A8;
	  3 :  lut_data6  <=  33'h0_20140100;
	  4 :  lut_data6  <=  33'h1_000061A8;
	  5 :  lut_data6  <=  33'h0_10140081;
	  6 :  lut_data6  <=  33'h0_A0140082;
	  7 :  lut_data6  <=  33'h0_00140083;
	  8 :  lut_data6  <=  33'h0_00140104;
	  9 :  lut_data6  <=  33'h0_80140C85;
	 10 :  lut_data6  <=  33'h0_01140006;
	 11 :  lut_data6  <=  33'h0_01110007;
	 12 :  lut_data6  <=  33'h0_00010008;
	 13 :  lut_data6  <=  33'h0_55555549;
	 14 :  lut_data6  <=  33'h0_9002410A;
	 15 :  lut_data6  <=  33'h0_3401100B;
	 16 :  lut_data6  <=  33'h0_130C006C;
	 17 :  lut_data6  <=  33'h0_3B02820D;
	 18 :  lut_data6  <=  33'h0_0200000E;
	 19 :  lut_data6  <=  33'h0_8000800F;
	 20 :  lut_data6  <=  33'h0_C1550410;
	 21 :  lut_data6  <=  33'h0_000000D8;
	 22 :  lut_data6  <=  33'h0_02C9C419;
	 23 :  lut_data6  <=  33'h0_8FA8001A;
	 24 :  lut_data6  <=  33'h0_10000F3B;
	 25 :  lut_data6  <=  33'h0_0080031C;
	 26 :  lut_data6  <=  33'h0_0080227D;
	 27 :  lut_data6  <=  33'h0_0300227E;
	 28 :  lut_data6  <=  33'h0_001F001F;
	 29 :  lut_data6  <=  33'h1_000061A8;
	 default  :  lut_data6  <=  33'h1_00000000;
   endcase
end

////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA 322.265625      FMCD/B/C  644.53125
always
begin
   case(lut_index)
	  0 :  lut_data7  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data7  <=  33'h0_80160200; 
	  2 :  lut_data7  <=  33'h1_000061A8;
	  3 :  lut_data7  <=  33'h0_20140100;
	  4 :  lut_data7  <=  33'h1_000061A8;
	  5 :  lut_data7  <=  33'h0_10140081;
	  6 :  lut_data7  <=  33'h0_A0140082;
	  7 :  lut_data7  <=  33'h0_00140083;
	  8 :  lut_data7  <=  33'h0_00140084;
	  9 :  lut_data7  <=  33'h0_80140C85;
	 10 :  lut_data7  <=  33'h0_01140006;
	 11 :  lut_data7  <=  33'h0_01110007;
	 12 :  lut_data7  <=  33'h0_00010008;
	 13 :  lut_data7  <=  33'h0_55555549;
	 14 :  lut_data7  <=  33'h0_9002410A;
	 15 :  lut_data7  <=  33'h0_3401100B;
	 16 :  lut_data7  <=  33'h0_130C006C;
	 17 :  lut_data7  <=  33'h0_3B02820D;
	 18 :  lut_data7  <=  33'h0_0200000E;
	 19 :  lut_data7  <=  33'h0_8000800F;
	 20 :  lut_data7  <=  33'h0_C1550410;
	 21 :  lut_data7  <=  33'h0_000000D8;
	 22 :  lut_data7  <=  33'h0_02C9C419;
	 23 :  lut_data7  <=  33'h0_8FA8001A;
	 24 :  lut_data7  <=  33'h0_10000F3B;
	 25 :  lut_data7  <=  33'h0_0080031C;
	 26 :  lut_data7  <=  33'h0_0080227D;
	 27 :  lut_data7  <=  33'h0_0300227E;
	 28 :  lut_data7  <=  33'h0_001F001F;
	 29 :  lut_data7  <=  33'h1_000061A8;
	 default  :  lut_data7  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA 644.53125      FMCD/B/C 322.265625
always
begin
   case(lut_index)
	  0 :  lut_data8  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data8  <=  33'h0_80160200; 
	  2 :  lut_data8  <=  33'h1_000061A8;
	  3 :  lut_data8  <=  33'h0_20140080;
	  4 :  lut_data8  <=  33'h1_000061A8;
	  5 :  lut_data8  <=  33'h0_10140101;
	  6 :  lut_data8  <=  33'h0_A0140082;
	  7 :  lut_data8  <=  33'h0_00140103;
	  8 :  lut_data8  <=  33'h0_00140104;
	  9 :  lut_data8  <=  33'h0_80140C85;
	 10 :  lut_data8  <=  33'h0_01140006;
	 11 :  lut_data8  <=  33'h0_01110007;
	 12 :  lut_data8  <=  33'h0_00010008;
	 13 :  lut_data8  <=  33'h0_55555549;
	 14 :  lut_data8  <=  33'h0_9002410A;
	 15 :  lut_data8  <=  33'h0_3401100B;
	 16 :  lut_data8  <=  33'h0_130C006C;
	 17 :  lut_data8  <=  33'h0_3B02820D;
	 18 :  lut_data8  <=  33'h0_0200000E;
	 19 :  lut_data8  <=  33'h0_8000800F;
	 20 :  lut_data8  <=  33'h0_C1550410;
	 21 :  lut_data8  <=  33'h0_000000D8;
	 22 :  lut_data8  <=  33'h0_02C9C419;
	 23 :  lut_data8  <=  33'h0_8FA8001A;
	 24 :  lut_data8  <=  33'h0_10000F3B;
	 25 :  lut_data8  <=  33'h0_0080031C;
	 26 :  lut_data8  <=  33'h0_0080227D;
	 27 :  lut_data8  <=  33'h0_0300227E;
	 28 :  lut_data8  <=  33'h0_001F001F;
	 29 :  lut_data8  <=  33'h1_000061A8;
	 default  :  lut_data8  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCD/B  322.265625 FMCA/C 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data9  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data9  <=  33'h0_80160200; 
	  2 :  lut_data9  <=  33'h1_000061A8;
	  3 :  lut_data9  <=  33'h0_20140080;
	  4 :  lut_data9  <=  33'h1_000061A8;
	  5 :  lut_data9  <=  33'h0_10140101;
	  6 :  lut_data9  <=  33'h0_A0140082;
	  7 :  lut_data9  <=  33'h0_00140103;
	  8 :  lut_data9  <=  33'h0_00140084;
	  9 :  lut_data9  <=  33'h0_80140C85;
	 10 :  lut_data9  <=  33'h0_01140006;
	 11 :  lut_data9  <=  33'h0_01110007;
	 12 :  lut_data9  <=  33'h0_00010008;
	 13 :  lut_data9  <=  33'h0_55555549;
	 14 :  lut_data9  <=  33'h0_9002410A;
	 15 :  lut_data9  <=  33'h0_3401100B;
	 16 :  lut_data9  <=  33'h0_130C006C;
	 17 :  lut_data9  <=  33'h0_3B02820D;
	 18 :  lut_data9  <=  33'h0_0200000E;
	 19 :  lut_data9  <=  33'h0_8000800F;
	 20 :  lut_data9  <=  33'h0_C1550410;
	 21 :  lut_data9  <=  33'h0_000000D8;
	 22 :  lut_data9  <=  33'h0_02C9C419;
	 23 :  lut_data9  <=  33'h0_8FA8001A;
	 24 :  lut_data9  <=  33'h0_10000F3B;
	 25 :  lut_data9  <=  33'h0_0080031C;
	 26 :  lut_data9  <=  33'h0_0080227D;
	 27 :  lut_data9  <=  33'h0_0300227E;
	 28 :  lut_data9  <=  33'h0_001F001F;
	 29 :  lut_data9  <=  33'h1_000061A8;
	 default  :  lut_data9  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/B 644.53125 FMCD/C 322.265625
always
begin
   case(lut_index)
	  0 :  lut_data10  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data10  <=  33'h0_80160200; 
	  2 :  lut_data10  <=  33'h1_000061A8;
	  3 :  lut_data10  <=  33'h0_20140080;
	  4 :  lut_data10  <=  33'h1_000061A8;
	  5 :  lut_data10  <=  33'h0_10140101;
	  6 :  lut_data10  <=  33'h0_A0140082;
	  7 :  lut_data10  <=  33'h0_00140083;
	  8 :  lut_data10  <=  33'h0_00140104;
	  9 :  lut_data10  <=  33'h0_80140C85;
	 10 :  lut_data10  <=  33'h0_01140006;
	 11 :  lut_data10  <=  33'h0_01110007;
	 12 :  lut_data10  <=  33'h0_00010008;
	 13 :  lut_data10  <=  33'h0_55555549;
	 14 :  lut_data10  <=  33'h0_9002410A;
	 15 :  lut_data10  <=  33'h0_3401100B;
	 16 :  lut_data10  <=  33'h0_130C006C;
	 17 :  lut_data10  <=  33'h0_3B02820D;
	 18 :  lut_data10  <=  33'h0_0200000E;
	 19 :  lut_data10  <=  33'h0_8000800F;
	 20 :  lut_data10  <=  33'h0_C1550410;
	 21 :  lut_data10  <=  33'h0_000000D8;
	 22 :  lut_data10  <=  33'h0_02C9C419;
	 23 :  lut_data10  <=  33'h0_8FA8001A;
	 24 :  lut_data10  <=  33'h0_10000F3B;
	 25 :  lut_data10  <=  33'h0_0080031C;
	 26 :  lut_data10  <=  33'h0_0080227D;
	 27 :  lut_data10  <=  33'h0_0300227E;
	 28 :  lut_data10  <=  33'h0_001F001F;
	 29 :  lut_data10  <=  33'h1_000061A8;
	 default  :  lut_data10  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCD 322.265625 FMCA/B/C
always
begin
   case(lut_index)
	  0 :  lut_data11  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data11  <=  33'h0_80160200; 
	  2 :  lut_data11  <=  33'h1_000061A8;
	  3 :  lut_data11  <=  33'h0_20140080;
	  4 :  lut_data11  <=  33'h1_000061A8;
	  5 :  lut_data11  <=  33'h0_10140101;
	  6 :  lut_data11  <=  33'h0_A0140082;
	  7 :  lut_data11  <=  33'h0_00140083;
	  8 :  lut_data11  <=  33'h0_00140084;
	  9 :  lut_data11  <=  33'h0_80140C85;
	 10 :  lut_data11  <=  33'h0_01140006;
	 11 :  lut_data11  <=  33'h0_01110007;
	 12 :  lut_data11  <=  33'h0_00010008;
	 13 :  lut_data11  <=  33'h0_55555549;
	 14 :  lut_data11  <=  33'h0_9002410A;
	 15 :  lut_data11  <=  33'h0_3401100B;
	 16 :  lut_data11  <=  33'h0_130C006C;
	 17 :  lut_data11  <=  33'h0_3B02820D;
	 18 :  lut_data11  <=  33'h0_0200000E;
	 19 :  lut_data11  <=  33'h0_8000800F;
	 20 :  lut_data11  <=  33'h0_C1550410;
	 21 :  lut_data11  <=  33'h0_000000D8;
	 22 :  lut_data11  <=  33'h0_02C9C419;
	 23 :  lut_data11  <=  33'h0_8FA8001A;
	 24 :  lut_data11  <=  33'h0_10000F3B;
	 25 :  lut_data11  <=  33'h0_0080031C;
	 26 :  lut_data11  <=  33'h0_0080227D;
	 27 :  lut_data11  <=  33'h0_0300227E;
	 28 :  lut_data11  <=  33'h0_001F001F;
	 29 :  lut_data11  <=  33'h1_000061A8;
	 default  :  lut_data11  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/D 644.53125  FMCB/C 322.265625
always
begin
   case(lut_index)
	  0 :  lut_data12  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data12  <=  33'h0_80160200; 
	  2 :  lut_data12  <=  33'h1_000061A8;
	  3 :  lut_data12  <=  33'h0_20140080;
	  4 :  lut_data12  <=  33'h1_000061A8;
	  5 :  lut_data12  <=  33'h0_10140081;
	  6 :  lut_data12  <=  33'h0_A0140082;
	  7 :  lut_data12  <=  33'h0_00140103;
	  8 :  lut_data12  <=  33'h0_00140104;
	  9 :  lut_data12  <=  33'h0_80140C85;
	 10 :  lut_data12  <=  33'h0_01140006;
	 11 :  lut_data12  <=  33'h0_01110007;
	 12 :  lut_data12  <=  33'h0_00010008;
	 13 :  lut_data12  <=  33'h0_55555549;
	 14 :  lut_data12  <=  33'h0_9002410A;
	 15 :  lut_data12  <=  33'h0_3401100B;
	 16 :  lut_data12  <=  33'h0_130C006C;
	 17 :  lut_data12  <=  33'h0_3B02820D;
	 18 :  lut_data12  <=  33'h0_0200000E;
	 19 :  lut_data12  <=  33'h0_8000800F;
	 20 :  lut_data12  <=  33'h0_C1550410;
	 21 :  lut_data12  <=  33'h0_000000D8;
	 22 :  lut_data12  <=  33'h0_02C9C419;
	 23 :  lut_data12  <=  33'h0_8FA8001A;
	 24 :  lut_data12  <=  33'h0_10000F3B;
	 25 :  lut_data12  <=  33'h0_0080031C;
	 26 :  lut_data12  <=  33'h0_0080227D;
	 27 :  lut_data12  <=  33'h0_0300227E;
	 28 :  lut_data12  <=  33'h0_001F001F;
	 29 :  lut_data12  <=  33'h1_000061A8;
	 default  :  lut_data12  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCB 322.265625 MCA/D/C 644.53125
always
begin
   case(lut_index)
	  0 :  lut_data13  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data13  <=  33'h0_80160200; 
	  2 :  lut_data13  <=  33'h1_000061A8;
	  3 :  lut_data13  <=  33'h0_20140080;
	  4 :  lut_data13  <=  33'h1_000061A8;
	  5 :  lut_data13  <=  33'h0_10140081;
	  6 :  lut_data13  <=  33'h0_A0140082;
	  7 :  lut_data13  <=  33'h0_00140103;
	  8 :  lut_data13  <=  33'h0_00140084;
	  9 :  lut_data13  <=  33'h0_80140C85;
	 10 :  lut_data13  <=  33'h0_01140006;
	 11 :  lut_data13  <=  33'h0_01110007;
	 12 :  lut_data13  <=  33'h0_00010008;
	 13 :  lut_data13  <=  33'h0_55555549;
	 14 :  lut_data13  <=  33'h0_9002410A;
	 15 :  lut_data13  <=  33'h0_3401100B;
	 16 :  lut_data13  <=  33'h0_130C006C;
	 17 :  lut_data13  <=  33'h0_3B02820D;
	 18 :  lut_data13  <=  33'h0_0200000E;
	 19 :  lut_data13  <=  33'h0_8000800F;
	 20 :  lut_data13  <=  33'h0_C1550410;
	 21 :  lut_data13  <=  33'h0_000000D8;
	 22 :  lut_data13  <=  33'h0_02C9C419;
	 23 :  lut_data13  <=  33'h0_8FA8001A;
	 24 :  lut_data13  <=  33'h0_10000F3B;
	 25 :  lut_data13  <=  33'h0_0080031C;
	 26 :  lut_data13  <=  33'h0_0080227D;
	 27 :  lut_data13  <=  33'h0_0300227E;
	 28 :  lut_data13  <=  33'h0_001F001F;
	 29 :  lut_data13  <=  33'h1_000061A8;
	 default  :  lut_data13  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
// for FMCA/D/B 644.53125 FMCC 322.265625
always
begin
   case(lut_index)
	  0 :  lut_data14  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data14  <=  33'h0_80160200; 
	  2 :  lut_data14  <=  33'h1_000061A8;
	  3 :  lut_data14  <=  33'h0_20140080;
	  4 :  lut_data14  <=  33'h1_000061A8;
	  5 :  lut_data14  <=  33'h0_10140081;
	  6 :  lut_data14  <=  33'h0_A0140082;
	  7 :  lut_data14  <=  33'h0_00140083;
	  8 :  lut_data14  <=  33'h0_00140104;
	  9 :  lut_data14  <=  33'h0_80140C85;
	 10 :  lut_data14  <=  33'h0_01140006;
	 11 :  lut_data14  <=  33'h0_01110007;
	 12 :  lut_data14  <=  33'h0_00010008;
	 13 :  lut_data14  <=  33'h0_55555549;
	 14 :  lut_data14  <=  33'h0_9002410A;
	 15 :  lut_data14  <=  33'h0_3401100B;
	 16 :  lut_data14  <=  33'h0_130C006C;
	 17 :  lut_data14  <=  33'h0_3B02820D;
	 18 :  lut_data14  <=  33'h0_0200000E;
	 19 :  lut_data14  <=  33'h0_8000800F;
	 20 :  lut_data14  <=  33'h0_C1550410;
	 21 :  lut_data14  <=  33'h0_000000D8;
	 22 :  lut_data14  <=  33'h0_02C9C419;
	 23 :  lut_data14  <=  33'h0_8FA8001A;
	 24 :  lut_data14  <=  33'h0_10000F3B;
	 25 :  lut_data14  <=  33'h0_0080031C;
	 26 :  lut_data14  <=  33'h0_0080227D;
	 27 :  lut_data14  <=  33'h0_0300227E;
	 28 :  lut_data14  <=  33'h0_001F001F;
	 29 :  lut_data14  <=  33'h1_000061A8;
	 default  :  lut_data14  <=  33'h1_00000000;
   endcase
end


////////////////////////////////////////////////////////////////////
/////////////////////	Registers map	  //////////////////////////
//  for FMCA/B/C/D 644.53125MHz
always
begin
   case(lut_index)
	  0 :  lut_data15  <=  33'h1_000061A8;//bit32 high for trigger delay
	  1 :  lut_data15  <=  33'h0_80160200; 
	  2 :  lut_data15  <=  33'h1_000061A8;
	  3 :  lut_data15  <=  33'h0_20140080;
	  4 :  lut_data15  <=  33'h1_000061A8;
	  5 :  lut_data15  <=  33'h0_10140081;
	  6 :  lut_data15  <=  33'h0_A0140082;
	  7 :  lut_data15  <=  33'h0_00140083;
	  8 :  lut_data15  <=  33'h0_00140084;
	  9 :  lut_data15  <=  33'h0_80140C85;
	 10 :  lut_data15  <=  33'h0_01140006;
	 11 :  lut_data15  <=  33'h0_01110007;
	 12 :  lut_data15  <=  33'h0_00010008;
	 13 :  lut_data15  <=  33'h0_55555549;
	 14 :  lut_data15  <=  33'h0_9002410A;
	 15 :  lut_data15  <=  33'h0_3401100B;
	 16 :  lut_data15  <=  33'h0_130C006C;
	 17 :  lut_data15  <=  33'h0_3B02820D;
	 18 :  lut_data15  <=  33'h0_0200000E;
	 19 :  lut_data15  <=  33'h0_8000800F;
	 20 :  lut_data15  <=  33'h0_C1550410;
	 21 :  lut_data15  <=  33'h0_000000D8;
	 22 :  lut_data15  <=  33'h0_02C9C419;
	 23 :  lut_data15  <=  33'h0_8FA8001A;
	 24 :  lut_data15  <=  33'h0_10000F3B;
	 25 :  lut_data15  <=  33'h0_0080031C;
	 26 :  lut_data15  <=  33'h0_0800227D;
	 27 :  lut_data15  <=  33'h0_0300227E;
	 28 :  lut_data15  <=  33'h0_001F001F;
	 29 :  lut_data15  <=  33'h1_000061A8;
	 default  :  lut_data15  <=  33'h1_00000000;
   endcase
end
////////////////////////////////////////////////////////////////////
endmodule
