// ============================================================================
// Copyright (c) 2013 by Terasic Technologies Inc.
// ============================================================================
//
// Permission:
//
//   Terasic grants permission to use and modify this code for use
//   in synthesis for all Terasic Development Boards and Altrea Development 
//   Kits made by Terasic.  Other use of this code, including the selling 
//   ,duplication, or modification of any portion is strictly prohibited.
//
// Disclaimer:
//
//   This VHDL or Verilog source code is intended as a design reference
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
//
//                     web: http://www.terasic.com/
//                     email: support@terasic.com
//
// ============================================================================
//
// Major Functions:i2c controller
//
// ============================================================================
//
// Revision History :
// ============================================================================
//   Ver  :| Author            :| Mod. Date :| Changes Made:
//   V1.0 :| Joe Yang          :| 05/07/10  :|      Initial Revision
// ============================================================================
module I2C_CDCM6208_Controller (
	CLOCK,
	I2C_SCLK,//I2C CLOCK
 	I2C_SDAT,//I2C DATA
	I2C_DATA,//DATA:[SLAVE_ADDR,SUB_ADDR,DATA]
	GO,      //GO transfor
	END,     //END transfor 
	ACK,      //ACK
	RST_n,
	//TEST
	SD_COUNTER,
	SDO
);
	input  CLOCK;
	input  [39:0]I2C_DATA;	
	input  GO;
	input  RST_n;	
 	inout  I2C_SDAT;	
	output I2C_SCLK;
	output END;	
	output ACK;

//TEST
	output [5:0] SD_COUNTER;
	output SDO;


reg SDO;
reg SCLK;
reg END;
reg [39:0]SD;
reg [5:0]SD_COUNTER;

wire			falling_edge /*synthesis keep*/;
wire	    	rising_edge  /*synthesis keep*/;

wire I2C_SCLK=SCLK | ( ((SD_COUNTER >= 3) & (SD_COUNTER <=47))? i2c_clk_cnt[1] :0 );
wire I2C_SDAT=SDO?1'bz:0 ;

reg ACK1,ACK2,ACK3,ACK4,ACK5;
wire ACK=ACK1 | ACK2 |ACK3| ACK4 |ACK5;


/////////////////////////
reg	[1:0]	i2c_clk_cnt;
always@(posedge CLOCK or negedge RST_n)
	begin
		if (!RST_n)
			i2c_clk_cnt <= 0;
		else
			i2c_clk_cnt <= i2c_clk_cnt + 1'b1;
	end
	
assign rising_edge  = (i2c_clk_cnt == 3) ? 1'b1 : 1'b0;
assign falling_edge = (i2c_clk_cnt == 0) ? 1'b1 : 1'b0;
	
//--I2C COUNTER
always @(negedge RST_n or posedge CLOCK ) begin
if (!RST_n) SD_COUNTER<=6'b111111;
else begin
   if (GO==0) 
	 SD_COUNTER<=0;
	else 
	if (SD_COUNTER < 6'b111111 & rising_edge) SD_COUNTER<=SD_COUNTER+1;	
end
end



always @(negedge RST_n or  posedge CLOCK ) begin
if (!RST_n) begin SCLK<=1;SDO<=1; ACK1<=0;ACK2<=0;ACK3<=0;ACK4<=0 ;ACK5<=0 ; END<=1; end
else
case (SD_COUNTER)
	6'd0  : begin ACK1<=0 ;ACK2<=0 ;ACK3<=0 ; ACK4<=0 ;ACK5<=0 ;END<=0; SDO<=1; SCLK<=1;end
	//start
	6'd1  : begin SD<=I2C_DATA;SDO<=0;end
	6'd2  : SCLK<=0;
	//SLAVE ADDR
	6'd3  : SDO<=SD[39];
	6'd4  : SDO<=SD[38];
	6'd5  : SDO<=SD[37];
	6'd6  : SDO<=SD[36];
	6'd7  : SDO<=SD[35];
	6'd8  : SDO<=SD[34];
	6'd9  : SDO<=SD[33];
	6'd10 : SDO<=SD[32];	
	6'd11 : begin
	  SDO<=1'b1;//ACK
	  if(falling_edge)ACK1<=I2C_SDAT;
	  else   ACK1<=ACK1;
	end
	//SUB ADDRH
	6'd12  : SDO<=SD[31]; 
	6'd13  : SDO<=SD[30];
	6'd14  : SDO<=SD[29];
	6'd15  : SDO<=SD[28];
	6'd16  : SDO<=SD[27];
	6'd17  : SDO<=SD[26];
	6'd18  : SDO<=SD[25];
	6'd19  : SDO<=SD[24];
	6'd20  : begin
     SDO<=1'b1;//ACK
	  if(falling_edge)ACK2<=I2C_SDAT;
	  else   ACK2<=ACK2;
	end
	//SUB ADDRL
	6'd21  : SDO<=SD[23]; 	  
	6'd22  : SDO<=SD[22];
	6'd23  : SDO<=SD[21];
	6'd24  : SDO<=SD[20];
	6'd25  : SDO<=SD[19];
	6'd26  : SDO<=SD[18];
	6'd27  : SDO<=SD[17];
	6'd28  : SDO<=SD[16];
	6'd29  :begin
     SDO<=1'b1;//ACK
	  if(falling_edge) ACK3<=I2C_SDAT; 
	  else   ACK3<=ACK3;
	end
	//DATAH
	6'd30  : SDO<=SD[15]; 	  
	6'd31  : SDO<=SD[14];
	6'd32  : SDO<=SD[13];
	6'd33  : SDO<=SD[12];
	6'd34  : SDO<=SD[11];
	6'd35  : SDO<=SD[10];
	6'd36  : SDO<=SD[9];
	6'd37  : SDO<=SD[8];
	6'd38  : begin 
	  SDO<=1'b1;//ACK
	  if(falling_edge) ACK4<=I2C_SDAT;
	  else   ACK4<=ACK4;
	end

	//DATAL
	6'd39  : SDO<=SD[7]; 
	6'd40  : SDO<=SD[6];
	6'd41  : SDO<=SD[5];
	6'd42  : SDO<=SD[4];
	6'd43  : SDO<=SD[3];
	6'd44  : SDO<=SD[2];
	6'd45  : SDO<=SD[1];
	6'd46  : SDO<=SD[0];
	6'd47  : begin
	   SDO<=1'b1;//ACK
 	   if(falling_edge) ACK5<=I2C_SDAT; 
	   else   ACK5<=ACK5;
	 end	
	//stop
    6'd48 : begin 
	   SDO<=1'b0;	
	   SCLK<=1'b0;
	 end	
    6'd49 : SCLK<=1'b1; 
    6'd50 : SDO<=1'b1; 
	 6'd51 : SDO<=1'b1;  
	 6'd52 : SDO<=1'b1;  
	 6'd53 : SDO<=1'b1;  
	 6'd54 : SDO<=1'b1;  
 	 6'd55 : begin SDO<=1'b1; END<=1; end 

    default:  begin SDO<=1'b1; END<=1; end 
endcase
end



endmodule
