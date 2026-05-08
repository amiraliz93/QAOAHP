module Fan_Control(
    input              CLK_2M,       // 2MHz clock provided by PLL
    input              RST_N,
    input       [12:0] Speed_Set,    // 13'd2000 or 13'd5000
    input              Alert_Clear,
    input              Alert,
    output      [3:0]  Alert_Type,
    output      [13:0] Speed_Detected,
    output             FAN_I2C_SCL,
    inout              FAN_I2C_SDA
);

reg [7:0] KTACH;

// 計算式: (12'd992 * 4) / (Speed_Set / 60) - 1
// Speed_Set = 2000 => 3968 / 33.33 - 1 ≒ 118 (8'h76)
// Speed_Set = 5000 => 3968 / 83.33 - 1 ≒ 46  (8'h2E)
always @(posedge CLK_2M or negedge RST_N) begin
    if (!RST_N) begin
        KTACH <= 8'hFF; // Default
    end else begin
        case (Speed_Set)
            13'd2000: KTACH <= 8'h76; // 118
            13'd5000: KTACH <= 8'h2E; // 46
            default:  KTACH <= 8'h76; // For the safety, default does not stop fan
        endcase
    end
end

wire        i2c_reg_control_start;
wire        wr_cmd;
wire [6:0]  slave_addr;
wire [7:0]  reg_addr;
wire [7:0]  reg_wdata;
wire        i2c_rdata_rdy;
wire [7:0]  i2c_rdata;
wire        i2c_cmd_finish;

I2C_Config u0 (
    .iClk           (CLK_2M), 
    .iRST_n         (RST_N),
    .oStart         (i2c_reg_control_start),
    .oSlave_Addr    (slave_addr),
    .oWord_Addr     (reg_addr),
    .owdata         (reg_wdata),
    .owcmd          (wr_cmd),
    .Speed_Set      (KTACH),          // Actual speed
    .Speed_Detected (Speed_Detected),
    .Alert_Type     (Alert_Type),
    .Alert          (Alert),
    .Alert_Clear    (Alert_Clear),
    .iReadData      (i2c_rdata),
    .iReadData_rdy  (i2c_rdata_rdy),
    .iCONFIG_DONE   (i2c_cmd_finish)
);

I2C_Bus_Controller u1 (
    .iCLK           (CLK_2M),
    .iRST_n         (RST_N),
    .iStart         (i2c_reg_control_start),
    .iSlave_addr    (slave_addr),
    .iWord_addr     (reg_addr),
    .iSequential_read(1'b0),
    .iRead_length   (8'd1),
    .i2c_clk        (FAN_I2C_SCL),
    .i2c_data       (FAN_I2C_SDA),
    .i2c_read_data  (i2c_rdata),
    .i2c_read_data_rdy(i2c_rdata_rdy),
    .wr_data        (reg_wdata),
    .wr_cmd         (wr_cmd),
    .oCONFIG_DONE   (i2c_cmd_finish)
);

endmodule