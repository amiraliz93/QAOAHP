// ============================================================================
// I2C Bus Controller (Improved SystemVerilog Version)
// ============================================================================
module I2C_Bus_Controller(
    input  logic        iCLK,           // System Clock (e.g. 2MHz or 80kHz)
    input  logic        iRST_n,         // Active Low Reset
    input  logic        iStart,         // High Pulse to Start
    input  logic [6:0]  iSlave_addr,    // 7-bit Slave Address
    input  logic [7:0]  iWord_addr,     // Register Address
    input  logic [7:0]  wr_data,        // Data to Write
    input  logic        wr_cmd,         // 1: Write, 0: Read
    input  logic        iSequential_read,
    input  logic [7:0]  iRead_length,
    
    output logic        i2c_clk,        // I2C SCL
    inout  wire         i2c_data,       // I2C SDA
    
    output logic [7:0]  i2c_read_data,
    output logic        i2c_read_data_rdy,
    output logic        oCONFIG_DONE,
    output logic [5:0]  oSYSTEM_STATE   // For Debugging
);

    //=========================================================================
    // Parameters
    //=========================================================================
    typedef enum logic [5:0] {
        ST_IDLE            = 6'd0,
        ST_START1          = 6'd1,
        ST_SLAVE_ADDR1     = 6'd2,
        ST_SLAVE_ADDR_ACK1 = 6'd3,
        ST_WORD_ADDR1      = 6'd4,
        ST_WORD_ADDR_ACK   = 6'd5,
        ST_START2          = 6'd6,
        ST_SLAVE_ADDR2     = 6'd7,
        ST_SLAVE_ADDR_ACK2 = 6'd8,
        ST_DATA1           = 6'd9, 
        ST_NON_ACK         = 6'd10,
        ST_MASTER_ACK      = 6'd11,
        ST_STOP            = 6'd12,
        ST_WR_DATA         = 6'd14,
        ST_WR_ACK          = 6'd15
    } state_t;

    state_t i2c_state;
    logic [1:0] i2c_clk_cnt;
    logic [2:0] i2c_bit_cnt;
    logic [7:0] shift_reg;
    logic [7:0] rlen_cnt;
    logic       sda_out;
    logic       sda_mode; // 1: Output (Master), 0: Input (Slave)

    // SDA Control: Tri-state Buffer logic
    assign i2c_data = (sda_mode) ? sda_out : 1'bz;

    //=========================================================================
    // Sequential Logic
    //=========================================================================
    always_ff @(posedge iCLK or negedge iRST_n) begin
        if (!iRST_n) begin
            i2c_state         <= ST_IDLE;
            i2c_clk_cnt       <= 2'd0;
            i2c_bit_cnt       <= 3'd0;
            oCONFIG_DONE      <= 1'b0;
            i2c_clk           <= 1'b1;
            sda_out           <= 1'b1;
            sda_mode          <= 1'b1;
            i2c_read_data_rdy <= 1'b0;
            rlen_cnt          <= 8'd0;
        end else begin
            i2c_clk_cnt++; // Automatic 2-bit rollover (0->1->2->3->0)

            case (i2c_state)
                ST_IDLE: begin
                    oCONFIG_DONE      <= 1'b0;
                    i2c_clk           <= 1'b1;
                    sda_out           <= 1'b1;
                    sda_mode          <= 1'b1;
                    i2c_read_data_rdy <= 1'b0;
                    if (iStart) begin
                        i2c_state     <= ST_START1;
                        shift_reg     <= {iSlave_addr, 1'b0}; // Write mode initially
                    end
                end

                ST_START1: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= 1'b1; i2c_clk <= 1'b1; end
                        2'd2: begin sda_out <= 1'b0; end // Start Condition
                        2'd3: begin i2c_clk <= 1'b0; i2c_state <= ST_SLAVE_ADDR1; i2c_bit_cnt <= 3'd7; end
                    endcase
                end

                ST_SLAVE_ADDR1: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= shift_reg[i2c_bit_cnt]; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            if (i2c_bit_cnt == 0) i2c_state <= ST_SLAVE_ADDR_ACK1;
                            else i2c_bit_cnt <= i2c_bit_cnt - 3'd1;
                        end
                    endcase
                end

                ST_SLAVE_ADDR_ACK1: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b0; end // Master Listen
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0; 
                            sda_mode <= 1'b1; 
                            i2c_state <= ST_WORD_ADDR1;
                            shift_reg <= iWord_addr;
                            i2c_bit_cnt <= 3'd7;
                        end
                    endcase
                end

                ST_WORD_ADDR1: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= shift_reg[i2c_bit_cnt]; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            if (i2c_bit_cnt == 0) i2c_state <= ST_WORD_ADDR_ACK;
                            else i2c_bit_cnt <= i2c_bit_cnt - 3'd1;
                        end
                    endcase
                end

                ST_WORD_ADDR_ACK: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            sda_mode <= 1'b1;
                            if (wr_cmd) begin
                                i2c_state <= ST_WR_DATA;
                                shift_reg <= wr_data;
                            end else begin
                                i2c_state <= ST_START2;
                            end
                            i2c_bit_cnt <= 3'd7;
                        end
                    endcase
                end

                ST_WR_DATA: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= shift_reg[i2c_bit_cnt]; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            if (i2c_bit_cnt == 0) i2c_state <= ST_WR_ACK;
                            else i2c_bit_cnt <= i2c_bit_cnt - 3'd1;
                        end
                    endcase
                end

                ST_WR_ACK: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            sda_mode <= 1'b1;
                            i2c_state <= ST_STOP;
                        end
                    endcase
                end

                ST_START2: begin // Repeated Start for Read
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= 1'b1; i2c_clk <= 1'b1; end
                        2'd2: begin sda_out <= 1'b0; end
                        2'd3: begin 
                            i2c_clk <= 1'b0; 
                            i2c_state <= ST_SLAVE_ADDR2; 
                            shift_reg <= {iSlave_addr, 1'b1}; // Read Mode
                            i2c_bit_cnt <= 3'd7; 
                        end
                    endcase
                end

                ST_SLAVE_ADDR2: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_out <= shift_reg[i2c_bit_cnt]; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            if (i2c_bit_cnt == 0) i2c_state <= ST_SLAVE_ADDR_ACK2;
                            else i2c_bit_cnt <= i2c_bit_cnt - 3'd1;
                        end
                    endcase
                end

                ST_SLAVE_ADDR_ACK2: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            i2c_state <= ST_DATA1;
                            i2c_bit_cnt <= 3'd7;
                            rlen_cnt <= 8'd0;
                        end
                    endcase
                end

                ST_DATA1: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd2: begin i2c_read_data[i2c_bit_cnt] <= i2c_data; end
                        2'd3: begin 
                            i2c_clk <= 1'b0;
                            if (i2c_bit_cnt == 0) begin
                                i2c_read_data_rdy <= 1'b1;
                                rlen_cnt++;
                                if (iSequential_read && (rlen_cnt < iRead_length))
                                    i2c_state <= ST_MASTER_ACK;
                                else
                                    i2c_state <= ST_NON_ACK;
                            end else begin
                                i2c_bit_cnt <= i2c_bit_cnt - 3'd1;
                            end
                        end
                    endcase
                end

                ST_MASTER_ACK: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b1; sda_out <= 1'b0; i2c_read_data_rdy <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin i2c_clk <= 1'b0; i2c_state <= ST_DATA1; i2c_bit_cnt <= 3'd7; end
                    endcase
                end

                ST_NON_ACK: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b1; sda_out <= 1'b1; i2c_read_data_rdy <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd3: begin i2c_clk <= 1'b0; i2c_state <= ST_STOP; end
                    endcase
                end

                ST_STOP: begin
                    case (i2c_clk_cnt)
                        2'd0: begin sda_mode <= 1'b1; sda_out <= 1'b0; end
                        2'd1: begin i2c_clk <= 1'b1; end
                        2'd2: begin sda_out <= 1'b1; end // Stop Condition
                        2'd3: begin oCONFIG_DONE <= 1'b1; i2c_state <= ST_IDLE; end
                    endcase
                end

                default: i2c_state <= ST_IDLE;
            endcase
        end
    end

    assign oSYSTEM_STATE = i2c_state;

endmodule
