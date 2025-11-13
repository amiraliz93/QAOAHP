// avalon_mm_rs232_master.v
// Fixed version to resolve the multiple driver error.

module avalon_mm_rs232_master (
    // Clock and Reset
    input  wire           clk_clk,
    input  wire           reset_reset_n,

    // Avalon Master Interface
    output reg            avm_read,
    input  wire  [31:0]   avm_readdata,
    input  wire           avm_readdatavalid,
    output reg            avm_write,
    output reg   [31:0]   avm_address,
    output reg   [31:0]   avm_writedata,
    output reg            avm_chipselect, // New chipselect signal
    output reg            avm_waitrequest,
    output reg   [3:0]    avm_byteenable,

    // --- Custom Signals ---
    // Transmit
    input  wire  [7:0]    tx_data_in,
    input  wire           tx_start,
    output wire           tx_ready,

    // Receive
    output reg   [7:0]    rx_data_out,
    output reg            rx_data_valid
);

    // --- State Machine Definitions ---
    localparam [3:0]
        IDLE            = 4'd0,
        TX_WAIT_TRDY    = 4'd1,
        TX_WRITE_DATA   = 4'd2,
        RX_WAIT_RRDY    = 4'd3,
        RX_READ_DATA    = 4'd4,
        RX_READ_WAIT    = 4'd5;

    reg [3:0] state;

    // --- RS232 Core Register Offsets ---
    localparam [31:0]
        DATA_REG_OFFSET   = 32'h0,
        STATUS_REG_OFFSET = 32'h4;

    // --- Register Bit Masks ---
    localparam [31:0]
        RRDY_MASK = 32'h00000001,
        TRDY_MASK = 32'h00000080;

    // --- Internal Registers and Wires ---
    reg  [7:0]  _tx_data_buffer;
    reg         _tx_ready_int;
    wire [31:0] status_reg_value;
    reg         _rx_data_valid_int;
    reg  [7:0]  _rx_data_out_reg;

    assign status_reg_value = avm_readdata;
    assign tx_ready = _tx_ready_int;
    
    // FSM for controlling Avalon transactions
    always @(posedge clk_clk or negedge reset_reset_n) begin
        if (!reset_reset_n) begin
            state <= IDLE;
            _tx_ready_int <= 1'b1;
            _rx_data_valid_int <= 1'b0;
            _rx_data_out_reg <= 8'h00;
            
            // Default signal values on reset
            avm_read <= 1'b0;
            avm_write <= 1'b0;
            avm_chipselect <= 1'b0;
            avm_address <= 32'h0;
            avm_writedata <= 32'h0;
            avm_byteenable <= 4'b0000;
            avm_waitrequest <= 1'b0;
            
        end else begin
            // Reset receive data valid flag
            _rx_data_valid_int <= 1'b0;

            case (state)
                IDLE: begin
                    _tx_ready_int <= 1'b1;
                    
                    // Default values for idle state
                    avm_read <= 1'b0;
                    avm_write <= 1'b0;
                    avm_chipselect <= 1'b0;
                    avm_address <= 32'h0;
                    avm_writedata <= 32'h0;
                    avm_byteenable <= 4'b0000;

                    // Check for a transmit request
                    if (tx_start) begin
                        _tx_data_buffer <= tx_data_in;
                        _tx_ready_int <= 1'b0;
                        state <= TX_WAIT_TRDY;
                    end
                    // Check for a receive request (by polling)
                    else begin
                        avm_read <= 1'b1;
                        avm_chipselect <= 1'b1;
                        avm_address <= STATUS_REG_OFFSET;
                        avm_byteenable <= 4'b0001;
                        state <= RX_WAIT_RRDY;
                    end
                end

                // --- Transmit States ---
                TX_WAIT_TRDY: begin
                    if (avm_readdatavalid) begin
                        if ((status_reg_value & TRDY_MASK) != 0) begin
                            state <= TX_WRITE_DATA;
                        end else begin
                            // Not ready, re-read status
                            avm_read <= 1'b1;
                            avm_chipselect <= 1'b1;
                            avm_address <= STATUS_REG_OFFSET;
                            avm_byteenable <= 4'b0001;
                        end
                    end
                end
                
                TX_WRITE_DATA: begin
                    avm_write <= 1'b1;
                    avm_chipselect <= 1'b1;
                    avm_address <= DATA_REG_OFFSET;
                    avm_writedata <= {24'h00, _tx_data_buffer};
                    avm_byteenable <= 4'b0001;
                    state <= IDLE;
                end

                // --- Receive States ---
                RX_WAIT_RRDY: begin
                    if (avm_readdatavalid) begin
                        if ((status_reg_value & RRDY_MASK) != 0) begin
                            state <= RX_READ_DATA;
                        end else begin
                            state <= IDLE;
                        end
                    end
                end

                RX_READ_DATA: begin
                    avm_read <= 1'b1;
                    avm_chipselect <= 1'b1;
                    avm_address <= DATA_REG_OFFSET;
                    avm_byteenable <= 4'b0001;
                    state <= RX_READ_WAIT;
                end

                RX_READ_WAIT: begin
                    if (avm_readdatavalid) begin
                        _rx_data_out_reg <= avm_readdata[7:0];
                        _rx_data_valid_int <= 1'b1;
                        state <= IDLE;
                    end
                end
                
                default: state <= IDLE;
            endcase
        end
    end

    // Register outputs
    always @(posedge clk_clk) begin
        rx_data_out   <= _rx_data_out_reg;
        rx_data_valid <= _rx_data_valid_int;
    end

endmodule
