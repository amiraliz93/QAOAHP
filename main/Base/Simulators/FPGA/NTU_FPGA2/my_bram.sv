// Manual implementation of BRAM, this kind of code should be inferred as a block that is suitable to BRAM, and compiler will assign BRAM to this block
// We can use this implementation on any vendor, while IP core provided by certain vendor can be only used in that vendor.
module my_bram #(parameter ADDRESS_WIDTH=8, parameter DEPTH=256, parameter DATA_WIDTH = 64) (
    input clk,          // Clock signal
    input wen, // Write enable signal
    input [ADDRESS_WIDTH-1:0] addr_r, // Address for read
    input [ADDRESS_WIDTH-1:0] addr_w, // Address for write
    input [DATA_WIDTH-1:0] data_in,   // Data to be written
    output reg [DATA_WIDTH-1:0] data_out  // Data to be read
);

// Define the memory block (BRAM)
// The synthesis tool infers this as a BRAM based on the structure
reg [DATA_WIDTH-1:0] ram_block [0:DEPTH-1];

reg _write_enable;
reg [DATA_WIDTH-1:0] _data_in;
reg [ADDRESS_WIDTH-1:0] _addr_r;
reg [ADDRESS_WIDTH-1:0] _addr_w;
// Always block to handle read/write operations on the positive edge of the clock
always @(posedge clk) begin
    if (_write_enable) begin
        // Write operation: store data_in at the specified address
        ram_block[_addr_w] <= _data_in;
    end
    // Read operation: output the data from the specified address
    // This is a synchronous read, so data is available in the next cycle
    data_out <= ram_block[_addr_r];

    _write_enable <= wen;
    _data_in <= data_in;
    _addr_r <= addr_r;
    _addr_w <= addr_w;
end

endmodule