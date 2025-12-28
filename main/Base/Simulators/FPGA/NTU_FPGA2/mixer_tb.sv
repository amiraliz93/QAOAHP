`timescale 10ns / 10ns

// todo
// test transmission
module mixer_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;

parameter UALRFREC = 215200;
parameter CLOCKWIDTH = 100;
parameter CLOCKWIDTH_HALF = 50;
parameter CLOCKWIDTH50 =  CLOCKWIDTH*10;
parameter P=64;
parameter Ni=32;

reg  [P-1:0]  cb; // cos beta
reg  [P-1:0]   p_ar;
wire  [P-1:0]  p_ar_o;
reg  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0] info_out; // information, like addresses, enabled signal, and so on.

mixer_test #(.P(P), // number of word width
  .Ni(32) // width of additional information
  )
  mix1
  (
   .CLK(CLK),
   .RST(RST),
   .cb(cb), // cos beta
   .p_ar(p_ar),
   .p_ar_o(p_ar_o),
   .info_in(info_in), // information, like addresses, enabled signal, and so on.
   .info_out(info_out) // information, like addresses, enabled signal, and so on.
);

always
begin
      #CLOCKWIDTH_HALF;
      CLK <= ~CLK; // clock generation, half period
end

reg [63:0] fp64;
reg [63:0] rfp64;
integer i;
// test for transmitter
logic [63: 0] data_array [];
reg [12:0] addr;
reg  en_data;
reg  en_data_out;
reg [12:0] addr_out;
initial begin 
      cb <= 64'b0; // 0
      p_ar <= 0;
      data_array = {
     64'h3fbecd39da16616b,
      64'h3fc0b5d462c343b7,
      64'h3fc2050bd87b56b8,
      64'h3fc354434e3369b9,
      64'h3fc4a37ac3eb7cba,
      64'h3fc5f2b239a38fbb,
      64'h3fc741e9af5ba2bc,
      64'h3fc891212513b5bd,
      64'h3fc9e0589acbc8be,
      64'h3fcb2f901083dbbf,
      64'h3fcc7ec7863beec0,
      64'h3fcdcdfefbf401c1,
      0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0,
      0,0,0,0,0,0,0,0,0,0,0
      };
      en_data <=0;
      addr <= 1;
      #CLOCKWIDTH50;
      
      for (i = 0; i <data_array.size() ; i = i + 1) begin
            cb <= 64'b0100000000010000100110001001001101110100101111000110101001111111; // 4.149
            en_data <= 1;
            p_ar <= data_array[i];
            info_in[12:0] <= addr;
            info_in[13] <= en_data;
            info_in[Ni-1:14] <= 0;
            addr <= addr + 1;
            rfp64 <= p_ar_o;

            en_data_out <= info_out[13];
            addr_out <= info_out[12:0];
            #CLOCKWIDTH;
      end
      en_data <= 0;

      $stop;
      
end
// Initialize signals
initial begin
      RST <= 1; // Reset active-high
      CLK <= 1;
      // Apply reset
      #CLOCKWIDTH50;
      RST <= 0;

      // End simulation
end

endmodule
