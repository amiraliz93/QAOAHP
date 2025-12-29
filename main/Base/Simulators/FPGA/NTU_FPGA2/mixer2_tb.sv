`timescale 10ns / 10ns

// todo
// test transmission
module mixer2_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;

parameter UALRFREC = 215200;
parameter CLOCKWIDTH = 100;
parameter CLOCKWIDTH_HALF = 50;
parameter CLOCKWIDTH50 =  CLOCKWIDTH*10;
parameter WAITINTERVAL =  CLOCKWIDTH*100;
parameter P=64;
parameter Ni=32;

reg  [P-1:0]  cb; // cos beta
reg  [P-1:0]  sb; // sin beta
reg  [P-1:0]   p_ar;
reg  [P-1:0]   p_ai;
wire  [P-1:0]  p_ar_o;
wire  [P-1:0]  p_ai_o;
reg  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0] info_out; // information, like addresses, enabled signal, and so on.

mixer2 #(.P(P), // number of word width
  .Ni(32) // width of additional information
  )
  mix2
  (
   .CLK(CLK),
   .RST(RST),
   .cb(cb), // cos beta
   .sb(sb),
   .p_ar(p_ar),
   .p_ai(p_ai),
   .p_ar_o(p_ar_o),
   .p_ai_o(p_ai_o),
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
logic [63: 0] data_arrayT [];
reg [12:0] addr;
reg  en_data;
reg  en_data_out;
reg [12:0] addr_out;
initial begin 
      cb <= 64'b0; // 0
      p_ar <= 0;
      data_array = {
        64'h3febf47fb376c290, // 0.8735960488466663
        64'hbfdf254d752e87f7, // -0.48665176814586125
        64'hbfe40e34ce910ff2, // -0.6267341646675944
        64'hbfe8ef7a56a86a24, // -0.7792331402336612
        64'h3f97e1db50a82394, // 0.023322512433910328
        64'hbfeffdc58f77dce0, // -0.999727993213039
        64'h3fe480e9b1c2785e, // 0.6407364341610096
        64'h3fe8917f5666f651, // 0.767760914568223
        64'h3feb4dad43339a5e, // 0.8532320320635554
        64'h3fe0b062d0efefa1, // 0.5215314942174595
        64'h3fce4602c63c3d19, // 0.23651156119783748
        64'hbfef179504b087c9, // -0.9716286746590806
        64'hbfdd778d8cb6cebf, // -0.46042193166361395
        64'h3fec680a3fe28be5, // 0.8877001998665689
        64'hbfeef0c93d33f963, // -0.9668928332683148
        64'hbfd054ea22f3417a, // -0.25518277562243663
        64'h3faf64c5f2bccf2f, // 0.06131571376900379
        64'h3feff0960b35b498, // 0.9981184214535856
        64'hbfeeb49ff3cf02c5, // -0.9595489274871275
        64'hbfd204c86c831d15, // -0.2815419254006469
        64'hbfe9f434009b0fe0, // -0.8110599529104512
        64'h3fe2b804664d3db2, // 0.5849630354004403
        64'hbfcbc58e8c94b947, // -0.21696645607284834
        64'hbfef3cdbde268dba, // -0.9761790598753841
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0,
        0,0,0,0,0,0,0,0,0,0,0
      };
      data_arrayT = {
        64'h3fe8ef7a56a86a24, // 0.7792331402336612
        64'hbfe40e34ce910ff2, // -0.6267341646675944
        64'h3fdf254d752e87f6, // 0.4866517681458612
        64'h3febf47fb376c290, // 0.8735960488466663
        64'hbfe8917f5666f651, // -0.767760914568223
        64'h3fe480e9b1c2785d, // 0.6407364341610094
        64'h3feffdc58f77dce0, // 0.999727993213039
        64'h3f97e1db50a823a2, // 0.023322512433910376
        64'h3fef179504b087c9, // 0.9716286746590806
        64'h3fce4602c63c3d1a, // 0.2365115611978375
        64'hbfe0b062d0efefa2, // -0.5215314942174596
        64'h3feb4dad43339a5d, // 0.8532320320635552
        64'h3fd054ea22f34179, // 0.2551827756224366
        64'hbfeef0c93d33f963, // -0.9668928332683148
        64'hbfec680a3fe28be5, // -0.8877001998665689
        64'hbfdd778d8cb6cebf, // -0.46042193166361395
        64'h3fd204c86c831d15, // 0.2815419254006469
        64'hbfeeb49ff3cf02c4, // -0.9595489274871274
        64'hbfeff0960b35b498, // -0.9981184214535856
        64'h3faf64c5f2bccf2d, // 0.06131571376900378
        64'h3fef3cdbde268dba, // 0.9761790598753841
        64'hbfcbc58e8c94b946, // -0.2169664560728483
        64'hbfe2b804664d3db3, // -0.5849630354004404
        64'hbfe9f434009b0fe1 // -0.8110599529104513
      };
      en_data <=0;
      addr <= 1;
      info_in <= '0;
      sb <= 0;
      cb <= 0;
      #CLOCKWIDTH50;
      
      for (i = 0; i <data_array.size()/2 ; i = i + 1) begin
            cb <= 64'h3fefbf675480d903; // 0.9921147013144779
            sb <= 64'h3fc00aeb5da15be0; // 0.12533323356430426
            en_data <= 1;
            p_ar <= data_array[i*2 + 0];
            p_ai <= data_array[i*2 + 1];
            info_in[12:0] <= addr;
            info_in[30] <= ~i[0];
            info_in[29:13] <= 0;
            info_in[31] <= 1;
            addr <= addr + 1;
            rfp64 <= p_ar_o;

            en_data_out <= info_out[13];
            addr_out <= info_out[12:0];
            #CLOCKWIDTH;
      end
      info_in <= 0;
      en_data <= 0;
      #WAITINTERVAL;

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
