`timescale 1ns / 1ns

// todo
// test transmission
module mixer2_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;

parameter CLOCKWIDTH = 2;
parameter CLOCKWIDTH_HALF = 1;
parameter CLOCKWIDTH50 =  CLOCKWIDTH*50;
parameter WAITINTERVAL =  CLOCKWIDTH*100;
parameter P=64;
parameter Ni=32;
parameter N1 = 1 + 11 ;
parameter N3 = 1 + 11 + 1 + 2; // suppose  add1_a visible at cycle 13 and add1_res on cycle 14


reg  [P-1:0]  cb; // cos beta
reg  [P-1:0]  sb; // sin beta
reg  [P-1:0]   p_r;
reg  [P-1:0]   p_i;
reg  [P-1:0]   sol_ar_in;
reg  [P-1:0]   sol_ai_in;
wire  [P-1:0]  p_r_o;
wire  [P-1:0]  p_i_o;
reg  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
reg  [1:0]  switch; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0] info_out; // information, like addresses, enabled signal, and so on.

Update_mixer #(.P(P), // number of word width
  .Ni(32) // width of additional information
  )
  mix2
  (
   .CLK(CLK),
   .RST(RST),
   .cos_beta(cb), // cos beta
   .sin_beta(sb),
   .p_r(p_r),
   .p_i(p_i),
   .p_r_o(p_r_o),
   .p_i_o(p_i_o),
   .switch_in(switch),
   .info_in(info_in), // information, like addresses, enabled signal, and so on.
   .info_out(info_out) // information, like addresses, enabled signal, and so on.
);
integer i;

reg [63:0] fp64;
reg [63:0] rfp64;
reg [63:0] cosb;
reg [63:0] sinb;
reg [63:0] costFR;
reg [63:0] costFI;
// test for transmitter
localparam D = 4;
logic [63: 0] data [0:D-1];
logic [63: 0] solM [0:D-1];
logic [63: 0] solC [0:D-1];
logic [63: 0] costF [0:D-1];
reg [12:0] addr;
reg [12:0] addr_out;
wire  en_data_out = info_out[13];
reg [23:0] countSol;
reg [63:0] sol_ar_Pipe [N3];
reg [63:0] sol_ai_Pipe [N3];
wire [63:0] sol_ar = sol_ar_Pipe[N3-1];
wire [63:0] sol_ai = sol_ai_Pipe[N3-1];

integer cycle_count;
integer start_cycle;
initial begin 
      
      for (i = 0; i < N3; i = i + 1) begin
            sol_ar_Pipe[i] <= 0;
            sol_ai_Pipe[i] <= 0;
      end
p_r <= 0;
//cosb = 64'h00c5d3eaeb8b5119;sinb = 64'h0087e42e07d59113;data = {64'h802d1530e1cca30e, 64'h00d91866f66c741c};costF = {64'h008a50bf27c61910, 64'h00453a557289a71b};solM = {64'h00569381d25c79f8, 64'h00870f876706bc1a};solC = {64'h00cebbb14f7bc6ee, 64'h00a234d938fcf71a};
cosb = 64'h19518bebead3c500;sinb = 64'h1391d5072ee48700;data = {64'h0c9d0561b6b10a00, 64'he2973c6b0f92a900, 64'h0a588d4eeeb1ca00, 64'he1b7f7359e1b3400};costF = {64'hecd52625b52b3700, 64'he6602583c72ab100,64'h02892a469ae444c0, 64'he019c2fc2af09100};solM = {64'h1c7f960748f0aa00, 64'hef0eef739aeea800, 64'h1a2bc666dfccd500, 64'hefc1450de5baf900};solC = {64'he0e51eb40d1ce200, 64'h0783e7526f3d3e80, 64'he2a2399cc783bc00, 64'hf34979f632a11600};
      addr <= 0;

      info_in <= '0;
      sb <= 0;
      cb <= 0;
      cycle_count <= 0;
      #CLOCKWIDTH50;
      #CLOCKWIDTH50;
      
      for (i = 0; i <D/2 ; i = i + 1) begin
            cb <= cosb; 
            sb <= sinb; 
            p_r <= data[i*2 + 0];
            p_i <= data[i*2 + 1];
            switch <= {i[0], ~i[0]};
            info_in[12:0] <= addr;
            info_in[13] <= 1;
            info_in[Ni-1:14] <= 0;
            addr <= addr + 1;
            rfp64 <= p_r_o;


            addr_out <= info_out[12:0];
            sol_ar_in <= solM[i*2 + 0];
            sol_ai_in <= solM[i*2 + 1];
            #CLOCKWIDTH;
      end
      for(i=0;i<N3;i=i+1) begin
            info_in <= 0;
            p_r <= 0;
            p_i <= 0;
            #CLOCKWIDTH;
      end
      for (i = 0; i <D/2 ; i = i + 1) begin
            cb <= costF[i*2 + 0];
            sb <= costF[i*2 + 1]; 
            p_r <= data[i*2 + 0];
            p_i <= data[i*2 + 1];
            switch <= {0, 0};
            info_in[12:0] <= addr;
            info_in[13] <= 1;
            info_in[Ni-1:14] <= 0;
            addr <= addr + 1;
            rfp64 <= p_r_o;

            addr_out <= info_out[12:0];
            sol_ar_in <= solC[i*2 + 0];
            sol_ai_in <= solC[i*2 + 1];
            #CLOCKWIDTH;
      end
      
      for(i=0;i<N3;i=i+1) begin
            info_in <= 0;
            p_r <= 0;
            p_i <= 0;
            #CLOCKWIDTH;
      end
      info_in <= 0;
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

always
begin
      #CLOCKWIDTH_HALF;
      CLK <= ~CLK; // clock generation, half period
end
always @(posedge CLK) begin
  if (RST) begin
    cycle_count <= 0;
    start_cycle <= 0;
  end else begin
    cycle_count <= cycle_count + 1;
    if (info_in[13]) start_cycle <= cycle_count;
    if (info_out[13]) $display("latency=%0d", cycle_count - start_cycle);
  end
end
integer j;
always @(posedge CLK) begin
      
      sol_ar_Pipe[0] <= sol_ar_in;
      sol_ai_Pipe[0] <= sol_ai_in;
      for (j = 0;j < N3-1; j = j + 1) begin
            sol_ai_Pipe[j+1] <= sol_ai_Pipe[j];
            sol_ar_Pipe[j+1] <= sol_ar_Pipe[j];
      end
end
endmodule
