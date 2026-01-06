
`timescale 1ns / 1ns

// todo
// test transmission
module gen_cost_tb ();

// Declare signals to connect to the UART module
reg RST;
reg CLK;

parameter CLOCKWIDTH = 2;
parameter CLOCKWIDTH_HALF = 1;
parameter CLOCKWIDTH50 =  CLOCKWIDTH*50;
parameter WAITINTERVAL =  CLOCKWIDTH*100;
parameter P=64;
parameter Ni=32;

localparam N0 = 21; // latency FP64 to Fix 56.53(sign)
localparam N1 = 11; // latency FP64 to Fix 56.53(sign)
localparam N2 = 167; // latency of CORDIC. 56.53(sign) to 55.53(sign)
localparam N3 = 15; // latency Fix 55.53(sign) to FP64
localparam NPip = 1 + N0 + 1 + N1 + 1 * N2 + 1 + N3 + 1; // number of pipeline. Depends on IP core like addFPF64 used in this module.

reg  [P-1:0]  gamma; // gamma
reg  [P-1:0]  H;     // sin beta
reg  [P-1:0]  sol_ar_in;
reg  [P-1:0]  sol_ai_in;
wire  [P-1:0]  Hr_o;
wire  [P-1:0]  Hi_o;
reg  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
reg  [1:0]  switch; // information, like addresses, enabled signal, and so on.
wire  [Ni-1:0] info_out; // information, like addresses, enabled signal, and so on.

gen_cost #(.P(P), // number of word width
  .Ni(32) // width of additional information
  )
  cogen
  (
   .CLK(CLK),
   .RST(RST),
   .gamma(gamma), // cos gamma
   .H(H),
   .Hr_o(Hr_o),
   .Hi_o(Hi_o),
   .info_in(info_in), // information, like addresses, enabled signal, and so on.
   .info_out(info_out) // information, like addresses, enabled signal, and so on.
);

integer i;

logic [63: 0] data [];
logic [63: 0] costF [];
reg [12:0] addr;
reg [12:0] addr_out;
wire  en_data_out = info_out[13];
reg [63:0] sol_ar_Pipe [NPip];
reg [63:0] sol_ai_Pipe [NPip];
wire [63:0] sol_ar = sol_ar_Pipe[NPip-1];
wire [63:0] sol_ai = sol_ai_Pipe[NPip-1];



initial begin 
      
      for (i = 0; i < NPip; i = i + 1) begin
            sol_ar_Pipe[i] <= 0;
            sol_ai_Pipe[i] <= 0;
      end
      H <= 0;
      addr <= 0;
      info_in <= '0;
gamma = 64'hc0036abd3314f738; // -2.427118681981394;
data = {
64'hbfe476b48e489c2e, // -0.6394903925873956
64'h3fe40089b3b4e7ce, // 0.6250656614204162
64'hbfbd72c553320860, // -0.11503251343859366
64'h3fdcb9f7342b9970, // 0.44885044188267553
64'h3fe74c91f17a1a58, // 0.728096934923836
64'hbfdbc770d127ae5c, // -0.43404789376538644
64'h3fe0d43fcc220ac8, // 0.5259093272182662
64'h3fe7bbff32d9ee38, // 0.7416988366307526
64'h3fe020d026bacb80, // 0.5040055042685339
64'h3fe92a6eb2eb5a10, // 0.7864297384756487
64'h3fcf330ce77bd6f8, // 0.24374543478887012
64'hbfe80adc3109a470, // -0.7513256986395742
64'h3fe8f70f10c29dc2, // 0.780158550963215
64'hbfc31a3414de50e0, // -0.14923716563192269
64'h3faf6a876c766b60, // 0.06135962676054052
64'h3fe49f76868acc2e, // 0.6444656970765033
64'hbfd7834a0d7c6da4, // -0.36738826100346444
64'hbfebb33a16d71488, // -0.8656282850357835
64'hbf61e1e660cbbe00, // -0.0021829127857626585
64'hbfdc034f8e9eb784, // -0.4377020733617132
64'h3fd782927e4836cc, // 0.3673444970344064
64'hbfd896630f616d60, // -0.38417889120110793
64'hbfcd37e6dbc2bc20, // -0.22826848726751425
64'h3fc641564ab0e1d0, // 0.17386892935770826
64'h3fd9fd04d7caa3c4, // 0.4060680491771189
64'h3fecdcdf179e8d88, // 0.9019618474307416
64'h3fe380ce7948b99c, // 0.6094734543637972
64'h3fd64cfff9163808, // 0.34844970059282376
64'h3fc920e027ef1b80, // 0.19631578397658345
64'hbfe025e1611542e4, // -0.5046240707543714
64'hbfdd49551c70a904, // -0.4576008584723612
64'hbfe283b889226d06 // -0.578579204407476
};
costF = {
64'h3f931fd80623ba49, 64'h3feffe9239ce93e2, // (0.018676162123421745+0.9998255852739215j)
64'h3fab7995c1671b5d, 64'hbfeff43255619e09, // (0.05366199479936171-0.9985591571430074j)
64'h3feec2c7bf06ebcf, 64'h3fd1a32ca2f6250f, // (0.9612768870350638+0.275584372655957j)
64'h3fdda1e1b9facd4c, 64'hbfec5d0609d80b42, // (0.4630054775436363-0.8863554184211824j)
64'hbfc8f9bde9632c9a, 64'hbfef628acc749c10, // (-0.1951215161883269-0.9807790749808891j)
64'h3fdfa69d97d52589, 64'h3febd0189cf9a953, // (0.4945444090595506+0.8691523614809661j)
64'h3fd2915274242f75, 64'hbfee9faa392aa866, // (0.2901197561447966-0.9569903484855444j)
64'hbfcd1b0e6fad30ee, 64'hbfef296769d1770d, // (-0.22738819554286444-0.9738041941415944j)
64'h3fd5cbc6e03cbdc9, 64'hbfee164c27fda5cd, // (0.34056255243464456-0.9402218609877139j)
64'hbfd5385c9f058f23, 64'hbfee30990f302f75, // (-0.3315650513597907-0.9434323593754771j)
64'h3fea8fc4e4d9e639, 64'hbfe1d896bdfa9a59, // (0.830049941039028-0.5576890669370399j)
64'hbfd001451e962b53, 64'h3feefbb4b682a8f3, // (-0.25007751452419774+0.96822581907807j)
64'hbfd44c780c648185, 64'hbfee590b36caf877, // (-0.31716729364600155-0.9483696050808835j)
64'h3fedec73b9ab8749, 64'h3fd6ada195472ce8, // (0.9351137758605798+0.3543476064484836j)
64'h3fefa5524ac9bd92, 64'hbfc2fe0616a679f3, // (0.9889308415130691-0.14837719064685437j)
64'h3f7b0a38e579f91a, 64'hbfefffd24d58d06d, // (0.006601545573742729-0.9999782095606072j)
64'h3fe41958b427e3ad, 64'h3fe8e6807707f91c, // (0.628094055048782+0.7781374287440346j)
64'hbfe02ea541bf2ba9, 64'h3feb9b5813ea1707, // (-0.505694034968987+0.8627128972009083j)
64'h3fefffe290d1ccda, 64'h3f75b386f75c2369, // (0.9999859646326528+0.005298163616097231j)
64'h3fdf280020434080, 64'h3febf3bf48b63ca4, // (0.48681643629698357+0.8735042972711153j)
64'h3fe41a0608cbef12, 64'hbfe8e5f48a98cf9b, // (0.6281767055329246-0.7780707079859782j)
64'h3fe3115dff596284, 64'h3fe9b2d6b43b6434, // (0.5958700167930862+0.8030808944851118j)
64'h3feb368c4ceb6347, 64'h3fe0d5ff2d4a4577, // (0.8504086973508515+0.5261226543972687j)
64'h3fed3153bebb03f4, 64'hbfda36a96f1cddcd, // (0.9122713780599256-0.40958629465918933j)
64'h3fe1ad205fca9b54, 64'hbfeaacc505bc04c7, // (0.5523836012770098-0.8335900413514076j)
64'hbfe28cfaeeb6257e, 64'hbfea13082b51e9e1, // (-0.5797094976743862-0.8148232313245068j)
64'h3fb766436108e08d, 64'hbfefddb510b6dafb, // (0.09140416258822111-0.9958138777209052j)
64'h3fe538d1d95f7743, 64'hbfe7f3571dcdde28, // (0.6631860013546674-0.7484546262848584j)
64'h3fec6f86d883c4f0, 64'hbfdd5a9d157bd6c2, // (0.8886141041567992-0.4586556157877163j)
64'h3fd5b4a4a45c7b84, 64'h3fee1a7a34268070, // (0.3391505818453242+0.9407320993959853j)
64'h3fdc6bc542386f7b, 64'h3fecabef841fdf2a, // (0.444077791851434+0.8959882336193676j)
64'h3fc53736aac906f0, 64'h3fef8eb0ab5733fe // (0.16574748362659575+0.9861682268616503j)
};


      #CLOCKWIDTH50;
      #CLOCKWIDTH50;
      
      for (i = 0; i <data.size() ; i = i + 1) begin
            H <= data[i];
            info_in[12:0] <= addr;
            info_in[13] <= 1;
            info_in[Ni-1:14] <= '0;
            addr <= addr + 1;

            addr_out <= info_out[12:0];
            sol_ar_in <= costF[i*2 + 0];
            sol_ai_in <= costF[i*2 + 1];
            #CLOCKWIDTH;
      end
      for(i=0;i<NPip;i=i+1) begin
            info_in <= '0;
            H <= '0;
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
integer j;
always @(posedge CLK) begin
      
      sol_ar_Pipe[0] <= sol_ar_in;
      sol_ai_Pipe[0] <= sol_ai_in;
      for (j = 0;j < NPip-1; j = j + 1) begin
            sol_ai_Pipe[j+1] <= sol_ai_Pipe[j];
            sol_ar_Pipe[j+1] <= sol_ar_Pipe[j];
      end
end
endmodule
