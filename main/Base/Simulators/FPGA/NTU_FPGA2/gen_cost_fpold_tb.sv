
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
localparam N0 = 20; // latency FP64 to Fix 56.53(sign)
localparam N1 = 11; // latency FP64 to Fix 56.53(sign)
localparam N2 = 172; // latency of CORDIC. 56.53(sign) to 55.53(sign)
localparam N3 = 15; // latency Fix 55.53(sign) to FP64
localparam NPip = 1 + N0 + 1 + N1 + 1 + N2 + 1 + N3 + 1; // number of pipeline. Depends on IP core like addFPF64 used in this module.

reg  [P-1:0]  gamma; // gamma
reg  [P-1:0]  H;     // sin beta
reg  [P-1:0]  sol_ar_in;
reg  [P-1:0]  sol_ai_in;
wire  [P-1:0]  Hr_o;
wire  [P-1:0]  Hi_o;
reg  [1:0]  switch; // information, like addresses, enabled signal, and so on.

gen_cost #(.P(P) // number of word width4
  )
  cogen
  (
   .CLK(CLK),
   .RST(RST),
   .gamma(gamma), // cos gamma
   .H(H),
   .Hr_o(Hr_o),
   .Hi_o(Hi_o)
   );

integer i;

logic [63: 0] data [];
logic [63: 0] costF [];
reg [12:0] addr;
reg [12:0] addr_out;
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
gamma = 64'h3feccf593242d6a8;data = {64'hbfec228b81305210,64'h3fd7fc2441814810,64'h3fb9ac1d4945a0a0,64'hbf7a14db997c9400,64'h3fdda487209a5ef8,64'h3fdc9f6233274c44,64'h3fdcd4e126dbce54,64'h3fec5218973ce588,64'hbfc0544d17d16f48,64'h3fee88ec4ee8315e,64'hbfd477a64e41f184,64'hbfeac081e01bdcae,64'h3fda5f21f7bafae4,64'hbfe22bbaa91b5e7c,64'h3fefb90512f6c012,64'hbfe13c3f888c99fa,64'h3fe6b1f6d920bd32,64'h3fee59c4ca65991c,64'h3febbec0624700e0,64'h3fe6d7c632ea48e0,64'hbfdbcdd2477d1578,64'hbfc3569ecfcd2868,64'hbfd96f70e7d4d348,64'hbfe8b78d1a289e82,64'hbf7f0de7cae9a400,64'hbfd9b51f1fd736c8,64'h3fe50646ab3703e6,64'h3fcf975796ea5c78,64'hbfc9aab58d9c8a38,64'hbfcc227d45427438,64'hbfcb1e1b8e295b10,64'hbfe54e2632ad1ef0};costF = {64'h3fe67cc3d3219d7a, 64'hbfe6c440835494f1,64'h3fee321c4c60d5c1, 64'h3fd52fbef6948ff6,64'h3fefdea26046637e, 64'h3fb714e97f9c1371,64'h3fefffdd89f97138, 64'hbf777b3757f4beb0,64'h3fed4207ddb2f5d8, 64'h3fd9ebc0b5b2e6e7,64'h3fed70de9cb5e700, 64'h3fd9141cc93bbf35,64'h3fed6766903c89d4, 64'h3fd940655a8ad318,64'h3fe65e3a7b7fb735, 64'h3fe6e241b84482be,64'h3fefca06b5850466, 64'hbfbd569ac423fe60,64'h3fe4e66ac8b3dc90, 64'h3fe83b60cf0e3ac9,64'h3feeaec8d8259b62, 64'hbfd22c684cbaed62,64'h3fe75b1f84baa03f, 64'hbfe5dfe804348600,64'h3fedd2b8cc8cb2b0, 64'h3fd733a89754a336,64'h3febe8985b62fb8b, 64'hbfdf4fe3e90225d2,64'h3fe41425b21b5f61, 64'h3fe8eab1ed597693,64'h3fec4f983f0d6fd4, 64'hbfddd5139e7813ed,64'h3fe9b1fd7ce23a7d, 64'h3fe31282b245967e,64'h3fe5067e3887bad9, 64'h3fe81f914059ddaa,64'h3fe6bc569e2f415d, 64'h3fe684c40fb96563,64'h3fe99da51514b981, 64'h3fe32dcd5f3f5257,64'h3fed9555ada15756, 64'hbfd8661b9bfc7b26,64'h3fefb455c4ea0ab7, 64'hbfc15b5e0a430c6f,64'h3fedf92cbc2df58a, 64'hbfd66a083ca4c98d,64'h3fe891c4369dede0, 64'hbfe4809728e63b95,64'h3fefffcf2529bd28, 64'hbf7bf55519f98abe,64'h3fedee22124f6239, 64'hbfd6a4c040403c3b,64'h3fea9022b0702c2f, 64'h3fe1d80b20a8d866,64'h3fef36986f88a406, 64'h3fcc355af8ae1275,64'h3fef7add9f92c778, 64'hbfc6fb99cb6c6d9a,64'h3fef601ed8f7164b, 64'hbfc92a3db0ee6380,64'h3fef6b6fd9a8f81f, 64'hbfc8443a8c95a7bf,64'h3fea6bd74b157a8d, 64'hbfe20d9d451db260};

      #CLOCKWIDTH50;
      #CLOCKWIDTH50;
      
      for (i = 0; i <data.size() ; i = i + 1) begin
            H <= data[i];
            addr <= addr + 1;

            sol_ar_in <= costF[i*2 + 0];
            sol_ai_in <= costF[i*2 + 1];
            #CLOCKWIDTH;
      end
      for(i=0;i<NPip;i=i+1) begin
            sol_ar_in <= 0;
            sol_ai_in <= 0;
            H <= '0;
            #CLOCKWIDTH;
      end
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
