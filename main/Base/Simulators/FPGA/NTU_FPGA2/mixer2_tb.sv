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
parameter N1 = 1 + 20 + 1;
parameter N3 = 1 + 20 + 1 + 2 + 27 + 1;
parameter NPip = N3; // number of pipeline. Depends on IP core like addFPF64 used in this module.

reg  [P-1:0]  cb; // cos beta
reg  [P-1:0]  sb; // sin beta
reg  [P-1:0]   p_ar;
reg  [P-1:0]   p_ai;
reg  [P-1:0]   sol_ar_in;
reg  [P-1:0]   sol_ai_in;
wire  [P-1:0]  p_ar_o;
wire  [P-1:0]  p_ai_o;
reg  [Ni-1:0]  info_in; // information, like addresses, enabled signal, and so on.
reg  [1:0]  switch; // information, like addresses, enabled signal, and so on.
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
logic [63: 0] data [];
logic [63: 0] solM [];
logic [63: 0] solC [];
logic [63: 0] costF [];
reg [12:0] addr;
reg [12:0] addr_out;
wire  en_data_out = info_out[13];
reg [23:0] countSol;
reg [63:0] sol_ar_Pipe [NPip];
reg [63:0] sol_ai_Pipe [NPip];
wire [63:0] sol_ar = sol_ar_Pipe[NPip-1];
wire [63:0] sol_ai = sol_ai_Pipe[NPip-1];



initial begin 
      
      for (i = 0; i < NPip; i = i + 1) begin
            sol_ar_Pipe[i] <= 0;
            sol_ai_Pipe[i] <= 0;
      end
p_ar <= 0;
cosb = 64'h3fd05aca2bd85315;sinb = 64'h3feef0029dfd45b5;data = {64'hbfcda1402ea358b6, 64'hbfef217ee850592b,64'h3fee0021abb1f147, 64'hbfd644ae95d61e0d,64'h3fef2fff9d1847b6, 64'hbfcca92c6445b413,64'hbfed9b41ca8254bc, 64'h3fd8494ea870a402,64'hbfe8ab1b720da0b3, 64'hbfe46212aa8178b2,64'h3fe00ad0a266239d, 64'hbfebb0399e9e8aaf,64'h3fca7644f4777555, 64'h3fef4f077c125834,64'hbfebd305561503f4, 64'h3fdf9c53a5b15786,64'hbfdfdac9365e8403, 64'h3febc130bcf3b585,64'hbfea3af63bed36d2, 64'h3fe25459ad300567,64'hbfedcfdc7b3ce916, 64'h3fd7425876f59b66,64'h3feee7e5204f1daf, 64'h3fd097b84c8aa0fd,64'h3fef4c9d42ad88ad, 64'hbfcaa3d349355154,64'h3fefff5c485a83b1, 64'hbf899704e1266c4b,64'h3fef7ec775b174dc, 64'h3fc6a52e461f879e,64'h3fef6a024e135110, 64'hbfc861bcb741bcfd,64'h3f86f2a795051fc9, 64'hbfefff7c583d736f,64'hbfe9168ffb302435, 64'hbfe3dd3aabf34e45,64'hbfd5da5a2daefcdb, 64'h3fee13a763f1ad7e,64'hbfecedbcebae2627, 64'h3fdb5bf910163c3e,64'hbfec0277c0ae66fa, 64'h3fdef2f15d20a23d,64'h3fefa2c2e0cf329e, 64'h3fc341cbb0c551a0,64'h3fef614d40f7202b, 64'h3fc912a1946061d9,64'h3fe8dae527e7c807, 64'hbfe427b159160eae,64'hbfe7f1af7a0d963e, 64'h3fe53aafcfae7dee,64'hbfef8e6e1a95025a, 64'h3fc53d65f37c97ad,64'hbfd0e0a22cbc2fa0, 64'h3feede04b8ffa7bc,64'hbfda43da68b07ecb, 64'hbfed2e5cc31a9830,64'hbfea57991ee9fd36, 64'h3fe22b1a11ec128f,64'h3fefabf46e60ed7b, 64'h3fc249c3e14ae879,64'h3fe599ffb82a2b11, 64'h3fe79bd67e3ccef8,64'h3fdaa9e713a3f8a6, 64'h3fed173070b586e2};costF = {64'h3feb22d2231fe303, 64'hbfe0f5b8a905db15,64'h3fefddea5a017529, 64'hbfb75417fe2f98c0,64'h3fea22e6698e78fe, 64'hbfe27698de511de2,64'h3fec92f407e37595, 64'hbfdccfb920be7a83,64'h3fe7073d8cf8e444, 64'hbfe638255c6251ac,64'h3fede3a1f2c84bbd, 64'h3fd6dbf7cf50a17b,64'h3fed9c5119737713, 64'h3fd844230d82e2b2,64'h3feffe717ca5c0e8, 64'hbf93f63b6f47eec3,64'h3fef63498d7c7231, 64'hbfc8eabded0d475c,64'h3fef915964a6018e, 64'hbfc4f78bb8c1949f,64'h3fefebef13099a73, 64'h3fb1e83874a02450,64'h3fef56e4d0672a08, 64'h3fc9dfa42b6a39de,64'h3fe8e9bb139aee6c, 64'hbfe41557f7b134e0,64'h3fe76f3b03bda6e9, 64'h3fe5ca5c16363891,64'h3fef189427b45fd6, 64'hbfce359d7ed81ad8,64'h3feffda3db536d7e, 64'h3f9893daaea46d5e,64'h3fea7e12c738e0c7, 64'h3fe1f2d13399ec98,64'h3fe7f562761f6a70, 64'h3fe53682fdd0fae6,64'h3febf4ea3936863b, 64'hbfdf23cef878e138,64'h3febf9c693889183, 64'hbfdf1253ef9a89d9,64'h3fef42bec6510b77, 64'h3fcb5ac7b2d84a8c,64'h3fe75f100ffd3986, 64'h3fe5dbb26a68f3d6,64'h3fef27cfa12d1d86, 64'h3fcd364a6f63024f,64'h3feffd773ee1f006, 64'h3f9977fbdf78560a,64'h3fe8be806bd12229, 64'h3fe44a8305bfa32a,64'h3feff20e3334acf7, 64'hbfaddc76ae1d2348,64'h3feb37750ec23f6c, 64'hbfe0d486ddd58fbe,64'h3feeef929d484811, 64'h3fd05e194cf52a7d,64'h3fe83ed41ff698d1, 64'h3fe4e269fb5ec91f,64'h3febc56c680eb5f6, 64'h3fdfcc03e6c43224,64'h3fefe37f0315d5fb, 64'h3fb55646c8aa6276,64'h3fef3d1447e704cf, 64'hbfcbc196fe31cb4d};solM = {64'h3fd1be3c6883e50e, 64'h3fe50c99d607abd6,64'h3ff2e1be0e51053e, 64'hbfd403768fbafa4c,64'hbfbe2996c49b5cfe, 64'hbfee7459696e71bd,64'hbf946d839684ab60, 64'h3ff0a0a6f4dccd0c,64'h3fe4771c9080de03, 64'h3fd49a0e1901f27e,64'h3fe7ce48c1b9bcd3, 64'hbfeeecc4bd1d2ce1,64'hbfdb2e17e1c7414e, 64'hbfe2e65a06db9608,64'hbff2b092230b5bf5, 64'h3fd4de99844efb31,64'hbfe5ca7f7f295d13, 64'hbfe24457ab94085e,64'hbff0c49e1c8c99db, 64'hbfd56dda42a00dee,64'hbfdf47333c18f1da, 64'h3ff06cff7ec75a5e,64'hbfbac4119cf106c0, 64'hbfeab3bc2d0df008,64'h3fd0c507f4a12e74, 64'h3fed3bb47f33934a,64'h3fdd3b28f90c2b37, 64'h3fee286c5f92e0ef,64'h3fdbe1ff401a46b8, 64'h3fefd15aa5129403,64'h3fb46f1ee9226316, 64'h3fece453009c5aca,64'h3fe34bd8b79e169f, 64'hbff0374ad8eb83a4,64'h3fe886485b963f4e, 64'hbfc2eaf8671dcfe6,64'hbfe004821521042c, 64'hbfe448460485e19e,64'hbff23c3ddcceec66, 64'hbfcc459be2d004ac,64'hbfd79fc2c7651513, 64'h3ff14517ef06e857,64'hbfcb817f9c7cd27c, 64'hbfe9d973ff5a5dba,64'h3feb8138e11a963e, 64'h3fe9a1b1078c9588,64'h3f82a693caa89100, 64'h3fe93010ee902930,64'hbfd68135a7cdbb03, 64'hbfe9156c7c78bc5b,64'hbfec969e3668a636, 64'hbfe5cacc60ad4a12,64'h3fea0e4831b18a30, 64'hbfc33c0c01203d20,64'hbff099624aee4a07, 64'hbfdf3b274eb4eb20,64'hbfd64db28ac5b63d, 64'h3ff1a1a6f6ba97ff,64'hbfd2f18471ee3726, 64'hbfe84c9a12a2c4ba,64'hbfe69ac8fb896c62, 64'h3fe2ec15e41a4403,64'hbfe36b05572784da, 64'h3fec51751c3df4af};solC = {64'hbfe6c7de9064c1ec, 64'hbfe67919af3ce8ba,64'h3fecdc6f14b26af2, 64'hbfdba4b24e4d2fe1,64'h3fe556aad165af70, 64'hbfe7d8c34f626477,64'hbfe4f85cd248dc3b, 64'h3fe82bdad4efae9a,64'hbfefe7c0dcf516c2, 64'h3fb3ae8840f706e0,64'h3fe8dfa904ccfc1d, 64'hbfe421cf99bd95c6,64'hbfc6ff6d931d109a, 64'h3fef7ab0e6ac3aee,64'hbfeb82ca9abe04b4, 64'h3fe05840265fa3ed,64'hbfd4708a04bf5895, 64'h3fee52fd2c4d5674,64'hbfe6dfa232207451, 64'h3fe660e92cc7e5ea,64'hbfee8d6b5d562932, 64'h3fd30811842b0ac6,64'h3fec97425f944dde, 64'h3fdcbe9db8b285fc,64'h3fe4300892f2b482, 64'hbfe8d41f5f30f546,64'h3fe7b476d745c08d, 64'h3fe57ef6a0327b0c,64'h3feff10ee7b3ba27, 64'hbfaee8d9b366af00,64'h3fef8d25236da629, 64'hbfc55bdb7a2d72ab,64'h3fe23e859fca5aa0, 64'hbfea4a299fb7993e,64'hbfc6765121ce1804, 64'hbfef80e05618f591,64'h3fc45a6a6c12b800, 64'h3fef97c61fb9fbff,64'hbfe2a6339f7f6e08, 64'h3fea01041b517061,64'hbfeeab691809788d, 64'h3fd2432202abf710,64'h3fe3d11e504e0d51, 64'h3fe92022055b5915,64'h3fed1f16457841a9, 64'h3fda87548f79ad03,64'h3fe95941e7b17033, 64'hbfe387d7163d5f49,64'hbfeff9eb3946c2ce, 64'h3fa3b96e623f2010,64'hbfef3165e61a4e34, 64'h3fcc90c385393c8a,64'h3fd21d3dfd5bb941, 64'h3feeb106b36a3d07,64'hbfc4ee84773f0ae4, 64'hbfef91b94134eacc,64'hbfefd0e80df93a90, 64'hbfbb6910db8ef60c,64'h3fe936fa59867bd0, 64'h3fe3b404c4f324a5,64'h3fe38f046c27d32d, 64'h3fe953b8a7aa5586,64'h3fe352a1d0916be8, 64'h3fe981e861133012};

      addr <= 0;
      info_in <= '0;
      sb <= 0;
      cb <= 0;
      #CLOCKWIDTH50;
      #CLOCKWIDTH50;
      
      for (i = 0; i <data.size()/2 ; i = i + 1) begin
            cb <= cosb; 
            sb <= sinb; 
            p_ar <= data[i*2 + 0];
            p_ai <= data[i*2 + 1];
            switch <= {i[0], ~i[0]};
            info_in[12:0] <= addr;
            info_in[13] <= 1;
            info_in[Ni-1:14] <= 0;
            addr <= addr + 1;
            rfp64 <= p_ar_o;

            addr_out <= info_out[12:0];
            sol_ar_in <= solM[i*2 + 0];
            sol_ai_in <= solM[i*2 + 1];
            #CLOCKWIDTH;
      end
      for(i=0;i<NPip;i=i+1) begin
            info_in <= 0;
            p_ar <= 0;
            p_ai <= 0;
            #CLOCKWIDTH;
      end
      for (i = 0; i <data.size()/2 ; i = i + 1) begin
            cb <= costF[i*2 + 0];
            sb <= costF[i*2 + 1]; 
            p_ar <= data[i*2 + 0];
            p_ai <= data[i*2 + 1];
            switch <= {0, 0};
            info_in[12:0] <= addr;
            info_in[13] <= 1;
            info_in[Ni-1:14] <= 0;
            addr <= addr + 1;
            rfp64 <= p_ar_o;

            addr_out <= info_out[12:0];
            sol_ar_in <= solC[i*2 + 0];
            sol_ai_in <= solC[i*2 + 1];
            #CLOCKWIDTH;
      end
      
      for(i=0;i<NPip;i=i+1) begin
            info_in <= 0;
            p_ar <= 0;
            p_ai <= 0;
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
