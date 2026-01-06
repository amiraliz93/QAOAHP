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
parameter N1 = 21 + 1;
parameter N3 = 21 + 1 + 2 + 27 + 1;
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
cosb = 64'hbfe8a077ab5ae95a;sinb = 64'hbfe46eec412b687f;data = {64'h3fee82396b42410d, 64'hbfd34f623756390f,64'hbfeffd03dec97132, 64'hbf9ba3efd9978948,64'hbfd48074689128fc, 64'h3fee504d70175a48,64'hbfd05d80505de99a, 64'h3feeefa6d9339488,64'h3feebca25dce0ea0, 64'h3fd1cdd51ac06dab,64'h3fe0c4c1b6f98ee7, 64'h3feb412f3d309992,64'h3febec3f7a1fad31, 64'h3fdf42da7058189e,64'h3fe610af16e13485, 64'hbfe72d1011095f31,64'hbfeafa776782d901, 64'h3fe135a190edca75,64'hbfe89e042fe76dfe, 64'h3fe471e027a73a67,64'h3fe32817e3a0418f, 64'hbfe9a1ea52c3e9cd,64'hbfeffe5f4df88c0a, 64'h3f94697fb5b8c59d,64'h3fdc73f6feb1c21b, 64'h3feca9e74d46c3e9,64'h3feb1896f69cbc40, 64'hbfe1060c6eff033a,64'h3fe11594d1c287ae, 64'h3feb0ece80b6ea6e,64'h3fdacbfe509e2158, 64'h3fed0f5a99c129df,64'hbfbb7ae001b7cf36, 64'h3fefd0aa9a76c55e,64'h3fec528fc7fe844b, 64'hbfddc9cdb364ac19,64'h3fef80c6727ed92c, 64'h3fc67895f7f97152,64'hbfec5891ba4ec355, 64'hbfddb2e967d2ac4f,64'hbfe50465fa181102, 64'hbfe82164785b7c69,64'hbfea7a954b5eb6a9, 64'h3fe1f7f6ee1f6e40,64'hbfe66712ba22ef4f, 64'h3fe6d999284b14d3,64'h3fb015ab74150426, 64'h3fefefd073d52f82,64'hbfef7b8477258c9d, 64'h3fc6ed4cc6879c39,64'h3fee0ff4d288649a, 64'h3fd5eeaaa17f6184,64'hbfd8814757cb6cb8, 64'h3fed8fb7d7212960,64'h3fd77b966fd8a905, 64'hbfedc4a2630dbfe3,64'h3fedd4918ebd7553, 64'h3fd72a27a6ed9206,64'h3fea6f7489ac89b9, 64'h3fe2085203d55230,64'h3fe6e39de48a69ce, 64'hbfe65cd63274977d,64'h3fed6cda48fdea71, 64'hbfd926f13c09ae2f};costF = {64'h3fefb5bf2113ba7d, 64'hbfc131ea8ee50d7c,64'hbfef1745a8ab7bf5, 64'h3fce4b1a6863e240,64'h3fefd99562ab388c, 64'hbfb8c3692635e6b1,64'hbfb77badf38ed241, 64'h3fefdd760c1d64f6,64'h3fe2156174bbc8b7, 64'h3fea66871ff8d02f,64'hbfee5f9a21c443ff, 64'hbfd42515410cbd72,64'h3fe8ed3d76745d3d, 64'h3fe410fcc24e6977,64'hbfeef5fba5281b12, 64'hbfd02d54fd39c7d5,64'hbfe9811c9bc76c3f, 64'hbfe353aec2fa1878,64'h3fdcadf94df8e5c4, 64'h3fec9b6fb054fcfa,64'h3fefe57fd7e6e005, 64'h3fb49328989fd733,64'h3fe84c6e89eea80e, 64'h3fe4d294b3537aac,64'h3fd3d41c3b34055f, 64'hbfee6ce94e9f3532,64'hbfe029ec6d59fa93, 64'h3feb9e1c1c9c4fd5,64'h3fea36d9f98f8568, 64'hbfe25a3a1a731133,64'h3fed92b482609cba, 64'h3fd872d88e628cd6,64'h3fc5095100fc2d5f, 64'h3fef909c390d12bc,64'hbfe9ffb0498efb25, 64'hbfe2a80d519eb327,64'hbfc37a1f61644893, 64'h3fefa09b152f5fb5,64'hbfebd826efe7b5b2, 64'hbfdf8a3bc6db3c63,64'h3fe1765f98eedfe1, 64'hbfead0bce22adb7e,64'hbfe13c96e1f1324e, 64'hbfeaf605d71b7ec7,64'hbfba07639e241254, 64'hbfefd58bcf81deb0,64'h3fdab031d70fb399, 64'hbfed15bf2366fa69,64'h3fb96f1d893bfeb1, 64'h3fefd7781502ba8b,64'h3fe6d419c77b3c0f, 64'hbfe66cacd5a7fce3,64'hbfeec0504544a5ac, 64'hbfd1b458efdb8da1,64'h3fc8210e4f721a56, 64'hbfef6d2137a0a10d,64'hbfe61b925bdfd881, 64'h3fe722ad8291be58,64'hbfdabcd919f387f1, 64'hbfed12d7388e0f05,64'hbfeda0ddcdea1f3d, 64'h3fd82de2403ab6d9,64'hbfe3da35fc9f645e, 64'h3fe918f37aa4901d};solM = {64'hbfe807da733c8308, 64'h3febdb33b9466ee0,64'h3fe273e530246374, 64'hbfe2d0f84f72b9bf,64'h3feba4979bb2c516, 64'hbfe21aa7bd1a7b67,64'h3fe9a75cb6cf0960, 64'hbfe143323eb2fa54,64'hbfc901944d90e630, 64'hbfe18eed02aa88f0,64'hbfcce1e6be11ab19, 64'hbff44d000e7cb096,64'hbff224daa93b0bf6, 64'hbfea1e5d306197d4,64'hbfcc005abf9c2b18, 64'h3f2921b111bd5000,64'h3ff0e896a1a9e806, 64'h3fb3cc8eddba1ad0,64'h3fedef21acd9a556, 64'h3fa7e20d5cfb16b0,64'hbfdcabb8f8ec6a2e, 64'h3ff413edad4844fe,64'h3fd0825569744106, 64'hbfd9724417f3f4d9,64'hbfe5d19dfc30ebb6, 64'hbff3ae3a8eb15b32,64'hbfb4662a2c056078, 64'h3fc01166c96ff504,64'h3fc5a1e22b3042b8, 64'hbfed60fb1acfce18,64'h3fcbdd711626c9e8, 64'hbff0a2f95cd32de6,64'hbfcb77c9cda8d1ec, 64'hbff548e884832b6e,64'hbfa7b3fc0e4c8f90, 64'h3fdb4fbe02555c89,64'hbff0dcf84ca305e6, 64'h3fdb8da9e853157e,64'h3fe966e2aeebcbf8, 64'hbfd16041e1efe6aa,64'h3feba5f471565042, 64'h3ff1bd395e25a190,64'h3fc3e0d0b293ab78, 64'hbf8a1e8c5f1375c0,64'h3ff2d11dcf901536, 64'hbfe2de71dc5ed907,64'h3fda162e90351c31, 64'hbfd48bdedaeddaf8,64'h3fef3b1b29c1d782, 64'hbfe79b6a448ff9e7,64'hbfe379c594226e4b, 64'h3fd753ac4e7be56f,64'hbfd32852db4bad52, 64'hbfee3f59227446d3,64'h3fd3ae23c7a7de4d, 64'h3feebb9f3c964190,64'hbfd6e29d25520b6d, 64'hbfe9cb2eb4987c16,64'hbfd9e5aa5d3fe60d, 64'hbff076763b3b53d7,64'hbfe9a541447b68d3, 64'hbfa944364e0966d0,64'hbff2765e0f4aac3c, 64'hbfc3bfbd061fb240};solC = {64'h3fecef6504ae48eb, 64'hbfdb54f643a04e2c,64'h3fef48b45c92631a, 64'hbfcaece9bc90a79a,64'hbfcd15039c3b2196, 64'h3fef29c1ac33a168,64'hbfee0e1b90d33ba4, 64'hbfd5f8cabb223cf8,64'h3fd40d010b552232, 64'h3fee6395922855a3,64'hbfcd5993fec5eba0, 64'hbfef25bceed770db,64'h3fd7e60507228bd6, 64'h3fedaf6ddaaec686,64'hbfeb34efe11fd004, 64'h3fe0d899860693fe,64'h3fefe5601e4c8f79, 64'h3fb49f7091b293b4,64'hbfed4efff1296de0, 64'hbfd9b0e2e57d1f3a,64'h3fe5279cf0702e74, 64'hbfe8028b2f6920a5,64'hbfe8b5743932c7e1, 64'hbfe45586871c4231,64'h3fefa934420b55eb, 64'hbfc2955c4d695ba6,64'h3fa016c20e2e5508, 64'h3feffbf45569bf7d,64'h3fed8370a38a45c2, 64'h3fd8bc334d65550e,64'h3fa47f4857ebb980, 64'h3feff96ec921d781,64'hbfeff270b0528b6c, 64'h3fad725c86a7ee8a,64'hbfefb1c80e988330, 64'hbfc1a55e9911f45c,64'hbfd4b124651cbdde, 64'h3fee48071ec09100,64'h3fe158cc0127ac87, 64'h3feae3e8674ad7ec,64'hbfefb09e4bfa9a25, 64'h3fc1c6ab9e383a1e,64'h3fed66e024d1165c, 64'h3fd942d74ec4e415,64'h3fe9026607f830cc, 64'h3fe3f697b46c5bf0,64'h3fedddabc527f579, 64'h3fd6fb0dab3de576,64'hbfd1a995a26d0b4a, 64'hbfeec1dc571308b3,64'h3fed2182e0a5954e, 64'hbfda7cacee5d0e56,64'h3fe3f3b621e6547c, 64'hbfe904b28339d410,64'hbfeb0563a6b10333, 64'hbfe1247602f1ebb7,64'hbfecfb8b4eb393b8, 64'h3fdb2147a94558dc,64'h3fc55b7f79fd6ad2, 64'hbfef8d2907e110be,64'hbfd97cfe581692d9, 64'h3fed5a51793a5b33,64'hbfd0c88892086446, 64'h3feee14d8b2c2aef};


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
            switch <= {0, 1};
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
