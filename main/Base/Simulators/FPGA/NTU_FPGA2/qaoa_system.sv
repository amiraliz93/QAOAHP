
module qaoa_system
  #(parameter UART_CLKS_PER_BIT = 434) // 50 MHz with 115200 baud rate
  (
   input       CLK,
   input       RST,
   input       i_Rx_Serial,
   output      o_Tx_Serial,
   output [31:0]  o_Status // counter to wait read FIFO latency. 
);

reg [63:0] r_data;
logic [63:0] n_r_data;
reg  [63:0] r_addr;
wire  [63:0] n_r_addr;
wire  n_r_req;
reg  r_req;
reg  rbram_vd;
logic  n_rbram_vd;
wire [63:0] n_w_addr;
reg [63:0] w_addr;
wire [63:0] n_w_data;
reg [63:0] w_data;
wire n_w_req;
reg w_req;

// need a logic to separate request from state machines and main unit.

reg [7:0] cmd;
parameter qa_WAIT = 1;
parameter qa_RUN = 2;
parameter qa_INIT = 4;

parameter MaxCS = 64;
reg [5:0] N; // number of qubits
reg [63:0] vSC [0:MaxCS-1];
reg [12:0] maxAddr;

reg [12:0] addr_cr;
reg [12:0] addr_cw;
reg [12:0] addr_gr;
reg [12:0] addr_gw;

logic [12:0] n_addr_cr;
logic [12:0] n_addr_cw;
logic [12:0] n_addr_gr;
logic [12:0] n_addr_gw;

wire [63:0] data_gq;
reg [63:0] data_ga;
reg [63:0] data_gb;
logic [63:0] n_data_ga;
logic [63:0] n_data_gb;
reg rbram_req [5]; // request sequence to BRAM;
logic n_rbram_rreq;
reg bram_wrena;
logic n_bram_wrena;
reg bram_wrenb;
logic n_bram_wrenb;

logic [63:0] n_testReg;
reg [63:0] testReg;

ntu_smachine #(.UART_CLKS_PER_BIT(UART_CLKS_PER_BIT)) ns// 50 MHz with 115200 baud rate
  (
    .CLK(CLK),
    .RST(RST),
   .i_Rx_Serial(i_Rx_Serial),
   .o_Tx_Serial(o_Tx_Serial),
   .o_Status(o_Status), // counter to wait read FIFO latency. 
   .r_data(r_data),
   .r_addr(n_r_addr),
   .r_req(n_r_req),
   .rbram_vd(rbram_vd),
   .w_addr(n_w_addr),
   .w_data(n_w_data),
   .w_req(n_w_req)
);

ram ram1 (.address_a(addr_gr), // 12 bit address
	.address_b(addr_gw),
	.clock(CLK),
	.data_a(), // 64 bit
	.data_b(data_gb),
	.wren_a(),
	.wren_b(bram_wrenb),
	.q_a(data_gq),
	.q_b());

always_comb begin: mainCombBlock
    n_rbram_vd = 0;
    n_addr_gr = 0;
    n_addr_gw = 0;
    n_data_ga = 0;
    n_data_gb = 0;
    n_testReg = testReg;
    n_r_data = 0;
    n_rbram_rreq = 0;
    n_bram_wrena = 0;
    n_bram_wrenb = 0;
    n_addr_cr = addr_cr;
    n_addr_cw = addr_cw;
    case(cmd)
    qa_WAIT: begin // accept operation from ntu_smachine.
        case(r_addr[63:56])
        1: begin //write N.
        // need a pipeline
            n_r_data = {58'd0, N}; 
            n_rbram_vd = r_req;
        end
        2: begin 
            n_r_data = 64'h01efef80aa80aaaa;
            n_rbram_vd = r_req;
        end
        4: begin
            n_r_data = testReg;
            n_rbram_vd = r_req;
        end
        8: begin
            n_r_data = 64'h04efef80aa80aaaa;
            n_rbram_vd = r_req;
        end
        16: begin
            n_addr_gr = r_addr[0+:12];
            n_rbram_rreq = r_req;
            n_rbram_vd = rbram_req[1];
            n_r_data = data_gq;
        end 
        endcase
        case(w_addr[63:56])
        1: begin //write N.
        // need a pipeline
            n_testReg = {58'd0, N}; 
        end
        2: begin 
            n_testReg = w_data + 8;
        end
        4: begin
            n_testReg = w_data;
        end
        8: begin
            n_testReg = w_data + 1;
        end
        16: begin
            n_addr_gw = w_addr[0+:12];
            n_data_gb = w_data;
            n_bram_wrenb = w_req;
        end 
        endcase
    end
    qa_INIT: begin
        
        n_addr_cr = 0;
        n_addr_cw = 0;
    end
    qa_RUN: begin 
        n_addr_gr = addr_cr;
        n_addr_gw = addr_cw;
        if(addr_cr != maxAddr) begin
            n_addr_cr = addr_cr + 1;
        end 
    end
    endcase
end
reg RSTL;

always@(posedge CLK) begin
    RSTL <= RST;
    if(RSTL)begin
        r_data <= 0;
        r_addr <= 0;
        r_req <= 0;
        w_addr <= 0;
        w_data <= 0;
        w_req <= 0;
        testReg <= 0;
        data_ga <= 0;
        data_gb <= 0;
        rbram_vd <= 0;
        bram_wrena <= 0;
        bram_wrenb <= 0;
        rbram_req[0] <= 0;
        rbram_req[1] <= 0;
        rbram_req[2] <= 0;
        addr_cr <= 0;
        addr_cw <= 0;
        cmd <= qa_INIT;
        N <= 9;
        maxAddr <= 13'h0800;
        addr_gr <= 0;
        addr_gw <= 0;
    end
    else begin
        N <= N;
        cmd <= o_Status[7:0];
        addr_gr <= n_addr_gr;
        addr_gw <= n_addr_gw;
        addr_cr <= n_addr_cr;
        addr_cw <= n_addr_cw;
        r_data <= n_r_data;
        r_addr <= n_r_addr;
        r_req <= n_r_req;
        w_addr <= n_w_addr;
        w_data <= n_w_data;
        w_req <= n_w_req;
        testReg <= n_testReg;
        data_ga <= n_data_ga;
        data_gb <= n_data_gb;
        rbram_vd <= n_rbram_vd;
        bram_wrena <= n_bram_wrena;
        bram_wrenb <= n_bram_wrenb;
        rbram_req[0] <= n_rbram_rreq;
        rbram_req[1] <= rbram_req[0];
        rbram_req[2] <= rbram_req[1];
        
        addr_cr <= n_addr_cr;
    end
end
endmodule