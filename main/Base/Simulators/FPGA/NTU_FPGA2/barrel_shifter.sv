// https://github.com/jsagoe1/Verilog-SystemVerilog/blob/master/parameterized_barrel_shift.sv
// 1400 MHz, 20260419
module barrel_shifter #(int n=4, parameter Ni=16)(
      input CLK,
      output  [n-1:0] d_out,   
      output [Ni-1:0] info_out,
      input  [n-1:0] d_in,
      input  [Ni-1:0] info_in,
      input  [$clog2(n)-1:0] sh_amt);
localparam iter=$clog2(n);
reg [n-1:0] out [iter+1];                      // n-bit out[0], out[1], .....up to out[iter] 
reg [Ni-1:0] info_out_r [iter+1];                    
// integer catw;
always@(posedge CLK) begin
      out[0] <= d_in;
      info_out_r[0] <= info_in;
end
genvar i;
generate
      for( i=1; i<= iter; i++) begin   : stage_gen
            localparam catw = 2**(i-1);              
            always@(posedge CLK) begin
                  if(sh_amt[i-1]) begin
                        out[i][catw - 1: 0] <= out[i-1][n-1:n - catw];
                        out[i][n-1:catw] <= out[i-1][n-catw-1:0];
                  end
                  else begin
                        out[i] <= out[i-1];
                  end
                  info_out_r[i] <= info_out_r[i-1];
            end
      end
endgenerate

assign d_out = out[iter];                       //last mux to module output
assign info_out = info_out_r[iter];
endmodule


module barrel_shifter_tb;
      localparam int n = 8;
      localparam int iter = $clog2(n);

      wire [n-1:0] d_out;
      reg [n-1:0] d_in;
      reg [$clog2(n)-1:0] sh_amt;
      reg CLK;

      barrel_shifter #(n) b0(.CLK(CLK),
            .d_out(d_out),
            .d_in(d_in),
            .sh_amt(sh_amt));
      
      initial begin
            d_in = 0;
            sh_amt = 0;
            CLK <= 0;

            #50;
            //testing to shift it by 1,2,3..'n' times
            for(int j=2;j<n;j++)begin
                  sh_amt <= j;
                  for (int i=0; i<2**n; i++) begin
                        d_in <= i;
                        #10;
                  end
                  #100;
            end

            #100;
            $stop;
      end
      always begin
            #5;
            CLK <= ~CLK;
      end
  
endmodule