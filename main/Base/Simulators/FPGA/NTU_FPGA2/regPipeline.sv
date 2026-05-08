module regPipeline #(
    parameter W_INFO = 32,
    parameter NPipe = 2
)(
   input  CLK,
   input  RST,
   input [W_INFO:0] info_in,
   output [W_INFO:0] info_out
);

reg [W_INFO-1:0] rp [NPipe];
assign info_out = rp[NPipe - 1];
integer i; 

always_ff @(posedge CLK) begin 
    if(RST) begin
        for(i =0;i<NPipe;i=i+1) begin 
            rp[i] <= 0;
        end
    end
    else begin 
        rp[0] = info_in;
        for(i =0;i<NPipe-1;i=i+1) begin 
            rp[i+1] <= rp[i];
        end
    end
end
endmodule