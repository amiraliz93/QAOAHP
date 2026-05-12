module regPipeline #(
    parameter W_INFO = 32,
    parameter NPipe = 2
)(
   input  CLK,
   input [W_INFO-1:0] info_in,
   output [W_INFO-1:0] info_out
);

generate 
    if(NPipe == 0) begin: no_pip_version
        assign info_out = info_in;
    end
    else begin: pipeline_version
        (* maxfan = 32 *) reg [W_INFO-1:0] rp [NPipe];
        assign info_out = rp[NPipe - 1];
        integer i; 

        always_ff @(posedge CLK) begin 
            rp[0] <= info_in;
            for(i =0;i<NPipe-1;i=i+1) begin 
                rp[i+1] <= rp[i];
            end
        end
    end
endgenerate
endmodule