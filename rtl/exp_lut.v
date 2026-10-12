`timescale 1ns/1ps

module exp_lut (
    input  wire        [3:0]  diff_addr,  // clamp(score - x_max, -15, 0) 的低 4 位
    output reg         [7:0]  exp_val     // UQ0.4: 值/16 = exp(score - x_max)
);
    always @(*) begin
        case (diff_addr)
        4'd0 : exp_val = 8'd16;   // diff = 0
        4'd1 : exp_val = 8'd0;   // diff = -15
        4'd2 : exp_val = 8'd0;   // diff = -14
        4'd3 : exp_val = 8'd0;   // diff = -13
        4'd4 : exp_val = 8'd0;   // diff = -12
        4'd5 : exp_val = 8'd0;   // diff = -11
        4'd6 : exp_val = 8'd0;   // diff = -10
        4'd7 : exp_val = 8'd0;   // diff = -9
        4'd8 : exp_val = 8'd0;   // diff = -8
        4'd9 : exp_val = 8'd0;   // diff = -7
        4'd10: exp_val = 8'd0;   // diff = -6
        4'd11: exp_val = 8'd0;   // diff = -5
        4'd12: exp_val = 8'd0;   // diff = -4
        4'd13: exp_val = 8'd1;   // diff = -3
        4'd14: exp_val = 8'd2;   // diff = -2
        4'd15: exp_val = 8'd6;   // diff = -1
        default: exp_val = 8'd0;
        endcase
    end
endmodule
