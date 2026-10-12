`timescale 1ns/1ps
//===========================================================================
//  tb_simple.v —— 最简测试平台：改几个数，跑一下，看波形
//===========================================================================
module tb_simple;

    //-----------------------------------------------------------------------
    // 时钟与复位（100 MHz，周期 10 ns）
    //-----------------------------------------------------------------------
    reg clk = 0;
    always #5 clk = ~clk;

    //-----------------------------------------------------------------------
    // 接 DUT 的信号
    //-----------------------------------------------------------------------
    reg               rst_n;
    reg               start;
    reg  signed [7:0] q_in, k_in, v_in;
    reg               q_valid, kv_valid;
    wire              data_ready;
    wire signed [7:0] out_data;
    wire              out_valid, done;

    attention_ip dut (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (start),
        .q_in       (q_in),
        .k_in       (k_in),
        .v_in       (v_in),
        .q_valid    (q_valid),
        .kv_valid   (kv_valid),
        .data_ready (data_ready),
        .out_data   (out_data),
        .out_valid  (out_valid),
        .done       (done),
        .q_pos(4'd0)          // 基础版（CAUSAL_EN=0）不使用，接 0 即可
    );

    //=======================================================================
    //  ①  只改这一段：输入数值
    //      每个数必须是 -128 ~ 127 的整数
    //      Q 是 1 个 Token 的 4 个元素；K/V 是 8 个 Token，每个 4 个元素
    //=======================================================================
    integer Q [0:3];
    integer K [0:7][0:3];
    integer V [0:7][0:3];

    initial begin
       // ---- Q: 1 个 Token, 4 个元素 ----
        Q[0] =   1;   Q[1] =  -2;   Q[2] =   3;   Q[3] =  -4;
        
        // ---- K: 8 个 Token, 每个 4 个元素 ----
        K[0][0] =   0;   K[0][1] =   1;   K[0][2] =  -1;   K[0][3] =   2;
        K[1][0] =  -2;   K[1][1] =   3;   K[1][2] =   0;   K[1][3] =   1;
        K[2][0] =   1;   K[2][1] =   0;   K[2][2] =   2;   K[2][3] =  -1;
        K[3][0] =   3;   K[3][1] =  -1;   K[3][2] =   1;   K[3][3] =   0;
        K[4][0] =  -1;   K[4][1] =   2;   K[4][2] =  -2;   K[4][3] =   3;
        K[5][0] =   0;   K[5][1] =  -3;   K[5][2] =   4;   K[5][3] =   1;
        K[6][0] =   2;   K[6][1] =   1;   K[6][2] =   0;   K[6][3] =  -2;
        K[7][0] =  -3;   K[7][1] =   0;   K[7][2] =   1;   K[7][3] =   2;
        
        // ---- V: 8 个 Token, 每个 4 个元素 ----
        V[0][0] =  10;   V[0][1] = -20;   V[0][2] =  30;   V[0][3] = -40;
        V[1][0] =  -5;   V[1][1] =  15;   V[1][2] = -25;   V[1][3] =  35;
        V[2][0] =  20;   V[2][1] = -10;   V[2][2] =  40;   V[2][3] = -30;
        V[3][0] = -15;   V[3][1] =  25;   V[3][2] = -35;   V[3][3] =  45;
        V[4][0] =   5;   V[4][1] =  -5;   V[4][2] =   5;   V[4][3] =  -5;
        V[5][0] = -10;   V[5][1] =  20;   V[5][2] = -30;   V[5][3] =  40;
        V[6][0] =  15;   V[6][1] = -25;   V[6][2] =  35;   V[6][3] = -45;
        V[7][0] =   0;   V[7][1] =   0;   V[7][2] =   0;   V[7][3] =   0;
    end
    //=======================================================================
    //  ①  改到这里为止
    //=======================================================================

    integer i, j, n;
    integer cyc, cyc0;

    // 在 posedge 数时钟拍（testbench 在 negedge 驱动，两者不冲突）
    integer gcyc;
    always @(posedge clk) gcyc = gcyc + 1;

    //=======================================================================
    //  ②  驱动 DUT + ③ 打印输出
    //=======================================================================
    initial begin
        // 想看 GTKWave 波形就把下面两行留着；不需要就注释掉
        $dumpfile("waves/simple.vcd");
        $dumpvars(0, tb_simple);

        // ---- 复位 ----
        gcyc     = 0;              // 拍数计数器清零
        rst_n    = 1'b0;
        start    = 1'b0;
        q_valid  = 1'b0;
        kv_valid = 1'b0;
        q_in = 8'sd0;  k_in = 8'sd0;  v_in = 8'sd0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        $display("");
        $display("================ tb_simple start ================");

        // ---- 1. start 拉高 1 拍，启动一次任务 ----
        cyc0  = gcyc;              // 记下启动时刻的拍数
        start = 1'b1;
        @(negedge clk);
        start = 1'b0;

        // ---- 2. 送 Q：连续 4 拍，每拍 1 个元素 ----
        //        data_ready=1 才允许送
        for (i = 0; i < 4; i = i + 1) begin
            @(negedge clk);
            while (!data_ready) @(negedge clk);
            q_in    = Q[i];
            q_valid = 1'b1;
        end
        @(negedge clk);
        q_valid = 1'b0;

        // ---- 3. 送 K/V：连续 32 拍 ----
        //        第 t 拍同时送 K[t/4][t%4] 和 V[t/4][t%4]
        //        所以是 i 从 0 到 7（Token）、j 从 0 到 3（维度）两层循环
        for (i = 0; i < 8; i = i + 1) begin
            for (j = 0; j < 4; j = j + 1) begin
                @(negedge clk);
                while (!data_ready) @(negedge clk);
                k_in     = K[i][j];
                v_in     = V[i][j];
                kv_valid = 1'b1;
            end
        end
        @(negedge clk);
        kv_valid = 1'b0;
        k_in = 8'sd0;
        v_in = 8'sd0;

        // ---- 4. 等 done，同时把每个输出打印出来 ----
        n = 0;
        while (!done) begin
            @(negedge clk);
            if (out_valid) begin
                $display("  out_valid #%0d :  O[%0d] = %0d",
                         n + 1, n, $signed(out_data));
                n = n + 1;
            end
        end
        cyc = gcyc - cyc0;         // 从 start 到 done 一共多少拍
        @(negedge clk);

        // ---- 5. 打印中间结果，方便和波形对照 ----
        $display("------------------------------------------------");
        $display("  cycles = %0d  (%.2f us @100MHz)", cyc, cyc * 10.0 / 1000.0);
        $display("  x_max  = %0d", $signed(dut.max_reg));
        $display("  L      = %0d", dut.l_sum);
        $display("  ACC    = %0d %0d %0d %0d",
                 $signed(dut.acc_reg[0]), $signed(dut.acc_reg[1]),
                 $signed(dut.acc_reg[2]), $signed(dut.acc_reg[3]));
        $display("  O      = %0d %0d %0d %0d",
                 $signed(dut.out_reg[0]), $signed(dut.out_reg[1]),
                 $signed(dut.out_reg[2]), $signed(dut.out_reg[3]));
        $display("  done pulsed 1 cycle -> task finished");
        $display("================================================");
        $display("");

        $finish;
    end

endmodule
