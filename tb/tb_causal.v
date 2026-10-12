`timescale 1ns/1ps
//===========================================================================
//  tb_causal.v —— 加分项 A（因果 Mask）独立测试
//---------------------------------------------------------------------------
//  被测对象：attention_ip #(.S(8), .D(4), .CAUSAL_EN(1'b1))
//
//  测试思路：
//    固定两组 Q/K/V，把 Query 的位置 q_pos 从 0 遍历到 7，
//    对每一个 q_pos 检查三件事：
//      ① 输出 O 是否等于 Golden Model 算出的期望值
//      ② L（权重和）是否等于期望值
//      ③ 所有 j > q_pos 的 weight_buf[j] 是否都被强制为 0（这就是"屏蔽"）
//
//  两组输入：
//    场景 0：Q=[4,0,0,0]，K_j=[j+1,0,0,0]，V_j=[8(j+1), -8(j+1), 4(j+1), -4(j+1)]
//            相邻 Token 的 Score 相差 4，所以只有 q_pos 自己那一路有权重，
//            期望 O 正好等于 V[q_pos]，一眼能看出 mask 有没有生效。
//    场景 1：Q=[1,0,0,0]，K_j=[j,0,0,0]，V_j=[10+3j, -5j, 7j-20, 2j+1]
//            相邻 Score 只差 1，最近 4 个 Token 的权重依次是 16/6/2/1，
//            能覆盖完整的指数表档位，验证 mask 与精度的配合。
//
//  运行：
//      iverilog -g2005 -o sim/tb_causal.vvp rtl/attention_ip.v rtl/exp_lut.v \
//               rtl/recip_lut.v tb/tb_causal.v
//      vvp sim/tb_causal.vvp
//  或 ModelSim：
//      vlog rtl/attention_ip.v rtl/exp_lut.v rtl/recip_lut.v tb/tb_causal.v
//      vsim -c -do "run -all; quit -f" work.tb_causal
//===========================================================================
module tb_causal;

    //-----------------------------------------------------------------------
    // 时钟与 DUT
    //-----------------------------------------------------------------------
    reg clk = 0;
    always #5 clk = ~clk;

    reg               rst_n, start, q_valid, kv_valid;
    reg  signed [7:0] q_in, k_in, v_in;
    reg  [3:0]        q_pos;
    wire              data_ready, out_valid, done;
    wire signed [7:0] out_data;

    attention_ip #(.S(8), .D(4), .CAUSAL_EN(1'b1)) dut (
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
        .q_pos      (q_pos)
    );

    //-----------------------------------------------------------------------
    // 输入数据
    //-----------------------------------------------------------------------
    reg signed [7:0] Q [0:3];
    reg signed [7:0] K [0:7][0:3];
    reg signed [7:0] V [0:7][0:3];

    //-----------------------------------------------------------------------
    // 期望值（由 model/golden_model.py 用 causal_i 参数算出）
    //-----------------------------------------------------------------------
    // 场景 0：L 恒为 16，O 就等于 V[q_pos]
    reg signed [7:0] E0_O [0:7][0:3];
    // 场景 1：L 和 O 都随 q_pos 变化
    reg signed [7:0] E1_O [0:7][0:3];
    integer          E1_L [0:7];

    integer i, j, n, fails, cases, got_n;
    reg signed [7:0] got [0:3];

    //-----------------------------------------------------------------------
    // 装载某一组输入
    //-----------------------------------------------------------------------
    task set_input;
        input integer scen;
        begin
            if (scen == 0) begin
                Q[0] = 8'sd4;  Q[1] = 8'sd0;  Q[2] = 8'sd0;  Q[3] = 8'sd0;
                for (i = 0; i < 8; i = i + 1) begin
                    K[i][0] = i + 1;   K[i][1] = 0;  K[i][2] = 0;  K[i][3] = 0;
                    V[i][0] =  8*(i+1);
                    V[i][1] = -8*(i+1);
                    V[i][2] =  4*(i+1);
                    V[i][3] = -4*(i+1);
                end
            end
            else begin
                Q[0] = 8'sd1;  Q[1] = 8'sd0;  Q[2] = 8'sd0;  Q[3] = 8'sd0;
                for (i = 0; i < 8; i = i + 1) begin
                    K[i][0] = i;        K[i][1] = 0;  K[i][2] = 0;  K[i][3] = 0;
                    V[i][0] = 10 + 3*i;
                    V[i][1] = -5*i;
                    V[i][2] = 7*i - 20;
                    V[i][3] = 2*i + 1;
                end
            end
        end
    endtask

    //-----------------------------------------------------------------------
    // 驱动一个用例：start -> 4 拍 Q -> 32 拍 K/V -> 收 4 个输出
    //-----------------------------------------------------------------------
    task run_dut;
        begin
            // start
            @(negedge clk);
            while (!data_ready) @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            // Q：4 拍
            for (i = 0; i < 4; i = i + 1) begin
                @(negedge clk);
                while (!data_ready) @(negedge clk);
                q_in    = Q[i];
                q_valid = 1'b1;
            end
            @(negedge clk);
            q_valid = 1'b0;

            // K/V：32 拍（每拍一对）
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
            k_in = 8'sd0;  v_in = 8'sd0;

            // 收输出
            got_n = 0;
            while (!done) begin
                @(negedge clk);
                if (out_valid) begin
                    if (got_n < 4) got[got_n] = out_data;
                    got_n = got_n + 1;
                end
            end
            repeat (2) @(negedge clk);
            q_valid = 1'b0;  kv_valid = 1'b0;
        end
    endtask

    //-----------------------------------------------------------------------
    // 主流程
    //-----------------------------------------------------------------------
    integer scen, pos, p;
    reg signed [7:0] eo;

    initial begin
        // ---- 期望值表 ----
        E0_O[0][0]=  8; E0_O[0][1]= -8; E0_O[0][2]=  4; E0_O[0][3]= -4;
        E0_O[1][0]= 16; E0_O[1][1]=-16; E0_O[1][2]=  8; E0_O[1][3]= -8;
        E0_O[2][0]= 24; E0_O[2][1]=-24; E0_O[2][2]= 12; E0_O[2][3]=-12;
        E0_O[3][0]= 32; E0_O[3][1]=-32; E0_O[3][2]= 16; E0_O[3][3]=-16;
        E0_O[4][0]= 40; E0_O[4][1]=-40; E0_O[4][2]= 20; E0_O[4][3]=-20;
        E0_O[5][0]= 48; E0_O[5][1]=-48; E0_O[5][2]= 24; E0_O[5][3]=-24;
        E0_O[6][0]= 56; E0_O[6][1]=-56; E0_O[6][2]= 28; E0_O[6][3]=-28;
        E0_O[7][0]= 64; E0_O[7][1]=-64; E0_O[7][2]= 32; E0_O[7][3]=-32;

        E1_O[0][0]= 10; E1_O[0][1]=  0; E1_O[0][2]=-20; E1_O[0][3]= 1;
        E1_O[1][0]= 12; E1_O[1][1]= -4; E1_O[1][2]=-15; E1_O[1][3]= 2;
        E1_O[2][0]= 15; E1_O[2][1]= -8; E1_O[2][2]= -9; E1_O[2][3]= 4;
        E1_O[3][0]= 17; E1_O[3][1]=-12; E1_O[3][2]= -3; E1_O[3][3]= 6;
        E1_O[4][0]= 20; E1_O[4][1]=-17; E1_O[4][2]=  4; E1_O[4][3]= 8;
        E1_O[5][0]= 23; E1_O[5][1]=-22; E1_O[5][2]= 11; E1_O[5][3]=10;
        E1_O[6][0]= 26; E1_O[6][1]=-27; E1_O[6][2]= 18; E1_O[6][3]=12;
        E1_O[7][0]= 29; E1_O[7][1]=-32; E1_O[7][2]= 25; E1_O[7][3]=14;

        E1_L[0]=16; E1_L[1]=22; E1_L[2]=24; E1_L[3]=25;
        E1_L[4]=25; E1_L[5]=25; E1_L[6]=25; E1_L[7]=25;

        // ---- 复位 ----
        rst_n = 1'b0;  start = 1'b0;  q_valid = 1'b0;  kv_valid = 1'b0;
        q_in = 8'sd0;  k_in = 8'sd0;  v_in = 8'sd0;   q_pos = 3'd0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        fails = 0;
        cases = 0;

        $display("");
        $display("==============================================================");
        $display(" 加分项 A：因果 Mask 独立测试  (S=8, D=4, CAUSAL_EN=1)");
        $display("==============================================================");

        for (scen = 0; scen < 2; scen = scen + 1) begin
            set_input(scen);
            $display("");
            $display("---- 场景 %0d ----", scen);
            for (pos = 0; pos < 8; pos = pos + 1) begin
                q_pos = pos[2:0];                // 当前 Query 的位置
                #1;                              // 让 q_pos 稳定
                run_dut;
                cases = cases + 1;

                // ① 输出 O
                for (p = 0; p < 4; p = p + 1) begin
                    eo = (scen == 0) ? E0_O[pos][p] : E1_O[pos][p];
                    if ($signed(got[p]) !== eo) begin
                        $display("  [FAIL] 场景%0d q_pos=%0d : O[%0d] RTL=%0d EXP=%0d",
                                 scen, pos, p, $signed(got[p]), eo);
                        fails = fails + 1;
                    end
                end

                // ② L
                if (scen == 1) begin
                    if (dut.l_sum !== E1_L[pos]) begin
                        $display("  [FAIL] 场景1 q_pos=%0d : L RTL=%0d EXP=%0d",
                                 pos, dut.l_sum, E1_L[pos]);
                        fails = fails + 1;
                    end
                end
                else begin
                    if (dut.l_sum !== 8'd16) begin
                        $display("  [FAIL] 场景0 q_pos=%0d : L RTL=%0d EXP=16",
                                 pos, dut.l_sum);
                        fails = fails + 1;
                    end
                end

                // ③ 屏蔽检查：j > q_pos 的权重必须为 0
                for (j = 0; j < 8; j = j + 1) begin
                    if ((j > pos) && (dut.weight_buf[j] !== 5'd0)) begin
                        $display("  [FAIL] 场景%0d q_pos=%0d : weight[%0d]=%0d 应被 mask 成 0",
                                 scen, pos, j, dut.weight_buf[j]);
                        fails = fails + 1;
                    end
                end

                $display("  q_pos=%0d  L=%3d  O=%4d %4d %4d %4d   %s",
                         pos, dut.l_sum,
                         $signed(dut.out_reg[0]), $signed(dut.out_reg[1]),
                         $signed(dut.out_reg[2]), $signed(dut.out_reg[3]),
                         "OK");
            end
        end

        $display("");
        $display("--------------------------------------------------------------");
        $display("  用例数 : %0d", cases);
        $display("  失败数 : %0d", fails);
        if (fails == 0) $display("  *** 因果 MASK 测试全部通过 (ALL PASS) ***");
        else            $display("  *** 存在失败 (FAILED) ***");
        $display("--------------------------------------------------------------");
        $finish;
    end

endmodule
