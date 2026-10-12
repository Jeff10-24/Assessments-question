`timescale 1ns/1ps
//===========================================================================
//  tb_attention_ip.v —— Attention IP 自检 testbench
//---------------------------------------------------------------------------
//  功能：
//    1. 从向量文件读入 Q/K/V 与期望结果（x_max / Score / Weight / L / ACC / O）
//    2. 按接口协议驱动 DUT（data_ready → start → q_valid → kv_valid → done）
//    3. 通过层次引用逐级比对 DUT 内部寄存器与 Golden Model
//    4. 统计用例数、通过数、比较次数、单任务周期数
//
//  运行（Icarus Verilog）：
//    iverilog -g2005 -o sim/tb.vvp rtl/*.v tb/tb_attention_ip.v
//    vvp sim/tb.vvp +VEC=sim/vectors/all.vec
//
//  可选 plusarg：
//    +VEC=<文件>     向量文件（默认 sim/vectors/all.vec）
//    +KVMODE=<0|1|2> 0=32 拍成对流(默认) 1=64 拍交错流 2=64 拍重复对流
//    +N=<个数>       只跑前 N 个用例
//    +TOL=<n>        输出允许的 LSB 误差（默认 0，即要求逐位一致）
//    +VCD=<文件>     导出 VCD 波形
//===========================================================================
module tb_attention_ip;

    //-----------------------------------------------------------------------
    // 时钟与复位（100 MHz）
    //-----------------------------------------------------------------------
    reg clk;
    reg rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;

    //-----------------------------------------------------------------------
    // DUT 接口
    //-----------------------------------------------------------------------
    reg  signed [7:0] q_in, k_in, v_in;
    reg               start, q_valid, kv_valid;
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

    //-----------------------------------------------------------------------
    // 波形观测网线
    //   Icarus Verilog 的 VCD 不抓取 unpacked array，这里把关心的存储体元素
    //   引成一维网线，方便波形截图与离线调试（不影响 DUT 逻辑）。
    //-----------------------------------------------------------------------
    wire signed [17:0] dbg_score0  = dut.score_buf[0];
    wire signed [17:0] dbg_score1  = dut.score_buf[1];
    wire signed [17:0] dbg_score2  = dut.score_buf[2];
    wire signed [17:0] dbg_score7  = dut.score_buf[7];
    wire        [4:0]  dbg_weight0 = dut.weight_buf[0];
    wire        [4:0]  dbg_weight1 = dut.weight_buf[1];
    wire        [4:0]  dbg_weight7 = dut.weight_buf[7];
    wire signed [15:0] dbg_acc0    = dut.acc_reg[0];
    wire signed [15:0] dbg_acc1    = dut.acc_reg[1];
    wire signed [15:0] dbg_acc2    = dut.acc_reg[2];
    wire signed [15:0] dbg_acc3    = dut.acc_reg[3];
    wire signed [7:0]  dbg_out0    = dut.out_reg[0];
    wire signed [7:0]  dbg_out1    = dut.out_reg[1];
    wire signed [7:0]  dbg_out2    = dut.out_reg[2];
    wire signed [7:0]  dbg_out3    = dut.out_reg[3];

    //-----------------------------------------------------------------------
    // 向量与期望值
    //-----------------------------------------------------------------------
    integer       fd;
    reg [8*256:1] vecfile;
    reg [8*256:1] vcdfile;
    reg [8*16:1]  kw;
    reg [8*16:1]  tag;

    integer e_q      [0:3];
    integer e_k      [0:31];
    integer e_v      [0:31];
    integer e_score  [0:7];
    integer e_weight [0:7];
    integer e_acc    [0:3];
    integer e_out    [0:3];
    integer e_xmax, e_l;

    //-----------------------------------------------------------------------
    // 统计
    //-----------------------------------------------------------------------
    integer n_case, n_pass, n_fail;
    integer n_cmp_out, n_cmp_mid;
    integer n_tol_hit;
    integer cyc_min, cyc_max, cyc_sum;
    integer fail_this;
    integer tol;
    integer kv_mode_sel;
    integer max_cases;

    integer i, j, r, tmp;

    // 协议监视：gcyc 统计 posedge 数（量周期），done_cyc 统计 done 为高的拍数。
    // 两个计数器都跑在 posedge，而 testbench 在 negedge 驱动/采样，
    // 这样不会和 testbench 在同一时刻竞争，两个仿真器结果一致。
    integer gcyc;
    integer done_cyc;
    always @(posedge clk) gcyc = gcyc + 1;
    always @(posedge clk) if (done) done_cyc = done_cyc + 1;

    //-----------------------------------------------------------------------
    // 读一个用例（按固定字段顺序读取，不依赖字符串比较）
    //-----------------------------------------------------------------------
    task read_case;
        output ok;
        integer t;
        begin
            ok = 0;
            r  = $fscanf(fd, "%s", kw);              // CASE
            if (r == 1) begin
                r = $fscanf(fd, "%d %s", n_case, tag);
                r = $fscanf(fd, "%s", kw);           // Q
                for (t = 0; t < 4; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_q[t] = tmp;
                end
                r = $fscanf(fd, "%s", kw);           // K
                for (t = 0; t < 32; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_k[t] = tmp;
                end
                r = $fscanf(fd, "%s", kw);           // V
                for (t = 0; t < 32; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_v[t] = tmp;
                end
                r = $fscanf(fd, "%s %d", kw, e_xmax);          // XMAX
                r = $fscanf(fd, "%s", kw);                     // SCORE
                for (t = 0; t < 8; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_score[t] = tmp;
                end
                r = $fscanf(fd, "%s", kw);                     // WEIGHT
                for (t = 0; t < 8; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_weight[t] = tmp;
                end
                r = $fscanf(fd, "%s %d", kw, e_l);              // L
                r = $fscanf(fd, "%s", kw);                     // ACC
                for (t = 0; t < 4; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_acc[t] = tmp;
                end
                r = $fscanf(fd, "%s", kw);                     // OUT
                for (t = 0; t < 4; t = t + 1) begin
                    r = $fscanf(fd, "%d", tmp); e_out[t] = tmp;
                end
                r = $fscanf(fd, "%s", kw);                     // END
                ok = 1;
            end
        end
    endtask

    //-----------------------------------------------------------------------
    // 驱动一个用例并逐级比对
    //-----------------------------------------------------------------------
    reg signed [7:0] got_out [0:3];
    integer          got_n;
    integer          done_cnt;

    task drive_and_check;
        integer cnt, guard, cyc, cyc0, dcyc0;
        begin
            fail_this = 0;
            got_n     = 0;
            done_cnt  = 0;
            cyc       = 0;

            // ---------------- 1. start ----------------
            @(negedge clk);
            while (!data_ready) @(negedge clk);
            cyc0  = gcyc;          // 与 start 同一拍采样，保证周期数从启动那拍算起
            dcyc0 = done_cyc;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            // ---------------- 2. 装载 Q（4 拍）----------------
            cnt   = 0;
            guard = 0;
            while (cnt < 4 && guard < 100) begin
                @(negedge clk);
                if (data_ready) begin
                    q_in    = e_q[cnt][7:0];
                    q_valid = 1'b1;
                    cnt     = cnt + 1;
                end
                guard = guard + 1;
            end
            @(negedge clk);
            q_valid = 1'b0;
            q_in    = 8'sd0;

            // ---------------- 3. 装载 K/V ----------------
            if (kv_mode_sel == 0) begin
                // 32 拍成对流：每拍同时送 K[j][k] 与 V[j][k]
                cnt   = 0;
                guard = 0;
                while (cnt < 32 && guard < 4000) begin
                    @(negedge clk);
                    if (data_ready) begin
                        k_in     = e_k[cnt][7:0];
                        v_in     = e_v[cnt][7:0];
                        kv_valid = 1'b1;
                        cnt      = cnt + 1;
                    end
                    guard = guard + 1;
                end
            end
            else if (kv_mode_sel == 1) begin
                // 64 拍交错流：偶数拍送 K 元素，奇数拍送 V 元素，另一路置 0
                cnt   = 0;
                guard = 0;
                while (cnt < 64 && guard < 4000) begin
                    @(negedge clk);
                    if (data_ready) begin
                        if (cnt[0] == 1'b0) begin
                            k_in = e_k[cnt>>1][7:0];
                            v_in = 8'sd0;
                        end
                        else begin
                            k_in = 8'sd0;
                            v_in = e_v[cnt>>1][7:0];
                        end
                        kv_valid = 1'b1;
                        cnt      = cnt + 1;
                    end
                    guard = guard + 1;
                end
            end
            else begin
                // 64 拍重复对流：每拍同时送同一个元素两遍（K/V 同址写入）
                cnt   = 0;
                guard = 0;
                while (cnt < 64 && guard < 4000) begin
                    @(negedge clk);
                    if (data_ready) begin
                        k_in     = e_k[cnt>>1][7:0];
                        v_in     = e_v[cnt>>1][7:0];
                        kv_valid = 1'b1;
                        cnt      = cnt + 1;
                    end
                    guard = guard + 1;
                end
            end
            @(negedge clk);
            kv_valid = 1'b0;
            k_in     = 8'sd0;
            v_in     = 8'sd0;

            // ---------------- 4. 等 done，同时采集输出 ----------------
            //   在 negedge 采样，避开与 DUT 同一时钟沿的竞争
            guard = 0;
            while (!done && guard < 20000) begin
                @(negedge clk);
                if (out_valid) begin
                    if (got_n < 4) got_out[got_n] = out_data;
                    got_n = got_n + 1;
                end
                if (done) done_cnt = done_cnt + 1;
                guard = guard + 1;
            end
            cyc = gcyc - cyc0;
            if (guard >= 20000) begin
                $display("  [FAIL] case %0d %0s : timeout waiting for done", n_case, tag);
                fail_this = fail_this + 1;
            end
            repeat (2) @(negedge clk);

            // ---------------- 5. 逐级比对 ----------------
            // 输出个数
            if (got_n !== 4) begin
                $display("  [FAIL] case %0d %0s : out_valid asserted %0d times (expect 4)",
                         n_case, tag, got_n);
                fail_this = fail_this + 1;
            end
            // done 必须恰好 1 拍
            if ((done_cyc - dcyc0) !== 1) begin
                $display("  [FAIL] case %0d %0s : done stayed high %0d cycles (expect 1)",
                         n_case, tag, done_cyc - dcyc0);
                fail_this = fail_this + 1;
            end

            // 5.1 x_max
            if ($signed(dut.max_reg) !== $signed(e_xmax)) begin
                $display("  [FAIL] case %0d %0s : x_max RTL=%0d EXP=%0d",
                         n_case, tag, $signed(dut.max_reg), e_xmax);
                fail_this = fail_this + 1;
            end
            n_cmp_mid = n_cmp_mid + 1;

            // 5.2 Score
            for (i = 0; i < 8; i = i + 1) begin
                if ($signed(dut.score_buf[i]) !== $signed(e_score[i])) begin
                    $display("  [FAIL] case %0d %0s : score[%0d] RTL=%0d EXP=%0d",
                             n_case, tag, i, $signed(dut.score_buf[i]), e_score[i]);
                    fail_this = fail_this + 1;
                end
                n_cmp_mid = n_cmp_mid + 1;
            end

            // 5.3 Weight
            for (i = 0; i < 8; i = i + 1) begin
                if (dut.weight_buf[i] !== e_weight[i][4:0]) begin
                    $display("  [FAIL] case %0d %0s : weight[%0d] RTL=%0d EXP=%0d",
                             n_case, tag, i, dut.weight_buf[i], e_weight[i]);
                    fail_this = fail_this + 1;
                end
                n_cmp_mid = n_cmp_mid + 1;
            end

            // 5.4 L
            if (dut.l_sum !== e_l[7:0]) begin
                $display("  [FAIL] case %0d %0s : L RTL=%0d EXP=%0d",
                         n_case, tag, dut.l_sum, e_l);
                fail_this = fail_this + 1;
            end
            n_cmp_mid = n_cmp_mid + 1;

            // 5.5 ACC
            for (i = 0; i < 4; i = i + 1) begin
                if ($signed(dut.acc_reg[i]) !== $signed(e_acc[i])) begin
                    $display("  [FAIL] case %0d %0s : acc[%0d] RTL=%0d EXP=%0d",
                             n_case, tag, i, $signed(dut.acc_reg[i]), e_acc[i]);
                    fail_this = fail_this + 1;
                end
                n_cmp_mid = n_cmp_mid + 1;
            end

            // 5.6 输出 O（容差 tol，默认 0 = 逐位一致）
            for (i = 0; i < 4; i = i + 1) begin
                j = $signed(got_out[i]) - $signed(e_out[i]);
                if (j < 0) j = -j;
                if (j > tol) begin
                    $display("  [FAIL] case %0d %0s : out[%0d] RTL=%0d EXP=%0d (diff %0d)",
                             n_case, tag, i, $signed(got_out[i]), e_out[i], j);
                    fail_this = fail_this + 1;
                end
                else if (j != 0) begin
                    n_tol_hit = n_tol_hit + 1;
                end
                n_cmp_out = n_cmp_out + 1;
            end

            // ---------------- 6. 统计 ----------------
            if (cyc < cyc_min) cyc_min = cyc;
            if (cyc > cyc_max) cyc_max = cyc;
            cyc_sum = cyc_sum + cyc;

            if (fail_this == 0) n_pass = n_pass + 1;
            else begin
                n_fail = n_fail + 1;
                $display("  ---- case %0d (%0s) FAILED ----", n_case, tag);
            end

            // 复位输入（下一用例重新驱动）
            q_valid  = 1'b0;
            kv_valid = 1'b0;
            q_in     = 8'sd0;
            k_in     = 8'sd0;
            v_in     = 8'sd0;
        end
    endtask

    //-----------------------------------------------------------------------
    // 主流程
    //-----------------------------------------------------------------------
    reg ok;

    initial begin
        vecfile     = "sim/vectors/all.vec";
        vcdfile     = "waves/attention_ip.vcd";
        kv_mode_sel = 0;
        max_cases   = 1000000;
        tol         = 0;

        if (!$value$plusargs("VEC=%s", vecfile))    vecfile = "sim/vectors/all.vec";
        if (!$value$plusargs("KVMODE=%d", kv_mode_sel)) kv_mode_sel = 0;
        if (!$value$plusargs("N=%d", max_cases))    max_cases = 1000000;
        if (!$value$plusargs("TOL=%d", tol))        tol = 0;

        n_case = 0; n_pass = 0; n_fail = 0;
        n_cmp_out = 0; n_cmp_mid = 0; n_tol_hit = 0;
        cyc_min = 1000000; cyc_max = 0; cyc_sum = 0;
        gcyc = 0; done_cyc = 0;

        // 初始化
        rst_n = 1'b0;
        start = 1'b0; q_valid = 1'b0; kv_valid = 1'b0;
        q_in = 8'sd0; k_in = 8'sd0; v_in = 8'sd0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        fd = $fopen(vecfile, "r");
        if (fd == 0) begin
            $display("*** cannot open vector file %0s ***", vecfile);
            $finish;
        end

        if ($test$plusargs("VCD")) begin
            if (!$value$plusargs("VCD=%s", vcdfile))
                vcdfile = "waves/attention_ip.vcd";
            $dumpfile(vcdfile);
            $dumpvars(0, tb_attention_ip);
        end

        $display("============================================================");
        $display(" Attention IP self-checking testbench");
        $display("   vector file   : %0s", vecfile);
        $display("   KV stream     : %0d  (0=32-beat pair / 1=64-beat interleave / 2=64-beat duplicated)",
                 kv_mode_sel);
        $display("   out tolerance : %0d LSB", tol);
        $display("============================================================");

        ok = 1;
        while ((n_case < max_cases) && (ok != 0)) begin
            read_case(ok);
            if (ok != 0) drive_and_check;
        end

        $display("------------------------------------------------------------");
        $display("  Total cases         : %0d", n_pass + n_fail);
        $display("  Passed              : %0d", n_pass);
        $display("  Failed              : %0d", n_fail);
        $display("  Output compares     : %0d", n_cmp_out);
        $display("  Mid-result compares : %0d  (x_max/Score/Weight/L/ACC)", n_cmp_mid);
        $display("  Outputs not bit-exact: %0d  (tol %0d LSB)", n_tol_hit, tol);
        if (n_pass + n_fail > 0)
            $display("  Cycles per task     : min=%0d  max=%0d  avg=%0d",
                     cyc_min, cyc_max, cyc_sum / (n_pass + n_fail));
        if (n_fail == 0)
            $display("  *** ALL TESTS PASSED ***");
        else
            $display("  *** TESTS FAILED ***");
        $display("------------------------------------------------------------");

        $fclose(fd);
        $finish;
    end

endmodule
