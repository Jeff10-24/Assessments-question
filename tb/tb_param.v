`timescale 1ns/1ps
//===========================================================================
//  tb_param.v —— 加分项 C（参数化 S / d）独立测试
//---------------------------------------------------------------------------
//  同一个 testbench 里同时例化 5 种不同规模的 attention_ip，
//  共用一组输入总线，一次只启动一个，逐个验证输出。
//
//  每种规模跑两组输入：
//    场景 A：所有 Token 的 Score 都相同  → 权重全 16，L = 16S（考验 L 与缩放）
//    场景 B：Score 有正有负、有大有小    → 覆盖指数表各档位与累加
//  期望值全部由 model/golden_model.py 用同样的输入算出。
//
//  被测配置：
//    cfg0: S=8 , d=4      cfg1: S=4 , d=4      cfg2: S=15, d=4
//    cfg3: S=2 , d=2      cfg4: S=8 , d=1
//
//  运行：
//      iverilog -g2005 -o sim/tb_param.vvp rtl/attention_ip.v rtl/exp_lut.v \
//               rtl/recip_lut.v tb/tb_param.v    &&    vvp sim/tb_param.vvp
//      （ModelSim: vlog ... tb/tb_param.v  →  vsim -c -do "run -all; quit -f" work.tb_param）
//===========================================================================
module tb_param;

    reg clk = 0;
    always #5 clk = ~clk;

    //-----------------------------------------------------------------------
    // 共用输入总线
    //-----------------------------------------------------------------------
    reg               rst_n;
    reg  signed [7:0] q_in, k_in, v_in;
    reg               q_valid, kv_valid;
    reg  [4:0]        starts;              // 5 个 start，一次只拉高一个
    wire [4:0]        drs, ovs, dns;
    wire signed [7:0] od0, od1, od2, od3, od4;

    attention_ip #(.S(8),  .D(4)) u0 (
        .clk(clk), .rst_n(rst_n), .start(starts[0]),
        .q_in(q_in), .k_in(k_in), .v_in(v_in),
        .q_valid(q_valid), .kv_valid(kv_valid),
        .data_ready(drs[0]), .out_data(od0), .out_valid(ovs[0]), .done(dns[0]),
        .q_pos(4'd0));          

    attention_ip #(.S(4),  .D(4)) u1 (
        .clk(clk), .rst_n(rst_n), .start(starts[1]),
        .q_in(q_in), .k_in(k_in), .v_in(v_in),
        .q_valid(q_valid), .kv_valid(kv_valid),
        .data_ready(drs[1]), .out_data(od1), .out_valid(ovs[1]), .done(dns[1]),
        .q_pos(4'd0));          

    attention_ip #(.S(15), .D(4)) u2 (
        .clk(clk), .rst_n(rst_n), .start(starts[2]),
        .q_in(q_in), .k_in(k_in), .v_in(v_in),
        .q_valid(q_valid), .kv_valid(kv_valid),
        .data_ready(drs[2]), .out_data(od2), .out_valid(ovs[2]), .done(dns[2]),
        .q_pos(4'd0));          

    attention_ip #(.S(2),  .D(2)) u3 (
        .clk(clk), .rst_n(rst_n), .start(starts[3]),
        .q_in(q_in), .k_in(k_in), .v_in(v_in),
        .q_valid(q_valid), .kv_valid(kv_valid),
        .data_ready(drs[3]), .out_data(od3), .out_valid(ovs[3]), .done(dns[3]),
        .q_pos(4'd0));          

    attention_ip #(.S(8),  .D(1)) u4 (
        .clk(clk), .rst_n(rst_n), .start(starts[4]),
        .q_in(q_in), .k_in(k_in), .v_in(v_in),
        .q_valid(q_valid), .kv_valid(kv_valid),
        .data_ready(drs[4]), .out_data(od4), .out_valid(ovs[4]), .done(dns[4]),
        .q_pos(4'd0));          

    //-----------------------------------------------------------------------
    // 激励数据（按最大规模 S=15, d=4 开数组）
    //-----------------------------------------------------------------------
    reg signed [7:0] Q [0:3];
    reg signed [7:0] K [0:14][0:3];
    reg signed [7:0] V [0:14][0:3];

    // 期望输出：EXP[cfg][scen][dim]
    reg signed [7:0] EXP_O [0:4][0:1][0:3];
    integer          EXP_L [0:4][0:1];

    // 每种配置的 S / d
    integer CFG_S [0:4];
    integer CFG_D [0:4];

    integer i, j, cfg, scen, p, fails, cases, got_n, guard;
    reg signed [7:0] got [0:3];
    integer          Dn, Sn;
    reg              dn, ov;
    reg signed [7:0] od;
    reg              dr;

    //-----------------------------------------------------------------------
    // 生成某配置 / 某场景的输入
    //-----------------------------------------------------------------------
    task gen_input;
        input integer Sn_, Dn_, sc_;
        integer a, b;
        begin
            if (sc_ == 0) begin
                // 场景 A：所有 Score 相同
                for (a = 0; a < Dn_; a = a + 1) Q[a] = 1;
                for (a = 0; a < Sn_; a = a + 1) begin
                    for (b = 0; b < Dn_; b = b + 1) begin
                        K[a][b] = 1;
                        V[a][b] = 0;
                    end
                end
                V[0][0] = 100;
            end
            else begin
                // 场景 B：Score 有正有负
                for (a = 0; a < Dn_; a = a + 1) Q[a] = a + 1;
                for (a = 0; a < Sn_; a = a + 1) begin
                    for (b = 0; b < Dn_; b = b + 1) begin
                        K[a][b] = (a + b) % 5 - 2;
                        V[a][b] = (a*3 + b*5) % 21 - 10;
                    end
                end
            end
        end
    endtask

    //-----------------------------------------------------------------------
    // 驱动 + 采集
    //-----------------------------------------------------------------------
    task run_cfg;
        input integer cfg_;
        input integer Sn_;
        input integer Dn_;
        begin
            // 只启动选中的那一个
            @(negedge clk);
            starts = (5'b00001 << cfg_);
            @(negedge clk);
            starts = 5'b00000;

            // Q：Dn_ 拍
            for (i = 0; i < Dn_; i = i + 1) begin
                @(negedge clk);
                q_in    = Q[i];
                q_valid = 1'b1;
            end
            @(negedge clk);
            q_valid = 1'b0;

            // K/V：Sn_ × Dn_ 拍
            for (i = 0; i < Sn_; i = i + 1) begin
                for (j = 0; j < Dn_; j = j + 1) begin
                    @(negedge clk);
                    k_in     = K[i][j];
                    v_in     = V[i][j];
                    kv_valid = 1'b1;
                end
            end
            @(negedge clk);
            kv_valid = 1'b0;
            k_in = 8'sd0;  v_in = 8'sd0;

            // 等 done 并采集（每次重新选一次信号）
            got_n = 0;
            guard = 0;
            while (guard < 20000) begin
                @(negedge clk);
                case (cfg_)
                    0: begin dn = dns[0]; ov = ovs[0]; od = od0; end
                    1: begin dn = dns[1]; ov = ovs[1]; od = od1; end
                    2: begin dn = dns[2]; ov = ovs[2]; od = od2; end
                    3: begin dn = dns[3]; ov = ovs[3]; od = od3; end
                    default: begin dn = dns[4]; ov = ovs[4]; od = od4; end
                endcase
                if (ov) begin
                    if (got_n < 4) got[got_n] = od;
                    got_n = got_n + 1;
                end
                if (dn) guard = 20000;
                guard = guard + 1;
            end
            repeat (2) @(negedge clk);
            q_valid = 1'b0; kv_valid = 1'b0;
        end
    endtask

    //-----------------------------------------------------------------------
    // 读回 L（每个配置的信号名不同，只能分开写）
    //-----------------------------------------------------------------------
    reg [15:0] l_now;
    task get_l;
        input integer cfg_;
        begin
            case (cfg_)
                0: l_now = u0.l_sum;
                1: l_now = u1.l_sum;
                2: l_now = u2.l_sum;
                3: l_now = u3.l_sum;
                default: l_now = u4.l_sum;
            endcase
        end
    endtask

    //-----------------------------------------------------------------------
    // 主流程
    //-----------------------------------------------------------------------
    initial begin
        // ---- 配置表 ----
        CFG_S[0]=8;  CFG_D[0]=4;
        CFG_S[1]=4;  CFG_D[1]=4;
        CFG_S[2]=15; CFG_D[2]=4;
        CFG_S[3]=2;  CFG_D[3]=2;
        CFG_S[4]=8;  CFG_D[4]=1;

        // ---- 期望值（golden_model.py 算出）----
        EXP_O[0][0][0]= 13; EXP_O[0][0][1]=  0; EXP_O[0][0][2]=  0; EXP_O[0][0][3]= 0;
        EXP_O[0][1][0]=  1; EXP_O[0][1][1]= -5; EXP_O[0][1][2]=  0; EXP_O[0][1][3]= 5;
        EXP_L[0][0]=128; EXP_L[0][1]=32;

        EXP_O[1][0][0]= 25; EXP_O[1][0][1]=  0; EXP_O[1][0][2]=  0; EXP_O[1][0][3]= 0;
        EXP_O[1][1][0]= -7; EXP_O[1][1][1]= -2; EXP_O[1][1][2]=  3; EXP_O[1][1][3]= 8;
        EXP_L[1][0]= 64; EXP_L[1][1]=16;

        EXP_O[2][0][0]=  7; EXP_O[2][0][1]=  0; EXP_O[2][0][2]=  0; EXP_O[2][0][3]= 0;
        EXP_O[2][1][0]=  1; EXP_O[2][1][1]= -1; EXP_O[2][1][2]= -3; EXP_O[2][1][3]= 2;
        EXP_L[2][0]=240; EXP_L[2][1]=48;

        EXP_O[3][0][0]= 50; EXP_O[3][0][1]=  0; EXP_O[3][0][2]=  0; EXP_O[3][0][3]= 0;
        EXP_O[3][1][0]= -7; EXP_O[3][1][1]= -2; EXP_O[3][1][2]=  0; EXP_O[3][1][3]= 0;
        EXP_L[3][0]= 32; EXP_L[3][1]=17;

        EXP_O[4][0][0]= 13; EXP_O[4][0][1]=  0; EXP_O[4][0][2]=  0; EXP_O[4][0][3]= 0;
        EXP_O[4][1][0]=  0; EXP_O[4][1][1]=  0; EXP_O[4][1][2]=  0; EXP_O[4][1][3]= 0;
        EXP_L[4][0]=128; EXP_L[4][1]=28;

        // ---- 复位 ----
        rst_n = 1'b0;  starts = 5'b00000;
        q_valid = 1'b0;  kv_valid = 1'b0;
        q_in = 8'sd0;  k_in = 8'sd0;  v_in = 8'sd0;
        repeat (4) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        fails = 0;  cases = 0;

        $display("");
        $display("==============================================================");
        $display(" 加分项 C：参数化 S/d 独立测试");
        $display("==============================================================");

        for (cfg = 0; cfg < 5; cfg = cfg + 1) begin
            Sn = CFG_S[cfg];
            Dn = CFG_D[cfg];
            for (scen = 0; scen < 2; scen = scen + 1) begin
                gen_input(Sn, Dn, scen);
                run_cfg(cfg, Sn, Dn);
                get_l(cfg);
                cases = cases + 1;

                // 检查输出
                for (p = 0; p < Dn; p = p + 1) begin
                    if ($signed(got[p]) !== EXP_O[cfg][scen][p]) begin
                        $display("  [FAIL] cfg%0d(S=%0d,d=%0d) 场景%0d : O[%0d] RTL=%0d EXP=%0d",
                                 cfg, Sn, Dn, scen, p, $signed(got[p]),
                                 EXP_O[cfg][scen][p]);
                        fails = fails + 1;
                    end
                end
                // 检查输出个数
                if (got_n !== Dn) begin
                    $display("  [FAIL] cfg%0d 场景%0d : out_valid 拍了 %0d 次（应为 %0d）",
                             cfg, scen, got_n, Dn);
                    fails = fails + 1;
                end
                // 检查 L
                if (l_now[7:0] !== EXP_L[cfg][scen][7:0]) begin
                    $display("  [FAIL] cfg%0d 场景%0d : L RTL=%0d EXP=%0d",
                             cfg, scen, l_now[7:0], EXP_L[cfg][scen]);
                    fails = fails + 1;
                end

                $display("  cfg%0d  S=%-2d d=%0d  场景%0d : L=%3d  O=%4d %4d %4d %4d   OK",
                         cfg, Sn, Dn, scen, l_now[7:0],
                         $signed(EXP_O[cfg][scen][0]), $signed(EXP_O[cfg][scen][1]),
                         $signed(EXP_O[cfg][scen][2]), $signed(EXP_O[cfg][scen][3]));
            end
        end

        $display("");
        $display("--------------------------------------------------------------");
        $display("  用例数 : %0d", cases);
        $display("  失败数 : %0d", fails);
        if (fails == 0) $display("  *** 参数化测试全部通过 (ALL PASS) ***");
        else            $display("  *** 存在失败 (FAILED) ***");
        $display("--------------------------------------------------------------");
        $finish;
    end

endmodule
