`timescale 1ns/1ps
//===========================================================================
//  attention_ip.v  —— 基础 Attention 加速器 IP 顶层
//===========================================================================
module attention_ip #(
    parameter S         = 8,        // Token 数（1..15）
    parameter D         = 4,        // Head 维度（1..4）
    parameter CAUSAL_EN = 1'b0      // 加分项 A：1 = 开启因果 Mask
) (
    input  wire               clk,
    input  wire               rst_n,       // 低有效异步复位
    input  wire               start,       // 空闲时拉高 1 拍，启动一次任务
    input  wire signed [7:0]  q_in,
    input  wire signed [7:0]  k_in,
    input  wire signed [7:0]  v_in,
    input  wire               q_valid,     // q_in 有效，共 d 拍
    input  wire               kv_valid,    // k_in/v_in 有效，S·d 或 2·S·d 拍
    output wire               data_ready,  // 为 1 时才能发送输入
    output wire signed [7:0]  out_data,    // out_valid=1 时有效，共 d 拍
    output wire               out_valid,
    output wire               done,        // 拉高 1 拍表示任务结束
    input  wire [3:0]          q_pos
);

    //-----------------------------------------------------------------------
    // 由 S / D 自动推导的位宽（加分项 C：改 S/D 不需要手改任何位宽）
    //-----------------------------------------------------------------------
    localparam KV  = 2*S*D;                       // K/V 原始缓冲深度
    localparam KVH = S*D;                         // "成对"流的拍数

    localparam SW  = $clog2(D*16384) + 2;         // Score/diff 位宽  (8/4 → 18)
    localparam AW  = $clog2(S*2048)  + 2;         // ACC 位宽        (8/4 → 16)
    localparam LW  = $clog2(S*16)    + 1;         // L 位宽          (8   → 8)
    localparam PW  = $clog2(S*2048)  + 34;        // 归一化乘积位宽  (8/4 → 48)
    localparam MG  = $clog2(S*2048) + 1;          // |ACC| 位宽

    localparam WW  = 5;                           // weight：UQ1.4，0..16

    // 计数器位宽
    localparam TW_C = (S > 1) ? $clog2(S) : 1;    // token 计数 / q_pos
    localparam DW_C = (D > 1) ? $clog2(D) : 1;    // 维度计数
    localparam QW_C = $clog2(D + 1);              // Q 元素计数（0..d）
    localparam KW_C = $clog2(KV + 1);             // K/V 拍计数（0..2·S·d）

    // 状态编码
    localparam [3:0] ST_IDLE    = 4'd0,
                     ST_LOAD_Q  = 4'd1,
                     ST_LOAD_KV = 4'd2,
                     ST_DOT     = 4'd3,
                     ST_MAX     = 4'd4,
                     ST_EXP     = 4'd5,
                     ST_UPDATE  = 4'd6,
                     ST_NORM    = 4'd7,
                     ST_OUT     = 4'd8,
                     ST_DONE    = 4'd9;

    //-----------------------------------------------------------------------
    // 状态与寄存器
    //-----------------------------------------------------------------------
    reg  [3:0]      state;
    reg  [QW_C-1:0] q_cnt;          // 已接收的 Q 元素数 0..d
    reg  [KW_C-1:0] kv_cnt;         // 已接收的 K/V 拍数 0..2·S·d
    reg             kv_mode;        // 0: 成对流; 1: 交错流
    reg  [5:0]      kv_idle;        // 交错流中途停顿的兜底计数
    reg  [TW_C-1:0] tok;            // Token 计数 0..S-1
    reg  [DW_C-1:0] dim;            // 维度计数 0..d-1
    reg  [TW_C-1:0] max_cnt;        // 求最大值扫描计数
    reg  [1:0]      nstep;          // 归一化子步骤 0..2
    reg  [DW_C-1:0] out_cnt;        // 输出计数

    // 存储体（Q/ACC/OUT 固定按 4 路声明，D < 4 时多余部分由综合优化掉）
    reg  signed [7:0]    q_reg    [0:3];          // Q 寄存器
    reg  signed [7:0]    raw_k    [0:KV-1];       // K 原始缓冲
    reg  signed [7:0]    raw_v    [0:KV-1];       // V 原始缓冲
    reg  signed [SW-1:0] score_buf [0:S-1];       // Score 缓冲
    reg  [WW-1:0]        weight_buf [0:S-1];      // Weight 缓冲
    reg  signed [AW-1:0] acc_reg  [0:3];          // 加权累加
    reg  signed [7:0]    out_reg  [0:3];          // 输出寄存器

    // 数据通路寄存器
    reg  signed [SW-1:0] max_reg;               // x_max
    reg  signed [SW-1:0] dot_acc;               // 点积累加器
    reg  [LW-1:0]        l_sum;                 // L（权重和）
    reg  [PW-1:0]        p_reg;                 // 归一化：|ACC|×recip
    reg  [15:0]          a0_reg;                // 归一化：近似商
    reg  signed [18:0]   rem_reg;               // 归一化：余数候选

    integer i, k;

    //-----------------------------------------------------------------------
    // K/V 读地址生成
    //   成对流：addr = tok·d + dim
    //   交错流：K 在偶数地址 2·idx，V 在奇数地址 2·idx+1
    //-----------------------------------------------------------------------
    wire [TW_C+DW_C-1:0] k_idx  = tok * D + dim;
    wire [TW_C+DW_C-1:0] v_idx0 = tok * D;
    wire [TW_C+DW_C-1:0] v_idx1 = tok * D + 1;
    wire [TW_C+DW_C-1:0] v_idx2 = tok * D + 2;
    wire [TW_C+DW_C-1:0] v_idx3 = tok * D + 3;

    wire [KW_C-1:0] k_addr  = kv_mode ? {k_idx,  1'b0} : k_idx;
    wire [KW_C-1:0] v_addr0 = kv_mode ? {v_idx0, 1'b1} : v_idx0;
    wire [KW_C-1:0] v_addr1 = kv_mode ? {v_idx1, 1'b1} : v_idx1;
    wire [KW_C-1:0] v_addr2 = kv_mode ? {v_idx2, 1'b1} : v_idx2;
    wire [KW_C-1:0] v_addr3 = kv_mode ? {v_idx3, 1'b1} : v_idx3;

    //-----------------------------------------------------------------------
    // 指数查找表（题目要求自行实现；本工程沿用已写好的 exp_lut）
    //   diff ≤ -16 时把地址钳到 1（该地址对应 diff=-15，输出 0），
    //   否则直接取 diff 的低 4 位（补码），0/-1/-2/-3 分别对应 16/6/2/1。
    //-----------------------------------------------------------------------
    wire signed [SW-1:0] diff     = score_buf[tok] - max_reg;
    wire [3:0]           exp_addr = (diff < -18'sd15) ? 4'd1 : diff[3:0];
    wire [7:0]           exp_val;

    exp_lut u_exp_lut (
        .diff_addr (exp_addr),
        .exp_val   (exp_val)
    );

    // ---- 加分项 A：因果 Mask -----------------------------------------------
    // CAUSAL_EN=1 时，位置在 tok > q_pos 的 Key 被屏蔽，weight 强制为 0。
    wire          causal_block = CAUSAL_EN && (tok > q_pos);
    wire [WW-1:0] w_new        = causal_block ? {WW{1'b0}} : exp_val[WW-1:0];

    //-----------------------------------------------------------------------
    // DOT：单乘法器 d 拍累加得到一个 score
    //-----------------------------------------------------------------------
    wire signed [15:0]   prod     = q_reg[dim] * raw_k[k_addr];
    wire signed [SW-1:0] dot_next = (dim == 0) ? prod : (dot_acc + prod);

    //-----------------------------------------------------------------------
    // UPDATE：4 路并行乘法器，一拍更新 4 个 ACC 分量
    //   weight 为无符号，需显式转成有符号再与 V 相乘
    //-----------------------------------------------------------------------
    wire signed [5:0]    w_s    = $signed({1'b0, weight_buf[tok]});
    wire signed [AW-1:0] acc_n0 = acc_reg[0] + w_s * raw_v[v_addr0];
    wire signed [AW-1:0] acc_n1 = acc_reg[1] + w_s * raw_v[v_addr1];
    wire signed [AW-1:0] acc_n2 = acc_reg[2] + w_s * raw_v[v_addr2];
    wire signed [AW-1:0] acc_n3 = acc_reg[3] + w_s * raw_v[v_addr3];

    //-----------------------------------------------------------------------
    // NORM：O = round(ACC/L) = 符号 × round_half_away(|ACC|/L)
    //   第一步：P = |ACC| × recip，recip = round(2^32/L)（倒数 LUT 输出 Q0.32）
    //   第二步：A0 = P >> 32（近似商），rem = |ACC| - A0·L
    //   第三步：余数修正（±1）后按 2·rem ≥ L 四舍五入
    //   已用穷举脚本证明：对全部可达 (ACC, L)，该通路与精确四舍五入逐位一致。
    //-----------------------------------------------------------------------
    wire           acc_sign = acc_reg[dim][AW-1];
    wire [AW-1:0]  mag16    = acc_sign ? (~acc_reg[dim] + 1'b1) : acc_reg[dim];
    wire [MG-1:0]  mag      = mag16[MG-1:0];       // |ACC| ≤ S×2048

    wire [32:0]    recip_val;

    // recip_lut 的地址是 8 位。L ≤ 240（S ≤ 15）时 8 位足够，
    // 但 S 很小时 LW < 8，直接写 l_sum[7:0] 会越界选到 x；
    // 用一条赋值统一适配：窄了自动零扩展，宽了截掉恒为 0 的高位。
    wire [7:0]     l_addr = l_sum;

    recip_lut u_recip_lut (
        .l_sum     (l_addr),
        .recip_val (recip_val)
    );

    wire [PW-1:0]      p_next = mag * recip_val;

    wire signed [18:0] l_s    = $signed({11'b0, l_addr});
    // rem_c 的真实取值范围是 (-L, 2L)，19 位有符号绰绰有余
    // （L ≥ 16 由算法保证：取得最大值的那个 Token 权重必然是 16）
    wire signed [18:0] rem_c  = $signed({4'b0, mag}) -
                                $signed({1'b0, (p_reg[PW-1:32] * l_addr)});

    // 余数修正：把 rem 归到 [0, L)
    wire signed [18:0] rem_f  = (rem_reg < 0)   ? (rem_reg + l_s) :
                               (rem_reg >= l_s) ? (rem_reg - l_s) : rem_reg;
    wire [16:0]        a_f    = (rem_reg < 0)   ? ({1'b0, a0_reg} - 17'd1) :
                               (rem_reg >= l_s) ? ({1'b0, a0_reg} + 17'd1) :
                                                  {1'b0, a0_reg};
    // 四舍五入（.5 远离 0）
    wire [16:0]        o_mag  = (2*rem_f >= l_s) ? (a_f + 17'd1) : a_f;
    wire signed [16:0] o_sgn  = acc_sign ? -$signed({1'b0, o_mag[14:0]})
                                         :  $signed({1'b0, o_mag[14:0]});

    //-----------------------------------------------------------------------
    // 主状态机
    //-----------------------------------------------------------------------
    wire cap_q  = (state == ST_LOAD_Q) || (state == ST_LOAD_KV) ||
                  ((state == ST_IDLE) && start);
    wire cap_kv = (state == ST_LOAD_Q) || (state == ST_LOAD_KV) ||
                  ((state == ST_IDLE) && start);

    // ---- 加分项 A：求最大值时也要跳过被 mask 的位置 ------------------------
    // 注意 max_cnt = 0 永远不会被 mask（因为 q_pos ≥ 0），
    // 所以 max_reg 一定能被初始化。
    wire max_skip = CAUSAL_EN && (max_cnt > q_pos);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= ST_IDLE;
            q_cnt    <= {QW_C{1'b0}};
            kv_cnt   <= {KW_C{1'b0}};
            kv_mode  <= 1'b0;
            kv_idle  <= 6'd0;
            tok      <= {TW_C{1'b0}};
            dim      <= {DW_C{1'b0}};
            max_cnt  <= {TW_C{1'b0}};
            nstep    <= 2'd0;
            out_cnt  <= {DW_C{1'b0}};
            dot_acc  <= {SW{1'b0}};
            max_reg  <= {SW{1'b0}};
            l_sum    <= {LW{1'b0}};
            p_reg    <= {PW{1'b0}};
            a0_reg   <= 16'd0;
            rem_reg  <= 19'sd0;
            for (i = 0; i < S; i = i + 1) begin
                score_buf[i]  <= {SW{1'b0}};
                weight_buf[i] <= {WW{1'b0}};
            end
            for (i = 0; i < 4; i = i + 1) begin
                q_reg[i]   <= 8'sd0;
                acc_reg[i] <= {AW{1'b0}};
                out_reg[i] <= 8'sd0;
            end
            for (i = 0; i < KV; i = i + 1) begin
                raw_k[i] <= 8'sd0;
                raw_v[i] <= 8'sd0;
            end
        end
        else begin
            //---------------------------------------------------------------
            // 输入捕获（装载阶段，只有 valid 有效时才消耗一拍）
            //---------------------------------------------------------------
            if (q_valid && cap_q && (q_cnt < D)) begin
                q_reg[q_cnt] <= q_in;
                q_cnt        <= q_cnt + 1'b1;
            end
            if (kv_valid && cap_kv && (kv_cnt < KV)) begin
                raw_k[kv_cnt] <= k_in;
                raw_v[kv_cnt] <= v_in;
                kv_cnt        <= kv_cnt + 1'b1;
            end

            //---------------------------------------------------------------
            case (state)
            //---------------------------------------------------------------
            ST_IDLE: begin
                if (start) begin
                    q_cnt   <= {QW_C{1'b0}};
                    kv_cnt  <= {KW_C{1'b0}};
                    kv_mode <= 1'b0;
                    kv_idle <= 6'd0;
                    state   <= ST_LOAD_Q;
                end
            end
            //---------------------------------------------------------------
            ST_LOAD_Q: begin
                if (q_cnt >= D) state <= ST_LOAD_KV;
            end
            //---------------------------------------------------------------
            // 收满 S·d 拍后若 kv_valid 仍为高 → 判为 2·S·d 拍交错流
            //---------------------------------------------------------------
            ST_LOAD_KV: begin
                if (kv_valid) begin
                    kv_idle <= 6'd0;
                    if (kv_cnt == KVH) kv_mode <= 1'b1;
                    if (kv_cnt == KV-1) begin                 // 已收满
                        kv_mode <= 1'b1;
                        state   <= ST_DOT;
                        tok     <= {TW_C{1'b0}};
                        dim     <= {DW_C{1'b0}};
                        dot_acc <= {SW{1'b0}};
                    end
                end
                else begin
                    if (kv_cnt == KVH) begin                  // S·d 拍成对流
                        kv_mode <= 1'b0;
                        state   <= ST_DOT;
                        tok     <= {TW_C{1'b0}};
                        dim     <= {DW_C{1'b0}};
                        dot_acc <= {SW{1'b0}};
                    end
                    else if (kv_cnt > KVH) begin              // 兜底：不再无限等待
                        kv_idle <= kv_idle + 6'd1;
                        if (kv_idle == 63) state <= ST_DOT;
                    end
                end
            end
            //---------------------------------------------------------------
            // 点积：d 拍一个 score，S 个 Token 共 S·d 拍
            //---------------------------------------------------------------
            ST_DOT: begin
                dot_acc <= dot_next;
                if (dim == D-1) begin
                    score_buf[tok] <= dot_next;
                    if (tok == S-1) begin
                        state   <= ST_MAX;
                        max_cnt <= {TW_C{1'b0}};
                    end
                    else begin
                        tok <= tok + 1'b1;
                        dim <= {DW_C{1'b0}};
                    end
                end
                else begin
                    dim <= dim + 1'b1;
                end
            end
            //---------------------------------------------------------------
            // 求 x_max：S 拍顺序比较；同时清零 L 与 ACC
            //---------------------------------------------------------------
            ST_MAX: begin
                if (max_cnt == 0) begin
                    max_reg <= score_buf[0];
                    l_sum   <= {LW{1'b0}};
                    for (k = 0; k < 4; k = k + 1) acc_reg[k] <= {AW{1'b0}};
                end
                else if (!max_skip && (score_buf[max_cnt] > max_reg)) begin
                    max_reg <= score_buf[max_cnt];
                end
                if (max_cnt == S-1) begin
                    state <= ST_EXP;
                    tok   <= {TW_C{1'b0}};
                end
                else begin
                    max_cnt <= max_cnt + 1'b1;
                end
            end
            //---------------------------------------------------------------
            // 权重：diff = score - x_max → exp_lut → weight，同时累加 L
            // （加分项 A：被 mask 的位置 weight 为 0，不贡献 L）
            //---------------------------------------------------------------
            ST_EXP: begin
                weight_buf[tok] <= w_new;
                l_sum           <= l_sum + {{(LW-WW){1'b0}}, w_new};
                if (tok == S-1) begin
                    state <= ST_UPDATE;
                    tok   <= {TW_C{1'b0}};
                end
                else begin
                    tok <= tok + 1'b1;
                end
            end
            //---------------------------------------------------------------
            // 加权累加：一拍更新 ACC，S 拍完成
            //---------------------------------------------------------------
            ST_UPDATE: begin
                acc_reg[0] <= acc_n0;
                if (D > 1) acc_reg[1] <= acc_n1;
                if (D > 2) acc_reg[2] <= acc_n2;
                if (D > 3) acc_reg[3] <= acc_n3;
                if (tok == S-1) begin
                    state <= ST_NORM;
                    nstep <= 2'd0;
                    dim   <= {DW_C{1'b0}};
                end
                else begin
                    tok <= tok + 1'b1;
                end
            end
            //---------------------------------------------------------------
            // 归一化：每个输出维度 3 拍（乘 → 取商求余 → 修正舍入）
            //---------------------------------------------------------------
            ST_NORM: begin
                case (nstep)
                2'd0: begin
                    p_reg <= p_next;
                    nstep <= 2'd1;
                end
                2'd1: begin
                    a0_reg  <= p_reg[PW-1:32];
                    rem_reg <= rem_c;
                    nstep   <= 2'd2;
                end
                default: begin
                    if (o_sgn > 17'sd127)        out_reg[dim] <= 8'sd127;
                    else if (o_sgn < -17'sd128)  out_reg[dim] <= -8'sd128;
                    else                         out_reg[dim] <= o_sgn[7:0];
                    nstep <= 2'd0;
                    if (dim == D-1) begin
                        state   <= ST_OUT;
                        out_cnt <= {DW_C{1'b0}};
                        dim     <= {DW_C{1'b0}};
                    end
                    else begin
                        dim <= dim + 1'b1;
                    end
                end
                endcase
            end
            //---------------------------------------------------------------
            // 输出：连续 d 拍，每拍一个 8-bit 有符号结果
            //---------------------------------------------------------------
            ST_OUT: begin
                if (out_cnt == D-1) state <= ST_DONE;
                else                out_cnt <= out_cnt + 1'b1;
            end
            //---------------------------------------------------------------
            ST_DONE: begin
                state   <= ST_IDLE;
                q_cnt   <= {QW_C{1'b0}};
                kv_cnt  <= {KW_C{1'b0}};
                kv_mode <= 1'b0;
                kv_idle <= 6'd0;
                tok     <= {TW_C{1'b0}};
                dim     <= {DW_C{1'b0}};
                max_cnt <= {TW_C{1'b0}};
                nstep   <= 2'd0;
                out_cnt <= {DW_C{1'b0}};
            end
            //---------------------------------------------------------------
            default: state <= ST_IDLE;
            endcase
        end
    end

    //-----------------------------------------------------------------------
    // 输出端口
    //-----------------------------------------------------------------------
    assign data_ready = (state == ST_IDLE) ||
                        (state == ST_LOAD_Q) ||
                        (state == ST_LOAD_KV);
    assign out_valid  = (state == ST_OUT);
    assign out_data   = out_reg[out_cnt];
    assign done       = (state == ST_DONE);

endmodule
