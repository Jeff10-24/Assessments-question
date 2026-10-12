#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention 计算器（改数值版）
============================
不用再一行行输入了 —— 直接把下面的 Q / K / V 改成你要的数，
然后按 F5（或右上角 ▷ 运行）就能立刻看到结果。

输出的 score / x_max / diff / weight / L / ACC / O
和 rtl/attention_ip.v 完全一致，可以逐行拿去和 ModelSim 波形上的
score_buf / max_reg / weight_buf / l_sum / acc_reg / out_data 对照。

计算规则（与 RTL 一致）：
    · 先求 x_max，再算 exp(score - x_max)
    · weight = round(16 × exp(diff))，4 位小数；diff ≤ -15 直接判 0
    · O = round(ACC / L)，遇到 .5 时远离 0
"""

import math

S = 8   # Token 数（K 和 V 都必须有 8 行）
D = 4   # 每行几个数（必须是 4 个）


# ===========================================================================
#
#   ★★★★★  要改的数值都在下面这三块，改完直接按 F5  ★★★★★
#
#   规则：每个数必须是 -128 ~ 127 的整数（int8）
#         K 和 V 各要有 8 行，每行 4 个数
#         行尾的 K[0]、K[1]… 只是注释，方便你数行号
#
# ===========================================================================

Q = [1, -2, 3, -4]
#   ↑ 当前 Query 的 4 个数

K = [
    [ 0,  1, -1,  2],  # K[0]  ← 第 0 个 Token 的 Key
    [-2,  3,  0,  1],  # K[1]
    [ 1,  0,  2, -1],  # K[2]
    [ 3, -1,  1,  0],  # K[3]
    [-1,  2, -2,  3],  # K[4]
    [ 0, -3,  4,  1],  # K[5]
    [ 2,  1,  0, -2],  # K[6]
    [-3,  0,  1,  2],  # K[7]
]

V = [
    [10, -20,  30, -40],  # V[0]  ← 第 0 个 Token 的 Value
    [-5,  15, -25,  35],  # V[1]
    [20, -10,  40, -30],  # V[2]
    [-15, 25, -35,  45],  # V[3]
    [ 5,  -5,   5,  -5],  # V[4]
    [-10, 20, -30,  40],  # V[5]
    [15, -25,  35, -45],  # V[6]
    [ 0,   0,   0,   0],  # V[7]
]

# ==================================================
#  下面是计算和打印部分，一般不用改
# ===========================================================================

LUT_MIN = -15   # 指数表只覆盖 diff ∈ [-15, 0]，更小的一律为 0


def quant_weight(diff):
    """weight = round(16 × exp(diff))，取值 0..16（对应 RTL 的 exp_lut）"""
    if diff < LUT_MIN:
        return 0
    return int(math.floor(16.0 * math.exp(diff) + 0.5))


def div_round_half_away(acc, L):
    """四舍五入算 acc / L，遇到 .5 时远离 0（和 RTL 一致）"""
    if acc >= 0:
        return (2 * acc + L) // (2 * L)
    return -((-2 * acc + L) // (2 * L))


def check_input():
    """检查输入合不合法，不合法就打印原因并退出（不会抛异常）"""
    errs = []
    if len(Q) != D:
        errs.append("Q 应该有 %d 个数，现在是 %d 个" % (D, len(Q)))
    for name, arr, want in (("K", K, S), ("V", V, S)):
        if len(arr) != want:
            errs.append("%s 应该有 %d 行，现在是 %d 行" % (name, want, len(arr)))
            continue
        for j, row in enumerate(arr):
            if len(row) != D:
                errs.append("%s[%d] 应该有 %d 个数，现在是 %d 个" % (name, j, D, len(row)))
            for c, val in enumerate(row):
                if not isinstance(val, int) or val < -128 or val > 127:
                    errs.append("%s[%d][%d] = %r 超出 int8 范围 [-128, 127]"
                                % (name, j, c, val))
    for c, val in enumerate(Q):
        if not isinstance(val, int) or val < -128 or val > 127:
            errs.append("Q[%d] = %r 超出 int8 范围 [-128, 127]" % (c, val))
    return errs


def fmt_row(row):
    return "[" + ", ".join("%4d" % v for v in row) + "]"


def fmt_f(row):
    return "[" + ", ".join("%g" % v for v in row) + "]"


def main():
    errs = check_input()
    if errs:
        print()
        print("!! 输入有问题，请先改好上面的 Q / K / V：")
        for e in errs:
            print("   ×", e)
        print()
        return 1

    # ---------------- 输入回显（方便截图时连输入一起看到）----------------
    print()
    print("=" * 62)
    print(" 输入")
    print("=" * 62)
    print(" Q      =", fmt_row(Q))
    for j in range(S):
        tag = " K[%d]   =" % j if j == 0 else "          "
        print(tag, fmt_row(K[j]))
    print()
    for j in range(S):
        tag = " V[%d]   =" % j if j == 0 else "          "
        print(tag, fmt_row(V[j]))

    # ---------------- 计算 ----------------
    score = [sum(Q[c] * K[j][c] for c in range(D)) for j in range(S)]
    x_max = max(score)
    diff = [s - x_max for s in score]
    weight = [quant_weight(d) for d in diff]
    L = sum(weight)
    acc = [sum(weight[j] * V[j][c] for j in range(S)) for c in range(D)]
    out = [div_round_half_away(acc[c], L) for c in range(D)]

    # ---------------- 输出 ----------------
    print()
    print("=" * 62)
    print(" 计算结果（左边是波形上的信号名，右边是对应的数）")
    print("=" * 62)
    print(" score  = Q·K_j           :", score)
    print(" x_max  = max(score)      :", x_max)
    print(" diff   = score - x_max   :", diff)
    print(" weight = round(16·e^diff):", weight)
    print("   真值 = weight / 16      :", fmt_f([round(w / 16.0, 4) for w in weight]))
    print(" L      = Σ weight        : %d   (= %.4f)" % (L, L / 16.0))
    print(" ACC    = Σ weight·V      :", acc)
    print("   真值 = ACC / 16         :", fmt_f([round(a / 16.0, 4) for a in acc]))
    print("-" * 62)
    print(" O      = round(ACC/L)    :", out)
    print("=" * 62)
    print()
    print(" 在波形上核对：score_buf → max_reg → weight_buf/l_sum → acc_reg → out_data")
    print()
    return 0


if __name__ == "__main__":
    main()
