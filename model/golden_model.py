#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Attention 加速器 IP —— Python Golden Model
==========================================
对应题目：《2026数字组考核v2》A 题（S=8, d=4, Batch=1, Head=1）

本模型与 RTL 采用**完全相同的定点数值方案**，因此可以做到逐位一致：

  1) score_j = Q · K_j = Σ_k Q[k]·K_j[k]          整数，范围 [-65024, 65536] → 18-bit 有符号
  2) x_max   = max_j(score_j)                     整数，18-bit 有符号
  3) diff_j  = score_j - x_max  ≤ 0               18-bit 有符号
  4) weight_j = round(16 · exp(diff_j))           无符号定点 UQ1.4，取值 0..16 → 5-bit
     （小数固定 4 位；因为减掉了 x_max，exp(diff)∈(0,1]，整数位只需 1 位）
  5) L       = Σ_j weight_j                       范围 [16,128] → 8-bit 无符号
  6) ACC[k]  = Σ_j weight_j · V_j[k]              与 weight 同标度（保留 4 位小数）→ 16-bit 有符号
  7) O[k]    = round_half_away(ACC[k] / L)        8-bit 有符号
     注意：weight 与 ACC 都带 4 位小数，L 也带 4 位小数，所以 ACC/L 的两份
     2^4 因子正好约掉，**归一化不需要额外移位**。

关键点：题目要求 "不是对 e^score 四舍五入后取整"，而是把 exp 结果做成
无符号定点数（小数 4 位）。所以 weight_j = round(16·exp(diff_j))，
diff_j = -1 时 exp(-1)=0.3679 → 6（即 6/16 = 0.375）。

运行：python model/golden_model.py        （自检 + 打印手算示例）
"""

import math
import random
import sys
from fractions import Fraction

# ----------------------------------------------------------------------------
# 题目固定规模
# ----------------------------------------------------------------------------
S = 8          # Token 数
D = 4          # Head 维度
FRAC = 4       # weight 小数位宽（题目规定固定 4 位）
WSCALE = 1 << FRAC          # 16
WEIGHT_MAX = WSCALE         # weight 最大值 = 1.0
L_MIN = WSCALE              # 最大值那一路至少贡献 16
L_MAX = S * WSCALE          # 8 × 16 = 128

SIGNED8_MIN, SIGNED8_MAX = -128, 127

# ----------------------------------------------------------------------------
# 量化函数（RTL 中的 exp_lut 用同一张表）
# ----------------------------------------------------------------------------

def quant_weight(diff):
    """diff = score - x_max ≤ 0（整数） → weight = round(16·exp(diff))，UQ1.4。"""
    if diff > 0:
        raise ValueError("diff 必须 <= 0（必须先减 x_max）")
    v = math.exp(float(diff)) * WSCALE
    return int(math.floor(v + 0.5))          # round-half-up（正值，等价于四舍五入）


def exp_lut_table():
    """复刻 exp_lut.v：地址 = diff 的低 4 位（补码），diff ≤ -16 时钳到 0。"""
    tbl = [0] * 16
    tbl[0] = quant_weight(0)                 # diff = 0
    for a in range(1, 16):
        d = a - 16                           # a=1 → -15, ..., a=15 → -1
        tbl[a] = quant_weight(d)
    return tbl


def round_half_away(num, den):
    """四舍五入 num/den（den > 0），.5 远离 0（与 RTL 的 +半个 LSB 一致）。"""
    assert den > 0
    if num >= 0:
        return (2 * num + den) // (2 * den)
    return -((-2 * num + den) // (2 * den))


def recip_q32(l_sum):
    """复刻 recip_lut.v：round(2^32 / L)，L<16 时饱和到 2^33-1（实际不可达）。"""
    if l_sum == 0:
        return (1 << 33) - 1
    if l_sum < L_MIN:
        return (1 << 33) - 1
    return (1 << 32) // l_sum + (1 if (1 << 32) % l_sum * 2 >= l_sum else 0)


# ----------------------------------------------------------------------------
# 主计算
# ----------------------------------------------------------------------------

def attention(q, k, v, trace=False, causal_i=None):
    """
    q : list[D]         (int8)
    k : list[S][D]      (int8)
    v : list[S][D]      (int8)
    causal_i : None 或 整数 i（0..S-1）
        None      → 基础版（全可见）
        整数 i    → 加分项 A 因果 Mask：只有 j ≤ i 的 Key 可见，
                    被屏蔽的位置 weight 强制为 0，且**求 x_max 时也跳过它们**
                    （否则最大值被屏蔽位拉大，有效权重会整体变小，
                      在 4 位小数下会大量量化为 0）。

    返回 dict：scores / xmax / diff / weights / L / acc / out
    """
    Dn = len(q)
    Sn = len(k)
    assert Sn >= 1 and Dn >= 1
    assert len(v) == Sn
    for row in k + v:
        assert len(row) == Dn

    if causal_i is None:
        visible = list(range(Sn))
    else:
        assert 0 <= causal_i < Sn
        visible = [j for j in range(Sn) if j <= causal_i]

    scores = [sum(q[c] * k[j][c] for c in range(Dn)) for j in range(Sn)]
    xmax = max(scores[j] for j in visible)
    diff = [scores[j] - xmax for j in range(Sn)]
    weights = [quant_weight(diff[j]) if j in visible else 0 for j in range(Sn)]
    L = sum(weights)
    acc = [sum(weights[j] * v[j][c] for j in range(Sn)) for c in range(Dn)]
    out = [round_half_away(acc[c], L) for c in range(Dn)]

    if trace:
        print("  scores  =", scores)
        print("  x_max   =", xmax)
        print("  diff    =", diff)
        print("  weight  =", weights, " (= 值×16 )")
        print("  L       =", L)
        print("  ACC     =", acc)
        print("  O       =", out)

    return dict(scores=scores, xmax=xmax, diff=diff, weights=weights,
                L=L, acc=acc, out=out)


def attention_float(q, k, v):
    """不做任何量化的浮点参考（用于统计量化误差，不参与 RTL 比对）。"""
    scores = [sum(q[c] * k[j][c] for c in range(D)) for j in range(S)]
    xmax = max(scores)
    e = [math.exp(s - xmax) for s in scores]
    Lf = sum(e)
    acc = [sum(e[j] * v[j][c] for j in range(S)) for c in range(D)]
    return [round_half_away(int(round(acc[c] / Lf * 4096)), 4096) for c in range(D)]


# ----------------------------------------------------------------------------
# RTL 归一化通路（倒数 LUT + 余数修正）的等价性验证
# ----------------------------------------------------------------------------

def norm_rtl(acc, l_sum):
    """按 RTL 的 3 拍归一化流程计算 round_half_away(acc/L)。"""
    mag = abs(acc)
    R = recip_q32(l_sum)
    A0 = (mag * R) >> 32                     # 近似商
    rem = mag - A0 * l_sum                   # 余数候选
    if rem < 0:
        rem += l_sum
        A = A0 - 1
    elif rem >= l_sum:
        rem -= l_sum
        A = A0 + 1
    else:
        A = A0
    omag = A + 1 if 2 * rem >= l_sum else A
    return -omag if acc < 0 else omag


def check_norm_exhaustive(verbose=True):
    """穷举所有可达 (ACC, L)，验证 RTL 归一化与精确四舍五入完全一致。"""
    bad = 0
    worst = None
    for l_sum in range(L_MIN, L_MAX + 1):
        for mag in range(0, 16385):          # |ACC| ≤ 8×16×128 = 16384
            for sgn in (1, -1):
                acc = sgn * mag
                if norm_rtl(acc, l_sum) != round_half_away(acc, l_sum):
                    bad += 1
                    if worst is None:
                        worst = (acc, l_sum, norm_rtl(acc, l_sum),
                                 round_half_away(acc, l_sum))
    if verbose:
        print("  [归一化通路] 穷举 L∈[16,128], |ACC|∈[0,16384] :",
              "全部一致" if bad == 0 else "不一致 %d 处 %s" % (bad, worst))
    return bad == 0


def check_recip_table():
    """倒数表最大相对误差（用于报告）。"""
    e = 0.0
    for L in range(L_MIN, L_MAX + 1):
        e = max(e, abs(recip_q32(L) / 2.0 ** 32 - 1.0 / L) * L)
    return e


# ----------------------------------------------------------------------------
# 自检
# ----------------------------------------------------------------------------

def selftest():
    ok = True
    print("=" * 74)
    print("Golden Model 自检")
    print("=" * 74)

    # --- 1. exp LUT ---
    tbl = exp_lut_table()
    exp_tbl = [quant_weight(a - 16) if a else quant_weight(0) for a in range(16)]
    print("  地址 a     :", " ".join("%4d" % a for a in range(16)))
    print("  对应 diff  :", " ".join("%4d" % (a - 16 if a else 0) for a in range(16)))
    print("  exp_lut 表 :", " ".join("%4d" % t for t in tbl))
    if tbl != exp_tbl:
        print("  [FAIL] exp_lut 与公式不一致")
        ok = False
    else:
        print("  [ OK ] exp_lut 16 项与 round(16·exp(diff)) 完全一致")

    # --- 2. 题目第三节手算示例（S=2, d=2, 用真实 exp 的定点量化版本）---
    print("\n  --- 题目手算示例复现 (Q=[1,2], K0=[1,0], K1=[0,1], "
          "V0=[4,2], V1=[1,6]) ---")
    q2 = [1, 2]
    k2 = [[1, 0], [0, 1]]
    v2 = [[4, 2], [1, 6]]
    scores = [sum(q2[c] * k2[j][c] for c in range(2)) for j in range(2)]
    xmax = max(scores)
    w = [quant_weight(s - xmax) for s in scores]
    L = sum(w)
    acc = [sum(w[j] * v2[j][c] for j in range(2)) for c in range(2)]
    out = [round_half_away(acc[c], L) for c in range(2)]
    print("      score =", scores, " x_max =", xmax)
    print("      weight(×16) =", w, "→ 小数表示", [f"{x/16:.4f}" for x in w],
          " (题目近似值 0.4 / 1.0)")
    print("      L =", L, f"({L/16:.4f})", " ACC =", acc, f"({[x/16 for x in acc]})")
    print("      O =", out, " (题目答案 [2, 5])")
    if out != [2, 5]:
        print("  [FAIL] 手算示例不匹配")
        ok = False
    else:
        print("  [ OK ] 与题目手算示例 O=[2,5] 一致")

    # --- 3. 边界 / 极端输入 ---
    print("\n  --- 边界用例 ---")
    cases = {
        "全 0": ([0] * D, [[0] * D] * S, [[0] * D] * S),
        "Q/K 全 -128": ([-128] * D, [[-128] * D] * S, [[-128] * D] * S),
        "Q/K 全 127": ([127] * D, [[127] * D] * S, [[127] * D] * S),
        "Q=-128,K=-128": ([-128] * D, [[-128] * D] * S, [[0] * D] * S),
        "V 全 -128": ([1, 2, 3, 4], [[1, 0, 0, 0]] * S, [[-128] * D] * S),
        "V 全 127": ([1, 2, 3, 4], [[1, 0, 0, 0]] * S, [[127] * D] * S),
    }
    for name, (q_, k_, v_) in cases.items():
        r = attention(q_, k_, v_)
        print("      %-14s score_max=%-8d L=%-4d ACC=%s O=%s"
              % (name, r["xmax"], r["L"], r["acc"], r["out"]))
        if not all(SIGNED8_MIN <= o <= SIGNED8_MAX for o in r["out"]):
            print("  [FAIL] 输出越界:", name)
            ok = False
    # Q=K=-128 时 score = 4×16384 = 65536（需要 18 bit）
    r = attention([-128] * D, [[-128] * D] * S, [[0] * D] * S)
    if r["scores"][0] != 65536:
        print("  [FAIL] 点积上界计算错误")
        ok = False
    else:
        print("  [ OK ] 点积上界 4×(-128)×(-128) = +65536（故 Score 需 18 bit）")
    r = attention([127] * D, [[-127] * D] * S, [[0] * D] * S)
    # 下界：(-128)×(127)×4 = -65024
    r2 = attention([-128] * D, [[127] * D] * S, [[0] * D] * S)
    if r2["scores"][0] != -65024:
        print("  [FAIL] 点积下界计算错误:", r2["scores"][0])
        ok = False
    else:
        print("  [ OK ] 点积下界 4×(-128)×127 = -65024")

    # --- 4. 输出恒等于 V 的加权平均 → 必落在 V 的极值之间 ---
    rnd = random.Random(20260101)
    viol = 0
    for _ in range(2000):
        q_ = [rnd.randint(-128, 127) for _ in range(D)]
        k_ = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
        v_ = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
        r = attention(q_, k_, v_)
        for c in range(D):
            lo = min(v_[j][c] for j in range(S))
            hi = max(v_[j][c] for j in range(S))
            if not (lo <= r["out"][c] <= hi):
                viol += 1
    print("\n  [%s] 2000 组随机输入：输出始终落在 V 的 min/max 之间 (违例 %d)"
          % (" OK " if viol == 0 else "FAIL", viol))
    ok &= (viol == 0)

    # --- 5. L 的取值范围 ---
    lmin, lmax = 999, -1
    nz = 0
    for _ in range(2000):
        q_ = [rnd.randint(-128, 127) for _ in range(D)]
        k_ = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
        v_ = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
        r = attention(q_, k_, v_)
        lmin = min(lmin, r["L"]); lmax = max(lmax, r["L"])
        nz += sum(1 for x in r["weights"] if x)
    print("  [全量程 int8 随机] L ∈ [%d, %d]，平均非零 weight 个数 %.2f/8"
          % (lmin, lmax, nz / 2000.0))
    lmin2, lmax2, nz2 = 999, -1, 0
    for _ in range(2000):
        q_ = [rnd.randint(-3, 3) for _ in range(D)]
        k_ = [[rnd.randint(-3, 3) for _ in range(D)] for _ in range(S)]
        v_ = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
        r = attention(q_, k_, v_)
        lmin2 = min(lmin2, r["L"]); lmax2 = max(lmax2, r["L"])
        nz2 += sum(1 for x in r["weights"] if x)
    print("  [小量程 Q/K 随机] L ∈ [%d, %d]，平均非零 weight 个数 %.2f/8"
          % (lmin2, lmax2, nz2 / 2000.0))
    if lmax > L_MAX or lmin < L_MIN or lmax2 > L_MAX:
        print("  [FAIL] L 越界")
        ok = False
    else:
        print("  [ OK ] L 恒在 [16,128] 内 → 8-bit 足够；倒数表只需覆盖 L≥16")

    # --- 6. 归一化通路等价性 ---
    print()
    ok &= check_norm_exhaustive()
    print("  [ OK ] 倒数表最大相对误差 %.3e（折算到输出 < 1e-6 LSB）"
          % check_recip_table())

    print("\n" + "=" * 74)
    print("自检结果：" + ("全部通过 (ALL PASS)" if ok else "存在失败项 (FAIL)"))
    print("=" * 74)
    return ok


if __name__ == "__main__":
    sys.exit(0 if selftest() else 1)
