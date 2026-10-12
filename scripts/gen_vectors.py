#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试向量生成器
==============
用 Golden Model 生成 directed / random / corner 三类向量，供自检 testbench 使用。

向量文件格式（token 流，TB 用 $fscanf 顺序读取）：
    CASE   <序号> <标签>
    Q      <4 个 int8>
    K      <32 个 int8>      按 K[j][k] 展开，j=0..7, k=0..3
    V      <32 个 int8>
    XMAX   <1 个整数>
    SCORE  <8 个整数>        18-bit 有符号
    WEIGHT <8 个整数>        UQ1.4 的整数表示（真值×16）
    L      <1 个整数>
    ACC    <4 个整数>
    OUT    <4 个 int8>
    END

运行：
    python scripts/gen_vectors.py                 # 默认 200 组随机
    python scripts/gen_vectors.py --random 500 --seed 123
"""

import argparse
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "model"))

from golden_model import attention, S, D          # noqa: E402

OUTDIR = os.path.join(ROOT, "sim", "vectors")


# ---------------------------------------------------------------------------
# 定向用例
# ---------------------------------------------------------------------------

def build_directed():
    """返回 [(tag, q, k, v), ...]，k/v 为 8×4 的二维表。"""
    C = []

    def add(tag, q, k, v):
        assert len(q) == D and len(k) == S and len(v) == S
        C.append((tag, list(q), [list(r) for r in k], [list(r) for r in v]))

    # 1) 题目第三节手算示例，嵌入到 S=8/d=4：
    #    Q=[1,2]，K0=[1,0]，K1=[0,1]，V0=[4,2]，V1=[1,6]
    #    其余 6 个 Token 用 K=[-8,0,0,0] 让 score=-8（diff=-10 → weight=0），
    #    这样 L 与 ACC 与题目的 2-Token 手算完全一致，O 应为 [2,5]。
    add("pdf_example",
        [1, 2, 0, 0],
        [[1, 0, 0, 0], [0, 1, 0, 0]] + [[-8, 0, 0, 0]] * 6,
        [[4, 2, 0, 0], [1, 6, 0, 0]] + [[0, 0, 0, 0]] * 6)

    # 2) 全 0
    add("all_zero", [0] * D, [[0] * D] * S, [[0] * D] * S)

    # 3) 点积上界：4×(-128)×(-128) = +65536（验证 Score 需要 18 bit）
    add("dot_max", [-128] * D, [[-128] * D] * S, [[1, -1, 2, -2]] * S)

    # 4) 点积下界：4×(-128)×127 = -65024
    add("dot_min", [-128] * D, [[127] * D] * S, [[1, -1, 2, -2]] * S)

    # 5) 输出上界：V 全 127，唯一最大权重的 Token 胜出
    add("v_max", [1, 0, 0, 0], [[4, 0, 0, 0]] + [[0, 0, 0, 0]] * 7,
        [[127] * D] * S)

    # 6) 输出下界：V 全 -128
    add("v_min", [1, 0, 0, 0], [[4, 0, 0, 0]] + [[0, 0, 0, 0]] * 7,
        [[-128] * D] * S)

    # 7) 8 个 Token 分数完全相同 → 权重全 16，L = 128
    add("all_equal", [1, 1, 1, 1], [[1, 1, 1, 1]] * S,
        [[1, 2, 3, 4], [-1, -2, -3, -4], [5, 6, 7, 8], [-5, -6, -7, -8],
         [9, 10, 11, 12], [13, 14, 15, 16], [-9, -10, -11, -12], [100, 50, 25, 12]])

    # 8) 逐级遍历指数表：diff = 0,-1,-2,-3,-4,-8,-15,-16
    #    Q=[1,0,0,0]，K_j[0] = -diff_j
    diffs = [0, -1, -2, -3, -4, -8, -15, -16]
    add("exp_walk", [1, 0, 0, 0],
        [[-d, 0, 0, 0] for d in diffs],
        [[16, 32, 48, 64], [16, 32, 48, 64], [16, 32, 48, 64], [16, 32, 48, 64],
         [16, 32, 48, 64], [16, 32, 48, 64], [16, 32, 48, 64], [16, 32, 48, 64]])

    # 9) 四舍五入边界（正）：L=32，ACC[0]=48 → 48/32 = 1.5 → 应为 2
    add("round_tie_pos", [1, 0, 0, 0],
        [[1, 0, 0, 0], [1, 0, 0, 0]] + [[-8, 0, 0, 0]] * 6,
        [[1, 0, 0, 0], [2, 0, 0, 0]] + [[0, 0, 0, 0]] * 6)

    # 10) 四舍五入边界（负）：L=32，ACC[0]=-48 → -1.5 → 应为 -2
    add("round_tie_neg", [1, 0, 0, 0],
        [[1, 0, 0, 0], [1, 0, 0, 0]] + [[-8, 0, 0, 0]] * 6,
        [[-1, 0, 0, 0], [-2, 0, 0, 0]] + [[0, 0, 0, 0]] * 6)

    # 11) 大负 diff（远超表范围）→ 权重必须为 0，且不能被低 4 位地址误命中
    add("huge_neg_diff", [127, 127, 127, 127], [[127] * D] + [[-128] * D] * 7,
        [[127] * D] + [[-128] * D] * 7)

    # 12) 并列最大分数（多个 Token 同为 x_max）
    add("tie_max", [1, 1, 0, 0],
        [[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0],
         [0, 0, 0, 0], [0, 0, 0, 0], [-4, -4, 0, 0], [-4, -4, 0, 0]],
        [[10, 20, 30, 40], [20, 30, 40, 50], [30, 40, 50, 60],
         [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16],
         [17, 18, 19, 20]])

    # 13) 负分值主导：max 为负数
    add("negative_scores", [-5, -5, -5, -5], [[3, 3, 3, 3]] * S,
        [[-1, -2, -3, -4], [-5, -6, -7, -8], [1, 2, 3, 4], [5, 6, 7, 8],
         [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]])

    # 14) 单个 Token 获胜，权重全部落在 0/6/16 三档
    add("mixed_weight", [2, -1, 3, 0],
        [[3, -2, 1, 0], [2, -1, 3, 0], [1, 0, 2, -1], [0, 0, 0, 0],
         [-1, 1, -1, 1], [-2, 2, -2, 2], [-3, 3, -3, 3], [-4, 4, -4, 4]],
        [[11, -22, 33, -44], [-55, 66, -77, 88], [99, -100, 101, -102],
         [1, 2, 3, 4], [-1, -2, -3, -4], [127, -128, 127, -128],
         [0, 0, 0, 0], [64, -64, 32, -32]])

    return C


def build_corner():
    """边界 / 极端组合，全部手工构造。"""
    C = []

    def add(tag, q, k, v):
        C.append((tag, list(q), [list(r) for r in k], [list(r) for r in v]))

    # 极值随机混合
    ex = [-128, 127, 0, -1, 1]
    rnd = random.Random(777)
    for n in range(6):
        q = [rnd.choice(ex) for _ in range(D)]
        k = [[rnd.choice(ex) for _ in range(D)] for _ in range(S)]
        v = [[rnd.choice(ex) for _ in range(D)] for _ in range(S)]
        add("corner_extreme_%d" % n, q, k, v)

    # 只有一个非零元素
    add("corner_sparse1", [1, 0, 0, 0], [[127, 0, 0, 0]] + [[0] * D] * 7,
        [[-128, 0, 0, 0]] + [[0] * D] * 7)
    # Q 有负值，K 有负值，点积抵消为 0
    add("corner_cancel", [1, -1, 1, -1], [[1, -1, 1, -1]] * S,
        [[1, 1, 1, 1]] * S)
    # 最大序号 Token 获胜（检查地址生成）
    add("corner_last_tok", [1, 0, 0, 0], [[0] * D] * 7 + [[5, 0, 0, 0]],
        [[0] * D] * 7 + [[-128, 127, -128, 127]])
    # 第 0 个 Token 获胜
    add("corner_first_tok", [1, 0, 0, 0], [[5, 0, 0, 0]] + [[0] * D] * 7,
        [[-128, 127, -128, 127]] + [[0] * D] * 7)
    # 权重和为最大 128 且 ACC 为极值
    add("corner_full_L", [1, 1, 1, 1], [[1, 1, 1, 1]] * S, [[127] * D] * S)
    add("corner_full_L_neg", [1, 1, 1, 1], [[1, 1, 1, 1]] * S, [[-128] * D] * S)
    # L 为奇数（不能整除），检查舍入
    add("corner_odd_L", [1, 0, 0, 0], [[0, 0, 0, 0], [-1, 0, 0, 0]] + [[-8, 0, 0, 0]] * 6,
        [[3, 0, 0, 0], [3, 0, 0, 0]] + [[0] * D] * 6)
    return C


def build_random(n, seed):
    """按多种分布生成随机用例，覆盖"权重退化"和"权重丰富"两种情况。"""
    rnd = random.Random(seed)
    C = []
    for i in range(n):
        m = i % 5
        if m == 0:                                   # 全量程 int8
            q = [rnd.randint(-128, 127) for _ in range(D)]
            k = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
            v = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
            tag = "rand_full"
        elif m == 1:                                 # 小量程 Q/K，权重丰富
            q = [rnd.randint(-4, 4) for _ in range(D)]
            k = [[rnd.randint(-4, 4) for _ in range(D)] for _ in range(S)]
            v = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
            tag = "rand_small"
        elif m == 2:                                 # 只有 {0,±1}，分数接近
            q = [rnd.choice([-1, 0, 1]) for _ in range(D)]
            k = [[rnd.choice([-1, 0, 1]) for _ in range(D)] for _ in range(S)]
            v = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
            tag = "rand_unit"
        elif m == 3:                                 # 小量程 V，检查累加精度
            q = [rnd.randint(-8, 7) for _ in range(D)]
            k = [[rnd.randint(-8, 7) for _ in range(D)] for _ in range(S)]
            v = [[rnd.randint(-16, 15) for _ in range(D)] for _ in range(S)]
            tag = "rand_both_small"
        else:                                        # 稀疏 / 带大量重复行
            q = [rnd.randint(-128, 127) for _ in range(D)]
            k = [[rnd.randint(-2, 2) for _ in range(D)] for _ in range(S)]
            v = [[rnd.randint(-128, 127) for _ in range(D)] for _ in range(S)]
            tag = "rand_sparse"
        C.append((tag, q, k, v))
    return C


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_vec(path, cases):
    n_out = 0
    with open(path, "w", encoding="ascii", newline="\n") as f:
        for idx, (tag, q, k, v) in enumerate(cases, start=1):
            r = attention(q, k, v)
            f.write("CASE %d %s\n" % (idx, tag))
            f.write("Q " + " ".join("%d" % x for x in q) + "\n")
            f.write("K " + " ".join("%d" % x for row in k for x in row) + "\n")
            f.write("V " + " ".join("%d" % x for row in v for x in row) + "\n")
            f.write("XMAX %d\n" % r["xmax"])
            f.write("SCORE " + " ".join("%d" % x for x in r["scores"]) + "\n")
            f.write("WEIGHT " + " ".join("%d" % x for x in r["weights"]) + "\n")
            f.write("L %d\n" % r["L"])
            f.write("ACC " + " ".join("%d" % x for x in r["acc"]) + "\n")
            f.write("OUT " + " ".join("%d" % x for x in r["out"]) + "\n")
            f.write("END\n")
            n_out += 1
    return n_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=200, help="随机用例个数")
    ap.add_argument("--seed", type=int, default=20260101)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    directed = build_directed()
    corner = build_corner()
    rand = build_random(args.random, args.seed)

    n1 = write_vec(os.path.join(args.outdir, "directed.vec"), directed)
    n2 = write_vec(os.path.join(args.outdir, "random.vec"), rand)
    n3 = write_vec(os.path.join(args.outdir, "corner.vec"), corner)
    n4 = write_vec(os.path.join(args.outdir, "all.vec"), directed + rand + corner)
    n5 = write_vec(os.path.join(args.outdir, "one.vec"), directed[:1])

    print("已生成测试向量 -> %s" % os.path.abspath(args.outdir))
    print("  directed.vec : %4d 例" % n1)
    print("  random.vec   : %4d 例" % n2)
    print("  corner.vec   : %4d 例" % n3)
    print("  all.vec      : %4d 例（合计）" % n4)
    print("  one.vec      : %4d 例（题目手算示例，用于出波形）" % n5)


if __name__ == "__main__":
    main()
