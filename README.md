## 1. 工具版本

| 工具 | 版本 | 路径 / 用途 |
|---|---|---|
| Icarus Verilog | **12.0** (devel) | `D:\iverilog`，主仿真器（批量回归） |
| ModelSim | **SE-64 10.4** | `D:\modelsim\APP\win64`，交叉验证 + 看波形 |
| GTKWave | 3.x | `D:\iverilog\gtkwave\bin\gtkwave.exe`，查看 VCD 波形 |
| Python | 3.8+ | 运行 `model/*.py`（只依赖标准库，无需第三方包） |
| VS Code | — | 编辑代码 + 运行 Python 参考模型 |

---

## 2. 文件结构

```
README.md                       本文件

rtl/                            可综合 RTL（3 个文件）
├── attention_ip.v        19.6 KB   顶层：状态机 + 存储体 + 数据通路（本次实现的主体）
├── exp_lut.v              1.0 KB   指数查找表（16 × 8 bit）
└── recip_lut.v           11.1 KB   倒数查找表（256 × 33 bit）

tb/                             testbench（4 个文件）
├── tb_attention_ip.v     17.8 KB   正式自检：读向量、驱动 DUT、六级中间结果逐位比对
├── tb_causal.v           10.7 KB   加分项 A 独立测试：因果 Mask（16 用例）
├── tb_param.v            11.7 KB   加分项 C 独立测试：参数化 S/d（5 种规模，10 用例）
└── tb_simple.v            6.8 KB   最简版：数值写死在文件里，改完直接跑看波形

model/                          Python 参考模型（2 个文件）
├── golden_model.py       13.3 KB   Golden Model：含自检、题目手算示例、归一化穷举证明
└── attention_calc.py      5.6 KB   计算器：改文件顶部的 Q/K/V，按 F5 直接看结果

scripts/                        仿真脚本（1 个文件）
└── gen_vectors.py        10.5 KB   生成测试向量（directed / random / corner / one / all）

sim/vectors/                    测试向量（5 个文件）
├── all.vec               77.7 KB   227 例合集 —— 主回归直接用这个
├── random.vec            68.9 KB   200 个随机用例
├── directed.vec           4.5 KB   14 个定向用例
├── corner.vec             4.3 KB   13 个边界用例
└── one.vec                0.3 KB   题目手算示例（单用例）

waves/                          波形与插图（11 个文件）
├── attention_ip.vcd      18.7 KB   ★ 仿真波形（145 个信号，GTKWave 可直接打开）
├── simple.vcd            48.2 KB   ★ tb_simple 的仿真波形（558 个信号）
├── wave_io.png           32.3 KB   图 3  接口时序
├── wave_overview.png     52.6 KB   图 4  整任务数据流
├── wave_dot.png          33.0 KB   图 5  点积细节
├── wave_norm.png         35.8 KB   图 6  归一化细节
├── arch_block.png       258.8 KB   图 1  整体架构与数据流
├── fsm.png               98.1 KB   图 2  状态转移图
├── datapath_bits.png    145.1 KB   图 7  位宽推导与数据表示
├── datapath.png         198.5 KB   数据通路模块图（仿 Patterson & Hennessy 画风）
└── sim_summary.html       1.2 KB   回归汇总（表格 + HTML）

docs/                           交付文档（3 个文件）
├── Attention加速器IP设计报告.docx   834.2 KB   设计报告（7 页）
├── Attention加速器IP设计报告.pdf    889.0 KB   同上 PDF 版
└── Attention加速器IP汇报.pptx       647.7 KB   汇报 PPT（6 页，含逐页讲稿备注）
```


## 3、验证

| 层次 | 手段 | 结果 |
|---|---|---|
| 算法 | Python Golden Model 自检（含题目手算示例 O = [2,5]） | ✅ 通过 |
| 数值 | 穷举证明归一化通路与精确四舍五入逐位一致 | ✅ 通过 |
| 查表 | 解析 Verilog 源码与 `math.exp` / 精确倒数比对两张 LUT | ✅ 256 项全对 |
| 模块 | 自检 testbench 逐级比对 x_max / Score / Weight / L / ACC / O | ✅ 逐位一致 |
| 协议 | `done` 恰好 1 拍、`out_valid` 恰好 4 拍、`data_ready` 时序 | ✅ 每条用例都检查 |
| 随机 | 200 组随机（全量程 / 小量程 / 单位值 / 稀疏 / 双小量程五种分布） | ✅ 通过 |
| 边界 | 点积上/下界、V 极值、L 最大、舍入 .5 边界、并列最大 | ✅ 通过 |
| 交叉 | ModelSim 编译 + 仿真（4 个 testbench） | ✅ 0 error / 0 warning，ALL PASSED |

**期望值来自独立的 Python 参考模型，不是 DUT 自己的行为** —— 否则测试永远会「通过」。