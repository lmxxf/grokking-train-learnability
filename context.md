# 上下文压缩：可学性边界实验

**日期**：2026-02-07
**项目**：arxiv/wechat62 (submodule of ai-theorys-study)
**GitHub**：git@github.com:lmxxf/grokking-train-learnability.git
**Zenodo**：https://doi.org/10.5281/zenodo.18524191

---

## 实验目标

验证 Epiplexity 论文（CMU+NYU, arXiv:2601.03220）的核心预测：找到神经网络可学性的边界——Epiplexity 从非零到零的相变点。

复用 wechat67 的 Grokking 实验架构（2 层 Transformer, 128 dim），改为序列预测任务。

---

## 实验结果

### 第一阶段：6 种序列 × 小模型 (0.3M)

| 序列 | 状态空间 | test acc | 结论 |
|-----|---------|----------|------|
| periodic | 7 | 100% | 可学 |
| lcg_simple | 256 | 100% | 可学 |
| lfsr_5 | 32 | 100% | 可学 |
| lcg_glibc | 2³¹ | 0.4% | 不可学（train 100% = 纯记忆） |
| lfsr_31 | 2³¹ | **50%** | **部分 Grokking（学会 1 bit）** |
| csprng | ∞ | 0.3% | 不可学（连记忆都失败） |

**核心发现**：相变边界在 256 → 2³¹ 之间。LFSR-31 的 50% = 部分 Grokking。

### 维度分析

- periodic/lcg_simple/lfsr_5：128 维（MLE 估计器在简单任务失效）
- lcg_glibc/csprng：10-14 维
- **lfsr_31：2-4 维**（极低，抓住了 1 bit 结构）

### 第二阶段：模型扩大 4x (8M)

| 序列 | 小模型 | 4x |
|-----|-------|-----|
| lcg_glibc | 100%/0.4% | 0.8%/0.5%（更差，欠拟合）|
| lfsr_31 | 100%/50% | 100%/50%（无变化）|

### 第三阶段：模型扩大 10x (33M)

| 序列 | 小模型 | 4x | 10x |
|-----|-------|-----|------|
| lcg_glibc | 100%/0.4% | 0.8%/0.5% | 0.8%/0.5% |
| lfsr_31 | 100%/50% | 100%/50% | **0.8%/0.5%（崩了）** |

**关键发现**：10x 模型的 lfsr_31 在 step 500 时还有 50%，到 step 100000 崩到 0.5%。不是训练不够，是 weight decay 太强把学到的东西"衰减"掉了——"学会了又忘了"。

### 第四阶段（进行中）：10x + 低 weight decay (0.01)

`train_xlarge_lowwd.py` 正在跑。验证假设：降低 weight decay 能否让 10x 模型保住 50%。

---

## 文件结构

```
arxiv/wechat62/
├── code/
│   ├── train_learnability.py      # 主训练脚本（6 种序列）
│   ├── train_large_model.py       # 4x 模型
│   ├── train_xlarge_model.py      # 10x 模型
│   ├── train_xlarge_lowwd.py      # 10x + 低 weight decay（进行中）
│   ├── estimate_dimension.py      # 维度分析
│   ├── analyze_dynamics.py        # Grokking/震荡检测
│   ├── plot_comparison.py         # 对比图
│   └── run_all.sh
├── paper/
│   ├── learnability-boundary.md   # 中文论文
│   ├── learnability-boundary-en.md # 英文论文
│   ├── learnability-boundary-en.pdf
│   └── zenodo-info.md             # Zenodo 填表信息
├── results/                       # 所有实验结果
└── README.md
```

---

## 公众号

No.80 已写好：`wechat/80.txt`，封面图 `wechat/cover80.svg/png`

---

## 待做

- [ ] 等 train_xlarge_lowwd.py 跑完，更新结论
- [ ] 如果低 wd 有效：更新论文 + 公众号 + push
- [ ] 赵磊拿去投 NeurIPS（他一作，他自己折腾 p-value）
