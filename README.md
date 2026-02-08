# 可学性边界实验：Epiplexity 相变点探索

验证论文《From Entropy to Epiplexity》的核心预测：找到神经网络能学会的伪随机复杂度临界点。

**核心问题**：Epiplexity 从 0 到非零的相变，在什么复杂度发生？

**关联文章**：微信公众号 No.62《信息量取决于谁在看》

---

## 实验设计

从简单到复杂，6 种序列生成器：

| 序列类型 | 生成规则 | 预期 |
|---------|---------|------|
| periodic | `(i % 7) > 3` → 0/1 | ✅ Grokking |
| lcg_simple | `x = (3*x + 7) % 256` | 边界 |
| lfsr_5 | `x = x[n-1] XOR x[n-5]` | 边界 |
| lcg_glibc | `x = (1103515245*x + 12345) % 2^31` | 边界 |
| lfsr_31 | 31位LFSR | 可能不可学 |
| csprng | `secrets.randbits` | ❌ 不可学 |

---

## 快速开始

```bash
# 进入 Docker 容器
sudo docker exec -it magical_bhabha bash

# 运行单个实验
cd /workspace/ai-theorys-study/arxiv/wechat62/code/
python train_learnability.py --seq_type periodic

# 运行全部实验 + 分析
bash run_all.sh

# 单独运行分析
python plot_comparison.py      # 学习曲线对比
python estimate_dimension.py   # 维度分析（找维度骤降）
python analyze_dynamics.py     # 动态分析（Grokking检测、震荡检测）
```

---

## 目录结构

```
.
├── README.md
├── code/
│   ├── train_learnability.py   # 主训练脚本（6 种序列，0.3M 模型）
│   ├── train_large_model.py    # 4x 模型（8M）
│   ├── train_xlarge_model.py   # 10x 模型（33M）
│   ├── train_xlarge_lowwd.py   # 10x + 低 weight decay (0.01)
│   ├── plot_comparison.py      # 对比图生成
│   ├── estimate_dimension.py   # 维度分析
│   ├── analyze_dynamics.py     # 动态分析
│   ├── analyze_attention.py    # Attention分析（备用）
│   └── run_all.sh              # 批量运行
├── paper/
│   ├── learnability-boundary.md    # 中文论文
│   ├── learnability-boundary-en.md # 英文论文
│   └── zenodo-info.md              # Zenodo 填表信息
└── results/
    ├── periodic/               # 周期序列结果
    ├── lcg_simple/
    ├── lfsr_5/
    ├── lcg_glibc/
    ├── lfsr_31/
    ├── csprng/
    ├── lcg_glibc_large/        # 4x 模型
    ├── lfsr_31_large/
    ├── lcg_glibc_xlarge/       # 10x 模型
    ├── lfsr_31_xlarge/
    ├── lcg_glibc_xlarge_lowwd/ # 10x + 低 wd
    ├── lfsr_31_xlarge_lowwd/
    ├── learning_curves.png
    ├── final_comparison.png
    ├── dimension_trajectory.png
    ├── dynamics_analysis.png
    └── summary.md
```

---

## 核心结论

1. **相变边界在 256 → 2³¹ 之间**：状态空间 ≤256 可学，2³¹ 不可学
2. **部分 Grokking 存在**：LFSR-31 学会了 7 bit（50% 准确率），剩下 1 bit 学不会
3. **扩大模型无法突破边界**：4x/10x 模型对 lcg_glibc 无效，10x 反而把 LFSR-31 的 7 bit 也弄丢
4. **不是 weight decay 的锅**：降低 50 倍 weight decay，结果一模一样
5. **可学性边界是任务-模型的联合性质**：不只取决于任务复杂度，模型规模方向反直觉——更大反而更差

---

## 已观察到的现象

1. **维度骤降** ✅：LFSR-31 的表示空间坍缩到 2-4 维，对应"学会 7 bit"
2. **部分 Grokking** ✅：LFSR-31 学会了 7 bit 规律（50%），剩下 1 bit 学不会
3. **反直觉的规模效应** ✅：大模型反而更差——10x 模型丢掉了小模型学会的 7 bit
4. **正则化不是原因** ✅：降低 weight decay 50 倍无效，排除了正则化假说

---

## 实验结果

### 第一阶段：小模型 (0.3M 参数)

| 序列类型 | 测试准确率 | 随机基线 | 结论 |
|---------|-----------|---------|------|
| periodic | 100% | 50% | ✅ 可学 |
| lcg_simple | 100% | 0.4% | ✅ 可学 |
| lfsr_5 | 100% | 50% | ✅ 可学 |
| lcg_glibc | 0.4% | 0.4% | ❌ 不可学 |
| lfsr_31 | 50% | 0.4% | ⚠️ 部分可学（7/8 bit） |
| csprng | 0.3% | 0.4% | ❌ 不可学 |

**核心发现**：
- 相变边界在 **256 → 2³¹** 之间
- lfsr_31 的 50% = 部分 Grokking，学会了 7 bit，剩 1 bit 学不会
- lcg_glibc 训练 100% + 测试 0% = 纯记忆解

### 第二阶段：大模型 (8M 参数，4x 扩大)

| 序列 | 小模型 train/test | 大模型 train/test | 变化 |
|-----|------------------|------------------|------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | 更差（连记忆都失败） |
| lfsr_31 | 100% / 50% | 100% / 50% | 无变化 |

### 第三阶段：超大模型 (33M 参数，10x 扩大)

| 序列 | 小模型 (0.3M) | 4x (8M) | 10x (33M) |
|-----|--------------|---------|-----------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | 0.8% / 0.5% |
| lfsr_31 | 100% / 50% | 100% / 50% | **0.8% / 0.5%** |

初始假设：10x 模型的崩溃是 weight decay 太强（0.5）导致的——"学会了又忘了"。

### 第四阶段：10x 模型 + 低 weight decay (0.01)

| 序列 | 小模型 (0.3M) | 4x (8M) | 10x (33M) | 10x + 低wd (0.01) |
|-----|--------------|---------|-----------|-------------------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | 0.8% / 0.5% | 0.8% / 0.5% |
| lfsr_31 | 100% / 50% | 100% / 50% | 0.8% / 0.5% | **0.8% / 0.5%** |

**假设被否定**：weight decay 从 0.5 降到 0.01（50 倍），结果一模一样。崩溃不是正则化太强，而是大模型在这类任务上的优化地形本身不稳定。

**结论**：
1. **lcg_glibc**：始终学不会，模型大小和正则化强度均无关
2. **lfsr_31**：10x 模型无论 weight decay 多少，都无法保住 50% 的部分 Grokking
3. **小模型是最优配置**：大模型参数空间太大，脆弱的部分学习结构无法稳定存在
4. **可学性边界不只取决于任务复杂度，还取决于模型规模**——方向反直觉：更大的模型反而更容易丢掉脆弱的部分学习

> **"不是模型太小，是规律太复杂。更大的模型不但没帮上忙，还把学会的 7 bit 也弄丢了。"**

---

## 参考

- Epiplexity 论文：https://arxiv.org/abs/2601.03220
- Grokking 流形发现实验：`../wechat67/`
- 本实验 Zenodo：https://zenodo.org/records/18524191
