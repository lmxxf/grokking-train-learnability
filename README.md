# Learnability Boundary Experiment: Exploring the Epiplexity Phase Transition
# 可学性边界实验：Epiplexity 相变点探索

Validating the core prediction of the paper "From Entropy to Epiplexity": finding the critical point of pseudorandom complexity that a neural network can learn.
验证论文《From Entropy to Epiplexity》的核心预测：找到神经网络能学会的伪随机复杂度临界点。

**Core Question / 核心问题**: At what complexity does the phase transition of Epiplexity from 0 to non-zero occur?
**核心问题**：Epiplexity 从 0 到非零的相变，在什么复杂度发生？

**Related Article / 关联文章**: WeChat Official Account No.62 "Information Depends on Who's Looking"
**关联文章**：微信公众号 No.62《信息量取决于谁在看》

---

## Experiment Design / 实验设计

6 sequence generators, from simple to complex:
从简单到复杂，6 种序列生成器：

| Sequence Type / 序列类型 | Generation Rule / 生成规则 | Expected Outcome / 预期 |
|---------|---------|------|
| periodic | `(i % 7) > 3` → 0/1 | ✅ Grokking |
| lcg_simple | `x = (3*x + 7) % 256` | Boundary / 边界 |
| lfsr_5 | `x = x[n-1] XOR x[n-5]` | Boundary / 边界 |
| lcg_glibc | `x = (1103515245*x + 12345) % 2^31` | Boundary / 边界 |
| lfsr_31 | 31-bit LFSR / 31位LFSR | Possibly unlearnable / 可能不可学 |
| csprng | `secrets.randbits` | ❌ Unlearnable / 不可学 |

---

## Runtime Environment / 运行环境

Requires NVIDIA GPU + Docker.
需要 NVIDIA GPU + Docker。

### 1. Pull Image / 拉取镜像

```bash
sudo docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

### 2. Create Container / 创建容器

```bash
# 把代码目录挂载进去（按实际路径修改）
sudo docker run -d --gpus all \
  -v /path/to/ai-theorys-study:/workspace/ai-theorys-study \
  --name my_container \
  nvcr.io/nvidia/pytorch:25.11-py3 \
  sleep infinity
```

### 3. Enter Container / 进入容器

```bash
sudo docker exec -it my_container bash
cd /workspace/ai-theorys-study/arxiv/wechat62
```

The image comes with PyTorch + CUDA pre-installed; no additional dependencies needed.
镜像自带 PyTorch + CUDA，无需额外安装依赖。

## Quick Start / 快速开始

```bash
# 运行单个实验
python code/train_learnability.py --seq_type periodic

# 运行全部 6 种序列实验 + 分析
cd code && bash run_all.sh

# 单独运行分析
python code/plot_comparison.py      # 学习曲线对比
python code/estimate_dimension.py   # 维度分析（找维度骤降）
python code/analyze_dynamics.py     # 动态分析（Grokking检测、震荡检测）

# 第五阶段：lfsr_31 上下文窗口扩展
python code/train_lfsr31_ctx32.py --output_dir results   # context_len=32
python code/train_lfsr31_ctx64.py --output_dir results   # context_len=64

# 第六阶段：lcg_glibc 数据量扩展
python code/train_lcg_200k.py --output_dir results       # seq_length=200000
```

---

## Directory Structure / 目录结构

```
.
├── README.md
├── code/
│   ├── train_learnability.py   # 主训练脚本（6 种序列，0.3M 模型）
│   ├── train_large_model.py    # 4x 模型（8M）
│   ├── train_xlarge_model.py   # 10x 模型（33M）
│   ├── train_xlarge_lowwd.py   # 10x + 低 weight decay (0.01)
│   ├── train_lfsr31_ctx32.py   # lfsr_31 上下文窗口扩展（context_len=32）
│   ├── train_lfsr31_ctx64.py   # lfsr_31 上下文窗口扩展（context_len=64）
│   ├── train_lcg_200k.py       # lcg_glibc 数据量扩展（seq_length=200000）
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
    ├── lfsr_31_ctx32/              # 上下文窗口扩展（context_len=32）
    ├── lfsr_31_ctx64/              # 上下文窗口扩展（context_len=64）
    ├── lcg_glibc_200k/            # 数据量扩展（seq_length=200000）
    ├── learning_curves.png
    ├── final_comparison.png
    ├── dimension_trajectory.png
    ├── dynamics_analysis.png
    └── summary.md
```

---

## Core Conclusions / 核心结论

1. **Phase transition boundary lies between 256 and 2^31**: State space <=256 is learnable, 2^31 is not
   **相变边界在 256 → 2³¹ 之间**：状态空间 ≤256 可学，2³¹ 不可学
2. **Partial Grokking exists**: LFSR-31 learned 7 bits (50% accuracy), but cannot learn the remaining 1 bit
   **部分 Grokking 存在**：LFSR-31 学会了 7 bit（50% 准确率），剩下 1 bit 学不会
3. **With the same 2^31 state space, LFSR is learnable but LCG is not**: State space size is not the sole determining factor; the internal geometric structure of the algorithm is equally critical
   **同样 2³¹，LFSR 能学而 LCG 不能**：状态空间大小不是唯一决定因素，算法的内部几何结构同样关键
   - LFSR (XOR) = axis-aligned: bits are relatively independent, arranged on quasi-orthogonal hyperplanes, Transformer can capture bit by bit
   - LFSR（XOR）= 轴向对齐：bit 之间相对独立，排列在准正交超平面上，Transformer 可逐 bit 捕获
   - LCG (modular multiplication) = topological shattering: multiplication makes every bit depend on all lower bits, the manifold is crushed, gradients cannot find a "handle"
   - LCG（乘法取模）= 拓扑粉碎：乘法让每一位都依赖所有低位，流形被揉碎，梯度找不到"把手"
   - **The physical threshold for Grokking = whether the local smoothness of the manifold exceeds the model's sampling frequency**
   - **Grokking 的物理门槛 = 流形的局部平滑度是否超过模型的采样频率**
4. **Scaling up the model cannot break the boundary**: 4x/10x models have no effect on lcg_glibc; 10x actually loses the 7 bits that the small model learned for LFSR-31
   **扩大模型无法突破边界**：4x/10x 模型对 lcg_glibc 无效，10x 反而把 LFSR-31 的 7 bit 也弄丢
5. **Weight decay is not to blame**: Reducing weight decay by 50x yields identical results
   **不是 weight decay 的锅**：降低 50 倍 weight decay，结果一模一样
6. **The learnability boundary is a joint property of task and model**: It depends not only on task complexity; model scale has a counterintuitive effect -- bigger is actually worse
   **可学性边界是任务-模型的联合性质**：不只取决于任务复杂度，模型规模方向反直觉——更大反而更差
7. **Insufficient information is the true cause of LFSR-31's 50% ceiling**: After expanding context_len from 16 to 32/64, test accuracy went from 50% to 99.8% to 100% -- the bottleneck is not model capacity, but input information. The pattern redundancy from longer windows further eliminates residual errors
   **信息不足是 LFSR-31 的 50% 天花板的真正原因**：context_len 从 16 扩到 32/64 后，测试准确率从 50% → 99.8% → 100%——瓶颈不是模型能力，是输入信息量。更长窗口提供的模式冗余进一步消除残余误差
8. **Topological shattering is irredeemable**: After expanding lcg_glibc data by 20x (10k to 200k), there is still zero generalization (test 0.3%), and training accuracy drops from 100% to 22% -- with more data it cannot even memorize everything; if the manifold doesn't exist, data volume is meaningless
   **拓扑粉碎不可救**：lcg_glibc 数据量扩大 20 倍（10k→200k）后依然零泛化（test 0.3%），且训练集准确率从 100% 降到 22%——数据多了连记忆都记不完，流形不存在则数据量无意义
9. **A challenge to Scaling Laws**: Current Scaling Laws (Chinchilla, etc.) assume performance = f(parameters, data, compute), with an implicit premise that a learnable manifold exists behind the data. This experiment reveals three blind spots: (a) When the manifold doesn't exist, all three axes fail simultaneously (LCG: 20x data + 100x parameters = 0% generalization); (b) Information density is a fourth axis independent of data volume (LFSR: 2x window > 100x parameters); (c) There exist inverse scaling intervals (10x model loses the 7 bits that the small model learned)
   **对 Scaling Law 的挑战**：当前 Scaling Law（Chinchilla 等）假设性能 = f(参数, 数据, 算力)，隐含前提是数据背后存在可学的流形。本实验揭示了三个盲区：（a）流形不存在时三个轴同时失效（LCG：20x 数据 + 100x 参数 = 0% 泛化）；（b）信息密度是独立于数据量的第四个轴（LFSR：2x 窗口 > 100x 参数）；（c）存在反向 scaling 区间（10x 模型反而丢掉小模型学会的 7 bit）

---

## Observed Phenomena / 已观察到的现象

1. **Dimension collapse** ✅: LFSR-31's representation space collapses to 2-4 dimensions, corresponding to "learned 7 bits"
   **维度骤降** ✅：LFSR-31 的表示空间坍缩到 2-4 维，对应"学会 7 bit"
2. **Partial Grokking** ✅: LFSR-31 learned the 7-bit pattern (50%), cannot learn the remaining 1 bit
   **部分 Grokking** ✅：LFSR-31 学会了 7 bit 规律（50%），剩下 1 bit 学不会
3. **Counterintuitive scaling effect** ✅: Larger models perform worse -- the 10x model loses the 7 bits learned by the small model
   **反直觉的规模效应** ✅：大模型反而更差——10x 模型丢掉了小模型学会的 7 bit
4. **Regularization is not the cause** ✅: Reducing weight decay by 50x has no effect, ruling out the regularization hypothesis
   **正则化不是原因** ✅：降低 weight decay 50 倍无效，排除了正则化假说
5. **Context window breakthrough** ✅: context_len 16->32->64, LFSR-31 goes from 50% to 99.8% to 100%, proving the 50% ceiling is due to insufficient information, not insufficient model capacity
   **上下文窗口突破** ✅：context_len 16→32→64，LFSR-31 从 50% → 99.8% → 100%，证明 50% 天花板是信息不足而非模型不足
6. **Data volume cannot save topological shattering** ✅: lcg_glibc data expanded 20x, test acc unchanged (0.3%), train acc drops from 100% to 22%
   **数据量无法救拓扑粉碎** ✅：lcg_glibc 数据扩大 20 倍，test acc 不变（0.3%），train acc 反而从 100% 降到 22%

---

## Experiment Results / 实验结果

### Phase 1: Small Model (0.3M Parameters) / 第一阶段：小模型 (0.3M 参数)

| Sequence Type / 序列类型 | Test Accuracy / 测试准确率 | Random Baseline / 随机基线 | Conclusion / 结论 |
|---------|-----------|---------|------|
| periodic | 100% | 50% | ✅ Learnable / 可学 |
| lcg_simple | 100% | 0.4% | ✅ Learnable / 可学 |
| lfsr_5 | 100% | 50% | ✅ Learnable / 可学 |
| lcg_glibc | 0.4% | 0.4% | ❌ Unlearnable / 不可学 |
| lfsr_31 | 50% | 0.4% | ⚠️ Partially learnable (7/8 bit) / 部分可学（7/8 bit） |
| csprng | 0.3% | 0.4% | ❌ Unlearnable / 不可学 |

**Core Findings / 核心发现**:
- The phase transition boundary lies between **256 and 2^31**
- 相变边界在 **256 → 2³¹** 之间
- lfsr_31's 50% = partial Grokking, learned 7 bits, cannot learn the remaining 1 bit
- lfsr_31 的 50% = 部分 Grokking，学会了 7 bit，剩 1 bit 学不会
- lcg_glibc train 100% + test 0% = pure memorization solution
- lcg_glibc 训练 100% + 测试 0% = 纯记忆解

### Phase 2: Large Model (8M Parameters, 4x Scale-up) / 第二阶段：大模型 (8M 参数，4x 扩大)

| Sequence / 序列 | Small model train/test / 小模型 train/test | Large model train/test / 大模型 train/test | Change / 变化 |
|-----|------------------|------------------|------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | Worse (even memorization fails) / 更差（连记忆都失败） |
| lfsr_31 | 100% / 50% | 100% / 50% | No change / 无变化 |

### Phase 3: Extra-Large Model (33M Parameters, 10x Scale-up) / 第三阶段：超大模型 (33M 参数，10x 扩大)

| Sequence / 序列 | Small (0.3M) / 小模型 (0.3M) | 4x (8M) | 10x (33M) |
|-----|--------------|---------|-----------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | 0.8% / 0.5% |
| lfsr_31 | 100% / 50% | 100% / 50% | **0.8% / 0.5%** |

Initial hypothesis: The 10x model's collapse was caused by weight decay being too strong (0.5) -- "learned then forgot".
初始假设：10x 模型的崩溃是 weight decay 太强（0.5）导致的——"学会了又忘了"。

### Phase 4: 10x Model + Low Weight Decay (0.01) / 第四阶段：10x 模型 + 低 weight decay (0.01)

| Sequence / 序列 | Small (0.3M) / 小模型 (0.3M) | 4x (8M) | 10x (33M) | 10x + low wd (0.01) / 10x + 低wd (0.01) |
|-----|--------------|---------|-----------|-------------------|
| lcg_glibc | 100% / 0.4% | 0.8% / 0.5% | 0.8% / 0.5% | 0.8% / 0.5% |
| lfsr_31 | 100% / 50% | 100% / 50% | 0.8% / 0.5% | **0.8% / 0.5%** |

**Hypothesis refuted**: Weight decay reduced from 0.5 to 0.01 (50x), results identical. The collapse is not due to overly strong regularization, but rather the inherently unstable optimization landscape of large models on this type of task.
**假设被否定**：weight decay 从 0.5 降到 0.01（50 倍），结果一模一样。崩溃不是正则化太强，而是大模型在这类任务上的优化地形本身不稳定。

**Conclusions / 结论**:
1. **lcg_glibc**: Always unlearnable; model size and regularization strength are both irrelevant
   **lcg_glibc**：始终学不会，模型大小和正则化强度均无关
2. **lfsr_31**: The 10x model, regardless of weight decay, cannot preserve the 50% partial Grokking
   **lfsr_31**：10x 模型无论 weight decay 多少，都无法保住 50% 的部分 Grokking
3. **The small model is the optimal configuration**: The parameter space of large models is too vast; fragile partial learning structures cannot stably exist
   **小模型是最优配置**：大模型参数空间太大，脆弱的部分学习结构无法稳定存在
4. **The learnability boundary depends not only on task complexity, but also on model scale** -- counterintuitively: larger models are more likely to lose fragile partial learning
   **可学性边界不只取决于任务复杂度，还取决于模型规模**——方向反直觉：更大的模型反而更容易丢掉脆弱的部分学习

> **"It's not that the model is too small -- the pattern is too complex. A larger model not only didn't help, it also lost the 7 bits that were already learned."**
> **"不是模型太小，是规律太复杂。更大的模型不但没帮上忙，还把学会的 7 bit 也弄丢了。"**

### Phase 5: Context Window Expansion (context_len 16->32) / 第五阶段：上下文窗口扩展（context_len 16→32）

LFSR-31's 50% ceiling has two possible explanations: (a) insufficient model capacity; (b) insufficient input information. Phases 2-4 ruled out (a); Phase 5 validates (b).
LFSR-31 的 50% 天花板有两种可能解释：（a）模型能力不足；（b）输入信息不足。第二到四阶段排除了（a），第五阶段验证（b）。

**Hypothesis / 假设**: With context_len=16, each step shifts 1 bit, adjacent steps have 7-bit overlap, independent information ~24 bits < 31-bit internal state -> insufficient information. Expanding to 32, independent information ~39 bits > 31 bits, theoretically sufficient to reconstruct the full state.
**假设**：context_len=16 时，每步移 1 bit，相邻步 7 bit 重叠，独立信息 ~24 bit < 31 bit 内部状态 → 信息不足。扩到 32 后，独立信息 ~39 bit > 31 bit，理论上足够还原完整状态。

| Configuration / 配置 | context_len | Independent Info / 独立信息 | test acc | Change / 变化 |
|-----|-------------|---------|----------|------|
| Small model (0.3M) / 小模型 (0.3M) | 16 | ~24 bit < 31 bit | 50% | — |
| Small model (0.3M) / 小模型 (0.3M) | **32** | ~39 bit > 31 bit | **99.8%** | Breakthrough / 突破 |
| Small model (0.3M) / 小模型 (0.3M) | **64** | ~71 bit >> 31 bit | **100%** | Pattern redundancy -> perfect / 模式冗余 → 完美 |

**Hypothesis perfectly validated**. Once the window is large enough, the 50% ceiling instantly disappears. The pattern redundancy at 64 further eliminates the 0.2% residual error from 32.
**假设完美验证**。窗口一开够，50% 天花板瞬间消失。64 的模式冗余进一步消除了 32 的 0.2% 残余误差。

**Conclusions / 结论**:
1. **LFSR-31's 50% is not a model capacity problem, but an information bottleneck**: The model cannot see enough bits to reconstruct the full internal state
   **LFSR-31 的 50% 不是模型能力问题，是信息瓶颈**：模型看不到足够的 bit 来还原完整内部状态
2. **What's needed is information, not parameters**: Scaling up the model (Phases 2-4) is ineffective or even harmful; expanding the window directly solves the problem
   **该给的是信息，不是参数**：加大模型（第二到四阶段）无效甚至有害，加大窗口直接解决
3. **Information has a saturation effect**: 32 already breaks through (99.8%); the additional redundancy at 64 only eliminates 0.2% residual -- diminishing returns
   **信息量存在饱和效应**：32 已突破（99.8%），64 的额外冗余只消除了 0.2% 残余——收益递减
4. **Three dimensions of the learnability boundary**: Task complexity, model capacity, and input information -- all three jointly determine the boundary position
   **可学性边界的三个维度**：任务复杂度、模型能力、输入信息量——三者共同决定边界位置

> **"It's not that the model is too dumb, nor that the pattern is too hard -- you didn't give it enough clues."**
> **"不是模型太笨，也不是规律太难——是你给的线索不够。"**

### Phase 6: Data Volume Expansion (seq_length 10k->200k) / 第六阶段：数据量扩展（seq_length 10k→200k）

LFSR-31's problem was solved by the window experiment (information bottleneck). What about lcg_glibc? Is its unlearnability due to insufficient data, or does the manifold itself not exist?
LFSR-31 的问题被窗口实验解决了（信息瓶颈）。那 lcg_glibc 呢？它的不可学是数据不够，还是流形本身不存在？

| Configuration / 配置 | seq_length | Training Samples / 训练样本 | train acc | test acc |
|-----|-----------|---------|-----------|----------|
| Original (0.3M) / 原始 (0.3M) | 10,000 | ~3,000 | 100% | 0.4% |
| 20x data (0.3M) / 20x 数据 (0.3M) | **200,000** | **~60,000** | **22%** | **0.3%** |

**More data actually makes it worse** -- originally 3000 samples could be memorized by brute force (train 100%), but 60000 cannot be fully memorized (train 22%). Yet test acc remains at approximately the random baseline.
**数据多了反而更差**——原来 3000 个样本能硬背（train 100%），60000 个背不完了（train 22%）。但 test acc 始终 ≈ 随机基线。

**Conclusions / 结论**:
1. **Topological shattering is an iron law**: LCG's modular multiplication shatters the manifold; 20x data cannot save it
   **拓扑粉碎是铁律**：LCG 的乘法取模把流形揉碎，20 倍数据救不了
2. **The original train 100% was an illusion**: 3000 samples happened to be within memorization capacity; 60000 exposes the truth
   **原来的 train 100% 是假象**：3000 样本刚好在记忆容量内，60000 暴露了真相
3. **Forms a perfect contrast with LFSR**: LFSR is solved by expanding the window (information); LCG cannot be saved even with more data -- the problem is not information volume, but that the manifold doesn't exist
   **与 LFSR 形成完美对比**：LFSR 靠加窗口（信息）直接解决，LCG 加数据也没用——问题不是信息量，是流形不存在

> **"No amount of data will help -- it's not that there aren't enough clues, it's that the answer doesn't exist."**
> **"给再多数据也没用——不是线索不够，是谜底不存在。"**

---

## TODO

- [x] ~~Expand data volume: lcg_glibc seq_length 10k->200k~~ -> **Completed: test 0.3%, topological shattering is irredeemable**
  ~~扩大数据量：lcg_glibc seq_length 10k→200k~~ → **已完成：test 0.3%，拓扑粉碎不可救**
- [x] ~~Expand context window: lfsr_31 context_len 16->32->64~~ -> **Completed: 50% -> 99.8% -> 100%**
  ~~扩大上下文窗口：lfsr_31 context_len 16→32→64~~ → **已完成：50% → 99.8% → 100%**

---

## References / 参考

- Epiplexity paper / Epiplexity 论文：https://arxiv.org/abs/2601.03220
- Grokking manifold discovery experiment / Grokking 流形发现实验：`../wechat67/`
- This experiment on Zenodo / 本实验 Zenodo：https://zenodo.org/records/18538126
