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
│   ├── train_learnability.py   # 主训练脚本
│   ├── plot_comparison.py      # 对比图生成
│   ├── estimate_dimension.py   # 维度分析
│   ├── analyze_dynamics.py     # 动态分析
│   ├── analyze_attention.py    # Attention分析（备用）
│   └── run_all.sh              # 批量运行
└── results/
    ├── periodic/               # 周期序列结果
    │   ├── train_log.json      # 训练日志
    │   ├── activations/        # 中间激活快照
    │   └── model_final.pt      # 最终模型
    ├── lcg_simple/
    ├── ...
    ├── learning_curves.png     # 学习曲线对比
    ├── final_comparison.png    # 最终准确率对比
    ├── dimension_trajectory.png # 维度变化曲线
    ├── dynamics_analysis.png   # 训练动态（Grokking点、震荡）
    └── summary.md              # 实验摘要
```

---

## 预期结果

- **periodic**：完美 Grokking，测试准确率 → 100%
- **lcg_simple / lfsr_5**：可能 Grokking（相变边界）
- **lcg_glibc / lfsr_31**：可能卡在随机基线附近
- **csprng**：测试准确率 ≈ 随机猜测 (1/256 ≈ 0.4%)

如果 lcg_simple 能学会而 lcg_glibc 学不会，说明：
> **Epiplexity 相变点在 "状态空间 256" 和 "状态空间 2^31" 之间**

---

## 要观察的现象

复用模加法实验的分析方法，看能不能发现：

1. **维度骤降**：Grokking 时内在维度突然下降（从高维噪声到低维结构）
2. **二阶段 Grokking**：先学会简单规律，再学会复杂规律
3. **临界态震荡**：在"学会/没学会"之间反复跳跃
4. **不同序列的维度差异**：可学序列维度低，不可学序列维度高（接近 embedding 维度）

---

## 参考

- Epiplexity 论文：https://arxiv.org/abs/2601.03220
- Grokking 流形发现实验：`../wechat67/`
