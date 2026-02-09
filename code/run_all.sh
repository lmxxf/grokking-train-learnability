#!/bin/bash
# 批量运行所有序列类型的实验

cd /workspace/ai-theorys-study/arxiv/wechat62/code/

echo "========== 可学性边界实验 =========="
echo ""

# 从简单到复杂
for seq_type in periodic lcg_simple lfsr_5 lcg_glibc lfsr_31 csprng; do
    echo ">>> Running: $seq_type"
    python train_learnability.py --seq_type $seq_type
    echo ""
done

echo "========== 训练完成，开始分析 =========="
echo ""

# 生成对比图
python plot_comparison.py

# 维度分析
python estimate_dimension.py

# 动态分析（Grokking检测、震荡检测）
python analyze_dynamics.py

echo "========== 全部完成 =========="
