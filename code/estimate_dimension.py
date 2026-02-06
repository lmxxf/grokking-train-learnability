"""
维度分析：追踪表示空间的内在维度变化

用 MLE (Maximum Likelihood Estimation) 方法估计激活向量的内在维度。
如果发生 Grokking，预期看到维度骤降。

用法：
    python estimate_dimension.py --seq_type periodic
"""

import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
from glob import glob
from sklearn.neighbors import NearestNeighbors


def estimate_intrinsic_dimension_mle(X, k=10):
    """
    用 MLE 方法估计内在维度 (Levina & Bickel, 2004)

    原理：在高维空间中，k近邻距离的比值分布与内在维度相关
    """
    n_samples = X.shape[0]

    # 找 k+1 近邻（包含自己）
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, _ = nbrs.kneighbors(X)

    # 去掉自己（距离为0）
    distances = distances[:, 1:]  # (n_samples, k)

    # 避免数值问题
    distances = np.maximum(distances, 1e-10)

    # MLE 估计
    # d_hat = 1 / (1/n * sum_i (1/(k-1) * sum_j log(r_k / r_j)))
    log_ratios = np.log(distances[:, -1:] / distances[:, :-1])  # log(r_k / r_j)
    mean_log_ratio = np.mean(log_ratios)

    if mean_log_ratio > 0:
        dim_estimate = (k - 1) / (mean_log_ratio * (k - 1) + 1e-10)
    else:
        dim_estimate = float('inf')

    return max(1, min(dim_estimate, X.shape[1]))  # 限制在合理范围


def analyze_dimension_trajectory(result_dir, seq_type):
    """分析维度随训练步数的变化"""
    act_dir = os.path.join(result_dir, seq_type, "activations")

    if not os.path.exists(act_dir):
        print(f"No activations found for {seq_type}")
        return None

    # 找所有激活文件
    act_files = sorted(glob(os.path.join(act_dir, "step_*.npz")))

    if len(act_files) == 0:
        print(f"No activation files found in {act_dir}")
        return None

    print(f"Found {len(act_files)} activation snapshots for {seq_type}")

    steps = []
    dimensions = []

    for f in act_files:
        # 解析步数
        step = int(os.path.basename(f).split('_')[1].split('.')[0])

        # 加载激活
        data = np.load(f)
        hidden = data['hidden']  # (n_samples, embed_dim)

        # 估计维度
        dim = estimate_intrinsic_dimension_mle(hidden, k=10)

        steps.append(step)
        dimensions.append(dim)
        print(f"  Step {step:6d}: dim = {dim:.1f}")

    return {"steps": steps, "dimensions": dimensions}


def plot_dimension_trajectory(results, output_dir):
    """绘制维度变化曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "periodic": "#2ecc71",
        "lcg_simple": "#3498db",
        "lfsr_5": "#9b59b6",
        "lcg_glibc": "#e67e22",
        "lfsr_31": "#e74c3c",
        "csprng": "#7f8c8d",
    }

    labels = {
        "periodic": "周期 (i%7>3)",
        "lcg_simple": "LCG简单",
        "lfsr_5": "LFSR-5",
        "lcg_glibc": "LCG-glibc",
        "lfsr_31": "LFSR-31",
        "csprng": "CSPRNG",
    }

    for seq_type, data in results.items():
        if data is not None:
            ax.plot(data["steps"], data["dimensions"],
                   label=labels.get(seq_type, seq_type),
                   color=colors.get(seq_type, "gray"),
                   linewidth=2, marker='o', markersize=4)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Intrinsic Dimension (MLE)", fontsize=12)
    ax.set_title("表示空间内在维度随训练变化", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dimension_trajectory.png"), dpi=150)
    print(f"\nSaved: {os.path.join(output_dir, 'dimension_trajectory.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_type", type=str, default=None,
                        help="分析指定序列类型，不指定则分析全部")
    parser.add_argument("--result_dir", type=str,
                        default="/workspace/ai-theorys-study/arxiv/wechat62/results")
    args = parser.parse_args()

    seq_types = ["periodic", "lcg_simple", "lfsr_5", "lcg_glibc", "lfsr_31", "csprng"]

    if args.seq_type:
        seq_types = [args.seq_type]

    results = {}
    for seq_type in seq_types:
        print(f"\n=== Analyzing {seq_type} ===")
        results[seq_type] = analyze_dimension_trajectory(args.result_dir, seq_type)

    # 保存结果
    dim_results = {}
    for seq_type, data in results.items():
        if data is not None:
            dim_results[seq_type] = data

    if dim_results:
        with open(os.path.join(args.result_dir, "dimension_analysis.json"), "w") as f:
            json.dump(dim_results, f, indent=2)

        plot_dimension_trajectory(results, args.result_dir)


if __name__ == "__main__":
    main()
