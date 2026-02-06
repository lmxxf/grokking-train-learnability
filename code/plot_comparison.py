"""
绘制所有序列类型的学习曲线对比图
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = "/workspace/ai-theorys-study/arxiv/wechat62/results"

SEQ_TYPES = ["periodic", "lcg_simple", "lfsr_5", "lcg_glibc", "lfsr_31", "csprng"]

LABELS = {
    "periodic": "周期 (i%7>3)",
    "lcg_simple": "LCG简单 (3x+7)",
    "lfsr_5": "LFSR-5",
    "lcg_glibc": "LCG-glibc",
    "lfsr_31": "LFSR-31",
    "csprng": "CSPRNG",
}

COLORS = {
    "periodic": "#2ecc71",      # 绿 - 可学
    "lcg_simple": "#3498db",    # 蓝
    "lfsr_5": "#9b59b6",        # 紫
    "lcg_glibc": "#e67e22",     # 橙
    "lfsr_31": "#e74c3c",       # 红
    "csprng": "#7f8c8d",        # 灰 - 不可学
}


def load_results():
    """加载所有实验结果"""
    results = {}
    for seq_type in SEQ_TYPES:
        log_path = os.path.join(RESULT_DIR, seq_type, "train_log.json")
        config_path = os.path.join(RESULT_DIR, seq_type, "config.json")

        if os.path.exists(log_path):
            with open(log_path) as f:
                log = json.load(f)
            with open(config_path) as f:
                config = json.load(f)

            results[seq_type] = {
                "log": log,
                "config": config,
                "baseline": 1.0 / config["vocab_size"]
            }
            print(f"Loaded: {seq_type} (final test acc: {log['test_acc'][-1]:.4f})")
        else:
            print(f"Missing: {seq_type}")

    return results


def plot_learning_curves(results):
    """绘制学习曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：测试准确率
    ax1 = axes[0]
    for seq_type in SEQ_TYPES:
        if seq_type not in results:
            continue
        log = results[seq_type]["log"]
        baseline = results[seq_type]["baseline"]

        ax1.plot(log["steps"], log["test_acc"],
                label=LABELS[seq_type], color=COLORS[seq_type], linewidth=2)
        # 画 baseline
        ax1.axhline(y=baseline, color=COLORS[seq_type], linestyle=':', alpha=0.5)

    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("可学性边界：测试集准确率", fontsize=14)
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # 右图：训练 loss
    ax2 = axes[1]
    for seq_type in SEQ_TYPES:
        if seq_type not in results:
            continue
        log = results[seq_type]["log"]

        ax2.plot(log["steps"], log["train_loss"],
                label=LABELS[seq_type], color=COLORS[seq_type], linewidth=2)

    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Train Loss", fontsize=12)
    ax2.set_title("训练 Loss 曲线", fontsize=14)
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "learning_curves.png"), dpi=150)
    print(f"\nSaved: {os.path.join(RESULT_DIR, 'learning_curves.png')}")


def plot_final_comparison(results):
    """绘制最终准确率对比柱状图"""
    fig, ax = plt.subplots(figsize=(10, 6))

    seq_types = [s for s in SEQ_TYPES if s in results]
    final_accs = [results[s]["log"]["test_acc"][-1] for s in seq_types]
    baselines = [results[s]["baseline"] for s in seq_types]
    labels = [LABELS[s] for s in seq_types]
    colors = [COLORS[s] for s in seq_types]

    x = np.arange(len(seq_types))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_accs, width, label='最终测试准确率', color=colors)
    bars2 = ax.bar(x + width/2, baselines, width, label='随机猜测基线', color='lightgray', edgecolor='gray')

    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('可学性边界：最终准确率 vs 随机基线', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.legend()
    ax.set_ylim([0, 1.1])

    # 在柱子上标注数值
    for bar, acc in zip(bars1, final_accs):
        height = bar.get_height()
        ax.annotate(f'{acc:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    # 画"可学/不可学"分界线
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% 分界')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "final_comparison.png"), dpi=150)
    print(f"Saved: {os.path.join(RESULT_DIR, 'final_comparison.png')}")


def generate_summary(results):
    """生成实验摘要"""
    summary = []
    summary.append("# 可学性边界实验结果\n")
    summary.append("| 序列类型 | 最终测试准确率 | 随机基线 | 是否可学 |")
    summary.append("|---------|--------------|---------|---------|")

    for seq_type in SEQ_TYPES:
        if seq_type not in results:
            continue
        final_acc = results[seq_type]["log"]["test_acc"][-1]
        baseline = results[seq_type]["baseline"]

        # 判断是否可学：准确率显著高于随机基线
        learnable = "✅" if final_acc > baseline * 1.5 else "❌"

        summary.append(f"| {LABELS[seq_type]} | {final_acc:.4f} | {baseline:.4f} | {learnable} |")

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(os.path.join(RESULT_DIR, "summary.md"), "w") as f:
        f.write(summary_text)
    print(f"\nSaved: {os.path.join(RESULT_DIR, 'summary.md')}")


if __name__ == "__main__":
    results = load_results()

    if len(results) > 0:
        plot_learning_curves(results)
        plot_final_comparison(results)
        generate_summary(results)
    else:
        print("No results found. Run experiments first.")
