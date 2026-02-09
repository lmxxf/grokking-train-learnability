"""
训练动态分析：发现二阶段 Grokking、临界态震荡等现象

分析指标：
1. 准确率跳变检测（Grokking 瞬间）
2. 震荡检测（在泛化/记忆之间反复跳跃）
3. Loss 曲线的导数变化
4. 训练/测试准确率的交叉点

用法：
    python analyze_dynamics.py
"""

import numpy as np
import os
import json
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter


RESULT_DIR = "/workspace/ai-theorys-study/arxiv/wechat62/results"

SEQ_TYPES = ["periodic", "lcg_simple", "lfsr_5", "lcg_glibc", "lfsr_31", "csprng"]

LABELS = {
    "periodic": "周期 (i%7>3)",
    "lcg_simple": "LCG简单",
    "lfsr_5": "LFSR-5",
    "lcg_glibc": "LCG-glibc",
    "lfsr_31": "LFSR-31",
    "csprng": "CSPRNG",
}


def detect_grokking_moment(test_acc, threshold=0.5, window=5):
    """
    检测 Grokking 发生的瞬间

    Grokking 特征：测试准确率在短时间内从低于阈值跳到高于阈值
    """
    acc = np.array(test_acc)

    # 找第一个超过阈值的点
    above_threshold = acc > threshold
    if not any(above_threshold):
        return None, None

    first_above = np.argmax(above_threshold)

    # 检查之前是否有一段时间低于阈值
    if first_above < window:
        return None, None

    before_mean = np.mean(acc[max(0, first_above-window):first_above])
    after_mean = np.mean(acc[first_above:min(len(acc), first_above+window)])

    # 跳变幅度
    jump = after_mean - before_mean

    if jump > 0.3:  # 显著跳变
        return first_above, jump

    return None, None


def detect_oscillations(test_acc, window=10):
    """
    检测临界态震荡

    震荡特征：测试准确率在一段时间内反复上下波动
    """
    acc = np.array(test_acc)

    if len(acc) < window * 2:
        return 0, []

    # 平滑后找峰值
    if len(acc) > 11:
        smoothed = savgol_filter(acc, min(11, len(acc)//2*2+1), 3)
    else:
        smoothed = acc

    # 找局部极大值
    peaks, _ = find_peaks(smoothed, distance=window//2)

    # 找局部极小值
    valleys, _ = find_peaks(-smoothed, distance=window//2)

    # 交替出现的峰谷 = 震荡
    oscillation_count = min(len(peaks), len(valleys))

    return oscillation_count, list(peaks)


def detect_phase_transitions(test_acc, train_acc):
    """
    检测多阶段相变

    分析训练/测试准确率的关系变化
    """
    test = np.array(test_acc)
    train = np.array(train_acc)

    phases = []

    # 阶段1：记忆期（训练高，测试低）
    memorization_end = None
    for i in range(len(test)):
        if train[i] > 0.9 and test[i] < 0.3:
            memorization_end = i
        elif test[i] > 0.5 and memorization_end is not None:
            phases.append(("memorization", 0, memorization_end))
            break

    # 阶段2：泛化期（测试开始上升）
    grok_start = None
    grok_end = None
    for i in range(len(test)):
        if test[i] > 0.5 and grok_start is None:
            grok_start = i
        if test[i] > 0.9 and grok_start is not None:
            grok_end = i
            phases.append(("grokking", grok_start, grok_end))
            break

    return phases


def analyze_all():
    """分析所有实验的训练动态"""
    results = {}

    for seq_type in SEQ_TYPES:
        log_path = os.path.join(RESULT_DIR, seq_type, "train_log.json")
        if not os.path.exists(log_path):
            continue

        with open(log_path) as f:
            log = json.load(f)

        test_acc = log["test_acc"]
        train_acc = log["train_acc"]
        steps = log["steps"]

        # Grokking 检测
        grok_idx, grok_jump = detect_grokking_moment(test_acc)
        grok_step = steps[grok_idx] if grok_idx is not None else None

        # 震荡检测
        oscillation_count, peak_indices = detect_oscillations(test_acc)

        # 相变检测
        phases = detect_phase_transitions(test_acc, train_acc)

        results[seq_type] = {
            "grok_step": grok_step,
            "grok_jump": grok_jump,
            "oscillation_count": oscillation_count,
            "phases": phases,
            "final_test_acc": test_acc[-1],
            "final_train_acc": train_acc[-1],
        }

        print(f"\n=== {LABELS[seq_type]} ===")
        print(f"  Final test acc: {test_acc[-1]:.4f}")
        if grok_step:
            print(f"  Grokking at step {grok_step} (jump: {grok_jump:.3f})")
        else:
            print(f"  No Grokking detected")
        print(f"  Oscillations: {oscillation_count}")
        if phases:
            for phase_name, start, end in phases:
                print(f"  Phase '{phase_name}': steps {steps[start]} - {steps[end]}")

    # 保存
    with open(os.path.join(RESULT_DIR, "dynamics_analysis.json"), "w") as f:
        # 转换 phases 为可序列化格式
        save_results = {}
        for k, v in results.items():
            save_results[k] = {
                "grok_step": v["grok_step"],
                "grok_jump": float(v["grok_jump"]) if v["grok_jump"] else None,
                "oscillation_count": v["oscillation_count"],
                "phases": [(p[0], int(p[1]), int(p[2])) for p in v["phases"]],
                "final_test_acc": v["final_test_acc"],
                "final_train_acc": v["final_train_acc"],
            }
        json.dump(save_results, f, indent=2)

    return results


def plot_dynamics(results):
    """可视化训练动态"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, seq_type in enumerate(SEQ_TYPES):
        if seq_type not in results:
            continue

        log_path = os.path.join(RESULT_DIR, seq_type, "train_log.json")
        if not os.path.exists(log_path):
            continue

        with open(log_path) as f:
            log = json.load(f)

        ax = axes[idx]
        steps = np.array(log["steps"])
        test_acc = np.array(log["test_acc"])
        train_acc = np.array(log["train_acc"])

        ax.plot(steps, train_acc, label='Train', color='blue', alpha=0.7)
        ax.plot(steps, test_acc, label='Test', color='red', alpha=0.7)

        # 标记 Grokking 点
        r = results[seq_type]
        if r["grok_step"]:
            ax.axvline(x=r["grok_step"], color='green', linestyle='--',
                      label=f'Grok @ {r["grok_step"]}')

        ax.set_title(f'{LABELS[seq_type]}\nOscillations: {r["oscillation_count"]}')
        ax.set_xlabel('Steps')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "dynamics_analysis.png"), dpi=150)
    print(f"\nSaved: {os.path.join(RESULT_DIR, 'dynamics_analysis.png')}")


if __name__ == "__main__":
    results = analyze_all()
    if results:
        plot_dynamics(results)
