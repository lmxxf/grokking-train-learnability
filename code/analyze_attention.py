"""
Attention 模式分析：不同序列类型的注意力分布差异

假说：
- 可学序列：attention 会聚焦到特定位置（发现了规律）
- 不可学序列：attention 均匀分布（没找到规律）

用法：
    python analyze_attention.py --seq_type periodic
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt


# 复用模型定义
class SequenceTransformer(nn.Module):
    """序列预测 Transformer"""

    def __init__(self, vocab_size, context_len, embed_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(context_len, embed_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(embed_dim, vocab_size)

        # 存储 attention weights
        self.attention_weights = []
        self._register_hooks()

    def _register_hooks(self):
        """注册 hook 捕获 attention weights"""
        def hook_fn(module, input, output):
            # TransformerEncoderLayer 的 self_attn 输出包含 attention weights
            if hasattr(module, 'self_attn'):
                # 这里简化处理，实际需要修改 forward 来获取 attention
                pass

        for layer in self.transformer.layers:
            layer.register_forward_hook(hook_fn)

    def forward(self, x):
        emb = self.embed(x)
        emb = emb + self.pos_embed.unsqueeze(0)
        h = self.transformer(emb)
        logits = self.output(h[:, -1, :])
        return logits


def compute_attention_entropy(attention_weights):
    """
    计算 attention 分布的熵

    熵高 = 均匀分布 = 没找到规律
    熵低 = 聚焦分布 = 找到了规律
    """
    # attention_weights: (batch, heads, seq_len, seq_len)
    # 取最后一个位置的 attention
    attn = attention_weights[:, :, -1, :]  # (batch, heads, seq_len)

    # 计算熵
    entropy = -torch.sum(attn * torch.log(attn + 1e-10), dim=-1)  # (batch, heads)
    return entropy.mean().item()


def analyze_gradient_flow(model, test_loader, device):
    """
    分析梯度流：哪些位置对预测影响最大

    可学序列：梯度集中在特定位置
    不可学序列：梯度均匀分布
    """
    model.eval()
    model.zero_grad()

    all_grad_norms = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # 启用输入的梯度
        batch_x_embed = model.embed(batch_x)
        batch_x_embed.requires_grad_(True)
        batch_x_embed.retain_grad()

        # Forward（手动，因为需要插入梯度计算）
        emb = batch_x_embed + model.pos_embed.unsqueeze(0)
        h = model.transformer(emb)
        logits = model.output(h[:, -1, :])

        # 对正确类别的 logit 求梯度
        correct_logits = logits.gather(1, batch_y.unsqueeze(1)).sum()
        correct_logits.backward()

        # 获取每个位置的梯度范数
        grad = batch_x_embed.grad  # (batch, context_len, embed_dim)
        grad_norm = grad.norm(dim=-1)  # (batch, context_len)

        all_grad_norms.append(grad_norm.detach().cpu())

        model.zero_grad()
        break  # 只用一个 batch

    grad_norms = torch.cat(all_grad_norms, dim=0)  # (total_samples, context_len)
    mean_grad_norm = grad_norms.mean(dim=0)  # (context_len,)

    return mean_grad_norm.numpy()


def plot_gradient_patterns(results, output_dir):
    """绘制不同序列的梯度分布模式"""
    n_types = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    labels = {
        "periodic": "周期 (i%7>3)",
        "lcg_simple": "LCG简单",
        "lfsr_5": "LFSR-5",
        "lcg_glibc": "LCG-glibc",
        "lfsr_31": "LFSR-31",
        "csprng": "CSPRNG",
    }

    for idx, (seq_type, grad_norm) in enumerate(results.items()):
        if idx >= len(axes):
            break

        ax = axes[idx]
        positions = np.arange(len(grad_norm))

        ax.bar(positions, grad_norm, color='steelblue', alpha=0.7)
        ax.set_title(labels.get(seq_type, seq_type))
        ax.set_xlabel('Position in Context')
        ax.set_ylabel('Gradient Norm')
        ax.grid(True, alpha=0.3)

        # 计算梯度集中度
        grad_entropy = -np.sum(grad_norm/grad_norm.sum() *
                              np.log(grad_norm/grad_norm.sum() + 1e-10))
        ax.text(0.95, 0.95, f'Entropy: {grad_entropy:.2f}',
               transform=ax.transAxes, ha='right', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 隐藏多余的 subplot
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gradient_patterns.png"), dpi=150)
    print(f"Saved: {os.path.join(output_dir, 'gradient_patterns.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_type", type=str, default=None)
    parser.add_argument("--result_dir", type=str,
                        default="/workspace/ai-theorys-study/arxiv/wechat62/results")
    args = parser.parse_args()

    seq_types = ["periodic", "lcg_simple", "lfsr_5", "lcg_glibc", "lfsr_31", "csprng"]
    if args.seq_type:
        seq_types = [args.seq_type]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 这个分析需要重新加载模型和数据
    # 简化版：只分析保存的激活模式
    print("\n注意：完整的 attention 分析需要重新推理。")
    print("当前版本只分析已保存的激活向量。")
    print("如需完整分析，请修改 train_learnability.py 保存 attention weights。")


if __name__ == "__main__":
    main()
