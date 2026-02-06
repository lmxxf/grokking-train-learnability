"""
可学性边界实验：找到神经网络能学会的伪随机复杂度临界点

核心问题：Epiplexity 从 0 到非零的相变，在什么复杂度发生？

序列类型（从简单到复杂）：
1. periodic: (i % 7) > 3 → 预期 Grokking
2. lcg_simple: x = (3*x + 7) % 256 → 边界探索
3. lcg_glibc: x = (1103515245*x + 12345) % 2^31 → 边界探索
4. lfsr_5: x = x[n-1] XOR x[n-5] → 边界探索
5. lfsr_31: 31位LFSR → 预期不可学
6. csprng: secrets.randbits → 预期不可学

用法：
    python train_learnability.py --seq_type periodic
    python train_learnability.py --seq_type lcg_simple
    python train_learnability.py --seq_type csprng
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import argparse
import secrets
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    "seq_length": 10000,        # 生成序列总长度
    "context_len": 16,          # 输入上下文长度
    "train_ratio": 0.3,         # 训练集比例
    "embed_dim": 128,           # embedding 维度
    "num_heads": 4,             # attention heads
    "num_layers": 2,            # transformer 层数
    "lr": 1e-3,                 # 学习率
    "weight_decay": 1.0,        # 权重衰减
    "batch_size": 512,          # batch size
    "total_steps": 100000,      # 总训练步数
    "eval_every": 500,          # 每隔多少步评估
    "save_activations_every": 2000,  # 每隔多少步保存激活
    "seed": 42,
}


# ============ 序列生成器 ============
class SequenceGenerator:
    """不同复杂度的序列生成器"""

    @staticmethod
    def periodic(length, seed=42):
        """简单周期：(i % 7) > 3 → 0/1"""
        return np.array([(i % 7) > 3 for i in range(length)], dtype=np.int64)

    @staticmethod
    def lcg_simple(length, seed=42):
        """简单线性同余：x = (3*x + 7) % 256"""
        seq = np.zeros(length, dtype=np.int64)
        x = seed % 256
        for i in range(length):
            seq[i] = x
            x = (3 * x + 7) % 256
        return seq

    @staticmethod
    def lcg_glibc(length, seed=42):
        """glibc 线性同余：x = (1103515245*x + 12345) % 2^31"""
        seq = np.zeros(length, dtype=np.int64)
        x = seed
        for i in range(length):
            x = (1103515245 * x + 12345) % (2**31)
            seq[i] = (x >> 16) & 0xFF  # 取高位，映射到 0-255
        return seq

    @staticmethod
    def lfsr_5(length, seed=42):
        """5位LFSR：x[n] = x[n-1] XOR x[n-5]"""
        # 初始化
        np.random.seed(seed)
        seq = np.zeros(length, dtype=np.int64)
        # 用随机初始状态
        for i in range(5):
            seq[i] = np.random.randint(0, 2)
        # 生成
        for i in range(5, length):
            seq[i] = seq[i-1] ^ seq[i-5]
        return seq

    @staticmethod
    def lfsr_31(length, seed=42):
        """31位LFSR：更长的反馈多项式"""
        # x^31 + x^3 + 1 (常用的31位LFSR)
        state = seed if seed != 0 else 1
        seq = np.zeros(length, dtype=np.int64)
        for i in range(length):
            bit = ((state >> 30) ^ (state >> 2)) & 1
            state = ((state << 1) | bit) & 0x7FFFFFFF
            seq[i] = state & 0xFF  # 取低8位
        return seq

    @staticmethod
    def csprng(length, seed=42):
        """密码学安全随机数（不可学）"""
        # secrets 模块不能设 seed，每次都是真随机
        # 但为了可复现，我们用固定文件或接受不可复现
        return np.array([secrets.randbelow(256) for _ in range(length)], dtype=np.int64)


def get_vocab_size(seq_type):
    """根据序列类型返回词表大小"""
    if seq_type == "periodic":
        return 2  # 0/1
    elif seq_type == "lfsr_5":
        return 2  # 0/1
    else:
        return 256  # 0-255


# ============ 数据准备 ============
def generate_dataset(seq_type, seq_length, context_len, train_ratio, seed=42):
    """生成训练/测试数据集"""

    # 生成序列
    generator = getattr(SequenceGenerator, seq_type)
    sequence = generator(seq_length, seed)

    # 滑动窗口切分
    X = []
    y = []
    for i in range(len(sequence) - context_len):
        X.append(sequence[i:i+context_len])
        y.append(sequence[i+context_len])

    X = np.array(X)
    y = np.array(y)

    # 打乱并分割
    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    n_train = int(len(X) * train_ratio)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_X, train_y = X[train_idx], y[train_idx]
    test_X, test_y = X[test_idx], y[test_idx]

    return (
        torch.tensor(train_X, dtype=torch.long),
        torch.tensor(train_y, dtype=torch.long),
        torch.tensor(test_X, dtype=torch.long),
        torch.tensor(test_y, dtype=torch.long),
    )


# ============ 模型 ============
class SequenceTransformer(nn.Module):
    """序列预测 Transformer"""

    def __init__(self, vocab_size, context_len, embed_dim, num_heads, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.embed_dim = embed_dim

        # Token embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)

        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(context_len, embed_dim) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出
        self.output = nn.Linear(embed_dim, vocab_size)

        # 保存激活
        self.last_hidden = None

    def forward(self, x, save_hidden=False):
        # x: (batch, context_len)

        # Embed
        emb = self.embed(x)  # (batch, context_len, embed_dim)
        emb = emb + self.pos_embed.unsqueeze(0)

        # Transformer
        h = self.transformer(emb)  # (batch, context_len, embed_dim)

        # 保存最后一个位置的激活
        if save_hidden:
            self.last_hidden = h[:, -1, :].detach().cpu()

        # 用最后一个位置预测下一个 token
        logits = self.output(h[:, -1, :])  # (batch, vocab_size)

        return logits


# ============ 训练 ============
def train(seq_type, output_dir):
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Sequence type: {seq_type}")

    # 随机种子
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # 创建输出目录
    exp_dir = os.path.join(output_dir, seq_type)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "activations"), exist_ok=True)

    # 数据
    vocab_size = get_vocab_size(seq_type)
    train_X, train_y, test_X, test_y = generate_dataset(
        seq_type,
        CONFIG["seq_length"],
        CONFIG["context_len"],
        CONFIG["train_ratio"],
        CONFIG["seed"]
    )

    train_dataset = TensorDataset(train_X, train_y)
    test_dataset = TensorDataset(test_X, test_y)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"Vocab size: {vocab_size}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Random baseline accuracy: {1/vocab_size:.4f}")

    # 模型
    model = SequenceTransformer(
        vocab_size=vocab_size,
        context_len=CONFIG["context_len"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)

    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    # 日志
    log = {
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    # 训练
    step = 0
    pbar = tqdm(total=CONFIG["total_steps"], desc=f"Training {seq_type}")

    while step < CONFIG["total_steps"]:
        for batch_x, batch_y in train_loader:
            if step >= CONFIG["total_steps"]:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Forward
            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            # 评估
            if step % CONFIG["eval_every"] == 0:
                model.eval()

                # 训练集
                train_correct = 0
                train_total = 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(bx).argmax(dim=1)
                    train_correct += (pred == by).sum().item()
                    train_total += len(by)
                train_acc = train_correct / train_total

                # 测试集
                test_correct = 0
                test_total = 0
                for bx, by in test_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(bx).argmax(dim=1)
                    test_correct += (pred == by).sum().item()
                    test_total += len(by)
                test_acc = test_correct / test_total

                log["steps"].append(step)
                log["train_loss"].append(loss.item())
                log["train_acc"].append(train_acc)
                log["test_acc"].append(test_acc)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "train": f"{train_acc:.3f}",
                    "test": f"{test_acc:.3f}"
                })

                model.train()

            # 保存激活
            if step % CONFIG["save_activations_every"] == 0:
                model.eval()
                all_hidden = []
                all_labels = []
                for bx, by in test_loader:
                    bx = bx.to(device)
                    with torch.no_grad():
                        _ = model(bx, save_hidden=True)
                    all_hidden.append(model.last_hidden)
                    all_labels.append(by)

                all_hidden = torch.cat(all_hidden, dim=0).numpy()
                all_labels = torch.cat(all_labels, dim=0).numpy()

                np.savez(
                    os.path.join(exp_dir, "activations", f"step_{step:06d}.npz"),
                    hidden=all_hidden,
                    labels=all_labels
                )
                model.train()

    pbar.close()

    # 保存结果
    with open(os.path.join(exp_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    config_save = CONFIG.copy()
    config_save["seq_type"] = seq_type
    config_save["vocab_size"] = vocab_size
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pt"))

    print(f"\n{'='*50}")
    print(f"Sequence type: {seq_type}")
    print(f"Final train acc: {log['train_acc'][-1]:.4f}")
    print(f"Final test acc: {log['test_acc'][-1]:.4f}")
    print(f"Random baseline: {1/vocab_size:.4f}")
    print(f"Results saved to: {exp_dir}")

    return log


# ============ 主函数 ============
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_type", type=str, required=True,
                        choices=["periodic", "lcg_simple", "lcg_glibc",
                                "lfsr_5", "lfsr_31", "csprng"],
                        help="序列类型")
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/ai-theorys-study/arxiv/wechat62/results",
                        help="输出目录")
    args = parser.parse_args()

    train(args.seq_type, args.output_dir)
