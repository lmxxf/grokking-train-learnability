"""
第三阶段测试：10x 大模型

在 4x 模型基础上再扩大 ~2.5x，总共约 10x

用法：
    python train_xlarge_model.py --seq_type lcg_glibc
    python train_xlarge_model.py --seq_type lfsr_31
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
import argparse
from tqdm import tqdm

# ============ 配置：10x 模型 ============
CONFIG = {
    "seq_length": 10000,
    "context_len": 16,
    "train_ratio": 0.3,
    # 10x 扩大（相对原始小模型）
    "embed_dim": 1024,          # 128 → 1024 (8x)
    "num_heads": 16,            # 4 → 16 (4x)
    "num_layers": 4,            # 2 → 4 (2x)
    # 其他
    "lr": 5e-4,                 # 大模型学习率稍低
    "weight_decay": 0.5,        # 稍微降低，防止欠拟合
    "batch_size": 128,          # 显存限制
    "total_steps": 100000,
    "eval_every": 500,
    "seed": 42,
}


# ============ 序列生成器 ============
class SequenceGenerator:
    @staticmethod
    def lcg_glibc(length, seed=42):
        seq = np.zeros(length, dtype=np.int64)
        x = seed
        for i in range(length):
            x = (1103515245 * x + 12345) % (2**31)
            seq[i] = (x >> 16) & 0xFF
        return seq

    @staticmethod
    def lfsr_31(length, seed=42):
        state = seed if seed != 0 else 1
        seq = np.zeros(length, dtype=np.int64)
        for i in range(length):
            bit = ((state >> 30) ^ (state >> 2)) & 1
            state = ((state << 1) | bit) & 0x7FFFFFFF
            seq[i] = state & 0xFF
        return seq


def generate_dataset(seq_type, seq_length, context_len, train_ratio, seed=42):
    generator = getattr(SequenceGenerator, seq_type)
    sequence = generator(seq_length, seed)

    X, y = [], []
    for i in range(len(sequence) - context_len):
        X.append(sequence[i:i+context_len])
        y.append(sequence[i+context_len])

    X, y = np.array(X), np.array(y)

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    n_train = int(len(X) * train_ratio)

    return (
        torch.tensor(X[indices[:n_train]], dtype=torch.long),
        torch.tensor(y[indices[:n_train]], dtype=torch.long),
        torch.tensor(X[indices[n_train:]], dtype=torch.long),
        torch.tensor(y[indices[n_train:]], dtype=torch.long),
    )


# ============ 10x 模型 ============
class XLargeSequenceTransformer(nn.Module):
    def __init__(self, vocab_size, context_len, embed_dim, num_heads, num_layers):
        super().__init__()
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

    def forward(self, x):
        emb = self.embed(x) + self.pos_embed.unsqueeze(0)
        h = self.transformer(emb)
        return self.output(h[:, -1, :])


# ============ 训练 ============
def train(seq_type, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sequence: {seq_type}")
    print(f"Model: embed_dim={CONFIG['embed_dim']}, layers={CONFIG['num_layers']}, heads={CONFIG['num_heads']}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    exp_dir = os.path.join(output_dir, f"{seq_type}_xlarge")
    os.makedirs(exp_dir, exist_ok=True)

    vocab_size = 256
    train_X, train_y, test_X, test_y = generate_dataset(
        seq_type, CONFIG["seq_length"], CONFIG["context_len"],
        CONFIG["train_ratio"], CONFIG["seed"]
    )

    train_loader = DataLoader(TensorDataset(train_X, train_y),
                             batch_size=CONFIG["batch_size"], shuffle=True)
    test_loader = DataLoader(TensorDataset(test_X, test_y),
                            batch_size=CONFIG["batch_size"], shuffle=False)

    print(f"Train: {len(train_X)}, Test: {len(test_X)}")

    model = XLargeSequenceTransformer(
        vocab_size=vocab_size,
        context_len=CONFIG["context_len"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Random baseline: {1/vocab_size:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    log = {"steps": [], "train_loss": [], "train_acc": [], "test_acc": []}

    step = 0
    pbar = tqdm(total=CONFIG["total_steps"], desc=f"Training {seq_type} (xlarge)")

    while step < CONFIG["total_steps"]:
        for batch_x, batch_y in train_loader:
            if step >= CONFIG["total_steps"]:
                break

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % CONFIG["eval_every"] == 0:
                model.eval()

                train_correct, train_total = 0, 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(bx).argmax(dim=1)
                    train_correct += (pred == by).sum().item()
                    train_total += len(by)
                train_acc = train_correct / train_total

                test_correct, test_total = 0, 0
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

                pbar.set_postfix({"loss": f"{loss.item():.4f}", "train": f"{train_acc:.3f}", "test": f"{test_acc:.3f}"})
                model.train()

    pbar.close()

    with open(os.path.join(exp_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    config_save = CONFIG.copy()
    config_save["seq_type"] = seq_type
    config_save["num_params"] = num_params
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pt"))

    print(f"\n{'='*50}")
    print(f"Sequence: {seq_type} (XLARGE MODEL - 10x)")
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    print(f"Final train acc: {log['train_acc'][-1]:.4f}")
    print(f"Final test acc: {log['test_acc'][-1]:.4f}")
    print(f"Random baseline: {1/vocab_size:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_type", type=str, required=True, choices=["lcg_glibc", "lfsr_31"])
    parser.add_argument("--output_dir", type=str, default="/workspace/ai-theorys-study/arxiv/wechat62/results")
    args = parser.parse_args()

    train(args.seq_type, args.output_dir)
