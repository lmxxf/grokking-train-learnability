"""
LFSR-31 上下文窗口扩展实验：context_len 16 → 64

假设：
  - LFSR-31 内部状态 31 bit
  - context_len=16 时，每步移 1 位，相邻步 7 bit 重叠，独立信息 ~24 bit < 31 bit → 50% 天花板
  - context_len=32 时，独立信息 ~71 bit > 31 bit → 应能突破 50%，接近 ~100%

基于 train_learnability.py，只跑 lfsr_31，context_len=32。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import json
from tqdm import tqdm

# ============ 配置 ============
CONFIG = {
    "seq_length": 10000,
    "context_len": 64,              # 16 → 64
    "train_ratio": 0.3,
    "embed_dim": 128,
    "num_heads": 4,
    "num_layers": 2,
    "lr": 1e-3,
    "weight_decay": 1.0,
    "batch_size": 512,
    "total_steps": 100000,
    "eval_every": 500,
    "save_activations_every": 2000,
    "seed": 42,
}


# ============ 序列生成 ============
def generate_lfsr31(length, seed=42):
    """31位LFSR：x^31 + x^3 + 1"""
    state = seed if seed != 0 else 1
    seq = np.zeros(length, dtype=np.int64)
    for i in range(length):
        bit = ((state >> 30) ^ (state >> 2)) & 1
        state = ((state << 1) | bit) & 0x7FFFFFFF
        seq[i] = state & 0xFF  # 取低8位
    return seq


# ============ 数据准备 ============
def generate_dataset(seq_length, context_len, train_ratio, seed=42):
    sequence = generate_lfsr31(seq_length, seed)

    X = []
    y = []
    for i in range(len(sequence) - context_len):
        X.append(sequence[i:i+context_len])
        y.append(sequence[i+context_len])

    X = np.array(X)
    y = np.array(y)

    np.random.seed(seed)
    indices = np.random.permutation(len(X))
    n_train = int(len(X) * train_ratio)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return (
        torch.tensor(X[train_idx], dtype=torch.long),
        torch.tensor(y[train_idx], dtype=torch.long),
        torch.tensor(X[test_idx], dtype=torch.long),
        torch.tensor(y[test_idx], dtype=torch.long),
    )


# ============ 模型 ============
class SequenceTransformer(nn.Module):
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
        self.last_hidden = None

    def forward(self, x, save_hidden=False):
        emb = self.embed(x) + self.pos_embed.unsqueeze(0)
        h = self.transformer(emb)
        if save_hidden:
            self.last_hidden = h[:, -1, :].detach().cpu()
        logits = self.output(h[:, -1, :])
        return logits


# ============ 训练 ============
def train(output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment: lfsr_31 with context_len={CONFIG['context_len']}")

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    exp_dir = os.path.join(output_dir, "lfsr_31_ctx64")
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "activations"), exist_ok=True)

    vocab_size = 256
    train_X, train_y, test_X, test_y = generate_dataset(
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

    model = SequenceTransformer(
        vocab_size=vocab_size,
        context_len=CONFIG["context_len"],
        embed_dim=CONFIG["embed_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"]
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,} ({param_count/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    log = {
        "steps": [],
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    step = 0
    pbar = tqdm(total=CONFIG["total_steps"], desc="Training lfsr_31 ctx64")

    while step < CONFIG["total_steps"]:
        for batch_x, batch_y in train_loader:
            if step >= CONFIG["total_steps"]:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            pbar.update(1)

            if step % CONFIG["eval_every"] == 0:
                model.eval()

                train_correct = 0
                train_total = 0
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    with torch.no_grad():
                        pred = model(bx).argmax(dim=1)
                    train_correct += (pred == by).sum().item()
                    train_total += len(by)
                train_acc = train_correct / train_total

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

    with open(os.path.join(exp_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    config_save = CONFIG.copy()
    config_save["seq_type"] = "lfsr_31"
    config_save["vocab_size"] = vocab_size
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config_save, f, indent=2)

    torch.save(model.state_dict(), os.path.join(exp_dir, "model_final.pt"))

    print(f"\n{'='*50}")
    print(f"Experiment: lfsr_31 context_len=32")
    print(f"Final train acc: {log['train_acc'][-1]:.4f}")
    print(f"Final test acc: {log['test_acc'][-1]:.4f}")
    print(f"Random baseline: {1/vocab_size:.4f}")
    print(f"Results saved to: {exp_dir}")

    return log


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/ai-theorys-study/arxiv/wechat62/results",
                        help="输出目录")
    args = parser.parse_args()
    train(args.output_dir)
