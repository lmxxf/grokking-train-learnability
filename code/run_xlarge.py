#!/usr/bin/env python3
"""
第三阶段：10x 大模型测试
一次性跑完 lcg_glibc 和 lfsr_31
"""

import subprocess
import sys

sequences = ["lcg_glibc", "lfsr_31"]

for seq in sequences:
    print(f"\n{'='*60}")
    print(f"Running: {seq} (xlarge)")
    print('='*60)
    subprocess.run([sys.executable, "train_xlarge_model.py", "--seq_type", seq])

print("\n" + "="*60)
print("All done!")
print("="*60)
