"""
Benchmark for the Triton kernel.
Runs 1000 iterations and reports median latency.
"""

import torch
import time
import numpy as np
from kernel import fused_linear_relu

torch.manual_seed(42)

# Test size: 1M elements
N = 1_000_000
x = torch.randn(N, device='cuda', dtype=torch.float32)
w = torch.randn(N, device='cuda', dtype=torch.float32)
b = torch.randn(N, device='cuda', dtype=torch.float32)

# Reference (PyTorch)
ref = torch.relu(x * w + b)

# Warmup
for _ in range(10):
    out = fused_linear_relu(x, w, b)
torch.cuda.synchronize()

# Correctness check
out = fused_linear_relu(x, w, b)
torch.cuda.synchronize()
max_error = (out - ref).abs().max().item()

# Benchmark
N_ITERS = 1000
latencies = []
for _ in range(N_ITERS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fused_linear_relu(x, w, b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1e6)  # microseconds

latencies = np.array(latencies)
kernel_latency_us = np.median(latencies)
bandwidth_gb_s = (N * 4 * 4) / (kernel_latency_us * 1e-6) / 1e9  # 4 tensors * 4 bytes

# Print metrics in key_value format
print("---")
print(f"kernel_latency_us: {kernel_latency_us:.6f}")
print(f"bandwidth_gb_s: {bandwidth_gb_s:.6f}")
print(f"max_error: {max_error:.10f}")
