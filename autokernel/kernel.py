"""
Triton GPU kernel for fused element-wise operations.
Modify this file to minimize kernel_latency_us.

The kernel performs: output = relu(x * w + b) where x, w, b are vectors.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_relu_kernel(
    x_ptr, w_ptr, b_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # Fused multiply-add + relu
    result = x * w + b
    result = tl.maximum(result, 0.0)

    tl.store(output_ptr + offsets, result, mask=mask)


def fused_linear_relu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Launch the Triton kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_linear_relu_kernel[grid](x, w, b, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
