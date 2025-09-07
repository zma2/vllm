# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import triton
import triton.language as tl

from vllm.platforms import current_platform


@triton.jit
def _compute_probs_kernel(
    probs_ptr,
    seeds_ptr,
    pos_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    seed = tl.load(seeds_ptr + req_idx)
    pos = tl.load(pos_ptr + req_idx)

    block_id = tl.program_id(1)
    r_offset = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    q = tl.rand(seed + pos, r_offset)

    # NOTE(woosuk): This logic makes sure q is not 0.
    RMAX = 0.9999999403953552
    RMAX_LOG = -5.960464477539063e-08
    q = tl.where(q >= RMAX, RMAX_LOG, tl.math.log(q))
    q = -1.0 * q

    p = tl.load(probs_ptr + req_idx * vocab_size + r_offset,
                mask=r_offset < vocab_size)
    p = p / q

    tl.store(probs_ptr + req_idx * vocab_size + r_offset,
             p,
             mask=r_offset < vocab_size)


def gumbel_sample_triton(
    # fp32[num_reqs, vocab_size]
    probs: torch.Tensor,
    # int64[num_reqs]
    seeds: torch.Tensor,
    # int64[num_reqs]
    pos: torch.Tensor,
) -> torch.Tensor:
    assert probs.is_contiguous()
    assert seeds.is_contiguous()
    assert pos.is_contiguous()

    num_reqs = probs.shape[0]
    vocab_size = probs.shape[1]
    BLOCK_SIZE = 8192
    _compute_probs_kernel[(num_reqs, triton.cdiv(vocab_size, BLOCK_SIZE))](
        probs,
        seeds,
        pos,
        vocab_size,
        BLOCK_SIZE,
    )
    return probs.argmax(dim=-1).view(-1)


def gumbel_sample_native(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div_(q).argmax(dim=-1).view(-1)


def gumbel_sample(
    probs: torch.Tensor,
    seeds: torch.Tensor,
    pos: torch.Tensor,
) -> torch.Tensor:
    if current_platform.is_cuda_alike() or current_platform.is_xpu():
        return gumbel_sample_triton(probs, seeds, pos)
    else:
        # The CPU backend does not support per-request seeds.
        return gumbel_sample_native(probs)
