# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def scaled_mm_kernel(a_ptr, b_ptr, scale_a_ptr, scale_b_ptr, c_ptr, bias_ptr,
                     M, N, K, stride_am, stride_ak, stride_bk, stride_bn,
                     stride_cm, stride_cn, ACCUMULATOR_DTYPE: tl.constexpr,
                     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                     BLOCK_SIZE_K: tl.constexpr,
                     BLOCK_SIZE_SCALE_A: tl.constexpr,
                     BLOCK_SIZE_SCALE_B: tl.constexpr):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N),
                           dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = (stride_am * offsets_am[:, None] +
                 stride_ak * offsets_k[None, :])
    offsets_b = (stride_bk * offsets_k[:, None] +
                 stride_bn * offsets_bn[None, :])

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (tl.arange(0, BLOCK_SIZE_SCALE_A) +
                        (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M)
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (tl.arange(0, BLOCK_SIZE_SCALE_B) +
                        (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N)
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = (c_ptr + stride_cm * offs_cm[:, None] +
              stride_cn * offs_cn[None, :])
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


@lru_cache
def _load_configs(N: int, K: int, dtype: str):
    # lookup pre-tuned config
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_filename = f"N={N},K={K},device_name={device_name},dtype={dtype}_w8a8.json"  # noqa: E501
    config_filepath = Path(__file__).parent / "triton_configs" / json_filename

    if not config_filepath.exists():
        logger.warning(
            "Using default W8A8 kernel config. Performance might "
            "be sub-optimal! Config file not found at %s", config_filepath)
        return None

    logger.info("Using configuration from %s for W8A8 Block FP8 kernel.",
                config_filepath)
    with open(config_filepath) as f:
        # return sorted key-value pair
        return sorted((int(k), v) for k, v in json.load(f).items())


def _select_config(M: int, N: int, K: int, dtype: str):
    # for small M, use pre-defined config.
    # it's bandwidth-bound, so tuned config is unnecessary
    if M <= 16:
        return dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256)
    if M <= 32:
        return dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=64, BLOCK_SIZE_K=256)

    configs = _load_configs(N, K, dtype)

    # no tuned config found. use heuristics
    if configs is None:
        if M <= 128:
            return dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128)
        else:
            return dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128)

    # smallest key that is >= M
    for k, v in configs:
        if k >= M:
            return v

    # otherwise, use the last config (largest key)
    _, v = configs[-1]
    return v


# input  - [M, K]
# weight - [K, N]
def triton_scaled_mm(input: torch.Tensor,
                     weight: torch.Tensor,
                     scale_a: torch.Tensor,
                     scale_b: torch.Tensor,
                     out_dtype: type[torch.dtype],
                     bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1
                                      or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1
                                      or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(
        N, META['BLOCK_SIZE_N']), )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    if input.is_floating_point():
        config = _select_config(M, N, K, "fp8")
        accumulator_dtype = tl.float32
    else:
        config = _select_config(M, N, K, "int8")
        accumulator_dtype = tl.int32

    block_size_sa = 1 if scale_a.numel() == 1 else config["BLOCK_SIZE_M"]
    block_size_sb = 1 if scale_b.numel() == 1 else config["BLOCK_SIZE_N"]

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](input,
                           weight,
                           scale_a,
                           scale_b,
                           result,
                           bias,
                           M,
                           N,
                           K,
                           input.stride(0),
                           input.stride(1),
                           weight.stride(0),
                           weight.stride(1),
                           result.stride(0),
                           result.stride(1),
                           accumulator_dtype,
                           BLOCK_SIZE_SCALE_A=block_size_sa,
                           BLOCK_SIZE_SCALE_B=block_size_sb,
                           **config)

    return result.to(out_dtype)
