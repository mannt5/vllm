# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from .cutlass import CutlassScaledMMLinearKernel
from .ScaledMMLinearKernel import ScaledMMLinearLayerConfig

logger = init_logger(__name__)


class TritonScaledMMLinearKernel(CutlassScaledMMLinearKernel):

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def can_implement(
            cls, c: ScaledMMLinearLayerConfig) -> tuple[bool, Optional[str]]:
        if current_platform.is_cpu():
            return (
                False,
                "TritonScaledMMLinearKernel requires Triton which is not " +
                "currently supported on CPU.")
        if not c.input_symmetric:
            return (False,
                    "TritonScaledMMLinearKernel only supports symmetric " +
                    "quantization.")
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        w_q, w_s, i_s, i_zp, _ = self._get_weight_params(layer)

        # ops.scaled_int8_quant supports both dynamic and static quant:
        # * dynamic, i_s is None and x_s computed from x.
        # * static, i_s is scalar and x_s is i_s.
        # symmetric = azp_adj is None
        x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(),
                                            i_s,
                                            i_zp,
                                            symmetric=True)

        return triton_scaled_mm(x_q, w_q, x_s, w_s, x.dtype)


@triton.jit
def _triton_scaled_mm_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    A_scale_ptr,
    B_scale_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACC_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    GROUP_M: tl.constexpr = 8

    # based on triton.ops.matmul
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A_ptr + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    for k in range(K, 0, -BLOCK_K):
        a = tl.load(A, mask=rk[None, :] < k, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k, other=0.0)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    idx_m = rm[:, None]
    idx_n = rn[None, :]
    mask = (idx_m < M) & (idx_n < N)

    row_scale = tl.load(A_scale_ptr + idx_m, mask=idx_m < M).to(tl.float32)
    col_scale = tl.load(B_scale_ptr + idx_n, mask=idx_n < N).to(tl.float32)
    acc = acc.to(tl.float32) * row_scale * col_scale

    xindex = idx_m * stride_cm + idx_n * stride_cn
    tl.store(C_ptr + tl.broadcast_to(xindex, mask.shape), acc, mask)


@lru_cache
def _load_configs(N: int, K: int):
    # lookup pre-tuned config
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_filename = f"N={N},K={K},device_name={device_name},dtype=int8_w8a8.json"  # noqa: E501
    config_filepath = Path(__file__).parent / "configs" / json_filename

    if not config_filepath.exists():
        logger.warning(
            "Using default W8A8 INT8 kernel config. Performance might "
            "be sub-optimal! Config file not found at %s", config_filepath)
        return None

    logger.info("Using configuration from %s for W8A8 Block FP8 kernel.",
                config_filepath)
    with open(config_filepath) as f:
        # return sorted key-value pair
        return sorted((int(k), v) for k, v in json.load(f).items())


def _select_config(M: int, N: int, K: int):
    # for small M, use pre-defined config
    if M <= 16:
        return dict(BLOCK_M=16,
                    BLOCK_N=64,
                    BLOCK_K=128,
                    num_stages=4,
                    num_warps=4)
    if M <= 32:
        return dict(BLOCK_M=32,
                    BLOCK_N=64,
                    BLOCK_K=128,
                    num_stages=4,
                    num_warps=4)
    if M <= 64:
        return dict(BLOCK_M=64,
                    BLOCK_N=64,
                    BLOCK_K=128,
                    num_stages=4,
                    num_warps=4)

    configs = _load_configs(N, K)
    if configs is None:
        return dict(BLOCK_M=64,
                    BLOCK_N=128,
                    BLOCK_K=128,
                    num_stages=4,
                    num_warps=4)

    # smallest key that is >= M
    for k, v in configs:
        if k >= M:
            return v

    # otherwise, use the last config (largest key)
    _, v = configs[-1]
    return v


def triton_scaled_mm(A: torch.Tensor, B: torch.Tensor, scale_A: torch.Tensor,
                     scale_B: torch.Tensor,
                     out_dtype: torch.dtype) -> torch.Tensor:
    assert (A.dtype == B.dtype == torch.int8)
    M, K = A.shape
    _, N = B.shape

    C = torch.empty(M, N, device=A.device, dtype=out_dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(
        N, META["BLOCK_N"]), )
    config = _select_config(M, N, K)
    _triton_scaled_mm_kernel[grid](
        A,
        B,
        C,
        scale_A,
        scale_B,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        ACC_DTYPE=tl.int32,
        **config,
    )
    return C
