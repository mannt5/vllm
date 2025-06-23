# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.layers.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
    BatchedPrepareAndFinalize, BatchedTritonExperts, NaiveBatchedExperts)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEModularKernel)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8)
from vllm.utils import round_up


def triton_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    quant_type: Optional[torch.dtype] = None,
    per_act_token_quant=False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale,
                         per_channel_quant=per_act_token_quant,
                         use_fp8_w8a8=quant_type == torch.float8_e4m3fn,
                         block_shape=block_shape)


def batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    qtype: Optional[torch.dtype] = None,
    per_act_token: bool = False,
    block_shape: Optional[list[int]] = None,
) -> torch.Tensor:
    max_num_tokens = round_up(a.shape[0], 64)

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(max_num_tokens,
                                  world_size=1,
                                  dp_size=1,
                                  rank=0),
        BatchedTritonExperts(max_num_tokens=max_num_tokens,
                             world_size=1,
                             dp_size=1,
                             FusedMoEQuantConfig.make(
                                 use_fp8_w8a8=qtype == torch.float8_e4m3fn,
                                 per_act_token_quant=per_act_token,
                                 block_shape=block_shape),
                             )

    return fused_experts(a,
                         w1,
                         w2,
                         topk_weight,
                         topk_ids,
                         w1_scale=w1_scale,
                         w2_scale=w2_scale)


def naive_batched_moe(
    a: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    num_experts = w1.shape[0]

    fused_experts = FusedMoEModularKernel(
        BatchedPrepareAndFinalize(a.shape[0], world_size=1, dp_size=1, rank=0),
        NaiveBatchedExperts(max_num_tokens=a.shape[0], dp_size=1,
                            world_size=1))

    return fused_experts(a, w1, w2, topk_weight, topk_ids, num_experts)


def per_block_cast_to_fp8(
        x: torch.Tensor,
        block_size_n: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.utils import cdiv
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (cdiv(m, 128) * 128, cdiv(n, block_size_n) * block_size_n),
        dtype=x.dtype,
        device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, block_size_n)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    x_scaled_sub = x_scaled.view_as(x_padded)[:m, :n].contiguous()
    scales = (x_amax / 448.0).view(x_view.size(0), x_view.size(2))
    return x_scaled_sub, scales


def chunk_scales(scales: Optional[torch.Tensor], start: int,
                 end: int) -> Optional[torch.Tensor]:
    if scales is not None:
        if scales.numel() == 1:
            return scales
        else:
            return scales[start:end]
    return None


def make_quantized_test_activations(
    E: int,
    m: int,
    k: int,
    in_dtype: torch.dtype,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
    per_act_token_quant: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    assert not per_act_token_quant, "NYI"

    a = torch.randn((E, m, k), device="cuda", dtype=in_dtype) / 10
    a_q = a
    a_scale = None

    if quant_dtype is not None:
        assert quant_dtype == torch.float8_e4m3fn, "only fp8 supported"
        a_q = torch.zeros_like(a, dtype=quant_dtype)
        a_scale = [None] * E
        for e in range(E):
            if block_shape is not None:
                a_q[e], a_scale[e] = per_token_group_quant_fp8(
                    a[e], block_shape[1])
            else:
                a_tmp, a_scale[e] = per_token_group_quant_fp8(
                    a[e].view(1, -1), a[e].numel())
                a_q[e] = a_tmp.view(*a[e].shape)
        a_scale = torch.stack(a_scale)

    return a, a_q, a_scale


def make_test_weights(
    e: int,
    n: int,
    k: int,
    in_dtype: torch.dtype = torch.bfloat16,
    quant_dtype: Optional[torch.dtype] = None,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor,
           torch.Tensor, Optional[torch.Tensor]]:
    w1_16 = torch.randn((e, 2 * n, k), device="cuda", dtype=in_dtype) / 15
    w2_16 = torch.randn((e, k, n), device="cuda", dtype=in_dtype) / 15

    if quant_dtype is not None:
        assert quant_dtype == torch.float8_e4m3fn, "only fp8 supported"
        w1_l = [None] * e
        w2_l = [None] * e
        w1_s_l = [None] * e
        w2_s_l = [None] * e
        for idx in range(e):
            if block_shape is not None:
                w1_l[idx], w1_s_l[idx] = per_block_cast_to_fp8(
                    w1_16[idx],
                    block_shape[1],
                )
                w2_l[idx], w2_s_l[idx] = per_block_cast_to_fp8(
                    w2_16[idx],
                    block_shape[1],
                )
            else:
                tmp, w1_s_l[idx] = per_token_group_quant_fp8(
                    w1_16[idx].view(1, -1), w1_16[idx].numel())
                w1_l[idx] = tmp.view(*w1_16[idx].shape)

                tmp, w2_s_l[idx] = per_token_group_quant_fp8(
                    w2_16[idx].view(1, -1), w2_16[idx].numel())
                w2_l[idx] = tmp.view(*w2_16[idx].shape)

        w1 = torch.stack(w1_l)
        w2 = torch.stack(w2_l)
        w1_s = torch.stack(w1_s_l)
        w2_s = torch.stack(w2_s_l)
        if w1_s.ndim == 2:
            assert w1_s.shape[-1] == 1
            w1_s = w1_s.view(-1, 1, 1)
            w2_s = w2_s.view(-1, 1, 1)

        if block_shape is not None:
            block_n, block_k = block_shape
            n_tiles_w1 = ((2 * n) + block_n - 1) // block_n
            k_tiles_w1 = (k + block_k - 1) // block_k
            n_tiles_w2 = (k + block_n - 1) // block_n
            k_tiles_w2 = (n + block_k - 1) // block_k
            assert w1_s.shape == (e, n_tiles_w1, k_tiles_w1)
            assert w2_s.shape == (e, n_tiles_w2, k_tiles_w2)
    else:
        w1 = w1_16
        w2 = w2_16
        w1_s = None
        w2_s = None

    return w1_16, w1, w1_s, w2_16, w2, w2_s
