# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Literal, Optional

import numpy as np
import torch
import triton
import triton.language as tl

from vllm import _custom_ops as ops
from vllm.attention.backends.utils import PAD_SLOT_ID

MAX_NUM_PROGRAMS = 1024

batch_ptr = torch.full(
    (MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32,
    device='cuda')  # tracking which seq-idx the Triton program is handling
token_chunk_offset_ptr = torch.full(
    (MAX_NUM_PROGRAMS, ), PAD_SLOT_ID, dtype=torch.int32, device='cuda'
)  # tracking BLOCK_M-based index in the sequence the Triton program is handling


def causal_conv1d_fn(x: torch.Tensor,
                     weight: torch.Tensor,
                     bias: Optional[torch.Tensor] = None,
                     query_start_loc: Optional[torch.Tensor] = None,
                     cache_indices: Optional[torch.Tensor] = None,
                     has_initial_state: Optional[torch.Tensor] = None,
                     conv_states: Optional[torch.Tensor] = None,
                     activation: Optional[str] = "silu",
                     pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3


    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    ops.causal_conv1d_fwd(x, weight, bias, conv_states, query_start_loc,
                          cache_indices, has_initial_state, activation
                          in ["silu", "swish"], pad_slot_id)
    return x


def causal_conv1d_update(x: torch.Tensor,
                         conv_state: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         activation: Optional[str] = None,
                         cache_seqlens: Optional[torch.Tensor] = None,
                         conv_state_indices: Optional[torch.Tensor] = None,
                         pad_slot_id: int = PAD_SLOT_ID):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    ops.causal_conv1d_update(x, conv_state, weight, bias, activation_val,
                             cache_seqlens, conv_state_indices, pad_slot_id)
    if unsqueeze:
        x = x.squeeze(-1)
    return x


@triton.jit()
def _causal_conv1d_update_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    cache_seqlens_ptr,  # circular buffer
    conv_state_indices_ptr,
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.
    constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_CONTINUOUS_BATCHING:
        # mask = idx_seq < batch
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq)
    else:
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    # STEP 1: READ init_state data
    conv_states_base = (conv_state_ptr +
                        (conv_state_batch_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))
    mask_w = idx_feats < dim

    prior_tokens = conv_states_base
    if KERNEL_WIDTH >= 2:
        conv_states_ptrs = prior_tokens  # [BLOCK_N]
        col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 3:
        conv_states_ptrs = prior_tokens + 1 * stride_conv_state_tok  # [BLOCK_N]
        col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH >= 4:
        conv_states_ptrs = prior_tokens + 2 * stride_conv_state_tok  # [BLOCK_N]
        col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
    if KERNEL_WIDTH == 5:
        conv_states_ptrs = prior_tokens + 3 * stride_conv_state_tok  # [BLOCK_N]
        col3 = tl.load(conv_states_ptrs, mask_w, 0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    conv_state_ptrs_source = (
        conv_state_ptr + (conv_state_batch_coord * stride_conv_state_seq) +
        (idx_feats * stride_conv_state_dim)[None, :] +
        ((idx_tokens + seqlen) * stride_conv_state_tok)[:, None]
    )  # [BLOCK_M, BLOCK_N]
    mask = ((conv_state_batch_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :])
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + (idx_seq * stride_x_seq) + (idx_feats * stride_x_dim
                                                 )  # [BLOCK_N]

    x_ptrs = x_base[None, :] + (
        (idx_tokens - VAL) * stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

    mask_x = ((idx_tokens - VAL >= 0)[:, None] &
              (idx_tokens - VAL < seqlen)[:, None] & (idx_feats < dim)[None, :]
              )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    conv_state_base = (conv_state_ptr +
                       (conv_state_batch_coord * stride_conv_state_seq) +
                       (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]
    conv_state_ptrs_target = conv_state_base + (
        idx_tokens * stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)  # ACTUAL MUST USE

    # STEP 3: init accumulator
    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # STEP 4:
    # PRE-LOAD WEIGHTS
    # first kernel column, configured for weights to handle BLOCK_N features in range
    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)

    x_base_1d = x_base  # starting of chunk [BLOCK_N]
    mask_x_1d = idx_feats < dim

    # STEP 5: compute each token
    for idx_token in tl.static_range(seqlen):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(
                KERNEL_WIDTH):  # <-- RESTORE THIS AFTER DEBUG3
            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            # TODO: add more if KERNEL_WIDTH>4

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < seqlen) & (idx_feats < dim
                                          )  # token-index  # feature-index
        o_ptrs = o_ptr + (
            idx_seq) * stride_o_seq + idx_token * stride_o_token + (
                idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_update_triton(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[Literal["silu", "swish"]] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """
    x: (batch, dim) or (batch, dim, seqlen)
        [shape=2: single token prediction]
        [shape=3: single or multiple tokens prediction]
    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen)
    """
    if validate_data:
        assert cache_seqlens is None  # not implemented yet - ok for vLLM
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        if activation is True:
            activation = "silu"
        else:
            activation = None
    unsqueeze = x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    batch, dim, seqlen = x.shape
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(
            -2
        ) == 1, f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch, ) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this
        assert cache_seqlens is None  # FOR NOW (not needed for vLLM) - circular buffer # fmt:off

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    if metadata is not None:
        stride_w_dim = metadata.stride_w_dim
        stride_w_width = metadata.stride_w_width
        stride_x_seq = metadata.stride_x_seq
        stride_x_dim = metadata.stride_x_dim
        stride_x_token = metadata.stride_x_token
        stride_o_seq = metadata.stride_x_seq
        stride_o_dim = metadata.stride_x_dim
        stride_o_token = metadata.stride_x_token
        stride_istate_seq = metadata.stride_istate_seq
        stride_istate_dim = metadata.stride_istate_dim
        stride_istate_token = metadata.stride_istate_token
        np2_statelen = metadata.np2_statelen
    else:
        stride_w_dim = weight.stride(0)
        stride_w_width = weight.stride(1)

        stride_x_seq, stride_x_dim, stride_x_token = x.stride(
        )  # X (batch, dim, seqlen)

        stride_o_seq, stride_o_dim, stride_o_token = out.stride()

        stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride(
        )
        state_len = width - 1
        np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        cache_seqlens,
        conv_state_indices,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_CONTINUOUS_BATCHING=conv_state_indices is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,  # better
        #BLOCK_N=64,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out


@triton.jit()
def _causal_conv1d_fwd_kernel_contbatch(  # continuous batching
    # Pointers to matrices
    x_ptr,  # (dim, cu_seqlen) holding `batch` of actual sequences + padded sequences
    w_ptr,  # (dim, width)
    bias_ptr,
    initial_states_ptr,  # conv_states_ptr
    cache_indices_ptr,  # conv_state_indices_ptr
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    o_ptr,  # (dim, seqlen) - actually pointing to x_ptr
    # Matrix dimensions
    batch: tl.int32,  # actually padded_batch
    dim: tl.constexpr,
    seqlen: tl.int32,  # cu_seqlen
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    # marking everything as many as possible as constant aexcept tl.constexpr
    # TODO: label them as tl.constexpr
    stride_x_seq: tl.constexpr,  # stride to get to next sequence,
    stride_x_dim: tl.constexpr,  # stride to get to next feature-value,
    stride_x_token: tl.
    constexpr,  # stride to get to next token (same feature-index, same sequence-index)
    stride_w_dim: tl.constexpr,  # stride to get to next dim-axis value
    stride_w_width: tl.constexpr,  # stride to get to next width-axis value
    stride_istate_seq: tl.constexpr,
    stride_istate_dim: tl.constexpr,
    stride_istate_token: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,  # maybe using this we don't need 'width'
    SILU_ACTIVATION: tl.constexpr,
    HAS_INITIAL_STATES: tl.constexpr,
    HAS_CACHE: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    DECODE_SEQLEN: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    conv_states_ptr = initial_states_ptr
    conv_state_indices_ptr = cache_indices_ptr
    stride_conv_state_seq = stride_istate_seq
    stride_conv_state_dim = stride_istate_dim
    stride_conv_state_tok = stride_istate_token
    state_len = KERNEL_WIDTH - 1  # can be passed via argument if it's not the same as this value

    # one program handles one chunk in a single sequence
    # rather than mixing sequences - to make updating initial_states across sequences efficiently

    # single-sequence id
    idx_seq = tl.load(batch_ptr + tl.program_id(0))
    chunk_offset = tl.load(token_chunk_offset_ptr + tl.program_id(0))

    # BLOCK_N elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if idx_seq == pad_slot_id:
        return

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    # find the actual sequence length
    seqlen = sequence_end_index - sequence_start_index

    token_offset = BLOCK_M * chunk_offset
    segment_len = min(BLOCK_M, seqlen - token_offset)

    # base of the sequence
    x_base = x_ptr + sequence_start_index * stride_x_token + idx_feats * stride_x_dim  # [BLOCK_N,]

    if IS_CONTINUOUS_BATCHING:
        # cache_idx
        conv_state_batch_coord = tl.load(conv_state_indices_ptr + idx_seq)
    else:
        # cache_idx
        conv_state_batch_coord = idx_seq
    if USE_PAD_SLOT:
        if conv_state_batch_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return
    conv_states_base = (conv_states_ptr +
                        (conv_state_batch_coord * stride_conv_state_seq) +
                        (idx_feats * stride_conv_state_dim))  # [BLOCK_N,]

    w_base = w_ptr + (idx_feats * stride_w_dim)  # [BLOCK_N,]

    # Does 2 things:
    # 1. READ prior-block init-state data - [done by every Triton programs]
    # 2. update conv_state with new data [only by the Triton program handles chunk_offset=0]
    if chunk_offset == 0:
        # read from conv_states
        load_init_state = False
        if HAS_INITIAL_STATES:  # the new HAS_INITIAL_STATES
            load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(
                tl.int1)
        if load_init_state:
            # load from conv_states
            prior_tokens = conv_states_base + (state_len -
                                               1) * stride_conv_state_tok
            mask_w = idx_feats < dim
            if KERNEL_WIDTH == 2:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 3:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 4:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
            if KERNEL_WIDTH == 5:
                conv_states_ptrs = prior_tokens  # [BLOCK_N]
                col3 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 1 * stride_conv_state_tok  # [BLOCK_N]
                col2 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 2 * stride_conv_state_tok  # [BLOCK_N]
                col1 = tl.load(conv_states_ptrs, mask_w, 0.0)
                conv_states_ptrs = prior_tokens - 3 * stride_conv_state_tok  # [BLOCK_N]
                col0 = tl.load(conv_states_ptrs, mask_w, 0.0)
        else:
            # prior-tokens are zeros
            if KERNEL_WIDTH >= 2:  # STRATEGY1
                # first chunk and does not have prior-token, so just set to 0
                col0 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 3:  # STRATEGY1
                col1 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 4:  # STRATEGY1
                col2 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)
            if KERNEL_WIDTH >= 5:  # STRATEGY1
                col3 = tl.zeros((BLOCK_N, ), dtype=x_ptr.dtype.element_ty)

        # STEP 2:
        # here prepare data for updating conv_state
        if state_len <= seqlen:  # SMALL_CACHE=True (only move part of 'x' into conv_state cache)
            # just read from 'x'
            # copy 'x' data to conv_state
            # load only 'x' data (and set 0 before 'x' if seqlen < state_len)
            idx_tokens_last = (seqlen - state_len) + tl.arange(
                0, NP2_STATELEN)  # [BLOCK_M]
            x_ptrs = x_ptr + (
                (sequence_start_index + idx_tokens_last) *
                stride_x_token)[:, None] + (
                    idx_feats * stride_x_dim)[None, :]  # [BLOCK_M,BLOCK_N,]
            mask_x = ((idx_tokens_last >= 0)[:, None] &
                      (idx_tokens_last < seqlen)[:, None] &
                      (idx_feats < dim)[None, :]
                      )  # token-index  # token-index  # feature-index
            loaded_x = tl.load(x_ptrs, mask_x, 0.0)
            new_conv_state = tl.load(x_ptrs, mask_x, 0.0)
            idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]
            conv_states_ptrs_target = conv_states_base[None, :] + (
                idx_tokens_conv * stride_conv_state_tok)[:, None]

            mask = (idx_tokens_conv < state_len)[:, None] & (idx_feats
                                                             < dim)[None, :]
            tl.debug_barrier()  #  NOTE: use this due to bug in Triton compiler
            tl.store(conv_states_ptrs_target, new_conv_state, mask)

        else:
            if load_init_state:
                # update conv_state by shifting left, i.e. take last few cols from conv_state + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                conv_states_ptrs_source = (
                    conv_states_ptr +
                    (conv_state_batch_coord * stride_conv_state_seq) +
                    (idx_feats * stride_conv_state_dim)[None, :] +
                    ((idx_tokens_conv + seqlen) * stride_conv_state_tok)[:,
                                                                         None]
                )  # [BLOCK_M, BLOCK_N]
                mask = ((conv_state_batch_coord < num_cache_lines)
                        & ((idx_tokens_conv + seqlen) < state_len)[:, None]
                        & (idx_feats < dim)[None, :])
                conv_state = tl.load(conv_states_ptrs_source, mask, other=0.0)

                VAL = state_len - seqlen

                x_ptrs = x_base[None, :] + (
                    (idx_tokens_conv - VAL) *
                    stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                          (idx_tokens_conv - VAL < seqlen)[:, None] &
                          (idx_feats < dim)[None, :]
                          )  # token-index  # token-index  # feature-index
                loaded_x = tl.load(x_ptrs, mask_x, 0.0)

                tl.debug_barrier(
                )  # need this due to the bug in tl.where not enforcing this when data is the result of another tl.load
                new_conv_state = tl.where(
                    mask, conv_state, loaded_x
                )  # BUG in 'tl.where'  which requires a barrier before this
                conv_states_ptrs_target = conv_states_base + (
                    idx_tokens_conv *
                    stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv
                        < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)
            else:  # load_init_state == False
                # update conv_state by shifting left, BUT
                # set cols prior to 'x' as zeros + cols from 'x'
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

                VAL = state_len - seqlen

                x_ptrs = x_base[None, :] + (
                    (idx_tokens_conv - VAL) *
                    stride_x_token)[:, None]  # [BLOCK_M, BLOCK_N]

                mask_x = ((idx_tokens_conv - VAL >= 0)[:, None] &
                          (idx_tokens_conv - VAL < seqlen)[:, None] &
                          (idx_feats < dim)[None, :]
                          )  # token-index  # token-index  # feature-index
                new_conv_state = tl.load(x_ptrs, mask_x, 0.0)

                conv_states_ptrs_target = conv_states_base + (
                    idx_tokens_conv *
                    stride_conv_state_tok)[:, None]  # [BLOCK_M, BLOCK_N]
                mask = (idx_tokens_conv
                        < state_len)[:, None] & (idx_feats < dim)[None, :]
                tl.store(conv_states_ptrs_target, new_conv_state, mask)

    else:  # chunk_offset > 0
        # read prior-token data from `x`
        load_init_state = True
        prior_tokens = x_base + (token_offset - 1) * stride_x_token
        mask_w = idx_feats < dim
        if KERNEL_WIDTH == 2:
            # FIXME TUAN - review if we need to add cache_modifier explicitly
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 3:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 4:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
        if KERNEL_WIDTH == 5:
            conv_states_ptrs = prior_tokens  # [BLOCK_N]
            col3 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 1 * stride_x_token  # [BLOCK_N]
            col2 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 2 * stride_x_token  # [BLOCK_N]
            col1 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')
            conv_states_ptrs = prior_tokens - 3 * stride_x_token  # [BLOCK_N]
            col0 = tl.load(conv_states_ptrs, mask_w, 0.0, cache_modifier='.ca')

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        mask_bias = idx_feats < dim
        acc_preload = tl.load(bias, mask=mask_bias,
                              other=0.0).to(tl.float32)  # [BLOCK_N]
    else:
        acc_preload = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    x_base_1d = x_base + token_offset * stride_x_token  # starting of chunk

    # PRE-LOAD WEIGHTS
    mask_w = idx_feats < dim
    if KERNEL_WIDTH >= 2:
        w_ptrs = w_base + (0 * stride_w_width)  # [BLOCK_N] tensor
        w_col0 = tl.load(w_ptrs, mask_w, other=0.0)
        w_ptrs = w_base + (1 * stride_w_width)  # [BLOCK_N] tensor
        w_col1 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 3:
        w_ptrs = w_base + (2 * stride_w_width)  # [BLOCK_N] tensor
        w_col2 = tl.load(w_ptrs, mask_w, other=0.0)
    if KERNEL_WIDTH >= 4:
        w_ptrs = w_base + (3 * stride_w_width)  # [BLOCK_N] tensor
        w_col3 = tl.load(w_ptrs, mask_w, other=0.0)
    mask_x_1d = idx_feats < dim
    for idx_token in range(segment_len):
        acc = acc_preload

        matrix_w = w_col0
        matrix_x = col0
        for j in tl.static_range(KERNEL_WIDTH):

            if KERNEL_WIDTH == 2:
                if j == 1:  # KERNEL_WIDTH-1:
                    matrix_w = w_col1
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 3:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            elif KERNEL_WIDTH == 4:
                if j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    x_ptrs_1d = x_base_1d + idx_token * stride_x_token  # [BLOCK_N]
                    matrix_x = tl.load(x_ptrs_1d, mask=mask_x_1d)
            # TODO: add more if KERNEL_WIDTH>4

            acc += matrix_x * matrix_w  # [BLOCK_N]

        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = (idx_token < segment_len) & (
            idx_feats < dim)  # token-index  # feature-index
        o_ptrs = o_ptr + (sequence_start_index + token_offset + idx_token
                          ) * stride_o_token + (idx_feats * stride_o_dim)

        tl.store(o_ptrs, acc, mask=mask_1d)


def causal_conv1d_fn_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    conv_states: Optional[
        torch.
        Tensor] = None,  # place the role of initial_state and incorporate `cache_indices`
    query_start_loc: Optional[
        torch.
        Tensor] = None,  # NEW (to use with varlen with x.shape=(dim, cu_seq_len))
    cache_indices: Optional[torch.Tensor] = None,  # NEW (used by conv_states)
    has_initial_states: Optional[
        torch.
        Tensor] = None,  # NEW interpretation sequence-level boolean (previously batch-level)
    # silu_activation: bool,
    activation: Optional[str] = "silu",
    # pad_slot_id: int,
    pad_slot_id: int = PAD_SLOT_ID,
    metadata=None,
    validate_data=False,
):
    """support varlen + continuous batching when x is 2D tensor

    Update Apr, 14, 2025: move position of 'conv_states' to the 4th position

    x: (dim,cu_seq_len)
        [len(shape)==2 means continuous-batching]
        cu_seq_len = total length  of all seqs in that batch
        x = [5, 1, 1, 1] <- continuous batching [here 4 sequences, with length 5, 1, 1, and 1 respectively]
            cu_seq_len = 8
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        if
        x = [5, 1, 1, 1] <- continuous batching (batch=4)
        then
        query_start_loc = [0, 5, 6, 7, 8] <- the starting index of the next sequence; while the last value is
           the ending index of the last sequence
        [length(query_start_loc)-1 == batch]

        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_states: (batch) bool
        [single boolean for each sequence in the batch: True or False]
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
        [mamba cache here is all sequences it can support, pass the whole thing in; and
        it use `cache_indices` to get the index to the cache of conv_state for that sequence

        conv_state[cache_indices[i]] for seq-i - to be used as initial_state when has_initial_states[i] = True
             and afther that conv_state[cache_indices[i]] need to be shift-left and updated with values from 'x'
        ]
    activation: either None or "silu" or "swish"
    pad_slot_id: int
        [something with cache management]
        // used to identify padding entries if cache_indices provided
        // in case of padding, the kernel will return early
        if cache_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    in_place: bool (default: False)
        whether to do in-place kernel update on 'x' or create a new tensor


    out: same shape as `x`
    """
    if isinstance(activation, bool):
        if activation:
            activation = "silu"

    args = None
    if metadata is not None:
        padded_batch = metadata.padded_batch
        dim = metadata.dim
        cu_seqlen = metadata.cu_seqlen
        is_channel_last = metadata.is_channel_last
        stride_x_seq = metadata.stride_x_seq
        stride_x_dim = metadata.stride_x_dim
        stride_x_token = metadata.stride_x_token
        stride_w_dim = metadata.stride_w_dim
        stride_w_width = metadata.stride_w_width
        width = metadata.width
        num_cache_lines = metadata.num_cache_lines
        stride_istate_seq = metadata.stride_istate_seq
        stride_istate_dim = metadata.stride_istate_dim
        stride_istate_token = metadata.stride_istate_token
        out = metadata.out
        stride_o_seq = metadata.stride_o_seq
        stride_o_dim = metadata.stride_o_dim
        stride_o_token = metadata.stride_o_token
        seqlens = metadata.seqlens
        np2_statelen = metadata.np2_statelen
        nums_dict = metadata.nums_dict
        #x = metadata.x
        args = nums_dict
    else:
        padded_batch = query_start_loc.size(0) - 1
        dim, cu_seqlen = x.shape
        is_channel_last = (x.stride(0) == 1) & (x.stride(1) > 1)
        stride_x_seq = 0
        stride_x_dim = x.stride(0)
        stride_x_token = x.stride(1)
        stride_w_dim = weight.stride(0)
        stride_w_width = weight.stride(1)

        _, width = weight.shape

        stride_istate_seq = 0
        stride_istate_dim = 0
        stride_istate_token = 0
        num_cache_lines = 0
        if conv_states is not None:
            # extensions to support vLLM:
            # 1. conv_states is used to replaced initial_states
            # 2. conv_states serve as a cache with num cache lines can be larger than batch size
            # 3. mapping from sequence x[idx] to a cache line at index as specified via cache_indices[idx]
            # 4. computation can be skipped if cache_indices[idx] == pad_slot_id
            num_cache_lines = conv_states.size(0)
            assert (num_cache_lines, dim, width - 1) == conv_states.shape
            stride_istate_seq = conv_states.stride(0)
            stride_istate_dim = conv_states.stride(1)
            stride_istate_token = conv_states.stride(2)
            assert stride_istate_dim == 1
        out = torch.zeros_like(x)
        if out.dim() == 2:
            stride_o_seq = 0
            stride_o_dim = out.stride(0)
            stride_o_token = out.stride(1)
        else:
            stride_o_seq = out.stride(0)
            stride_o_dim = out.stride(1)
            stride_o_token = out.stride(2)
        seqlens = np.diff(query_start_loc.to('cpu'))
        state_len = width - 1
        np2_statelen = triton.next_power_of_2(state_len)
        args = seqlens

    if validate_data:
        assert x.dim() == 2
        assert query_start_loc is not None
        assert query_start_loc.dim() == 1
        assert x.stride(0) == 1 or x.stride(1) == 1
        if bias is not None:
            assert bias.dim() == 1
            assert dim == bias.size(0)
        if cache_indices is not None:
            assert cache_indices.dim() == 1
            assert padded_batch == cache_indices.size(0)
        if has_initial_states is not None:
            assert has_initial_states.size() == (padded_batch, )
            assert conv_states is not None, "ERROR: `has_initial_states` is used, which needs also `conv_states`"
        assert weight.stride(1) == 1
        assert (dim, width) == weight.shape
        assert return_final_states is False
        assert is_channel_last, "Need to run in channel-last layout"

    if metadata is None:

        def num_program(META, seqlens):
            query_start_loc = META["query_start_loc_ptr"]
            batch_tensor = META["batch_ptr"]

            tot = 0

            mlist = []
            offsetlist = []
            tot_per_request = []

            nums = -(-seqlens // META["BLOCK_M"])

            tot = nums.sum().item()
            mlist = np.repeat(np.arange(len(nums)), nums)
            tot_per_request = np.repeat(nums, nums)
            start = 0
            for idx, num in enumerate(nums):
                offsetlist.extend(
                    range(num)
                )  # chunk-idx if a sequence is splitted into multiple chunks

            if META["batch_ptr"].nelement() < len(mlist):
                newlen = len(mlist) + 1
                META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                META["token_chunk_offset_ptr"].resize_(newlen).fill_(
                    PAD_SLOT_ID)

            if META["batch_ptr"].nelement() >= len(mlist):
                META["batch_ptr"][0:len(mlist)].copy_(
                    torch.from_numpy(np.array(mlist)))
                META["token_chunk_offset_ptr"][0:len(mlist)].copy_(
                    torch.from_numpy(np.array(offsetlist)))

            META["batch_ptr"] = META["batch_ptr"].to(META["x_ptr"].device)
            META["token_chunk_offset_ptr"] = META["token_chunk_offset_ptr"].to(
                META["x_ptr"].device)
            return tot
    else:

        def num_program(META, nums_dict):
            nums = nums_dict[META["BLOCK_M"]]['nums']
            tot = nums_dict[META["BLOCK_M"]]['tot']

            mlist = nums_dict[META["BLOCK_M"]]['mlist']
            mlist_len = nums_dict[META["BLOCK_M"]]['mlist_len']

            offsetlist = nums_dict[META["BLOCK_M"]]['offsetlist']

            if nums_dict[META["BLOCK_M"]]["batch_ptr"] is not None:
                META["batch_ptr"] = nums_dict[META["BLOCK_M"]]["batch_ptr"]
                META["token_chunk_offset_ptr"] = nums_dict[
                    META["BLOCK_M"]]["token_chunk_offset_ptr"]
            else:
                if META["batch_ptr"].nelement() < mlist_len:
                    newlen = mlist_len + 1
                    META["batch_ptr"].resize_(newlen).fill_(PAD_SLOT_ID)
                    META["token_chunk_offset_ptr"].resize_(newlen).fill_(
                        PAD_SLOT_ID)

                if META["batch_ptr"].nelement() >= mlist_len:
                    META["batch_ptr"][0:mlist_len].copy_(mlist)
                    META["token_chunk_offset_ptr"][0:mlist_len].copy_(
                        offsetlist)
            return tot

    def grid(META):
        return (
            num_program(META, args),
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    global batch_ptr, token_chunk_offset_ptr

    _causal_conv1d_fwd_kernel_contbatch[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_states,
        query_start_loc,
        batch_ptr,
        token_chunk_offset_ptr,
        out,
        # Matrix dimensions
        padded_batch,
        dim,
        cu_seqlen,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        HAS_INITIAL_STATES=has_initial_states is not None,
        HAS_CACHE=conv_states is not None,
        IS_CONTINUOUS_BATCHING=cache_indices is not None,
        USE_PAD_SLOT=pad_slot_id is not None,
        NP2_STATELEN=np2_statelen,
        DECODE_SEQLEN=1,
        #launch_cooperative_grid=True
        BLOCK_M=8,
        BLOCK_N=256,
        num_stages=2,
    )
    return out
