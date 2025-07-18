# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, Optional

import torch

from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonImpl,
                                                   MLACommonMetadata)

FLASHINFER_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024

logger = init_logger(__name__)


class FlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl


class FlashInferMLAImpl(MLACommonImpl[MLACommonMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashInferMLA V1 with FP8 KV cache not yet supported")
        

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None


        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        q = q.unsqueeze(1)
        o = torch.zeros(B,
                        1,  # acc_q_len = # MTP draft tokens + 1
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        page_size = kv_c_and_k_pe_cache.size(1)
        max_seq_len = attn_metadata.decode.seq_lens.max().item()

        workspace_buffer = torch.empty(
            FLASHINFER_WORKSPACE_BUFFER_SIZE,
            dtype=torch.uint8,
            device=q.device,
        )
        kv_cache_duplicate = torch.stack([kv_c_and_k_pe_cache, kv_c_and_k_pe_cache], dim=1)
        trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache_duplicate,
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            block_tables=attn_metadata.decode.block_table,
            seq_lens=attn_metadata.decode.seq_lens,
            block_size=page_size,
            max_seq_len=max_seq_len,
            out=o,
            bmm1_scale=self.scale,
        )
        o = o.squeeze(1)

        return self._v_up_proj(o)
