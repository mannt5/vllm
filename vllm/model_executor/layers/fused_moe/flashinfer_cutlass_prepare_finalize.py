# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional, Tuple

import torch
# from flashinfer.comm.trtllm_alltoall import MnnvlMoe as MnnvlMoe
# from flashinfer.comm.trtllm_alltoall import MoEAlltoallInfo as MoEAlltoallInfo

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import (get_dp_group, get_tp_group, get_ep_group)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import (
    moe_kernel_quantize_input)

_alltoall_manager = None

def get_global_num_tokens_cpu():
    cu_sizes = get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
    sizes = [cu_sizes[0].item()]
    for i in range(1, len(cu_sizes)):
        sizes.append((cu_sizes[i] - cu_sizes[i - 1]).item())
    return sizes

def get_local_sizes(local_tokens):
    sizes = get_global_num_tokens_cpu()
    max_num_tokens = envs.VLLM_MOE_DP_CHUNK_SIZE
    sizes_chunked = [max_num_tokens] * len(sizes)
    if local_tokens < max_num_tokens:
        # When the number of local tokens is less than max_num_tokens, all other
        # ranks will also have fewer than max_num_tokens. The remaining tokens
        # are accounted for as residual.
        sizes_chunked = [x % max_num_tokens for x in sizes]

    return sizes_chunked


class FlashInferCutlassMoEPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):

    def __init__(
        self,
        quant_dtype: Optional[torch.dtype] = None,
        per_channel_quant: bool = False,
        block_shape: Optional[list[int]] = None,
        num_dispatchers: int = 1,
        ep_rank: int = 0,
        ep_size: int = 1,
    ):
        super().__init__()
        self.per_channel_quant = per_channel_quant
        self.block_shape = block_shape
        self.quant_dtype = quant_dtype
        self.num_dispatchers_ = num_dispatchers
        self.ep_rank = ep_rank
        self.ep_size = ep_size

    # @property
    # def alltoall_info(self) -> MoEAlltoallInfo:
    #     return self.alltoall_info

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> Optional[int]:
        return None

    def topk_indices_dtype(self) -> Optional[torch.dtype]:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],  # Not used
        a2_scale: Optional[torch.Tensor],  # Not used
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        a1_gscale: torch.Tensor,
        use_dp: Optional[bool] = True,
        local_tokens: int = -1,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], Optional[torch.Tensor]]:

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, \
                "apply_router_weight_on_input is only implemented for topk=1"
            a1.mul_(topk_weights.to(a1.dtype))

        if not use_dp:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                a1_gscale,
                quant_config.quant_dtype,
                self.per_channel_quant,
                self.block_shape,
                is_fp4_scalar_swizzled=True,
            )
        else:
            #TODO(shuw): make env var
            enable_flashinfer_fp4_allgather = True
            enable_flashinfer_alltoall = True

            if enable_flashinfer_alltoall:
                global_num_tokens_cpu = get_global_num_tokens_cpu()
                top_k = topk_ids.size(1)
                all2all_manager = get_ep_group().device_communicator.all2all_manager
                print("xxxx"*100)
                print(all2all_manager)
                print(f"ep_size:{self.ep_size}, {self.ep_rank}")
                assert all2all_manager is not None
                # TODO(shuw): need to consider chunking for global_num_tokens_cpu
                x1, topk_ids1, topk_weights1, alltoall_info = all2all_manager.dispatch(
                    get_dp_group().device_communicator,
                    global_num_tokens_cpu,
                    a1,
                    topk_ids,
                    topk_weights,
                    top_k,
                    num_experts,
                    self.ep_rank,
                    self.ep_size,
                )
                self.alltoall_info = alltoall_info

            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                a1_gscale,
                quant_config.quant_dtype,
                self.per_channel_quant,
                self.block_shape,
                is_fp4_scalar_swizzled=False  # delay swizzle to after comm
            )

            if enable_flashinfer_fp4_allgather:
                topk_weights, topk_ids, a1q, a1q_scale = \
                    get_dp_group().all_gatherv([topk_weights, topk_ids, a1q, a1q_scale],
                                            dim=0,
                                            sizes=get_local_sizes(local_tokens))

            # if enable_flashinfer_alltoall:
            #     print("all2allcalling"*100)
            #     a1q = MnnvlMoe.mnnvl_moe_alltoallv(a1q, self.alltoall_info,
            #                                        self.alltoall_workspace,
            #                                        self.ep_rank, self.ep_size)
            #     a1q_scale = MnnvlMoe.mnnvl_moe_alltoallv(
            #         a1q_scale, alltoall_info, self.alltoall_workspace,
            #         self.ep_rank, self.ep_size)

            from flashinfer import fp4_swizzle_blockscale
            a1_m, a1_n = a1q.shape
            a1q_scale = fp4_swizzle_blockscale(a1q_scale, a1_m, a1_n * 2)

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        use_dp: bool = False,
        local_tokens: int = -1,
    ) -> None:
        if use_dp:
            # TODO(shuw): env var later
            enable_flashinfer_fp4_allgather = True
            enable_flashinfer_alltoall = False
            if enable_flashinfer_fp4_allgather:
                fused_expert_output = get_dp_group().reduce_scatterv(
                    fused_expert_output,
                    dim=0,
                    sizes=get_local_sizes(local_tokens),
                )

            # if enable_flashinfer_alltoall:
            #     top_k = topk_ids.size(1)
            #     token_count = fused_expert_output.shape[0]
            #     _ = flashinfer_alltoall_combine(
            #         fused_expert_output,
            #         # TODO(shuw): need to consider chunking for global_num_tokens_cpu
            #         self.alltoall_info,
            #         ep_rank=self.ep_rank,
            #         ep_size=self.ep_size,
            #         top_k=top_k,
            #         token_count=token_count,
            #     )
        output.copy_(fused_expert_output)




# def flashinfer_alltoall_combine(
#     output: torch.Tensor,
#     alltoall_info: MoEAlltoallInfo,
#     top_k: int,
#     ep_rank: int,
#     ep_size: int,
#     token_count: int,
# ):
#     # TODO(shuw): add later
#     # assert (
#     #     ensure_alltoall_workspace_initialized()
#     # ), "FlashInfer AllToAll workspace not available"
#     return _flashinfer_mnnvlmoe.mnnvl_moe_alltoallv_combine(
#         output,
#         alltoall_info,
#         _alltoall_manager.workspace_tensor,
#         ep_rank=ep_rank,
#         ep_size=ep_size,
#         top_k=top_k,
#         token_count=token_count,
#     )