# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch
import torch.nn as nn

from vllm.attention.layer import Attention
from vllm.config import (CompilationLevel, VllmConfig,
                         get_layers_from_vllm_config)
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.models import supports_multimodal
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.utils import prepare_eagle_input_kernel

logger = init_logger(__name__)

PADDING_SLOT_ID = -1


class EagleProposer:

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.draft_model_config = self.speculative_config.draft_model_config
        self.kv_sharing_mapping = self.speculative_config.kv_sharing_mapping
        self.method = self.speculative_config.method

        self.runner = runner

        self.dtype = vllm_config.model_config.dtype

        self.max_model_len = vllm_config.model_config.max_model_len
        self.block_size = vllm_config.cache_config.block_size
        self.num_speculative_tokens = (
            self.speculative_config.num_speculative_tokens)
        self.max_num_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens)
        # We need to get the hidden size from the draft model config because
        # the draft model's hidden size can be different from the target model's
        # hidden size (e.g., Llama 3.3 70B).
        self.hidden_size = self.draft_model_config.get_hidden_size()

        self.is_multimodal_model = vllm_config.model_config \
            .is_multimodal_model

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE and
                               not self.vllm_config.model_config.enforce_eager)
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))
        self.draft_prefill_kv_sharing_from_base = self.kv_sharing_mapping is not None

        # persistent buffers for cuda graph
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
        self.hidden_states = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)
        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        self.arange = torch.arange(vllm_config.scheduler_config.max_num_seqs +
                                   1,
                                   device=device,
                                   dtype=torch.int32)
        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=device)

    def _prepare_adjusted_tensors(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_slot_mapping: torch.Tensor,
        cu_num_tokens: torch.Tensor,
        decode_mask: torch.Tensor,
        full_prefill_mask: torch.Tensor,
        prefill_first_hiddens: torch.Tensor,
        block_table: torch.Tensor,
        batch_size: int,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int,
               torch.Tensor]:
        """
        Prepare adjusted tensors for different request types (partial prefill, full prefill, full decode).
        
        Args:
            target_token_ids: Input token IDs tensor
            target_positions: Input position IDs tensor
            target_hidden_states: Input hidden states tensor
            target_slot_mapping: Input slot mapping tensor
            cu_num_tokens: Cumulative number of tokens per request
            decode_mask: Mask indicating which tokens are for decoding
            full_prefill_mask: Mask indicating which requests are full prefill
            prefill_first_hiddens: First hidden states for prefill requests
            block_table: Block table for KV cache mapping
            batch_size: Number of requests in the batch
            num_tokens: Total number of tokens
            
        Returns:
            tuple: (target_positions, target_hidden_states, target_slot_mapping,
                    cu_num_tokens, current_pos, partial_prefill_mask)

        """
        # Count total number of full prefill requests to determine the size needed for adjusted tensors
        num_full_prefill = full_prefill_mask.sum().item()

        # Create tensors with extra space for the additional positions from full prefill requests
        adjusted_token_ids = torch.zeros(
            num_tokens + num_full_prefill,
            dtype=target_token_ids.dtype,
            device=target_token_ids.device,
        )
        adjusted_positions = torch.zeros(
            num_tokens + num_full_prefill,
            dtype=target_positions.dtype,
            device=target_positions.device,
        )
        adjusted_slot_mapping = torch.zeros(
            num_tokens + num_full_prefill,
            dtype=target_slot_mapping.dtype,
            device=target_slot_mapping.device,
        )
        adjusted_hidden_states = torch.zeros(
            num_tokens + num_full_prefill,
            self.hidden_size,
            dtype=target_hidden_states.dtype,
            device=target_hidden_states.device,
        )

        # Create updated cumulative token counts
        updated_cu_num_tokens = torch.zeros_like(cu_num_tokens)

        # Track which requests are partial prefill (no decode tokens)
        partial_prefill_mask = torch.zeros_like(full_prefill_mask)

        # Create masks for each category
        has_decode_mask = torch.zeros(batch_size,
                                      dtype=torch.bool,
                                      device=decode_mask.device)
        for i in range(batch_size):
            start_idx = cu_num_tokens[i].item()
            end_idx = cu_num_tokens[i + 1].item()
            has_decode_mask[i] = decode_mask[start_idx:end_idx].any().item()

        # Category 1: Partial prefill (no decode tokens)
        partial_prefill_mask = ~has_decode_mask

        # Process batched operations using masks
        current_pos = 0
        cu_num_tokens_index = 0

        # Process each request in the batch
        # Process all requests in batch order but with optimized operations
        # Create arrays to track request properties
        req_starts = cu_num_tokens[:-1]
        req_ends = cu_num_tokens[1:]
        req_lens = req_ends - req_starts

        # Process each request in order
        for i in range(batch_size):
            # Get the start and end indices for this request
            start_idx = req_starts[i].item()
            end_idx = req_ends[i].item()
            req_len = req_lens[i].item()

            # Check category
            is_partial_prefill = partial_prefill_mask[i].item()
            is_full_prefill = full_prefill_mask[i].item()

            if is_partial_prefill:
                # Category 1: Partial prefill - just copy all tokens
                if not self.draft_prefill_kv_sharing_from_base:
                    # Use torch operations for copying blocks of data
                    adjusted_token_ids[current_pos:current_pos +
                                       req_len].copy_(
                                           target_token_ids[start_idx:end_idx])
                    adjusted_positions[current_pos:current_pos +
                                       req_len].copy_(
                                           target_positions[start_idx:end_idx])
                    adjusted_slot_mapping[current_pos:current_pos +
                                          req_len].copy_(target_slot_mapping[
                                              start_idx:end_idx])
                    adjusted_hidden_states[current_pos + 1:current_pos +
                                           req_len].copy_(
                                               target_hidden_states[start_idx +
                                                                    1:end_idx])
                    adjusted_hidden_states[
                        current_pos] = prefill_first_hiddens[i]
                    current_pos += req_len
                    cu_num_tokens_index += 1

            elif is_full_prefill:
                # Category 2: Full prefill with decode - copy tokens and add one position
                pos = target_positions[end_idx - 1] + 1
                block_number = pos // self.block_size
                block_number = block_table[i][block_number].item()
                block_offset = pos % self.block_size

                if not self.draft_prefill_kv_sharing_from_base:
                    # Copy token IDs, positions, slot mappings, and hidden states
                    adjusted_token_ids[current_pos:current_pos +
                                       req_len].copy_(
                                           target_token_ids[start_idx:end_idx])
                    adjusted_positions[current_pos:current_pos +
                                       req_len].copy_(
                                           target_positions[start_idx:end_idx])
                    adjusted_positions[current_pos +
                                       req_len] = adjusted_positions[
                                           current_pos + req_len - 1] + 1

                    adjusted_slot_mapping[current_pos:current_pos +
                                          req_len].copy_(target_slot_mapping[
                                              start_idx:end_idx])
                    adjusted_slot_mapping[
                        current_pos +
                        req_len] = block_number * self.block_size + block_offset

                    adjusted_hidden_states[
                        current_pos + 1:current_pos + req_len + 1].copy_(
                            target_hidden_states[start_idx:end_idx])
                    adjusted_hidden_states[
                        current_pos] = prefill_first_hiddens[i]
                    current_pos += req_len + 1
                else:
                    adjusted_positions[current_pos] = 0
                    adjusted_slot_mapping[
                        current_pos] = block_number * self.block_size + block_offset
                    adjusted_hidden_states[current_pos] = target_hidden_states[
                        end_idx - 1]
                    current_pos += 1

                cu_num_tokens_index += 1

            else:
                # Category 3: Full decode - shift tokens
                # Shift operations using optimized copy operations
                adjusted_token_ids[current_pos:current_pos + req_len -
                                   1].copy_(target_token_ids[start_idx +
                                                             1:end_idx])
                adjusted_positions[current_pos:current_pos + req_len].copy_(
                    target_positions[start_idx:end_idx] + 1)

                adjusted_slot_mapping[current_pos:current_pos + req_len -
                                      1].copy_(target_slot_mapping[start_idx +
                                                                   1:end_idx])
                pos = adjusted_positions[current_pos + req_len - 1]
                block_number = pos // self.block_size
                block_number = block_table[i][block_number].item()
                block_offset = pos % self.block_size
                adjusted_slot_mapping[
                    current_pos + req_len -
                    1] = block_number * self.block_size + block_offset

                adjusted_hidden_states[current_pos:current_pos +
                                       req_len].copy_(target_hidden_states[
                                           start_idx:end_idx])

                current_pos += req_len
                cu_num_tokens_index += 1

            # Update the cumulative token count for this request
            updated_cu_num_tokens[cu_num_tokens_index] = current_pos

        # Copy the adjusted tensors to the input buffers
        self.input_ids[:current_pos] = adjusted_token_ids[:current_pos]

        # Update the variables used by the rest of the function
        target_positions = adjusted_positions[:current_pos]
        target_hidden_states = adjusted_hidden_states[:current_pos]
        target_slot_mapping = adjusted_slot_mapping[:current_pos]
        cu_num_tokens = updated_cu_num_tokens

        return (
            target_positions,
            target_hidden_states,
            target_slot_mapping,
            cu_num_tokens,
            current_pos,
            partial_prefill_mask,
        )

    def propose(
        self,
        # [num_tokens]
        target_token_ids: torch.Tensor,
        # [num_tokens]
        target_positions: torch.Tensor,
        # [num_tokens, hidden_size]
        target_hidden_states: torch.Tensor,
        # [num_tokens]
        target_slot_mapping: torch.Tensor,
        # [batch_size]
        next_token_ids: torch.Tensor,
        # [batch_size + 1] starting with 0
        cu_num_tokens: torch.Tensor,
        # [batch_size, max_num_blocks_per_req]
        block_table: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        prefill_first_hiddens: torch.Tensor,
        mm_embeds: Optional[list[torch.Tensor]] = None,
        decode_mask: torch.Tensor = None,
        full_prefill_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if self.method == "eagle3":
            assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states)
            assert target_hidden_states.shape[-1] == self.hidden_size

        prefill_shift_tokens = True
        has_prefill = decode_mask is not None and (
            ~decode_mask.bool()).any().item()
        if not self.speculative_config.eagle_shift_prefill_token() and (
                self.method in ["eagle", "eagle3"]):
            assert decode_mask is not None
            assert full_prefill_mask is not None
            prefill_shift_tokens = False

        if not prefill_shift_tokens and has_prefill:
            # Adjust the tensors for full prefill requests
            (
                target_positions,
                target_hidden_states,
                target_slot_mapping,
                cu_num_tokens,
                num_tokens,
                partial_prefill_mask,
            ) = self._prepare_adjusted_tensors(
                target_token_ids,
                target_positions,
                target_hidden_states,
                target_slot_mapping,
                cu_num_tokens,
                decode_mask,
                full_prefill_mask,
                prefill_first_hiddens,
                block_table,
                batch_size,
                num_tokens,
            )
        else:
            # Original behavior: shift all tokens by one
            partial_prefill_mask = None
            self.input_ids[:num_tokens - 1] = target_token_ids[1:]
            if not prefill_shift_tokens:
                target_positions += 1
                max_num_blocks_per_req = block_table.shape[1]
                segment_indices = torch.arange(len(target_positions),
                                               device=target_positions.device)
                segment_indices = (segment_indices.unsqueeze(0)
                                   >= cu_num_tokens[:-1].unsqueeze(1)).sum(
                                       dim=0) - 1
                # Calculate the block table indices
                block_table_indices = (
                    target_positions // self.block_size +
                    segment_indices * max_num_blocks_per_req)
                block_numbers = block_table.flatten()[block_table_indices]
                block_offsets = target_positions % self.block_size
                target_slot_mapping = block_numbers * self.block_size + block_offsets

            # Use the original last token indices
        last_token_indices = cu_num_tokens[1:] - 1

        # Replace the last token with the next token, but only for non-partial prefill requests
        if not prefill_shift_tokens and has_prefill:
            mask = ~partial_prefill_mask
            self.input_ids[last_token_indices[mask]] = next_token_ids[mask]
        else:
            # Original behavior: apply to all requests
            self.input_ids[last_token_indices] = next_token_ids

        # FA requires seq_len to have dtype int32.
        seq_lens = (target_positions[last_token_indices] + 1).int()

        if self.method in ["eagle", "eagle3"]:
            # FIXME(woosuk): The below two ops cause synchronization. Optimize.
            max_seq_len = seq_lens.max().item()
            max_num_tokens = (cu_num_tokens[1:] -
                              cu_num_tokens[:-1]).max().item()
            attn_metadata = FlashAttentionMetadata(
                num_actual_tokens=num_tokens,
                max_query_len=max_num_tokens,
                query_start_loc=cu_num_tokens,
                max_seq_len=max_seq_len,
                seq_lens=seq_lens,
                block_table=block_table,
                slot_mapping=target_slot_mapping,
                # TODO(woosuk): Support cascade attention.
                use_cascade=False,
                common_prefix_len=0,
                cu_prefix_query_lens=None,
                prefix_kv_lens=None,
                suffix_kv_lens=None,
            )
        elif self.method == "deepseek_mtp":
            query_lens = cu_num_tokens[1:] - cu_num_tokens[:-1]
            max_query_len = query_lens.max().item()

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=cu_num_tokens,
                seq_lens=seq_lens,
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
            )

            assert self.runner is not None

            # FIXME: need to consider multiple kv_cache_groups
            attn_metadata = self.runner.attn_metadata_builders[0].build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
        if self.use_cuda_graph and num_tokens <= self.cudagraph_batch_sizes[-1]:
            num_input_tokens = self.vllm_config.pad_for_cudagraph(num_tokens)
        else:
            num_input_tokens = num_tokens
        # copy inputs to buffer for cudagraph
        self.positions[:num_tokens] = target_positions
        self.hidden_states[:num_tokens] = target_hidden_states
        if self.is_multimodal_model:
            input_ids = self.input_ids[:num_tokens]
            if mm_embeds:
                inputs_embeds = self.model.get_input_embeddings(
                    input_ids, mm_embeds)
            else:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
            self.inputs_embeds[:num_tokens] = inputs_embeds
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
            input_ids = None
        else:
            inputs_embeds = None
            input_ids = self.input_ids[:num_input_tokens]

        with set_forward_context(per_layer_attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=self.positions[:num_input_tokens],
                hidden_states=self.hidden_states[:num_input_tokens],
                inputs_embeds=inputs_embeds,
            )
            if self.method == "deepseek_mtp":
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states
        sample_hidden_states = last_hidden_states[last_token_indices]
        logits = self.model.compute_logits(sample_hidden_states, None)
        draft_token_ids = logits.argmax(dim=-1)
        # print("draft_tokens topK:", logits.topk(3, dim=-1))

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            # [batch_size, 1]
            return draft_token_ids.view(-1, 1)

        # TODO: Currently, MTP module released by deepseek only has
        # one layer. Adapt this code to support multiple layers once
        # there's a multi-layer MTP module.

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        positions = target_positions[last_token_indices]
        hidden_states = hidden_states[last_token_indices]
        if self.use_cuda_graph and batch_size <= self.cudagraph_batch_sizes[-1]:
            input_batch_size = self.vllm_config.pad_for_cudagraph(batch_size)
        else:
            input_batch_size = batch_size
        attn_metadata.num_actual_tokens = batch_size
        attn_metadata.max_query_len = 1
        attn_metadata.query_start_loc = self.arange[:batch_size + 1]
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs.
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax() returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions += 1

            # NOTE(woosuk): We should handle the case where the draft model
            # generates tokens beyond the max model length. Since it is complex
            # to remove such requests from the batch, we keep them in the batch
            # but adjust the position ids and slot mappings to avoid the
            # out-of-range access during the model execution. The draft tokens
            # generated with this adjustment should be ignored.
            exceeds_max_model_len = positions >= self.max_model_len
            # Mask out the position ids that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)

            # Increment the sequence lengths.
            attn_metadata.max_seq_len += 1
            attn_metadata.seq_lens += 1
            # Consider max model length.
            attn_metadata.max_seq_len = min(attn_metadata.max_seq_len,
                                            self.max_model_len)
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = block_table.gather(dim=1,
                                           index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            attn_metadata.slot_mapping = (block_ids * self.block_size +
                                          clamped_positions % self.block_size)
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                    PADDING_SLOT_ID)

            # copy inputs to buffer for cudagraph
            self.input_ids[:batch_size] = input_ids
            self.positions[:batch_size] = clamped_positions
            self.hidden_states[:batch_size] = hidden_states
            if self.is_multimodal_model:
                inputs_embeds = self.model.get_input_embeddings(input_ids)
                self.inputs_embeds[:batch_size] = inputs_embeds
                inputs_embeds = self.inputs_embeds[:input_batch_size]
                input_ids = None
            else:
                inputs_embeds = None
                input_ids = self.input_ids[:input_batch_size]

            # Run the model.
            with set_forward_context(per_layer_attn_metadata,
                                     self.vllm_config,
                                     num_tokens=input_batch_size):
                last_hidden_states, hidden_states = self.model(
                    input_ids=input_ids,
                    positions=self.positions[:input_batch_size],
                    hidden_states=self.hidden_states[:input_batch_size],
                    inputs_embeds=inputs_embeds,
                )
            hidden_states = hidden_states[:batch_size]
            logits = self.model.compute_logits(last_hidden_states[:batch_size],
                                               None)

            # TODO(wenlong): get more than one token for tree attention
            draft_token_ids = logits.argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        # print("draft_token_ids:", draft_token_ids)
        return draft_token_ids

    @staticmethod
    def prepare_inputs(
        # [batch_size + 1]
        cu_target_query_lens: torch.Tensor,
        # [batch_size]
        num_rejected_tokens: torch.Tensor,
        num_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # cu_target_query_lens: [0, a, a + b, a + b + c]
        # num_rejected_tokens: [n1, n2, n3]
        # num_tokens_per_req: [a - n1, b - n2, c - n3]
        # cu_num_tokens: [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        # token_indices: [0, 1, ..., a - n1 - 1,
        #                 a, a + 1, ..., a + b - n2 - 1,
        #                 a + b, a + b + 1, ..., a + b + c - n3 - 1]

        # [0, a, a + b, a + b + c] -> [a, b, c]
        query_len_per_req = cu_target_query_lens[1:] - cu_target_query_lens[:-1]
        # [a, b, c] -> [a - n1, b - n2, c - n3]
        num_tokens_per_req = query_len_per_req - num_rejected_tokens

        # [a - n1, b - n2, c - n3] ->
        # [0, a - n1, a + b - n1 - n2, a + b + c - n1 - n2 - n3]
        cu_num_tokens = torch.zeros_like(cu_target_query_lens)
        torch.cumsum(num_tokens_per_req, dim=0, out=cu_num_tokens[1:])
        token_indices = torch.empty(
            num_tokens,
            dtype=torch.int32,
            device=cu_target_query_lens.device,
        )
        batch_size = num_rejected_tokens.shape[0]
        BLOCK_SIZE = 1024
        prepare_eagle_input_kernel[(batch_size, )](
            token_indices,
            cu_target_query_lens,
            cu_num_tokens,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return cu_num_tokens, token_indices

    def load_model(self, target_model: nn.Module) -> None:
        draft_model_config = self.vllm_config.speculative_config.draft_model_config
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, Attention).keys())

        from vllm.compilation.backends import set_model_tag
        with set_model_tag("eagle_head"):
            self.model = get_model(vllm_config=self.vllm_config,
                                   model_config=draft_model_config)

        draft_attn_layer_names = (
            get_layers_from_vllm_config(self.vllm_config, Attention).keys() -
            target_attn_layer_names)

        self.attn_layer_names = list(draft_attn_layer_names)

        if supports_multimodal(target_model):
            # handle multimodality
            self.model.config.image_token_index = (
                target_model.config.image_token_index)
            target_language_model = target_model.get_language_model()
        else:
            target_language_model = target_model
        # share embed_tokens with the target model if needed
        if get_pp_group().world_size == 1 \
            and self.model.model.embed_tokens.weight.shape \
                == target_language_model.model.embed_tokens.weight.shape:
            logger.info(
                "Assuming the EAGLE head shares the same vocab embedding" \
                " with the target model."
            )
            del self.model.model.embed_tokens
            self.model.model.embed_tokens = (
                target_language_model.model.embed_tokens)
        else:
            logger.info(
                "The EAGLE head's vocab embedding will be loaded separately" \
                " from the target model."
            )

        # share lm_head with the target model if needed
        # some model definition do not define lm_head explicitly
        # and reuse embed_tokens for lm_head, e.g., CohereForCausalLM
        if self.vllm_config.speculative_config.method != "eagle3" and \
                hasattr(target_language_model, "lm_head"):
            logger.info("Loading EAGLE LM head weights from the target model.")
            self.model.lm_head = target_language_model.lm_head

    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
    ) -> None:
        with set_forward_context(None, self.vllm_config,
                                 num_tokens=num_tokens):
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            self.model(
                input_ids=input_ids,
                positions=self.positions[:num_tokens],
                hidden_states=self.hidden_states[:num_tokens],
                inputs_embeds=inputs_embeds,
            )

    def validate_same_kv_cache_group(self,
                                     kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (len(
            set([
                kv_cache_groups[layer_name]
                for layer_name in self.attn_layer_names
            ])) == 1
                ), "All eagle layers should belong to the same kv cache group"


# NOTE(woosuk): Currently, the below code is not used and we always use argmax
# to sample the draft tokens. We will use this after we find a way to manage
# the draft prob tensor.
# Refer to https://github.com/vllm-project/vllm/pull/16899 for the details.
# FIXME(woosuk): The logic here is duplicated with the main sampling code.
# We should refactor this to reuse the same sampling implementation.
def compute_probs_and_sample_next_token(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sampling_metadata.all_greedy:
        # For greedy requests, draft_probs is not used in rejection sampling.
        # Therefore, we can just return the logits.
        probs = logits
        next_token_ids = logits.argmax(dim=-1)
        return next_token_ids, probs

    is_greedy = sampling_metadata.temperature == -1
    temperature = torch.where(is_greedy, 1.0, sampling_metadata.temperature)
    logits.div_(temperature.view(-1, 1))
    probs = logits.softmax(dim=-1, dtype=torch.float32)

    # NOTE(woosuk): Currently, we ignore most of the sampling parameters in
    # generating the draft tokens. We only use the temperature. While this
    # could degrade the acceptance rate, it does not affect the distribution
    # of the generated tokens after rejection sampling.

    # TODO(woosuk): Consider seeds.
    q = torch.empty_like(probs)
    q.exponential_()
    # NOTE(woosuk): We shouldn't use `probs.div_(q)` because the draft_probs
    # will be used later for rejection sampling.
    next_token_ids = probs.div(q).argmax(dim=-1).view(-1)
    if not sampling_metadata.all_random:
        greedy_token_ids = probs.argmax(dim=-1)
        next_token_ids = torch.where(
            is_greedy,
            greedy_token_ids,
            next_token_ids,
        )
    return next_token_ids, probs
