import dataclasses
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set,
                    Tuple, Type, TypeVar)

import torch
import torch.nn as nn

from vllm.attention import get_attn_backend
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.inputs import INPUT_REGISTRY, InputRegistry
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.model_executor import SamplingMetadataCache
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
from vllm.model_executor.models.interfaces import (supports_lora,
                                                   supports_multimodal)
from vllm.multimodal import (MULTIMODAL_REGISTRY, BatchedTensorInputs,
                             MultiModalKwargs, MultiModalPlaceholderMap,
                             MultiModalRegistry)
from vllm.sampling_params import SamplingParams
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import DeviceMemoryProfiler, make_tensor_with_pad
from vllm.worker.model_runner import AttentionMetadata, SamplingMetadata
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase, ModelRunnerInputBuilderBase,
    _add_attn_metadata_broadcastable_dict,
    _add_sampling_metadata_broadcastable_dict,
    _init_attn_metadata_from_tensor_dict,
    _init_sampling_metadata_from_tensor_dict)
from vllm.attention.backends.utils import is_block_tables_empty
if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]

TModelInputForXPU = TypeVar('TModelInputForXPU', bound="ModelInputForXPU")


@dataclass(frozen=True)
class ModelInputForXPU(ModelRunnerInputBase):
    """
    Used by the NeuronModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    lora_mapping: Optional["LoRAMapping"] = None
    lora_requests: Optional[Set[LoRARequest]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    multi_modal_kwargs: Optional[BatchedTensorInputs] = None
    virtual_engine: Optional[int] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    async_callback: Optional[Callable] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)

        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForXPU],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForXPU:
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


@dataclass(frozen=True)
class ModelInputForXPUWithSamplingMetadata(ModelInputForXPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "multi_modal_kwargs": self.multi_modal_kwargs,
            "lora_requests": self.lora_requests,
            "lora_mapping": self.lora_mapping,
        }
        _add_attn_metadata_broadcastable_dict(tensor_dict, self.attn_metadata)
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls,
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> "ModelInputForXPUWithSamplingMetadata":
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        if attn_backend is not None:
            tensor_dict = _init_attn_metadata_from_tensor_dict(
                attn_backend, tensor_dict)
        return cls(**tensor_dict)


class ModelInputForXPUBuilder(ModelRunnerInputBuilderBase[ModelInputForXPU]):

    def __init__(self,
                 runner: "XPUModelRunnerBase",
                 finished_requests_ids: Optional[List[str]] = None) -> None:
        super().__init__()
        self.seq_group_metadata_list: List[SequenceGroupMetadata] = []
        self.runner = runner
        self.model_input_cls = self.runner._model_input_cls
        self.attn_backend = self.runner.attn_backend
        self.sliding_window = self.runner.sliding_window
        self.block_size = self.runner.block_size
        self.device = self.runner.device
        self.scheduler_config = self.runner.scheduler_config
        self.cache_config = self.runner.cache_config
        # Multi-modal data support
        self.multi_modal_input_mapper = self.runner.multi_modal_input_mapper

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        self.seq_group_metadata_list.append(seq_group_metadata)

    def build(self) -> ModelInputForXPU:
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        seq_lens: List[int] = []
        # Prefill's seq_len: query_length + chunked_prefill length
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        # One for each sequence, physical blocks
        block_tables: List[List[int]] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []

        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0
        mrope_input_positions = None

        if len(self.seq_group_metadata_list) == 0:
            return None

        for seq_group_metadata in self.seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            # is_prompt indicates that it is still in prompt states
            # TODO: remove this is_prompt
            is_prompt = seq_group_metadata.is_prompt

            # Iterate over all the seqs in the seq_group
            for seq_id in seq_ids:
                # Check for prefix caching
                computed_block_nums = seq_group_metadata.computed_block_nums
                if (self.scheduler_config is not None
                    and self.scheduler_config.chunked_prefill_enabled
                    and not (computed_block_nums is None or computed_block_nums == [])):
                    raise RuntimeError("chunked prefill cannot be used with prefix caching")
                seq_data = seq_group_metadata.seq_data[seq_id]
                # Context_len: how many tokens that have been computed
                if is_prompt:
                    if self.scheduler_config.chunked_prefill_enabled:
                        context_len = seq_data.get_num_computed_tokens()
                    elif computed_block_nums is not None:
                        context_len = len(computed_block_nums) * self.block_size
                    else:
                        context_len = 0
                else:
                    context_len = seq_data.get_len() - 1

                # Get tokens for this sequence
                # For prefill, the seq_len will be the second one.
                # For decoding, the seq_len will be the first one.
                seq_len = min(seq_data.get_len(), context_len + seq_group_metadata.token_chunk_size)

                if is_prompt:
                    tokens = seq_data.get_token_ids()[context_len: seq_len]
                else:
                    # Last token
                    tokens = [seq_data.get_last_token_id()]

                if (computed_block_nums is not None or self.scheduler_config.chunked_prefill_enabled or not is_prompt):
                    # Chunked prefill or decoding
                    # For chunked prefill, the block tables may not be None
                    if seq_group_metadata.block_tables is not None:
                        block_table = seq_group_metadata.block_tables[seq_id]
                    else:
                        block_table = []
                else:
                    # Prefill without chunked prefill
                    block_table = []
                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)
                # Total seq_lens
                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)
                input_tokens.extend(tokens)
                query_len = seq_len - context_len
                query_lens.append(query_len)
                if is_prompt or self.scheduler_config.chunked_prefill_enabled:
                    context_lens.append(context_len)
                    input_positions.extend(list(range(context_len, seq_len)))
                else:
                    context_lens = seq_lens
                    #query_lens = seq_lens
                    position = seq_len - 1
                    input_positions.append(position)
                if is_prompt:
                    mm_data = seq_group_metadata.multi_modal_data
                    if mm_data and not self.runner.model_is_mrope:
                        mm_kwargs = self.multi_modal_input_mapper(mm_data)
                    else:
                        mm_kwargs = mm_data
                    if mm_kwargs is not None:
                        multi_modal_kwargs_list.append(mm_kwargs)
                    if self.runner.model_is_mrope and mm_data:
                        image_grid_thw = mm_kwargs.get("image_grid_thw", None)
                        video_grid_thw = mm_kwargs.get("video_grid_thw", None)
                        assert image_grid_thw is not None or video_grid_thw is not None, (
                            "mrope embedding type requires multi-modal input mapper "
                            "returns 'image_grid_thw' or 'video_grid_thw'.")

                        hf_config = self.runner.model_config.hf_config
                        token_ids = seq_data.get_token_ids()
                        temp_mrope_input_positions, mrope_position_delta = \
                            MRotaryEmbedding.get_input_positions(
                                token_ids,
                                image_grid_thw=image_grid_thw,
                                video_grid_thw=video_grid_thw,
                                image_token_id=hf_config.image_token_id,
                                video_token_id=hf_config.video_token_id,
                                vision_start_token_id=hf_config.vision_start_token_id,
                                vision_end_token_id=hf_config.vision_end_token_id,
                                spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                                context_len=0,
                            )
                        seq_data.mrope_position_delta = mrope_position_delta
                        if mrope_input_positions is None:
                            mrope_input_positions = [[] for _ in range(3)]
                        for idx in range(3):
                            # msections = temp_mrope_input_positions
                            # for _seq_mrope_input_positions in msections:
                            mrope_input_positions[idx].extend(
                                temp_mrope_input_positions[idx])
                if is_prompt:
                    assert len(seq_ids) == 1
                    num_prefills += 1
                    num_prefill_tokens += len(tokens)
                    prefill_seq_lens.append(seq_len)
                else:
                    assert query_len == 1, "Wrong query length in decoding"
                    num_decode_tokens += 1
                    decode_seq_lens = seq_lens
                    #decode_seq_lens.append(seq_len)
                if is_block_tables_empty(seq_group_metadata.block_tables):
                    slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                    continue
                # seq_id: List[int]
                block_table = seq_group_metadata.block_tables[seq_id]

                start_idx=0
                if self.sliding_window is not None:
                    start_idx = max(0, seq_len - self.sliding_window)
                # TODO: add sliding window
                for i in range(context_len, seq_len):
                    if i < start_idx:
                        slot_mapping.append(_PAD_SLOT_ID)
                        continue
                    block_number = block_table[i // self.block_size]
                    block_offset = i % self.block_size
                    # slot_mapping is when we flatteren the blocks, and see which block it is located
                    # block_table is a logical -> to physical transition...
                    # i // block_size is the logical block number
                    slot_mapping.append(block_number * self.block_size + block_offset)
        max_decode_seq_len = max(decode_seq_lens, default=0)
        is_prompt = (self.seq_group_metadata_list[0].is_prompt
                if self.seq_group_metadata_list else None)
        need_block_table = False
        if  self.scheduler_config.chunked_prefill_enabled or not is_prompt or self.cache_config.enable_prefix_caching:
            need_block_table = True
            # max_block_table_len = max(
            #     len(block_table) for block_table in block_tables)
            block_tables = make_tensor_with_pad(
                block_tables,
                # max_len=max_block_table_len,
                pad=0,
                dtype=torch.int,
                device=self.device,
            )

        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device=self.device)

        if self.runner.model_is_mrope and mrope_input_positions is not None:
            input_positions_tensor = torch.tensor(mrope_input_positions,
                                                  dtype=torch.long,
                                                  device=self.device)
        else:
            input_positions_tensor = torch.tensor(input_positions,
                                                  dtype=torch.long,
                                                  device=self.device)

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device=self.device)
        if need_block_table:
            seq_lens_tensor = torch.tensor(seq_lens,
                                        dtype=torch.int,
                                        device=self.device)
        else:
            seq_lens_tensor = torch.tensor([])
        if self.scheduler_config.chunked_prefill_enabled or self.cache_config.enable_prefix_caching:
            context_lens_tensor = torch.tensor(context_lens,
                                            dtype=torch.int,
                                            device=self.device)
            query_lens_tensor = torch.tensor(query_lens,
                                            dtype=torch.long,
                                            device=self.device)
            query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                        dtype=torch.int32,
                                        device=self.device)

            torch.cumsum(query_lens_tensor,
                        dim=0,
                        dtype=query_start_loc.dtype,
                        out=query_start_loc[1:])
        else:
            context_lens_tensor = context_lens
            query_start_loc = query_lens

        # Generate attn_metadata
        attn_metadata = self.attn_backend.make_metadata(
            # FIXME: Later maybe we can get rid of this parameter
            is_prompt=is_prompt, #1
            num_prefills=num_prefills, # 6
            slot_mapping=slot_mapping_tensor, # 2
            num_prefill_tokens=num_prefill_tokens, # 7
            num_decode_tokens=num_decode_tokens, # 8
            seq_lens=seq_lens, # 3
            seqlen_q=torch.tensor([]), # 4
            multi_modal_placeholder_index_maps=None,
            # max_seqlen=max_seqlen, # 5
            max_seqlen=max(query_lens),
            seq_lens_tensor=seq_lens_tensor, # 9
            # max_query_len=max_query_len,
            max_decode_seq_len=max_decode_seq_len, # 10
            query_start_loc=query_start_loc,
            # seq_start_loc=seq_start_loc,
            context_lens=context_lens_tensor,
            block_tables=block_tables if need_block_table else torch.tensor([], device=self.device, dtype=torch.int) # 11
        )
        if multi_modal_kwargs_list is not None and len(multi_modal_kwargs_list) > 0:
            multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)
        else:
            multi_modal_kwargs = None
        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            multi_modal_kwargs=multi_modal_kwargs,
            seq_lens=seq_lens,
            query_lens=query_lens,
        )

    def _prepare_prompt(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               BatchedTensorInputs, List[int], List[int], Set[LoRARequest]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        multi_modal_kwargs_list: List[MultiModalKwargs] = []
        multi_modal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        for seq_group_metadata in seq_group_metadata_list:
            assert seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            computed_len = seq_data.get_num_computed_tokens()
            seq_len = len(prompt_tokens)

            seq_lens.append(seq_len)  # Prompt token num
            input_tokens.extend(prompt_tokens)  # Token ids

            # Token position ids
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(list(range(computed_len, seq_len)))
            mm_data = seq_group_metadata.multi_modal_data
            if mm_data:
                mm_kwargs = self.multi_modal_input_mapper(mm_data)
                multi_modal_kwargs_list.append(mm_kwargs)
            if self.runner.model_is_mrope and mm_data:
                image_grid_thw = mm_kwargs.get("image_grid_thw", None)
                video_grid_thw = mm_kwargs.get("video_grid_thw", None)
                assert image_grid_thw is not None or video_grid_thw is not None, (
                    "mrope embedding type requires multi-modal input mapper "
                    "returns 'image_grid_thw' or 'video_grid_thw'.")

                hf_config = self.runner.model_config.hf_config
                token_ids = seq_data.get_token_ids()
                temp_mrope_input_positions, mrope_position_delta = \
                    MRotaryEmbedding.get_input_positions(
                        token_ids,
                        image_grid_thw=image_grid_thw,
                        video_grid_thw=video_grid_thw,
                        image_token_id=hf_config.image_token_id,
                        video_token_id=hf_config.video_token_id,
                        vision_start_token_id=hf_config.vision_start_token_id,
                        vision_end_token_id=hf_config.vision_end_token_id,
                        spatial_merge_size=hf_config.vision_config.spatial_merge_size,
                        context_len=0,
                    )
                seq_data.mrope_position_delta = mrope_position_delta
                mrope_input_positions = [[] for _ in range(3)]
                for idx in range(3):
                    # msections = temp_mrope_input_positions
                    # for _seq_mrope_input_positions in msections:
                    mrope_input_positions[idx].extend(
                        temp_mrope_input_positions[idx])
                input_positions = mrope_input_positions


            if self.runner.lora_config:
                lora_id = seq_group_metadata.lora_int_id

                if lora_id > 0:
                    lora_requests.add(seq_group_metadata.lora_request)

                lora_index_mapping += [lora_id] * (seq_len - computed_len)
                lora_prompt_mapping.extend(
                    [lora_id] *
                    (seq_len -
                     computed_len if seq_group_metadata.sampling_params
                     and seq_group_metadata.sampling_params.prompt_logprobs
                     is not None else 1))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([_PAD_SLOT_ID] * seq_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            # Mask the [0, start_idx) tokens of the prompt with _PAD_SLOT_ID,
            # where start_idx is max(0, seq_len - sliding_window).
            # For example, if the prompt len is 10, sliding window is 8, and
            # block size is 4, the first two tokens are masked and the slot
            # mapping will be [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
            start_idx = 0
            if self.sliding_window is not None:
                start_idx = max(0, seq_len - self.sliding_window)

            for i in range(computed_len, seq_len):
                if i < start_idx:
                    slot_mapping.append(_PAD_SLOT_ID)
                    continue

                block_number = block_table[i //
                                           self.block_size]  # type: ignore
                block_offset = i % self.block_size  # type: ignore
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        num_prompt_tokens = len(input_tokens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)  # type: ignore
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)  # type: ignore
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            multi_modal_placeholder_maps.items()
        }

        max_seqlen = max(seq_lens)
        tmp = [0]
        tmp.extend(seq_lens)
        seqlen = torch.tensor(tmp)
        seqlen_q = torch.cumsum(seqlen, dim=0).to(device=self.device)

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=True,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            seq_lens=seq_lens,
            seqlen_q=seqlen_q,
            max_seqlen=max_seqlen,
            seq_lens_tensor=torch.tensor([]),
            max_decode_seq_len=0,
            num_prefills=len(seq_lens),
            num_prefill_tokens=num_prompt_tokens,
            num_decode_tokens=0,
            block_tables=torch.tensor([], device=self.device, dtype=torch.int),
        )

        multi_modal_kwargs = MultiModalKwargs.batch(multi_modal_kwargs_list)
        return (input_tokens, input_positions, attn_metadata, seq_lens,
                multi_modal_kwargs, lora_index_mapping, lora_prompt_mapping,
                lora_requests)

    def _prepare_decode(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata, List[int],
               List[int], Set[LoRARequest]]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []
        seq_lens: List[int] = []
        block_tables: List[List[int]] = []
        lora_index_mapping: List[int] = []
        lora_prompt_mapping: List[int] = []
        lora_requests: Set[LoRARequest] = set()

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            assert seq_group_metadata.token_chunk_size == 1

            seq_ids = list(seq_group_metadata.seq_data.keys())
            lora_id = seq_group_metadata.lora_int_id

            if lora_id > 0:
                lora_requests.add(seq_group_metadata.lora_request)

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                seq_len = seq_data.get_len()
                position = seq_len - 1
                input_positions.append(position)

                seq_len = seq_len if self.sliding_window is None else min(
                    seq_len, self.sliding_window)
                seq_lens.append(seq_len)

                block_table = seq_group_metadata.block_tables[seq_id]
                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)
                lora_index_mapping.append(lora_id)
                lora_prompt_mapping.append(lora_id)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                block_tables.append(block_table)

        max_decode_seq_len = max(seq_lens)

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        input_positions = torch.tensor(input_positions,
                                       dtype=torch.long,
                                       device=self.device)
        slot_mapping = torch.tensor(slot_mapping,
                                    dtype=torch.long,
                                    device=self.device)
        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device=self.device)

        block_tables = make_tensor_with_pad(
            block_tables,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )

        attn_metadata = self.attn_backend.make_metadata(
            is_prompt=False,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            seq_lens=seq_lens,
            seqlen_q=torch.tensor([]),
            max_seqlen=0,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_seq_len=max_decode_seq_len,
            num_prefill_tokens=0,
            num_decode_tokens=len(input_tokens),
            num_prefills=0,
            block_tables=block_tables,
        )
        return (input_tokens, input_positions, attn_metadata,
                lora_index_mapping, lora_prompt_mapping, lora_requests)


class XPUModelRunnerBase(ModelRunnerBase[TModelInputForXPU]):

    _model_input_cls: Type[TModelInputForXPU]
    _builder_cls: Type[ModelInputForXPUBuilder]

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        return_hidden_states: bool = False,
        input_registry: InputRegistry = INPUT_REGISTRY,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
    ):

        ModelRunnerBase.__init__(self, vllm_config=vllm_config)
        model_config = self.model_config
        cache_config = self.cache_config
        self.is_driver_worker = is_driver_worker
        self.return_hidden_states = return_hidden_states

        self.device = self.device_config.device

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        )

        # Multi-modal data support
        self.input_registry = input_registry
        self.mm_registry = mm_registry
        self.multi_modal_input_mapper = mm_registry \
            .create_input_mapper(model_config)
        self.mm_registry.init_mm_limits_per_prompt(self.model_config)

        # Lazy initialization.
        self.model: nn.Module  # Set after init_Model
        # Set after load_model.
        self.lora_manager: Optional[LRUCacheWorkerLoRAManager] = None

        self.sampling_metadata_cache: SamplingMetadataCache = \
              SamplingMetadataCache() \
                if self.parallel_config.pipeline_parallel_size == 1 else None

    def load_model(self) -> None:
        with DeviceMemoryProfiler() as m:
            self.model = get_model(vllm_config=self.vllm_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

        if self.lora_config:
            assert supports_lora(self.model), "Model does not support LoRA"
            assert not supports_multimodal(
                self.model
            ), "To be tested: Multi-modal model with LoRA settings."

            self.lora_manager = LRUCacheWorkerLoRAManager(
                self.scheduler_config.max_num_seqs,
                self.scheduler_config.max_num_batched_tokens,
                self.vocab_size,
                self.lora_config,
                self.device,
                self.model.embedding_modules,
                self.model.embedding_padding_modules,
                max_position_embeddings=self.model.config.
                max_position_embeddings,
            )
            self.model = self.lora_manager.create_lora_manager(self.model)

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

    @property
    def model_is_mrope(self) -> bool:
        """Detect if the model has "mrope" rope_scaling type.
        mrope requires keep "rope_deltas" between prompt and decoding phases."""
        rope_scaling = getattr(self.model_config.hf_config, "rope_scaling", {})
        if rope_scaling is None:
            return False
        return rope_scaling.get("type", None) == "mrope" or rope_scaling.get("mrope_section", None) is not None

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs

        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for multi-modal encoding, which
        # needs to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            dummy_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: dummy_data.seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_data.multi_modal_data,
                multi_modal_placeholders=dummy_data.multi_modal_placeholders)
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        # use an empty tensor instead of `None`` to force Dynamo to pass
        # it by reference, rather by specializing on the value ``None``.
        # the `dtype` argument does not matter, and we use `float32` as
        # a placeholder (it has wide hardware support).
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.xpu.synchronize()
        return

    def save_sharded_state(
        self,
        path: str,
        pattern: Optional[str] = None,
        max_size: Optional[int] = None,
    ) -> None:
        from vllm.model_executor.model_loader.loader import ShardedStateLoader
        ShardedStateLoader.save_model(
            self.model,
            path,
            pattern=pattern,
            max_size=max_size,
        )

    def save_tensorized_model(
        self,
        tensorizer_config: TensorizerConfig,
    ) -> None:
        from vllm.model_executor.model_loader.loader import TensorizerLoader
        TensorizerLoader.save_model(
            self.model,
            tensorizer_config=tensorizer_config,
        )

    def _prepare_model_input_tensors(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        finished_requests_ids: Optional[List[str]] = None
    ) -> TModelInputForXPU:
        """Helper method to prepare the model input based on a given sequence
        group. Prepares metadata needed for the base model forward pass but not
        metadata for possible additional steps, e.g., sampling.

        """
        builder = self._builder_cls(weakref.proxy(self), finished_requests_ids)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)

        return builder.build()  # type: ignore


class XPUModelRunner(XPUModelRunnerBase[ModelInputForXPUWithSamplingMetadata]):
    _model_input_cls: Type[ModelInputForXPUWithSamplingMetadata] = (
        ModelInputForXPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForXPUBuilder] = ModelInputForXPUBuilder

    def make_model_input_from_broadcasted_tensor_dict(
            self,
            tensor_dict: Dict[str,
                              Any]) -> ModelInputForXPUWithSamplingMetadata:
        return (
            ModelInputForXPUWithSamplingMetadata.from_broadcasted_tensor_dict(
                tensor_dict,
                attn_backend=self.attn_backend,
            ))

    def prepare_model_input(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[List[str]] = None
    ) -> ModelInputForXPUWithSamplingMetadata:
        """Prepare the model input based on a given sequence group, including
        metadata for the sampling step.

        """
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list, finished_requests_ids)
        # Sampling metadata is only required for the final pp group
        generators = self.get_generators(finished_requests_ids)
        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            model_input.seq_lens,
            model_input.query_lens,
            self.device,
            pin_memory=False,
            generators=generators,
            cache=self.sampling_metadata_cache)

        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   virtual_engine=virtual_engine)

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForXPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
    ) -> Optional[List[SamplerOutput]]:
        if num_steps > 1:
            raise ValueError(
                "XPUModelRunner does not support multi-step execution.")

        if self.lora_config:
            assert model_input.lora_requests is not None
            assert model_input.lora_mapping is not None
            self.set_active_loras(model_input.lora_requests,
                                  model_input.lora_mapping)

        model_executable = self.model
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_start_time = time.time()

        hidden_or_intermediate_states = model_executable(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            kv_caches=kv_caches,
            attn_metadata=model_input.attn_metadata,
            intermediate_tensors=intermediate_tensors,
            **MultiModalKwargs.as_kwargs(model_input.multi_modal_kwargs or {},
                                         device=self.device))
        # Compute the logits in the last pipeline stage.
        if not get_pp_group().is_last_rank:
            return hidden_or_intermediate_states

        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time):
            model_forward_end_time = time.time()

        # Compute the logits.
        logits = self.model.compute_logits(hidden_or_intermediate_states,
                                           model_input.sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        if model_input.async_callback is not None:
            model_input.async_callback()

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        if (self.observability_config is not None
                and self.observability_config.collect_model_forward_time
                and output is not None):
            model_forward_time = (model_forward_end_time -
                                  model_forward_start_time)
            # If there are multiple workers, we are still tracking the latency
            # from the start time of the driver worker to the end time of the
            # driver worker. The model forward time will then end up covering
            # the communication time as well.
            output.model_forward_time = model_forward_time

        return [output]

    def set_active_loras(self, lora_requests: Set[LoRARequest],
                         lora_mapping: LoRAMapping) -> None:
        if not self.lora_manager:
            raise RuntimeError("LoRA is not enabled.")
        self.lora_manager.set_active_adapters(lora_requests, lora_mapping)
