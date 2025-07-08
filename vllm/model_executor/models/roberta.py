# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Iterable
from typing import Optional, Union

import torch
from torch import nn
from transformers import RobertaConfig

from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.pooler import ClassifierPooler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.bert import (BertEmbeddingModel,
                                             BertMMTokenIdsMixin, BertModel,
                                             TokenTypeInputBuilder,
                                             TokenTypeMultiModalProcessor,
                                             TokenTypeProcessingInfo)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors, PoolerOutput

from .bert_with_rope import BertWithRope, JinaRobertaModel
from .interfaces import SupportsCrossEncoding


class RobertaEmbedding(nn.Module):

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.size = config.hidden_size
        self.word_embeddings = VocabParallelEmbedding(config.vocab_size,
                                                      config.hidden_size)
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size,
                                                padding_idx=self.padding_idx)

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size,
                                      eps=config.layer_norm_eps)
        self.position_ids = nn.Parameter(
            torch.empty((1, config.max_position_embeddings)), )

        self.position_embedding_type = config.position_embedding_type
        if self.position_embedding_type != "absolute":
            raise ValueError("Only 'absolute' position_embedding_type" +
                             " is supported")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        apply_layer_norm: bool = True,
    ) -> torch.Tensor:

        # forward was called directly without going
        # throught the multi-modal flow
        if input_ids is not None and position_ids is not None \
            and inputs_embeds is None and token_type_ids is None:
            token_type_ids = torch.zeros(input_ids.size(),
                                         dtype=torch.long,
                                         device=input_ids.device)

        tensors_to_add: list[torch.Tensor] = []

        if inputs_embeds is not None:
            tensors_to_add.append(inputs_embeds)

        if token_type_ids is not None:
            tensors_to_add.append(self.token_type_embeddings(token_type_ids))

        if position_ids is not None:
            tensors_to_add.append(self.position_embeddings(position_ids))

        if input_ids is not None:
            tensors_to_add.append(self.word_embeddings(input_ids))

        embeds = torch.stack(tensors_to_add, dim=0).sum(dim=0)

        if apply_layer_norm:
            return self.LayerNorm(embeds)
        else:
            return embeds


# Adapted from transformers
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: RobertaConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[0, :]  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class RobertaEmbeddingModel(BertEmbeddingModel):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       model: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.padding_idx = vllm_config.model_config.hf_config.pad_token_id

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Fix Roberta positions here outside of the CUDA graph.
        # Because we need the to extract the sequences from
        # input_ids the control flow is data dependent.
        replace_roberta_positions(input_ids=input_ids,
                                  position_ids=positions,
                                  padding_idx=self.padding_idx)

        return self.model(input_ids=input_ids,
                          position_ids=positions,
                          token_type_ids=token_type_ids,
                          inputs_embeds=inputs_embeds,
                          intermediate_tensors=intermediate_tensors)

    def _build_model(self,
                     vllm_config: VllmConfig,
                     prefix: str = "") -> Union[BertModel, BertWithRope]:
        if (vllm_config.model_config.hf_config.position_embedding_type ==
                "rotary"):
            return JinaRobertaModel(vllm_config=vllm_config, prefix=prefix)
        else:
            return BertModel(vllm_config=vllm_config,
                             prefix=prefix,
                             embedding_class=RobertaEmbedding)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        weights = self.hf_to_vllm_mapper.apply(weights)
        # Separate weights in "roberta"-prefixed and all else (not in memory).
        # For use with models like FacebookAI/roberta-base.
        bert_weights, task_weights = roberta_task_weights_filter(weights)
        loaded = self.model.load_weights(bert_weights)
        if not len(loaded):
            # Fix for models like `sentence-transformers/stsb-roberta-base-v2`
            # which use the same architecture, but have no "roberta" prefix.
            loaded = self.model.load_weights(task_weights)
        assert len(loaded), "Unable to load RobertaEmbeddingModel"


@MULTIMODAL_REGISTRY.register_processor(TokenTypeMultiModalProcessor,
                                        info=TokenTypeProcessingInfo,
                                        dummy_inputs=TokenTypeInputBuilder)
class RobertaForSequenceClassification(nn.Module, BertMMTokenIdsMixin,
                                       SupportsCrossEncoding):
    """A model that uses Roberta to provide embedding functionalities.

   This class encapsulates the BertModel and provides an interface for
   embedding operations and customized pooling functions.

   Attributes:
       roberta: An instance of BertModel used for forward operations.
       _pooler: An instance of Pooler used for pooling operations.
   """

    jina_to_vllm_mapper = WeightsMapper(
        orig_to_new_substr={
            'emb_ln': "embeddings.LayerNorm",
            'layers': "layer",
            'mixer.Wqkv': "attention.self.qkv_proj",
            'mixer.out_proj': "attention.output.dense",
            'norm1': "attention.output.LayerNorm",
            'mlp.fc1': "intermediate.dense",
            'mlp.fc2': "output.dense",
            'norm2': "output.LayerNorm",
        })

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.padding_idx = vllm_config.model_config.hf_config.pad_token_id

        self.num_labels = config.num_labels
        self.roberta = BertModel(vllm_config=vllm_config,
                                 prefix=maybe_prefix(prefix, "bert"),
                                 embedding_class=RobertaEmbedding,
                                 add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)

        self._pooler = ClassifierPooler(vllm_config.model_config,
                                        self.classifier)
        self.input_ids: Optional[torch.Tensor] = None

    def maybe_store_input_ids(self, input_ids: torch.Tensor):
        self.input_ids = input_ids

    def get_language_model(self) -> torch.nn.Module:
        return self.roberta

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        bert_weights, task_weights = roberta_task_weights_filter(weights)
        bert_weights = self.jina_to_vllm_mapper.apply(bert_weights)

        self.roberta.load_weights(bert_weights)

        params_dict = dict(self.named_parameters())

        for name, loaded_weight in task_weights:
            if name.startswith("classifier"):
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    def pooler(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> Optional[PoolerOutput]:
        return self._pooler(hidden_states, pooling_metadata)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _input_ids = input_ids if input_ids is not None else self.input_ids
        replace_roberta_positions(input_ids=_input_ids,
                                  position_ids=positions,
                                  padding_idx=self.padding_idx)
        return self.roberta(input_ids=input_ids,
                            position_ids=positions,
                            inputs_embeds=inputs_embeds,
                            intermediate_tensors=intermediate_tensors,
                            token_type_ids=token_type_ids)


# Adapted from transformers
def create_position_ids_from_input_ids(input_ids,
                                       padding_idx,
                                       past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()

    incremental_indices = (torch.cumsum(mask, dim=0).type_as(mask) +
                           past_key_values_length) * mask

    return incremental_indices.long() + padding_idx


def replace_roberta_positions(input_ids: torch.Tensor,
                              position_ids: torch.Tensor,
                              padding_idx: int) -> None:

    seq_lens: Optional[torch.Tensor] = None
    attn_metadata = get_forward_context().attn_metadata
    if attn_metadata is not None:  # can be None during warmup
        if isinstance(attn_metadata, dict):
            attn_metadata = next(iter(attn_metadata.values()))
        # TODO: remove "seq_lens_tensor" after V0 is removed
        seq_lens = getattr(attn_metadata, "seq_lens_tensor",
                           getattr(attn_metadata, "seq_lens", None))

    if seq_lens is not None:
        assert isinstance(seq_lens, torch.Tensor)

        # Replace position ids because in RoBERTa models
        # they have to start at padding_idx + 1 and ignore
        # existing padding tokens
        # References:
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L133
        # - https://github.com/huggingface/transformers/blob/a3d69a8994d673899608a7c17fbf4f953f50474e/src/transformers/models/roberta/modeling_roberta.py#L1669
        token_list = torch.split(input_ids[:torch.sum(seq_lens)],
                                 seq_lens.tolist())

        offset = 0
        for tokens in token_list:
            length = tokens.shape[0]
            position_ids[offset:offset+length] = \
                create_position_ids_from_input_ids(tokens, padding_idx)
            offset = offset + length


def roberta_task_weights_filter(
    all_weights: Iterable[tuple[str, torch.Tensor]]
) -> tuple[Iterable[tuple[str, torch.Tensor]], Iterable[tuple[str,
                                                              torch.Tensor]]]:
    """
    Separate task-specific weights that are applied on top
    of the encoder-decoder bert base.
    To do so, return two generators over the original iterator.
    Also, remove the "roberta." prefix to make it loadable
    from vanilla BertModel.
    """
    # Copy of a lazy iterator without in-memory overhead so both
    # iterators can be iterated upon independently.
    all_weights1, all_weights2 = itertools.tee(all_weights)

    def encoder_decoder_weights():
        for name, weight in all_weights1:
            if name.startswith("roberta."):
                yield (name[len("roberta."):], weight)

    return encoder_decoder_weights(), ((n, w) for n, w in all_weights2
                                       if not n.startswith("roberta."))
