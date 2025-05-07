# SPDX-License-Identifier: Apache-2.0

# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/models/ovis/modeling_ovis.py
# Copyright 2023 The vLLM team.
# Copyright 2023 HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Ovis2 model."""
from typing import (Iterable, List, Literal, Mapping, Optional, Set, Tuple,
                    TypedDict, Union)

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import gumbel_softmax, pad, softmax
from transformers import BatchFeature

from vllm.config import VllmConfig
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.models.aimv2 import AIMv2Model
from vllm.model_executor.models.siglip import SiglipVisionModel
from vllm.model_executor.models.utils import (AutoWeightsLoader, flatten_bn,
                                              init_vllm_registered_model,
                                              maybe_prefix)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargs)
from vllm.multimodal.parse import ImageSize, MultiModalDataItems
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement)
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.ovis2 import (BaseVisualTokenizerConfig,
                                                   OvisConfig)
from vllm.transformers_utils.processors.ovis2 import OvisProcessor

from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .utils import merge_multimodal_embeddings

# Cannot find the following number from hf config.
IMAGE_TOKEN = "<image>"
NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT = 256
IMAGE_INDICATOR_IDS = [-301, -302, -303, -304, -305]

IMAGE_PAD_TOKEN_MAP = {
    "gemma2": "<unused0>",
    "llama": "<|reserved_special_token_0|>",
    "qwen2": "<|image_pad|>",
}
IMAGE_PAD_TOKEN_ID_MAP = {
    "gemma2": 7,
    "llama": 128002,
    "qwen2": 151655,
}


def st_argmax(y_soft: torch.Tensor, dim: int):  # straight-through softmax
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(
        y_soft, memory_format=torch.legacy_contiguous_format).scatter_(
            dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret


class VisualTokenizer(torch.nn.Module):

    def __init__(
        self,
        config: BaseVisualTokenizerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.backbone = self._init_backbone(
            config=config.backbone_config,
            quant_config=quant_config,
            prefix=f"{prefix}.backbone",
        )
        # reserved tokens for IMAGE_INDICATORS
        head_dim = config.vocab_size - len(IMAGE_INDICATOR_IDS)
        self.head = torch.nn.Sequential(
            ReplicatedLinear(
                config.backbone_config.hidden_size * config.hidden_stride *
                config.hidden_stride,
                head_dim,
                bias=False,
            ), torch.nn.LayerNorm(head_dim))

    def _init_backbone(
        self,
        config: BaseVisualTokenizerConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        model_type = config.backbone_config.model_type
        if model_type == "aimv2":
            return AIMv2Model(
                config=config.backbone_config,
                quant_config=quant_config,
                prefix=prefix,
            )
        elif model_type == "siglip_vision_model":
            return SiglipVisionModel(
                config=config.backbone_config,
                quant_config=quant_config,
                prefix=prefix,
            )
        raise ValueError(
            f"Unsupported visual tokenizer model_type: {model_type}")

    @property
    def dtype(self):
        return self.backbone.dtype

    @property
    def device(self):
        return self.backbone.device

    def tokenize(self, logits):
        if self.config.tokenize_function == 'softmax':
            tokens = softmax(logits, dim=-1)
        elif self.config.tokenize_function == 'gumbel_argmax':
            tokens = gumbel_softmax(logits, tau=self.config.tau, hard=True)
        elif self.config.tokenize_function == 'st_argmax':
            tokens = st_argmax(logits, dim=-1)
        else:
            raise ValueError(
                'Invalid `max_type`, expected softmax or gumbel_argmax '
                f'or st_argmax, but got {self.config.tokenize_function}')
        return tokens

    def encode(self, pixel_values):
        features = self.backbone(pixel_values)
        if self.config.drop_cls_token:
            features = features[:, 1:, :]

        # merge number of `hidden_stride * hidden_stride` hidden states together
        # to reduce token sequence length
        # e.g., for hidden_stride=2, this leads to a token length reduction:
        # 1024 -> 256 for aimv2
        if self.config.hidden_stride > 1:
            # this `d` maybe different from the above `d``
            n, L, d = features.shape
            sqrt_l = int(L**0.5)
            assert sqrt_l**2 == L, (
                "The token sequence length should be a perfect square.")
            features = features.reshape(n, sqrt_l, sqrt_l, d)
            pl = (self.config.hidden_stride -
                  (sqrt_l %
                   self.config.hidden_stride)) % self.config.hidden_stride
            features = pad(features, (0, 0, 0, pl, 0, pl), "constant", 0)
            sqrt_l += pl
            features = features.reshape(n, sqrt_l // self.config.hidden_stride,
                                        self.config.hidden_stride,
                                        sqrt_l // self.config.hidden_stride,
                                        self.config.hidden_stride, d)
            # [n, sqrt_l/hs, sqrt_l/hs, hs, hs, d]
            features = features.permute(0, 1, 3, 2, 4, 5)
            # [n, sqrt_l/hs, sqrt_l/hs, hs*hs*d]
            features = features.flatten(3)
            # [n, sqrt_l/hs*sqrt_l/hs, hs*hs*d]
            features = features.reshape(
                n, -1,
                self.config.hidden_stride * self.config.hidden_stride * d)

        return features

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[BatchSize, ImageShape] -> [BatchSize, Token, VocabSize]"""
        features = self.encode(pixel_values)
        logits, _ = self.head[0](
            features)  # we spllit the sequncial here for not throwing an error
        logits = self.head[1](logits)
        tokens = self.tokenize(logits)
        # tokens' shape is [BatchSize, #Token, VocabSize-5], so padding with
        # [BatchSize, #Token, 5], after which, tokens' shape should become
        # [BatchSize, #Token, VocabSize]
        batch_size, token_len, _ = tokens.shape
        padding_tensor = torch.zeros(size=(batch_size, token_len,
                                           len(IMAGE_INDICATOR_IDS)),
                                     dtype=tokens.dtype,
                                     device=tokens.device,
                                     layout=tokens.layout,
                                     requires_grad=False)
        tokens = torch.cat((tokens, padding_tensor), dim=2)
        return tokens


class Ovis2ImagePatchInputs(TypedDict):
    type: Literal["image_patches"]
    flat_data: torch.Tensor
    """
    Shape: 
    `(batch_size * num_patches, patch_size_x * patch_size_y * num_channels)`
    """

    inducator_tokens: torch.Tensor
    """
    Shape: 
    `(batch_size * (num_patches + 1))`
    """

    patches_per_image: List[int]
    """
    List of number of total patches for each image in the batch.
    This is used to restore the first two dimensions of `flat_data`.
    """


class VisualEmbedding(torch.nn.Embedding):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, visual_tokens: Tensor) -> Tensor:
        if visual_tokens.dtype in [
                torch.int8, torch.int16, torch.int32, torch.int64, torch.long
        ]:
            return super().forward(visual_tokens)
        return torch.matmul(visual_tokens, self.weight)

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class Ovis2ProcessingInfo(BaseProcessingInfo):

    def get_hf_config(self):
        return self.ctx.get_hf_config(OvisConfig)

    def get_hf_processor(self, **kwargs):
        return self.ctx.get_hf_processor(OvisProcessor)

    def get_image_pad_token(self) -> str:
        hf_text_config = self.get_hf_config().get_text_config()
        text_model_type = hf_text_config.model_type
        return IMAGE_PAD_TOKEN_MAP.get(text_model_type)

    def get_image_processor(self) -> OvisProcessor:
        return self.get_hf_processor().image_processor  # type: ignore

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {  # 32k is model token limit at the moment
            "image":
            self.get_hf_config().multimodal_max_length //
            ((9 + 1) * NUMBER_OF_TOKEN_TO_RESERVE_FOR_SEGMENT)
        }

    def get_image_size_with_most_features(self) -> ImageSize:
        image_processor = self.get_image_processor()
        return ImageSize(width=image_processor.size['shortest_edge'] * 9 * 2,
                         height=image_processor.size['shortest_edge'] * 9 * 2)


class Ovis2DummyInputsBuilder(BaseDummyInputsBuilder[Ovis2ProcessingInfo]):

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return IMAGE_TOKEN * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)

        target_width, target_height = \
            self.info.get_image_size_with_most_features()

        mm_data = {
            "image":
            self._get_dummy_images(width=target_width,
                                   height=target_height,
                                   num_images=num_images),
        }
        return mm_data


class Ovis2MultiModalProcessor(BaseMultiModalProcessor[Ovis2ProcessingInfo]):

    def image_indicators_to_visual_tokens(
        self,
        image_indicators: list[int],
    ) -> list[int]:
        """
        Filter image indicators placeholders and convert them to corresponding 
        tokens in visual tokenizer.
        For example, [-301, -300, -302, -300, -303, -300, -304, -300, -305]
        should return [vocab_size-1, vocab_size-2, ..., vocab_size-5]
        """
        hf_config = self.info.get_hf_config()
        vte_vocab_size = hf_config.visual_tokenizer_config.vocab_size
        # -300 is image_atom token, filter them out
        return [vte_vocab_size + x + 300 for x in image_indicators if x < -300]

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        if not mm_data:
            #    # Avoid warning from HF logger for text-only input
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            # prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids) nope
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        processed_outputs = super()._call_hf_processor(
            prompt=prompt,
            mm_data=mm_data,
            mm_kwargs=mm_kwargs,
        )

        hf_processor = self.info.get_hf_processor()
        image_indicators = [
            hf_processor.construct_image_indicators(grid)
            for grid in processed_outputs["grids"]
        ]
        indicator_tokens = [
            self.image_indicators_to_visual_tokens(indicator)
            for indicator in image_indicators
        ]
        processed_outputs["indicator_tokens"] = indicator_tokens
        return processed_outputs

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:

        return prompt_tokens

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(pixel_values=MultiModalFieldConfig.batched("image"),
                    grids=MultiModalFieldConfig.batched("image"),
                    indicator_tokens=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargs,
    ) -> list[PromptReplacement]:

        def get_replacement_ovis(item_idx):
            grid = out_mm_kwargs["grids"][item_idx]

            hf_processor = self.info.get_hf_processor()
            return hf_processor.construct_image_placeholders(grid)

        return [
            PromptReplacement(
                modality="image",
                target=IMAGE_TOKEN,
                replacement=get_replacement_ovis,
            ),
        ]


@MULTIMODAL_REGISTRY.register_processor(Ovis2MultiModalProcessor,
                                        info=Ovis2ProcessingInfo,
                                        dummy_inputs=Ovis2DummyInputsBuilder)
class Ovis2ForConditionalGeneration(nn.Module, SupportsMultiModal):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config: OvisConfig = config
        self.llm = init_vllm_registered_model(
            vllm_config=vllm_config.with_hf_config(config.get_text_config()),
            prefix=maybe_prefix(prefix, "llm"),
        )

        self.visual_tokenizer = VisualTokenizer(
            config=config.visual_tokenizer_config,
            quant_config=quant_config,
            prefix=f"{prefix}.visual_tokenizer",
        )

        self.vte = VisualEmbedding(
            self.config.visual_tokenizer_config.vocab_size,
            self.config.hidden_size)

        text_model_type = self.config.get_text_config().model_type
        self.image_pad_token_id = IMAGE_PAD_TOKEN_ID_MAP[text_model_type]

        # TODO(Isotr0py): PP support
        # self.make_empty_intermediate_tensors = (
        #    self.language_model.make_empty_intermediate_tensors)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Ovis2ImagePatchInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        indicator_tokens = kwargs.pop("indicator_tokens", None)

        if pixel_values is None and indicator_tokens is None:
            return None

        if pixel_values is not None and indicator_tokens is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            if not isinstance(indicator_tokens, (torch.Tensor, list)):
                raise ValueError("Incorrect type of indicator_tokens. "
                                 f"Got type: {type(pixel_values)}")

            return Ovis2ImagePatchInputs(
                type="image_patches",
                flat_data=flatten_bn(flatten_bn(pixel_values), concat=True),
                patches_per_image=[
                    x.shape[0] for x in flatten_bn(pixel_values)
                ],
                indicator_tokens=flatten_bn(flatten_bn(indicator_tokens),
                                            concat=True),
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_input(
            self, image_input: Ovis2ImagePatchInputs) -> MultiModalEmbeddings:
        image_patches_flat = image_input["flat_data"]
        patches_per_image = image_input["patches_per_image"]
        indicator_tokens = image_input["indicator_tokens"]

        indicator_per_image = list(
            map(lambda x: x + 1 if x > 1 else x + 2, patches_per_image))

        target_dtype = self.visual_tokenizer.dtype
        visual_tokens = self.visual_tokenizer(
            image_patches_flat.to(target_dtype))
        visual_embeds = self.vte(visual_tokens)  # 1:1 numeric eq.

        indicator_embeds = self.vte(indicator_tokens)
        indicator_embeds_per_image = indicator_embeds.split(
            indicator_per_image)

        visual_embeds_per_image = visual_embeds.split(patches_per_image, dim=0)
        vision_embeddings = []
        for indicator, visual in zip(indicator_embeds_per_image,
                                     visual_embeds_per_image):
            vision_embeddings_per_image = []
            for i in range(visual.shape[0]):
                vision_embeddings_per_image.append(
                    torch.cat([indicator[i:i + 1], visual[i]], dim=0))
            vision_embeddings_per_image.append(indicator[i + 1:])
            vision_embeddings.append(
                torch.cat(vision_embeddings_per_image, dim=0))

        return tuple(vision_embeddings)

    def get_multimodal_embeddings(
            self, **kwargs: object) -> Optional[MultiModalEmbeddings]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return None

        image_features = self._process_image_input(image_input)

        return image_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.llm.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.image_pad_token_id)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        if intermediate_tensors is not None:
            inputs_embeds = None

        # NOTE: In v1, inputs_embeds is always generated at model runner, this
        # condition is for v0 compatibility.
        elif inputs_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            inputs_embeds = self.get_input_embeddings(input_ids,
                                                      vision_embeddings)
            input_ids = None

        # up until here we have a inputs_embeds 100% numerical identity
        # between the OG HF Transformers implementation and ours
        hidden_states = self.llm(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.llm.logits_processor(self.llm.lm_head, hidden_states,
                                           sampling_metadata)
        return logits

    def load_weights(self, weights: Iterable[Tuple[str,
                                                   torch.Tensor]]) -> Set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_language_model(self) -> torch.nn.Module:
        return self.llm
