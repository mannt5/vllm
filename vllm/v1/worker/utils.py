# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

import torch

from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.v1.kv_cache_interface import KVCacheGroupSpec


def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings,
    expected_num_items: int,
) -> None:
    """
    Perform sanity checks for the result of
    [`vllm.model_executor.models.SupportsMultiModal.get_multimodal_embeddings`][].
    """
    assert isinstance(mm_embeddings, (list, tuple, torch.Tensor)), (
        "Expected multimodal embeddings to be a list/tuple of 2D tensors, "
        f"or a single 3D tensor, but got {type(mm_embeddings)} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert len(mm_embeddings) == expected_num_items, (
        "Expected number of multimodal embeddings to match number of "
        f"input items: {expected_num_items}, but got {len(mm_embeddings)=} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")

    assert all(e.ndim == 2 for e in mm_embeddings), (
        "Expected multimodal embeddings to be a sequence of 2D tensors, "
        f"but got tensors with shapes {[e.shape for e in mm_embeddings]} "
        "instead. This is most likely due to incorrect implementation "
        "of the model's `get_multimodal_embeddings` method.")


def scatter_mm_placeholders(
    embeds: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Scatter the multimodal embeddings into a contiguous tensor that represents
    the placeholder tokens.

    [`vllm.multimodal.processing.PromptUpdateDetails.is_embed`][].

    Args:
        embeds: The multimodal embeddings.
          Shape: `(num_embeds, embed_dim)`
        is_embed: A boolean mask indicating which positions in the placeholder
          tokens need to be filled with multimodal embeddings.
          Shape: `(num_placeholders, num_embeds)`
    """
    if is_embed is None:
        return embeds

    placeholders = embeds.new_full(
        (is_embed.shape[0], embeds.shape[-1]),
        fill_value=torch.nan,
    )
    placeholders[is_embed] = embeds
    return placeholders


def gather_mm_placeholders(
    placeholders: torch.Tensor,
    is_embed: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Reconstructs the embeddings from the placeholder tokens.

    This is the operation of [scatter_mm_placeholders][].
    """
    if is_embed is None:
        return placeholders

    return placeholders[is_embed]


def initialize_kv_cache_for_kv_sharing(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    kv_caches: dict[str, torch.Tensor],
) -> None:
    """
    Sets up KV cache sharing by reusing the allocated KV caches in `kv_caches`
    for layers that do not allocate its own KV cache, based on the mapping in
    `shared_kv_cache_layers`. Adds these layers to the corresponding KV cache
    group, which is needed to ensure that attention metadata is assigned later.

    Args:
        shared_kv_cache_layers: Layer pairings for cross-layer KV sharing.
            If an Attention layer `layer_name` is in the keys of this dict, it
            means this layer will perform attention using the keys and values
            from the KV cache of `shared_kv_cache_layers[layer_name]`.
        kv_cache_groups: The KV cache groups of the model.
        kv_caches: The allocated kv_caches with layer names as keys.
            Note that layers in shared_kv_cache_layers.keys() are not
            originally included as it only contains layers which have its own
            KV cache allocation.
    """
    # Record index of KV cache group for each layer that allocates a KV cache.
    layer_to_kv_cache_group_idx: dict[str, int] = {}
    for i, kv_cache_group in enumerate(kv_cache_groups):
        for layer_name in kv_cache_group.layer_names:
            layer_to_kv_cache_group_idx[layer_name] = i

    for layer_name, target_layer_name in shared_kv_cache_layers.items():
        kv_caches[layer_name] = kv_caches[target_layer_name]
        group_idx = layer_to_kv_cache_group_idx[target_layer_name]
        kv_cache_groups[group_idx].layer_names.append(layer_name)


def copy_kv_cache_for_layers(
    kv_caches: dict[str, torch.Tensor],
    kv_sharing_layers_mapping: dict[str, str],
    copy_positions_mask: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """
    Copies values from one set of layers to another
    based on a mapping and a mask.

    This function is primarily used when KV sharing
    is enabled especially for spec decoding to copy 
    target model's cache for the specified positions into
    corresponding draft model layer's KV cache.

    Args:
        kv_caches:
            Dictionary mapping layer names to their cache tensors.
        kv_sharing_layers_mapping:
            Mapping the target layers to the source layers.
        copy_positions_mask:
            Boolean mask where True indicates positions to copy.
        slot_mapping:
            Mapping tensor that maps positions to cache slots.
    """
    # Get the positions to copy
    positions = torch.nonzero(copy_positions_mask, as_tuple=True)[0]

    if positions.numel() == 0:
        # No positions to copy
        return

    # Get the corresponding slot mappings for the positions
    slots = slot_mapping[positions]

    # Copy KV cache values from source layers to target layers
    for target_layer, source_layer in kv_sharing_layers_mapping.items():
        if target_layer not in kv_caches or source_layer not in kv_caches:
            continue

        target_kv_cache = kv_caches[target_layer]
        source_kv_cache = kv_caches[source_layer]

        block_size = source_kv_cache.shape[2]

        kv_dim = 2
        # Process in smaller batches to reduce memory overhead
        batch_size = 8192
        num_positions = positions.size(0)

        for start_idx in range(0, num_positions, batch_size):
            end_idx = min(start_idx + batch_size, num_positions)

            # Get batch of slots
            batch_slots = slots[start_idx:end_idx]
            batch_block_indices = batch_slots // block_size
            batch_block_offsets = batch_slots % block_size

            # Create batch-sized indexing tensors
            batch_block_indices_expanded = batch_block_indices.view(
                1, -1, 1, 1, 1)
            batch_block_offsets_expanded = batch_block_offsets.view(
                1, 1, -1, 1, 1)

            # Copy values for this batch
            for kv_idx in range(kv_dim):
                target_kv_cache[
                    kv_idx,
                    batch_block_indices_expanded.squeeze(),
                    batch_block_offsets_expanded.squeeze(), :, :] = (
                        source_kv_cache[
                            kv_idx,
                            batch_block_indices_expanded.squeeze(),
                            batch_block_offsets_expanded.squeeze(), :, :])
