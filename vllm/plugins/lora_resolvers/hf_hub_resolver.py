# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from typing import Optional

from huggingface_hub import HfApi, snapshot_download

import vllm.envs as envs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolverRegistry
from vllm.plugins.lora_resolvers.filesystem_resolver import FilesystemResolver


class HfHubResolver(FilesystemResolver):

    def __init__(self, repo_name: str):
        self.repo_name = repo_name
        # At initialization time, get all directories
        # in the repo containing an adapter_config.json;
        # these are the dirs we will allows downloads for
        # for potential LoRA requests.
        repo_files = HfApi().list_repo_files(repo_id=repo_name)
        self.adapter_dirs = [
            name.split("/")[0] for name in repo_files
            if name.endswith("adapter_config.json")
        ]

    async def resolve_lora(self, base_model_name: str,
                           lora_name: str) -> Optional[LoRARequest]:
        """Resolves potential LoRA requests in a remote repo on HF Hub.
        This is effectively the same behavior as the filesystem resolver, but
        with an extra guard + snapshot_download on dirs containing an adapter
        config prior to inspecting the cached dir to build a potential LoRA
        request.
        """
        if lora_name in self.adapter_dirs:
            repo_path = snapshot_download(repo_id=self.repo_name,
                                          allow_patterns=f"{lora_name}/*")
            lora_path = os.path.join(repo_path, lora_name)
            maybe_lora_request = await self._get_lora_req_from_path(
                lora_name, lora_path, base_model_name)
            return maybe_lora_request
        return None


def register_hf_hub_resolver():
    """Register the Hf hub LoRA Resolver with vLLM"""

    hf_repo = envs.VLLM_LORA_RESOLVER_HF_REPO
    if hf_repo:
        hf_hub_resolver = HfHubResolver(hf_repo)
        LoRAResolverRegistry.register_resolver("Hf Hub Resolver",
                                               hf_hub_resolver)

    return
