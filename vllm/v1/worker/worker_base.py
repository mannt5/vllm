# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheSpec
from vllm.v1.worker.intermediates_logging import IntermediatesLogger, register_intermediate_hooks
from vllm.v1.worker.il_config import IntermediateLoggingConfig
from vllm.worker.worker_base import WorkerBase as WorkerBaseV0

logger = init_logger(__name__)


class WorkerBase(WorkerBaseV0):
    """
    Abstract class for v1 worker, mainly define some methods for v1.
    For methods shared by v0 and v1, define them in v0 WorkerBase
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        """
        Initialize common worker components.
        
        Args:
            vllm_config: Complete vLLM configuration
            local_rank: Local device index
            rank: Global rank in distributed setup
            distributed_init_method: Distributed initialization method
            is_driver_worker: Whether this worker handles driver 
            responsibilities
        """
        # Configuration storage
        super().__init__(vllm_config=vllm_config)

        self.parallel_config.rank = rank
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        # Device and model state
        self.device: Optional[torch.device] = None
        self.model_runner: Optional[nn.Module] = None

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get specifications for KV cache implementation."""
        raise NotImplementedError

    def compile_or_warm_up_model(self) -> None:
        """Prepare model for execution through compilation/warmup."""
        raise NotImplementedError

    def check_health(self) -> None:
        """Basic health check (override for device-specific checks)."""
        return
        
    def register_intermediate_hooks(self, 
                                   config: Optional[IntermediateLoggingConfig] = None,
                                   **kwargs) -> None:
        """Register hooks for intermediate tensor logging.
        
        This method is called via collective_rpc from the engine core.
        It registers hooks on the model to dump intermediate tensors during execution.
        
        Args:
            config: Configuration for intermediate logging. If provided, this takes precedence over kwargs.
            **kwargs: Configuration parameters that can include:
                - output_dir: Directory where to save the intermediate tensors.
                - module_name_regex: Optional regex pattern to filter modules by name.
                - log_step_ids: List of step IDs to log.
                - max_tensor_size: Maximum number of elements in tensors to log (None = no limit).
                - enabled: Whether logging is enabled.
        """
        logger.info(f"register_intermediate_hooks called on worker {self.rank}")
        logger.info(f"Config: {config}")
        logger.info(f"Kwargs: {kwargs}")
        
        if self.model_runner is None:
            logger.error("Could not register intermediate hooks: model_runner is None")
            return
            
        if not hasattr(self.model_runner, "model"):
            logger.error("Could not register intermediate hooks: model_runner has no 'model' attribute")
            return
            
        model = self.model_runner.model
        if model is None:
            logger.error("Could not register intermediate hooks: model is None")
            return
            
        # Log model information
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model device: {next(model.parameters(), torch.tensor(0)).device}")
        
        # Create config from kwargs if not provided
        if config is None:
            logger.info("Creating config from kwargs")
            config = IntermediateLoggingConfig.from_dict(kwargs)
        
        logger.info(f"Registering intermediate hooks for model with config: {config.to_dict()}")
        
        try:
            # Register hooks
            logger_instance = register_intermediate_hooks(model, config)
            # Store the logger instance for potential later hook removal
            self._intermediates_logger = logger_instance
            logger.info("Successfully registered intermediate hooks")
        except Exception as e:
            logger.error(f"Error registering intermediate hooks: {e}")
            import traceback
            logger.error(traceback.format_exc())
