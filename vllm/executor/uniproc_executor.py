# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import threading
import torch.distributed as dist

from concurrent.futures import ThreadPoolExecutor
import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.executor.executor_base import ExecutorBase
from vllm.logger import init_logger
from vllm.utils import (get_distributed_init_method, get_ip, get_open_port,
                        run_method)
from vllm.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)


class UniProcExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        local_rank = 0
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(
            ":")
        if len(device_info) > 1:
            local_rank = int(device_info[1])
        rank = 0
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)
        return [answer]

    def check_health(self) -> None:
        # UniProcExecutor will always be healthy as long as
        # it's running.
        return


UniProcExecutorAsync = UniProcExecutor


class ExecutorWithExternalLauncher(UniProcExecutor):
    """An executor that uses external launchers to launch engines,
    specially designed for torchrun-compatible launchers, for
    offline inference with tensor parallelism.

    see https://github.com/vllm-project/vllm/issues/11400 for
    the motivation, and examples/offline_inference/torchrun_example.py
    for the usage example.

    The key idea: although it is tensor-parallel inference, we only
    create one worker per executor, users will launch multiple
    engines with torchrun-compatible launchers, and all these engines
    work together to process the same prompts. When scheduling is
    deterministic, all the engines will generate the same outputs,
    and they don't need to synchronize the states with each other.
    """
    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        assert self.vllm_config.scheduler_config.delay_factor == 0.0, \
            ("ExecutorWithExternalLauncher needs deterministic "
            "execution, so it"
            "does not support delay_factor in scheduling")
        if envs.VLLM_USE_V1:
            assert not envs.VLLM_ENABLE_V1_MULTIPROCESSING, \
            ("To get deterministic execution in V1, "
            "please set VLLM_ENABLE_V1_MULTIPROCESSING=0")
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        self.non_blocking: bool = False
        self.excecute_model_thread_pool: Optional[ThreadPoolExecutor] = None
        # engines are launched in torchrun-compatible launchers
        # so we can use the env:// method.
        # required env vars:
        # - RANK
        # - LOCAL_RANK
        # - MASTER_ADDR
        # - MASTER_PORT
        distributed_init_method = "env://"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        self.local_rank = local_rank
        is_driver_worker = True
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        self.pp_rank = get_pp_group().rank_in_group
        self.pp_lock: Optional[threading.Lock] = None
        if self.max_concurrent_batches > 1:
            self.pp_lock = threading.Lock()
            self.excecute_model_thread_pool = ThreadPoolExecutor(
                max_workers=self.max_concurrent_batches, thread_name_prefix="external_exec_")

    @property
    def max_concurrent_batches(self) -> int:
        return self.parallel_config.pipeline_parallel_size
    
    def execute_model_pp(self, device: int, args: Tuple = (),kwargs: Optional[Dict] = None):
        assert self.pp_lock is not None, "self.pp_lock is not initialized."
        with self.pp_lock:
            device = torch.cuda.set_device(device)
            answer = run_method(self.driver_worker, "execute_model", args, kwargs)
            # transfer the answer from the last PP rank
            # to first PP rank with async torch.dist P2P
            answer = get_pp_group().broadcast_object_async(
                answer, src=self.parallel_config.pipeline_parallel_size - 1)
        return answer.wait()

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        assert kwargs is not None
        if method == "execute_model" \
                    and self.parallel_config.pipeline_parallel_size > 1:
            device = torch.cuda.current_device()
            assert self.excecute_model_thread_pool is not None, \
                "self.excecute_model_thread_pool is not initialized."
            answer = self.excecute_model_thread_pool.submit(
                self.execute_model_pp, device, args, kwargs)

        else:
            answer = run_method(self.driver_worker, method, args, kwargs)

        return [answer]

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """
        Determine the number of available KV blocks.
        Add an additional all_reduce to get the min across all ranks.
        Note that even if we have the same `gpu_memory_utilization` and 
        `swap_space`, the available memory in every rank might still 
        differ because NCCL can take different amounts of memory in 
        different ranks. Therefore, it is necessary to test if all ranks 
        agree on the same KV cache configuration.
        """
        a, b = super().determine_num_available_blocks()
        from vllm.distributed.parallel_state import get_world_group
        cpu_group = get_world_group().cpu_group
        a_tensor = torch.tensor([a], device="cpu", dtype=torch.int64)
        b_tensor = torch.tensor([b], device="cpu", dtype=torch.int64)
        dist.all_reduce(a_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        dist.all_reduce(b_tensor, group=cpu_group, op=dist.ReduceOp.MIN)
        return a_tensor.item(), b_tensor.item()
