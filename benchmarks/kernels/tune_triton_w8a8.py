# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from benchmark_w8a8_block_fp8.py

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime
from typing import Any

import torch
from tqdm import tqdm

from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
    scaled_mm_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import FlexibleArgumentParser

mp.set_start_method("spawn", force=True)

assert current_platform.is_cuda(), "Only support tune w8a8 int8 kernel on CUDA device."


def w8a8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    config: dict[str, Any],
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    M, K = A.shape
    _, N = B.shape
    C = torch.empty(M, N, device=A.device, dtype=output_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    scaled_mm_kernel[grid](
        A,
        B,
        As,
        Bs,
        C,
        None,
        M,
        N,
        K,
        *A.stride(),
        *B.stride(),
        *C.stride(),
        ACCUMULATOR_DTYPE=tl.float32 if A.is_floating_point() else tl.int32,
        BLOCK_SIZE_SCALE_A=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_SCALE_B=config["BLOCK_SIZE_N"],
        **config,
    )
    return C


def get_configs_compute_bound():
    configs = [
        {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            # "GROUP_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
        for block_m in [32, 64, 128, 256]
        for block_n in [32, 64, 128, 256]
        for block_k in [64, 128, 256]
        # for group_m in [1, 8]
        for num_warps in [4, 8]
        for num_stages in [3, 4, 5]
    ]
    return configs


def get_weight_shapes():
    # change shapes as needed
    weight_shapes = [
        (6144, 2560),
        (2560, 4096),
        (19456, 2560),
        (2560, 9728),
    ]
    return weight_shapes


def benchmark_config(A, B, As, Bs, config, out_dtype=torch.bfloat16, num_iters=10):
    def run():
        w8a8_matmul(A, B, As, Bs, config, out_dtype)

    torch.cuda.synchronize()
    # JIT complication & warmup
    for _ in range(5):
        run()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies: list[float] = []
    for i in range(num_iters):
        torch.cuda.synchronize()
        start_event.record()
        run()
        end_event.record()
        end_event.synchronize()
        latencies.append(start_event.elapsed_time(end_event))
    avg = sum(latencies) / (num_iters * 10) * 1000  # us
    return avg


def tune(M, N, K, out_dtype, search_space, input_type):
    factor_for_scale = 1e-2

    if input_type == "int8":
        A = torch.randint(-128, 127, size=(M, K), dtype=torch.int8, device="cuda")
        B = torch.randint(-128, 127, size=(N, K), dtype=torch.int8, device="cuda").T

    elif input_type == "fp8":
        dtype = torch.float8_e4m3fn
        fp8_max = torch.finfo(dtype).max

        A = (torch.rand(M, K, dtype=torch.float32, device="cuda") * 2 - 1) * fp8_max
        B = (torch.rand(N, K, dtype=torch.float32, device="cuda") * 2 - 1) * fp8_max

        A = A.clip(-fp8_max, fp8_max).to(dtype)
        B = B.clip(-fp8_max, fp8_max).to(dtype).T

    As = torch.rand(M, 1, dtype=torch.float32, device="cuda") * factor_for_scale
    Bs = torch.rand(N, 1, dtype=torch.float32, device="cuda") * factor_for_scale

    best_config = None
    best_time = float("inf")
    for config in tqdm(search_space):
        try:
            kernel_time = benchmark_config(
                A,
                B,
                As,
                Bs,
                config,
                out_dtype,
                num_iters=10,
            )
        except triton.runtime.autotuner.OutOfResources:
            # Some configurations may be invalid and fail to compile.
            continue

        if kernel_time < best_time:
            best_time = kernel_time
            best_config = config
    now = datetime.now()
    print(f"{now.ctime()}] Completed tuning for batch_size={M}")
    assert best_config is not None
    return best_config


def save_configs(
    N,
    K,
    configs,
    save_path,
    input_type="int8",
) -> None:
    os.makedirs(save_path, exist_ok=True)
    device_name = current_platform.get_device_name().replace(" ", "_")
    json_file_name = (
        f"N={N},K={K},device_name={device_name},dtype={input_type}_w8a8.json"
    )

    config_file_path = os.path.join(save_path, json_file_name)
    print(f"Writing best config to {config_file_path}...")

    with open(config_file_path, "w") as f:
        json.dump(configs, f, indent=4)
        f.write("\n")


def tune_on_gpu(args_dict):
    """Run tuning on a specific GPU."""
    gpu_id = args_dict["gpu_id"]
    batch_sizes = args_dict["batch_sizes"]
    weight_shapes = args_dict["weight_shapes"]
    args = args_dict["args"]

    torch.cuda.set_device(gpu_id)
    print(f"Starting tuning on GPU {gpu_id} with batch sizes {batch_sizes}")

    out_dtype = torch.bfloat16
    save_path = args.save_path
    input_type = args.input_type

    search_space = get_configs_compute_bound()

    start = time.time()
    for shape in tqdm(weight_shapes, desc=f"GPU {gpu_id} - Shapes"):
        N, K = shape[0], shape[1]
        print(f"[GPU {gpu_id}] Tune for weight shape of `N: {N}, K: {K}`")
        benchmark_results = [
            tune(
                batch_size,
                N,
                K,
                out_dtype,
                search_space,
                input_type,
            )
            for batch_size in tqdm(batch_sizes, desc=f"GPU {gpu_id} - Batch sizes")
        ]
        best_configs = {M: config for M, config in zip(batch_sizes, benchmark_results)}
        save_configs(N, K, best_configs, save_path, input_type)

    end = time.time()
    print(f"Tuning on GPU {gpu_id} took {end - start:.2f} seconds")


def distribute_batch_sizes(batch_sizes, num_gpus):
    """Distribute batch sizes across available GPUs."""
    batches_per_gpu = []
    for i in range(num_gpus):
        start_idx = i * len(batch_sizes) // num_gpus
        end_idx = (i + 1) * len(batch_sizes) // num_gpus
        batches_per_gpu.append(batch_sizes[start_idx:end_idx])
    return batches_per_gpu


def main(args):
    print(args)
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPU available for tuning")
    print(f"Found {num_gpus} GPUs for parallel tuning")

    torch.cuda.init()

    if args.batch_size is None:
        # M <= 32 will be selected based on heuristic. don't need to tune.
        batch_sizes = [
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            3072,
            4096,
        ]
    else:
        batch_sizes = [args.batch_size]
        num_gpus = 1  # If only one batch size, use only one GPU

    weight_shapes = get_weight_shapes()

    batches_per_gpu = distribute_batch_sizes(batch_sizes, num_gpus)

    process_args = []
    for gpu_id in range(num_gpus):
        process_args.append(
            {
                "gpu_id": gpu_id,
                "batch_sizes": batches_per_gpu[gpu_id],
                "weight_shapes": weight_shapes,  # Each GPU processes all weight shapes
                "args": args,
            }
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_gpus) as pool:
        pool.map(tune_on_gpu, process_args)

    print("Multi-GPU tuning completed")


if __name__ == "__main__":
    parser = FlexibleArgumentParser(
        description="""
Tune triton w8a8:
    python3 tune_triton_w8a8.py
Then copy to model_executor/layers/quantization/compressed_tensors/triton_configs
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-type", type=str, choices=["int8", "fp8"], default="int8"
    )
    parser.add_argument("--batch-size", type=int, required=False)
    parser.add_argument("--save-path", type=str, default="./")
    args = parser.parse_args()

    main(args)
