"""
Example script demonstrating how to get logits values at specific indices
when using vLLM with tensor parallelism.

This solves the problem where logits are distributed across multiple workers
and cannot be directly accessed by index.
"""

import argparse
from typing import List, Dict, Optional
import torch
from vllm import LLM, SamplingParams
from vllm.model_executor.layers.logits_processor_ext import LogitsProcessorWithIndexedOutput


class LogitsCapturingSampler:
    """Custom sampler that captures logits at specified indices."""
    
    def __init__(self, indices: List[int]):
        self.indices = indices
        self.captured_logits: Dict[int, List[float]] = {idx: [] for idx in indices}
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """Capture logits at specified indices and return unchanged logits."""
        for idx in self.indices:
            if 0 <= idx < logits.shape[-1]:
                # Store the logit value for each sequence in the batch
                for seq_idx in range(logits.shape[0]):
                    self.captured_logits[idx].append(logits[seq_idx, idx].item())
        return logits
    
    def get_captured_logits(self) -> Dict[int, List[float]]:
        """Return the captured logits values."""
        return self.captured_logits


def get_logits_at_indices(
    model_name: str,
    prompts: List[str],
    logit_indices: List[int],
    temperature: float = 0.8,
    max_tokens: int = 16,
    tensor_parallel_size: int = 1
) -> Dict[str, Dict[int, float]]:
    """
    Get logits values at specific indices for given prompts.
    
    Args:
        model_name: Name or path of the model
        prompts: List of input prompts
        logit_indices: List of token indices to get logits for
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        tensor_parallel_size: Number of GPUs for tensor parallelism
    
    Returns:
        Dictionary mapping prompts to their logits values at specified indices
    """
    
    # Create custom logits processor
    logits_sampler = LogitsCapturingSampler(logit_indices)
    
    # Create sampling parameters with custom logits processor
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        logits_processors=[logits_sampler]
    )
    
    # Initialize LLM
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True
    )
    
    # Generate outputs
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect results
    results = {}
    captured_logits = logits_sampler.get_captured_logits()
    
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        prompt_logits = {}
        for idx in logit_indices:
            if idx in captured_logits and i < len(captured_logits[idx]):
                prompt_logits[idx] = captured_logits[idx][i]
        results[prompt] = prompt_logits
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Get logits at specific indices from vLLM model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model name or path"
    )
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["Hello, my name is", "The capital of France is"],
        help="Input prompts"
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=[0, 10, 100, 1000],
        help="Token indices to get logits for"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism"
    )
    
    args = parser.parse_args()
    
    print(f"Model: {args.model}")
    print(f"Prompts: {args.prompts}")
    print(f"Logit indices: {args.indices}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("-" * 50)
    
    # Get logits at specified indices
    results = get_logits_at_indices(
        model_name=args.model,
        prompts=args.prompts,
        logit_indices=args.indices,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # Print results
    for prompt, logits in results.items():
        print(f"\nPrompt: '{prompt}'")
        print("Logits at indices:")
        for idx, value in sorted(logits.items()):
            print(f"  Index {idx}: {value:.4f}")


if __name__ == "__main__":
    main()