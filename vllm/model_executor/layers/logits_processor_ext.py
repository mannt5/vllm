"""Extended LogitsProcessor with support for retrieving specific logit values."""
import torch
from typing import Optional, List, Dict
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.sampling_metadata import SamplingMetadata


class LogitsProcessorWithIndexedOutput(LogitsProcessor):
    """
    Extended LogitsProcessor that can return specific logit values by index.
    
    This processor always uses all_gather to ensure all workers have access
    to the complete logits tensor, allowing retrieval of specific indices.
    """
    
    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None,
                 return_indices: Optional[List[int]] = None) -> None:
        super().__init__(vocab_size, org_vocab_size, scale, logits_as_input, soft_cap)
        # Force all_gather to ensure all workers have complete logits
        self.use_all_gather = True
        self.return_indices = return_indices
        self.indexed_logits: Optional[Dict[int, torch.Tensor]] = None
    
    def set_return_indices(self, indices: Optional[List[int]]) -> None:
        """Set the indices of logits to return."""
        self.return_indices = indices
        
    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
        embedding_bias: Optional[torch.Tensor] = None,
        prune_hidden_states: bool = True,
    ) -> Optional[torch.Tensor]:
        # Call parent forward method
        logits = super().forward(lm_head, hidden_states, sampling_metadata, 
                               embedding_bias, prune_hidden_states)
        
        # If we need to return specific indices, store them
        if logits is not None and self.return_indices is not None:
            self.indexed_logits = {}
            for idx in self.return_indices:
                if 0 <= idx < logits.shape[-1]:
                    # Store logits values for each sequence in the batch
                    self.indexed_logits[idx] = logits[:, idx].clone()
        
        return logits
    
    def get_indexed_logits(self) -> Optional[Dict[int, torch.Tensor]]:
        """Get the stored indexed logits values."""
        return self.indexed_logits
    
    def _gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Always use all_gather to ensure all workers have complete logits."""
        return tensor_model_parallel_all_gather(logits)