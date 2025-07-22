"""Extensions to OpenAI protocol for vLLM-specific features."""
from typing import Optional, Union, List
from vllm.entrypoints.openai.protocol import CompletionResponseChoice, ChatCompletionResponseChoice


class CompletionResponseChoiceWithLogits(CompletionResponseChoice):
    """Extended completion response choice that includes logits values."""
    logits_values: Optional[dict[int, float]] = None


class ChatCompletionResponseChoiceWithLogits(ChatCompletionResponseChoice):
    """Extended chat completion response choice that includes logits values."""
    logits_values: Optional[dict[int, float]] = None