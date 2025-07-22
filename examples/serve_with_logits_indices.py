"""
Example of serving vLLM model with support for returning logits at specific indices.

This example shows how to extend the OpenAI-compatible API to return logits values
at specified token indices, solving the issue where logits are distributed across
multiple workers in tensor parallel settings.

Usage:
    python serve_with_logits_indices.py --model facebook/opt-125m --tensor-parallel-size 2

Then make requests:
    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "facebook/opt-125m",
            "prompt": "Hello world",
            "max_tokens": 10,
            "logits_indices": [0, 10, 100],
            "temperature": 0.8
        }'
"""

import argparse
import json
from typing import List, Dict, Optional, Union, AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from pydantic import BaseModel, Field

from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
from vllm.entrypoints.openai.protocol import CompletionRequest, CompletionResponse
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion


class CompletionRequestWithLogits(CompletionRequest):
    """Extended completion request that includes logits indices."""
    logits_indices: Optional[List[int]] = Field(
        default=None,
        description="List of token indices to return logits values for"
    )


class CompletionChoiceWithLogits(BaseModel):
    """Completion choice extended with logits values."""
    index: int
    text: str
    logprobs: Optional[dict] = None
    finish_reason: Optional[str] = None
    logits_values: Optional[Dict[int, float]] = None


class CompletionResponseWithLogits(BaseModel):
    """Completion response extended with logits values."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoiceWithLogits]
    usage: dict


class LogitsCapturingProcessor:
    """Logits processor that captures values at specified indices."""
    
    def __init__(self, indices: List[int]):
        self.indices = indices or []
        self.captured_logits: Dict[int, List[float]] = {idx: [] for idx in self.indices}
        self.current_step = 0
    
    def __call__(self, input_ids, logits):
        """Capture logits at specified indices."""
        import torch
        
        if len(self.indices) > 0:
            # Handle both single sequence and batch
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)
            
            for seq_idx in range(logits.shape[0]):
                for idx in self.indices:
                    if 0 <= idx < logits.shape[-1]:
                        if self.current_step == 0:  # Only capture first token's logits
                            self.captured_logits[idx].append(logits[seq_idx, idx].item())
        
        self.current_step += 1
        return logits
    
    def get_captured_logits(self, seq_idx: int = 0) -> Dict[int, float]:
        """Get captured logits for a specific sequence."""
        result = {}
        for idx in self.indices:
            if idx in self.captured_logits and seq_idx < len(self.captured_logits[idx]):
                result[idx] = self.captured_logits[idx][seq_idx]
        return result
    
    def reset(self):
        """Reset captured logits."""
        self.captured_logits = {idx: [] for idx in self.indices}
        self.current_step = 0


class ExtendedOpenAIServingCompletion(OpenAIServingCompletion):
    """Extended serving class that supports returning logits at specific indices."""
    
    async def create_completion(
        self,
        request: CompletionRequestWithLogits,
        raw_request: Optional[str] = None
    ) -> Union[CompletionResponseWithLogits, AsyncGenerator[str, None]]:
        """Create completion with optional logits values."""
        
        # Create logits processor if indices are specified
        logits_processor = None
        if request.logits_indices:
            logits_processor = LogitsCapturingProcessor(request.logits_indices)
        
        # Convert extended request to standard request
        standard_request = CompletionRequest(**request.dict(exclude={'logits_indices'}))
        
        # Add logits processor to sampling parameters
        if logits_processor:
            if standard_request.logits_processors is None:
                standard_request.logits_processors = []
            standard_request.logits_processors.append(logits_processor)
        
        # Generate completion
        generator = await super().create_completion(standard_request, raw_request)
        
        # If streaming, we can't easily inject logits values
        if request.stream:
            async for chunk in generator:
                yield chunk
        else:
            # For non-streaming, we can modify the response
            response = await generator.__anext__()
            
            if logits_processor and isinstance(response, CompletionResponse):
                # Convert to our extended response format
                choices_with_logits = []
                for i, choice in enumerate(response.choices):
                    choice_dict = choice.dict()
                    choice_dict['logits_values'] = logits_processor.get_captured_logits(i)
                    choices_with_logits.append(CompletionChoiceWithLogits(**choice_dict))
                
                response_with_logits = CompletionResponseWithLogits(
                    id=response.id,
                    object=response.object,
                    created=response.created,
                    model=response.model,
                    choices=choices_with_logits,
                    usage=response.usage.dict()
                )
                
                yield response_with_logits
            else:
                yield response


async def create_app(engine_args: AsyncEngineArgs) -> FastAPI:
    """Create FastAPI app with extended completion endpoint."""
    app = FastAPI(title="vLLM Server with Logits Support")
    
    # Initialize engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    
    # Create extended serving instance
    serving = ExtendedOpenAIServingCompletion(
        engine=engine,
        served_model_names=[engine_args.model],
        lora_modules=None
    )
    
    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequestWithLogits):
        """Create completion with optional logits indices."""
        generator = await serving.create_completion(request)
        
        if request.stream:
            return StreamingResponse(
                (json.dumps(chunk.dict()) + "\n" async for chunk in generator),
                media_type="text/event-stream"
            )
        else:
            response = await generator.__anext__()
            return JSONResponse(content=response.dict())
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


def main():
    parser = argparse.ArgumentParser(
        description="vLLM API server with logits index support"
    )
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--trust-remote-code", action="store_true", help="Trust remote code")
    
    args = parser.parse_args()
    
    # Create engine args
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Create and run app
    import asyncio
    app = asyncio.run(create_app(engine_args))
    
    print(f"Starting server on {args.host}:{args.port}")
    print(f"Model: {args.model}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print("\nExample request:")
    print(f"""
    curl http://{args.host}:{args.port}/v1/completions \\
        -H "Content-Type: application/json" \\
        -d '{{
            "model": "{args.model}",
            "prompt": "Hello world",
            "max_tokens": 10,
            "logits_indices": [0, 10, 100],
            "temperature": 0.8
        }}'
    """)
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()