# 在vLLM中获取指定索引的Logits值

## 问题描述

在使用vLLM进行分布式推理（tensor parallel）时，logits张量会被切分到多个worker上。这导致无法直接通过索引访问完整的logits张量，因为：

1. 在tensor parallel模式下，每个worker只持有logits的一部分
2. 默认情况下，只有rank 0的worker会收集完整的logits
3. 其他worker上的logits访问会返回None

## 解决方案

本解决方案提供了三种方法来获取指定索引的logits值：

### 方法1：使用自定义LogitsProcessor（推荐）

通过自定义的logits processor，在生成过程中捕获指定索引的logits值。

```python
from typing import List, Dict
import torch
from vllm import LLM, SamplingParams

class LogitsCapturingSampler:
    """捕获指定索引logits值的处理器"""
    
    def __init__(self, indices: List[int]):
        self.indices = indices
        self.captured_logits: Dict[int, List[float]] = {idx: [] for idx in indices}
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """在处理logits时捕获指定索引的值"""
        for idx in self.indices:
            if 0 <= idx < logits.shape[-1]:
                for seq_idx in range(logits.shape[0]):
                    self.captured_logits[idx].append(logits[seq_idx, idx].item())
        return logits

# 使用示例
logits_sampler = LogitsCapturingSampler([0, 10, 100])  # 捕获索引0, 10, 100的logits
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=16,
    logits_processors=[logits_sampler]
)

llm = LLM(model="facebook/opt-125m", tensor_parallel_size=2)
outputs = llm.generate(["Hello world"], sampling_params)

# 获取捕获的logits值
captured_logits = logits_sampler.get_captured_logits()
```

### 方法2：扩展的LogitsProcessor类

创建了一个扩展的`LogitsProcessorWithIndexedOutput`类，强制使用all_gather确保所有worker都有完整的logits。

```python
from vllm.model_executor.layers.logits_processor_ext import LogitsProcessorWithIndexedOutput

# 这个processor会在所有worker上使用all_gather
# 确保每个worker都能访问完整的logits
processor = LogitsProcessorWithIndexedOutput(
    vocab_size=50257,
    return_indices=[0, 10, 100]
)
```

### 方法3：扩展的API服务

提供了一个扩展的OpenAI兼容API，支持在请求中指定要返回的logits索引。

启动服务：
```bash
python examples/serve_with_logits_indices.py \
    --model facebook/opt-125m \
    --tensor-parallel-size 2
```

发送请求：
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "facebook/opt-125m",
        "prompt": "Hello world",
        "max_tokens": 10,
        "logits_indices": [0, 10, 100],
        "temperature": 0.8
    }'
```

响应示例：
```json
{
    "id": "cmpl-...",
    "object": "text_completion",
    "created": 1234567890,
    "model": "facebook/opt-125m",
    "choices": [{
        "index": 0,
        "text": " is a beautiful day",
        "finish_reason": "length",
        "logits_values": {
            "0": -3.4521,
            "10": 2.1234,
            "100": -0.9876
        }
    }],
    "usage": {
        "prompt_tokens": 2,
        "completion_tokens": 10,
        "total_tokens": 12
    }
}
```

## 使用示例

### 示例1：离线推理获取logits

```bash
python examples/get_logits_by_index.py \
    --model facebook/opt-125m \
    --prompts "Hello world" "The sky is" \
    --indices 0 10 100 1000 \
    --tensor-parallel-size 2
```

### 示例2：API服务获取logits

1. 启动服务：
```bash
python examples/serve_with_logits_indices.py \
    --model your-model-path \
    --tensor-parallel-size 4
```

2. 使用Python客户端：
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "your-model-path",
        "prompt": "Test prompt",
        "max_tokens": 10,
        "logits_indices": [0, 42, 100, 1000]
    }
)

data = response.json()
logits_values = data["choices"][0]["logits_values"]
print(f"Logit at index 42: {logits_values['42']}")
```

## 注意事项

1. **性能影响**：使用all_gather会增加通信开销，特别是在大词表模型上
2. **内存使用**：强制all_gather会使每个worker都保存完整的logits，增加内存使用
3. **索引有效性**：确保请求的索引在词表范围内（0 到 vocab_size-1）
4. **批处理**：当前实现只返回每个序列第一个token的logits值

## 扩展建议

1. **支持所有token的logits**：当前实现只返回第一个生成token的logits，可以扩展为返回所有token
2. **优化通信**：可以只gather需要的索引对应的logits，而不是整个张量
3. **缓存机制**：对于频繁请求的索引，可以实现缓存机制
4. **流式返回**：支持在流式生成中返回每个token的logits值

## 相关文件

- `vllm/model_executor/layers/logits_processor_ext.py` - 扩展的LogitsProcessor实现
- `vllm/entrypoints/openai/protocol_ext.py` - 扩展的协议定义
- `examples/get_logits_by_index.py` - 离线推理示例
- `examples/serve_with_logits_indices.py` - API服务示例