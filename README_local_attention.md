# vLLM Local Attention 配置指南

本文档介绍如何在vLLM中配置和使用局部注意力（Local Attention）机制进行模型服务。

## 什么是Local Attention？

Local Attention（局部注意力）是一种优化的注意力机制，它将注意力计算限制在一个固定大小的窗口内，而不是计算整个序列的注意力。这种方法在处理长序列时特别有效。

### 主要优势：

1. **计算效率**：将O(n²)的复杂度降低到O(n×w)，其中w是窗口大小
2. **内存效率**：显著减少内存使用
3. **适合长序列**：在处理长文本时性能更好

## vLLM中的Local Attention实现

在vLLM中，Local Attention通过以下方式实现：

1. **ChunkedLocalAttentionSpec**：分块局部注意力
2. **FullAttentionSpec with attention_chunk_size**：带窗口限制的全注意力
3. **SlidingWindowSpec**：滑动窗口注意力

## 配置方法

### 方法1：修改模型配置文件

在模型的`config.json`中添加`attention_chunk_size`参数：

```json
{
  "architectures": ["LlamaForCausalLM"],
  "attention_chunk_size": 512,  // 设置局部注意力窗口大小
  "use_irope": true,            // 某些模型需要启用
  // ... 其他配置
}
```

### 方法2：使用环境变量

某些情况下可以通过环境变量设置：

```bash
export VLLM_ATTENTION_CHUNK_SIZE=512
```

### 方法3：在代码中配置

```python
from vllm import LLM

# 注意：attention_chunk_size需要在模型配置中预先设置
llm = LLM(
    model="your-model-path",
    trust_remote_code=True,
    max_model_len=2048,  # 限制最大序列长度
)
```

## 使用示例

### 1. 离线推理

```bash
python local_attention_serve_example.py --mode offline --model Qwen/Qwen3-0.6B --attention-chunk-size 512
```

### 2. 启动模型服务

```bash
# 使用vLLM命令行工具
vllm serve your-model-path \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 2048

# 或使用示例脚本
python local_attention_serve_example.py --mode serve --model your-model-path
```

### 3. 客户端调用

```bash
# 简单测试
python local_attention_client.py --prompt "解释local attention的原理"

# 长上下文测试
python local_attention_client.py --test-mode long-context

# 性能基准测试
python local_attention_client.py --test-mode benchmark
```

## 注意事项

1. **模型支持**：不是所有模型都支持local attention，需要检查模型架构
2. **窗口大小**：attention_chunk_size的选择需要平衡性能和效果
   - 太小：可能丢失重要的长距离依赖
   - 太大：计算效率提升不明显
3. **与其他优化的兼容性**：
   - Local attention可以与量化、并行等其他优化技术结合使用
   - 某些优化可能需要特定的配置调整

## 性能调优建议

1. **窗口大小选择**：
   - 一般任务：256-512
   - 代码生成：512-1024
   - 长文档处理：1024-2048

2. **批处理优化**：
   - 使用相似长度的序列进行批处理
   - 设置合适的`max_num_seqs`

3. **内存管理**：
   - 监控GPU内存使用
   - 调整`gpu_memory_utilization`参数

## 故障排除

### 问题1：模型不支持attention_chunk_size

**解决方案**：
- 检查模型架构是否支持
- 确认config.json中正确设置了参数
- 某些模型可能需要`use_irope=true`

### 问题2：性能没有明显提升

**解决方案**：
- 检查序列长度是否足够长
- 调整窗口大小
- 确认GPU支持相关优化

### 问题3：精度下降

**解决方案**：
- 增大attention_chunk_size
- 对特定任务进行微调
- 考虑使用sliding window而非fixed chunk

## 相关资源

- [vLLM官方文档](https://docs.vllm.ai/)
- [注意力机制优化论文](https://arxiv.org/abs/2004.05150)
- [Local Attention实现](https://github.com/lucidrains/local-attention)

## 示例文件说明

1. `local_attention_serve_example.py`：主要示例脚本，展示配置和使用方法
2. `local_attention_client.py`：客户端示例，展示如何调用服务
3. `model_config_with_local_attention.json`：包含local attention配置的示例文件