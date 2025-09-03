#!/usr/bin/env python3
"""
使用vLLM配置Local Attention进行模型服务的示例

在vLLM中，local attention（局部注意力）通过设置attention_chunk_size参数来实现。
这种注意力机制将注意力限制在局部窗口内，可以有效降低长序列的计算复杂度。

配置方式：
1. 在模型的config.json中设置attention_chunk_size参数
2. 或者在运行时通过环境变量设置
3. 对于支持的模型，vLLM会自动检测并应用local attention
"""

import os
import json
from typing import Dict, Any
import argparse

# vLLM的导入
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.serving.server import create_serve_app
import uvicorn


def create_model_config_with_local_attention(
    base_config_path: str,
    attention_chunk_size: int,
    output_path: str
) -> None:
    """
    创建一个包含attention_chunk_size配置的模型配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        attention_chunk_size: 局部注意力窗口大小
        output_path: 输出配置文件路径
    """
    # 读取基础配置
    with open(base_config_path, 'r') as f:
        config = json.load(f)
    
    # 添加attention_chunk_size配置
    config['attention_chunk_size'] = attention_chunk_size
    
    # 保存新配置
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"已创建包含local attention配置的文件: {output_path}")
    print(f"attention_chunk_size: {attention_chunk_size}")


def run_offline_inference_with_local_attention(
    model_name: str,
    attention_chunk_size: int = 512
) -> None:
    """
    使用local attention运行离线推理示例
    
    Args:
        model_name: 模型名称或路径
        attention_chunk_size: 局部注意力窗口大小
    """
    # 创建LLM实例
    # 注意：attention_chunk_size需要在模型配置中设置
    # 某些模型可能需要修改config.json
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=2048,  # 限制最大序列长度
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )
    
    # 测试提示词
    prompts = [
        "Local attention是一种限制注意力范围的技术，它",
        "在处理长序列时，local attention的优势包括",
        "vLLM中配置attention_chunk_size参数可以",
    ]
    
    print(f"\n使用模型: {model_name}")
    print(f"Local attention窗口大小: {attention_chunk_size}")
    print("\n开始推理...\n")
    
    # 执行推理
    outputs = llm.generate(prompts, sampling_params)
    
    # 打印结果
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt {i+1}: {prompt}")
        print(f"Generated: {generated_text}")
        print("-" * 80)


def serve_model_with_local_attention(
    model_name: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    attention_chunk_size: int = 512
) -> None:
    """
    启动一个使用local attention的模型服务
    
    Args:
        model_name: 模型名称或路径
        host: 服务器主机地址
        port: 服务器端口
        attention_chunk_size: 局部注意力窗口大小
    """
    print(f"\n启动vLLM服务器...")
    print(f"模型: {model_name}")
    print(f"Local attention窗口大小: {attention_chunk_size}")
    print(f"服务地址: http://{host}:{port}")
    
    # 使用命令行启动服务器
    # 注意：在实际使用中，attention_chunk_size需要在模型配置中设置
    cmd = f"""
vllm serve {model_name} \\
    --host {host} \\
    --port {port} \\
    --trust-remote-code \\
    --max-model-len 2048
"""
    
    print(f"\n可以使用以下命令启动服务器:")
    print(cmd)
    
    print("\n或者使用Python API:")
    print("""
from vllm import LLM
from vllm.serving.server import create_serve_app
import uvicorn

# 创建LLM实例（确保模型配置中包含attention_chunk_size）
llm = LLM(model=model_name, trust_remote_code=True)

# 创建FastAPI应用
app = create_serve_app(llm)

# 启动服务器
uvicorn.run(app, host=host, port=port)
""")


def demonstrate_attention_types():
    """
    演示vLLM中不同类型的注意力机制
    """
    print("\nvLLM支持的注意力机制类型:")
    print("\n1. FullAttentionSpec - 全局注意力")
    print("   - 标准的全注意力机制")
    print("   - 可选配置sliding_window或attention_chunk_size")
    
    print("\n2. ChunkedLocalAttentionSpec - 分块局部注意力")
    print("   - 需要设置attention_chunk_size参数")
    print("   - 将注意力限制在固定大小的窗口内")
    
    print("\n3. SlidingWindowSpec - 滑动窗口注意力")
    print("   - 需要设置sliding_window参数")
    print("   - 注意力窗口会随着位置滑动")
    
    print("\n配置示例:")
    print("""
# 在模型的config.json中添加:
{
  "attention_chunk_size": 512,  # 设置局部注意力窗口大小
  "use_irope": true,            # 某些模型需要启用irope
  ...
}

# 或在代码中:
kv_cache_spec = ChunkedLocalAttentionSpec(
    block_size=16,
    num_kv_heads=32,
    head_size=128,
    dtype=torch.float16,
    attention_chunk_size=512,
    use_mla=False
)
""")


def main():
    parser = argparse.ArgumentParser(
        description="vLLM Local Attention 配置示例"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="模型名称或路径"
    )
    parser.add_argument(
        "--attention-chunk-size",
        type=int,
        default=512,
        help="局部注意力窗口大小"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["offline", "serve", "info"],
        default="info",
        help="运行模式: offline（离线推理）, serve（启动服务）, info（显示信息）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务器主机地址"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="服务器端口"
    )
    
    args = parser.parse_args()
    
    if args.mode == "offline":
        run_offline_inference_with_local_attention(
            args.model,
            args.attention_chunk_size
        )
    elif args.mode == "serve":
        serve_model_with_local_attention(
            args.model,
            args.host,
            args.port,
            args.attention_chunk_size
        )
    else:  # info
        demonstrate_attention_types()
        print("\n" + "="*80)
        print("使用说明:")
        print("\n1. 显示注意力机制信息:")
        print("   python local_attention_serve_example.py --mode info")
        print("\n2. 运行离线推理:")
        print("   python local_attention_serve_example.py --mode offline --model <model_name>")
        print("\n3. 启动模型服务:")
        print("   python local_attention_serve_example.py --mode serve --model <model_name>")
        print("\n注意: attention_chunk_size需要在模型配置文件中设置才能生效")


if __name__ == "__main__":
    main()