#!/usr/bin/env python3
"""
Local Attention模型服务的客户端示例

这个客户端演示如何与配置了local attention的vLLM服务器进行交互
"""

import requests
import json
import argparse
from typing import Dict, Any, List


class LocalAttentionClient:
    """使用local attention的vLLM服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        发送生成请求到服务器
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大生成token数
            temperature: 采样温度
            top_p: nucleus采样参数
            stream: 是否使用流式输出
        
        Returns:
            服务器响应
        """
        endpoint = f"{self.base_url}/v1/completions"
        
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream
        }
        
        response = requests.post(
            endpoint,
            headers=self.headers,
            json=payload,
            stream=stream
        )
        
        if stream:
            return self._handle_stream_response(response)
        else:
            return response.json()
    
    def _handle_stream_response(self, response):
        """处理流式响应"""
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]":
                        yield json.loads(data)
    
    def test_long_context(self):
        """测试长上下文处理能力"""
        # 创建一个长上下文
        context = """在自然语言处理领域，注意力机制是一项革命性的技术。
        传统的序列到序列模型在处理长序列时会遇到信息瓶颈问题。
        为了解决这个问题，研究人员提出了各种改进方案。
        
        Local attention（局部注意力）是其中一种有效的解决方案。
        与全局注意力不同，局部注意力将注意力范围限制在一个固定大小的窗口内。
        这种方法有以下几个优点：
        
        1. 计算效率：将O(n²)的复杂度降低到O(n×w)，其中w是窗口大小
        2. 内存效率：减少了需要存储的注意力权重
        3. 局部性：某些任务中，局部信息比全局信息更重要
        
        在vLLM中，可以通过设置attention_chunk_size参数来启用局部注意力。
        """ * 5  # 重复5次以创建更长的上下文
        
        prompt = context + "\n\n基于以上内容，请总结local attention的主要优势："
        
        print("测试长上下文处理...")
        print(f"上下文长度: {len(prompt)} 字符")
        
        response = self.generate(prompt, max_tokens=200)
        return response
    
    def benchmark_performance(self, num_requests: int = 5):
        """简单的性能基准测试"""
        import time
        
        prompts = [
            "解释什么是local attention：",
            "比较local attention和global attention的区别：",
            "local attention在哪些场景下特别有用：",
            "如何在vLLM中配置local attention：",
            "local attention的实现原理是：",
        ]
        
        print(f"\n执行{num_requests}次请求的性能测试...")
        
        total_time = 0
        results = []
        
        for i in range(min(num_requests, len(prompts))):
            prompt = prompts[i % len(prompts)]
            
            start_time = time.time()
            response = self.generate(prompt, max_tokens=50)
            end_time = time.time()
            
            elapsed = end_time - start_time
            total_time += elapsed
            
            results.append({
                "prompt": prompt,
                "time": elapsed,
                "response": response
            })
            
            print(f"请求 {i+1}: {elapsed:.2f}秒")
        
        avg_time = total_time / num_requests
        print(f"\n平均响应时间: {avg_time:.2f}秒")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Local Attention vLLM服务客户端"
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM服务器URL"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="解释local attention的工作原理：",
        help="生成提示词"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="最大生成token数"
    )
    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["simple", "long-context", "benchmark"],
        default="simple",
        help="测试模式"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式输出"
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = LocalAttentionClient(args.server_url)
    
    if args.test_mode == "simple":
        # 简单生成测试
        print(f"提示词: {args.prompt}")
        print("\n生成中...")
        
        if args.stream:
            for chunk in client.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                stream=True
            ):
                if "choices" in chunk:
                    print(chunk["choices"][0]["text"], end="", flush=True)
            print()
        else:
            response = client.generate(
                args.prompt,
                max_tokens=args.max_tokens
            )
            if "choices" in response:
                print("\n生成结果:")
                print(response["choices"][0]["text"])
    
    elif args.test_mode == "long-context":
        # 长上下文测试
        response = client.test_long_context()
        if "choices" in response:
            print("\n生成结果:")
            print(response["choices"][0]["text"])
    
    elif args.test_mode == "benchmark":
        # 性能基准测试
        client.benchmark_performance()


if __name__ == "__main__":
    main()