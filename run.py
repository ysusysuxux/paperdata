"""
MomoL-VL 模型推理脚本
支持文本和图像输入，调用远程Linux服务器上的模型
"""

import requests
import base64
import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


class MomoLVLInferencer:
    """MomoL-VL 推理客户端"""
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """
        初始化推理客户端
        
        Args:
            server_url: 模型服务器地址 (默认: http://localhost:5000)
                       如果在远程Linux服务器，改为 http://<your_server_ip>:5000
        """
        self.server_url = server_url
        # 尝试使用标准的 vLLM API 端点，如果不行再回退到 /invocations
        self.infer_endpoint = f"{server_url}/v1/chat/completions"
        
    def load_image_as_base64(self, image_path: str) -> str:
        """
        加载图像文件并转换为base64编码
        
        Args:
            image_path: 图像文件路径 (支持 .jpg/.jpeg/.png)
            
        Returns:
            base64编码的图像字符串
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的图像格式
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        if image_path.suffix.lower() not in supported_formats:
            raise ValueError(f"不支持的图像格式: {image_path.suffix}。支持格式: {supported_formats}")
        
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return image_base64
    
    def infer(self, text: str, image_path: str, max_tokens: int = 2048, 
              temperature: float = 0.7, top_p: float = 0.9, system_prompt: str = None) -> Dict[str, Any]:
        """
        调用模型进行推理
        
        Args:
            text: 输入文本
            image_path: 图像路径 (png/jpg/jpeg)
            max_tokens: 最大生成长度（默认：2048）
            temperature: 温度参数，控制随机性（默认：0.7）
            top_p: nucleus sampling 参数（默认：0.9）
            system_prompt: 可选的系统提示词
            
        Returns:
            {
                'success': bool,
                'result': str (模型输出) 或 'error': str (错误信息)
            }
        """
        try:
            # 加载图像
            print(f"加载图像: {image_path}")
            image_base64 = self.load_image_as_base64(image_path)
            
            # 准备请求 - vLLM Chat Completion 格式（多模态）
            # 使用 messages 格式传递图像
            payload = {
                'model': 'momolVL',  # 模型名称（与 vLLM serve 路径的最后部分一致）
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/jpeg;base64,{image_base64}'
                                }
                            },
                            {
                                'type': 'text',
                                'text': text
                            }
                        ]
                    }
                ],
                # 生成参数
                'max_tokens': max_tokens,
                'temperature': temperature,
                'top_p': top_p
            }
            
            print(f"生成参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
            print(f"使用 Chat Completion API 格式")
            
            print(f"发送请求到 {self.infer_endpoint}")
            print(f"输入文本: {text[:100]}..." if len(text) > 100 else f"输入文本: {text}")
            
            # 调用服务器
            response = requests.post(
                self.infer_endpoint,
                json=payload,
                timeout=300,  # 300秒超时
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                raw_result = response.json()
                
                # 打印原始响应以便调试
                print(f"\n🔍 原始响应: {json.dumps(raw_result, ensure_ascii=False, indent=2)[:500]}...")
                
                # 处理两种响应格式
                if 'choices' in raw_result and len(raw_result['choices']) > 0:
                    choice = raw_result['choices'][0]
                    
                    # Chat Completion API 格式: {"message": {"content": "..."}}
                    if 'message' in choice:
                        text_output = choice['message'].get('content', '')
                    # Completion API 格式: {"text": "..."}
                    else:
                        text_output = choice.get('text', '')
                    
                    # 检查是否为空
                    if not text_output or text_output.strip() == '':
                        print(f"⚠️  警告: 模型返回了空响应")
                        print(f"🔍 完整 choices: {raw_result['choices']}")
                        if 'usage' in raw_result:
                            print(f"🔍 Token 使用: {raw_result['usage']}")
                    
                    return {
                        'success': True,
                        'result': text_output,
                        'raw_response': raw_result  # 保留原始响应
                    }
                else:
                    return {
                        'success': False,
                        'error': f"响应格式不正确，缺少 choices 字段",
                        'raw_response': raw_result
                    }
            else:
                error_detail = ""
                try:
                    error_detail = response.text
                except:
                    pass
                return {
                    'success': False,
                    'error': f"服务器返回错误: HTTP {response.status_code}\n详细信息: {error_detail}"
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': f"无法连接到服务器 {self.server_url}。请确保服务器已启动。"
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"推理失败: {str(e)}"
            }
    
    def infer_batch(self, prompts: list) -> list:
        """
        批量推理
        
        Args:
            prompts: 列表，每个元素为 {'text': str, 'image_path': str}
            
        Returns:
            结果列表
        """
        results = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n处理第 {i}/{len(prompts)} 个样本...")
            result = self.infer(
                text=prompt['text'],
                image_path=prompt['image_path']
            )
            results.append(result)
            print(f"结果: {result}")
        
        return results


def main():
    """主函数 - 命令行接口"""
    parser = argparse.ArgumentParser(description='MomoL-VL 模型推理脚本')
    parser.add_argument('--text', type=str, required=True, help='输入文本')
    parser.add_argument('--image', type=str, required=True, help='图像路径 (jpg/png)')
    parser.add_argument('--server', type=str, default='http://localhost:5000', 
                       help='模型服务器地址 (默认: http://localhost:5000)')
    parser.add_argument('--output', type=str, help='输出结果保存路径 (可选)')
    
    args = parser.parse_args()
    
    # 初始化推理客户端
    inferencer = MomoLVLInferencer(server_url=args.server)
    
    # 执行推理
    print("=" * 50)
    print("MomoL-VL 推理开始")
    print("=" * 50)
    
    result = inferencer.infer(
        text=args.text,
        image_path=args.image
    )
    
    print("\n" + "=" * 50)
    print("推理结果:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 保存输出（如果指定）
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()

