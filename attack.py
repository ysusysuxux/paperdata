#!/usr/bin/env python3
"""
MomoL-VL 攻击脚本
使用 vLLM 部署的模型进行推理，并将结果保存到 JSON 文件

使用方式:
    python attack.py --text "你的问题" --image_path "/path/to/image.jpg"
    python attack.py --text "分析这张图片" --image_path "./test.png" --output "./eomol.json"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
from run import MomoLVLInferencer


class AttackRunner:
    """MomoL-VL 攻击运行器"""
    
    def __init__(self, output_file: str = "./eomol.json", server_url: str = "http://localhost:5000",
                 max_tokens: int = 2048, temperature: float = 0.7, top_p: float = 0.9):
        """
        初始化攻击运行器
        
        Args:
            output_file: 输出 JSON 文件路径
            server_url: vLLM 服务器地址
            max_tokens: 最大生成长度
            temperature: 温度参数
            top_p: Top-p 采样参数
        """
        self.output_file = output_file
        self.server_url = server_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.inferencer = MomoLVLInferencer(server_url=server_url)
        
        print(f"📝 输出文件: {self.output_file}")
        print(f"🖥️  服务器: {self.server_url}")
        print(f"⚙️  生成参数: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}")
    
    def load_results(self) -> list:
        """
        加载现有的结果文件
        
        Returns:
            结果列表（如果文件不存在则返回空列表）
        """
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    # 读取 JSONL 格式（每行一个 JSON 对象）
                    results = []
                    for line in f:
                        if line.strip():
                            results.append(json.loads(line))
                    print(f"✓ 已加载 {len(results)} 条现有结果")
                    return results
            except Exception as e:
                print(f"⚠️  加载结果文件出错: {e}")
                return []
        else:
            print(f"✓ 创建新的结果文件: {self.output_file}")
            return []
    
    def save_result(self, result: Dict[str, Any]):
        """
        保存单个结果到文件（追加模式）
        
        Args:
            result: 包含 text, image_path, attackllm_output 的字典
        """
        try:
            # 检查文件是否已存在
            file_exists = os.path.exists(self.output_file)
            
            # JSONL 格式（每行一个 JSON 对象）
            with open(self.output_file, 'a', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
            
            # 计算文件中的记录数
            with open(self.output_file, 'r', encoding='utf-8') as f:
                line_count = sum(1 for line in f if line.strip())
            
            if file_exists:
                print(f"✓ 结果已追加到: {self.output_file} (当前共 {line_count} 条记录)")
            else:
                print(f"✓ 结果已保存到: {self.output_file} (新建文件)")
        except Exception as e:
            print(f"❌ 保存结果失败: {e}")
    
    def attack(self, text: str, image_path: str) -> Dict[str, Any]:
        """
        执行攻击（调用模型）
        
        Args:
            text: 输入文本
            image_path: 图像路径
            
        Returns:
            包含结果的字典
        """
        # 验证输入
        if not text:
            raise ValueError("文本不能为空")
        
        if not image_path:
            raise ValueError("图像路径不能为空")
        
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        
        supported_formats = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        if image_path.suffix.lower() not in supported_formats:
            raise ValueError(f"不支持的图像格式: {image_path.suffix}")
        
        print(f"\n{'='*60}")
        print(f"🚀 开始推理")
        print(f"{'='*60}")
        print(f"📝 文本: {text[:100]}..." if len(text) > 100 else f"📝 文本: {text}")
        print(f"🖼️  图像: {image_path}")
        
        # 调用推理器
        try:
            result = self.inferencer.infer(
                text=text,
                image_path=str(image_path),
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p
            )
            
            if result['success']:
                output = result['result']
                print(f"\n✓ 推理成功")
                print(f"📤 输出长度: {len(output)} 字符")
                print(f"📤 输出预览: {output[:300]}..." if len(output) > 300 else f"📤 输出: {output}")
                
                # 构建结果对象
                record = {
                    'text': text,
                    'image_path': str(image_path.absolute()),
                    'attackllm_output': output
                }
                
                print(f"💾 准备保存完整输出 ({len(output)} 字符)")
                
                return record
            else:
                error = result.get('error', '未知错误')
                print(f"❌ 推理失败: {error}")
                raise RuntimeError(f"推理失败: {error}")
                
        except Exception as e:
            print(f"❌ 推理异常: {e}")
            raise
    
    def run(self, text: str, image_path: str):
        """
        运行完整的攻击流程
        
        Args:
            text: 输入文本
            image_path: 图像路径
        """
        try:
            # 执行攻击
            record = self.attack(text, image_path)
            
            # 保存结果
            self.save_result(record)
            
            print(f"\n{'='*60}")
            print(f"✅ 攻击完成")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ 攻击失败: {e}")
            print(f"{'='*60}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MomoL-VL 攻击脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基础用法
  python attack.py --text "这是什么?" --image_path "image.jpg"
  
  # 指定服务器
  python attack.py --text "分析图片" --image_path "test.png" --server "http://192.168.1.100:5000"
  
  # 指定输出文件
  python attack.py --text "问题" --image_path "img.jpg" --output "./results.json"
        """
    )
    
    parser.add_argument(
        '--text',
        type=str,
        required=True,
        help='输入文本'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        required=True,
        help='图像路径 (jpg/png/gif/webp)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./eomol.json',
        help='输出 JSON 文件路径 (默认: ./eomol.json)'
    )
    
    parser.add_argument(
        '--server',
        type=str,
        default='http://localhost:5000',
        help='vLLM 服务器地址 (默认: http://localhost:5000)'
    )
    
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=2048,
        help='最大生成长度 (默认: 2048)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='温度参数 (默认: 0.7)'
    )
    
    parser.add_argument(
        '--top_p',
        type=float,
        default=0.9,
        help='Top-p 采样参数 (默认: 0.9)'
    )
    
    args = parser.parse_args()
    
    # 创建运行器
    runner = AttackRunner(
        output_file=args.output,
        server_url=args.server,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # 执行攻击
    runner.run(
        text=args.text,
        image_path=args.image_path
    )


if __name__ == '__main__':
    main()

