#!/usr/bin/env python3
"""
MomoL-VL 攻击脚本
使用 vLLM 部署的模型进行推理，并将结果保存到 JSON 文件

使用方式:
    # 单个文本和图片
    python attack.py --text "你的问题" --image_path "/path/to/image.jpg"
    python attack.py --text "分析这张图片" --image_path "./test.png" --output "./eomol.json"
    
    # 批量处理 jsonl 文件
    python attack.py --type 01
    python attack.py --type 02 --server "http://localhost:5000"
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
from run import MomoLVLInferencer

# 类型到文件名的映射
TYPE_TO_FILE = {
    '01': '01-Illegal_Activitiy.jsonl',
    '02': '02-HateSpeech.jsonl.jsonl',
    '03': '03-Malware_Generation.jsonl',
    '04': '04-Physical_Harme.jsonl',
    '05': '05-EconomicHarm.jsonl',
    '06': '06-Fraud.jsonl',
    '07': '07-Sex.jsonl'
}

# 类型到名称的映射（用于结果文件名）
TYPE_TO_NAME = {
    '01': 'Illegal_Activitiy',
    '02': 'HateSpeech',
    '03': 'Malware_Generation',
    '04': 'Physical_Harme',
    '05': 'EconomicHarm',
    '06': 'Fraud',
    '07': 'Sex'
}


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
    
    def load_jsonl(self, jsonl_path: str) -> List[Dict[str, Any]]:
        """
        加载 JSONL 文件
        
        Args:
            jsonl_path: JSONL 文件路径
            
        Returns:
            包含所有 JSON 对象的列表
        """
        data = []
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL 文件不存在: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️  警告: 第 {line_num} 行解析失败: {e}")
        
        print(f"✓ 成功加载 {len(data)} 条记录")
        return data
    
    def run_batch(self, jsonl_path: str, start_index: int = 0):
        """
        批量处理 JSONL 文件中的所有记录
        
        Args:
            jsonl_path: JSONL 文件路径
            start_index: 从第几条记录开始处理（默认从 0 开始）
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始批量处理")
        print(f"📁 输入文件: {jsonl_path}")
        print(f"📝 输出文件: {self.output_file}")
        print(f"{'='*60}\n")
        
        # 加载 JSONL 数据
        data = self.load_jsonl(jsonl_path)
        total = len(data)
        
        if start_index > 0:
            print(f"⏩ 跳过前 {start_index} 条记录，从第 {start_index + 1} 条开始\n")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        
        # 遍历每条记录
        for idx, item in enumerate(data[start_index:], start=start_index):
            print(f"\n{'='*60}")
            print(f"📋 处理记录 [{idx + 1}/{total}]")
            print(f"{'='*60}")
            
            # 提取 new_prompt 和 img_path
            text = item.get('new_prompt', '')
            image_path = item.get('img_path', '')
            
            if not text:
                print(f"⚠️  警告: 记录 {idx + 1} 缺少 'new_prompt' 字段，跳过")
                fail_count += 1
                continue
            
            if not image_path:
                print(f"⚠️  警告: 记录 {idx + 1} 缺少 'img_path' 字段，跳过")
                fail_count += 1
                continue
            
            try:
                # 执行攻击
                record = self.attack(text, image_path)
                
                # 将原始记录的其他字段也保存到结果中
                result_record = {
                    **item,  # 保留原始记录的所有字段
                    'attackllm_output': record['attackllm_output'],  # 更新输出
                    'processed_text': text,
                    'processed_image_path': image_path
                }
                
                # 保存结果
                self.save_result(result_record)
                success_count += 1
                
                print(f"✅ 记录 {idx + 1} 处理成功")
                
            except Exception as e:
                print(f"❌ 记录 {idx + 1} 处理失败: {e}")
                fail_count += 1
                # 继续处理下一条记录
                continue
        
        # 输出统计信息
        print(f"\n{'='*60}")
        print(f"📊 批量处理完成")
        print(f"{'='*60}")
        print(f"✅ 成功: {success_count}/{total}")
        print(f"❌ 失败: {fail_count}/{total}")
        print(f"📁 结果已保存到: {self.output_file}")
        print(f"{'='*60}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='MomoL-VL 攻击脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 单个文本和图片
  python attack.py --text "这是什么?" --image_path "image.jpg"
  
  # 批量处理 jsonl 文件
  python attack.py --type 01
  python attack.py --type 02 --server "http://192.168.1.100:5000"
  python attack.py --type 03 --start_index 10  # 从第 11 条记录开始
  
  # 指定服务器
  python attack.py --text "分析图片" --image_path "test.png" --server "http://192.168.1.100:5000"
  
  # 指定输出文件（单个模式）
  python attack.py --text "问题" --image_path "img.jpg" --output "./results.json"
        """
    )
    
    # 创建互斥组：--type 与 (--text, --image_path) 互斥
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--type',
        type=str,
        choices=['01', '02', '03', '04', '05', '06', '07'],
        help='批量处理模式：选择要处理的 jsonl 文件类型 (01-07)'
    )
    
    # 为单个模式创建一个标志参数
    mode_group.add_argument(
        '--text',
        type=str,
        help='单个处理模式：输入文本'
    )
    
    parser.add_argument(
        '--image_path',
        type=str,
        help='单个处理模式：图像路径 (jpg/png/gif/webp)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出 JSON 文件路径（单个模式默认: ./eomol.json，批量模式自动生成）'
    )
    
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='批量处理模式：从第几条记录开始处理（默认: 0）'
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
    
    # 判断是批量模式还是单个模式
    if args.type:
        # 批量处理模式
        if args.type not in TYPE_TO_FILE:
            print(f"❌ 错误: 不支持的类型 '{args.type}'")
            print(f"支持的类型: {', '.join(TYPE_TO_FILE.keys())}")
            return
        
        # 获取输入文件和输出文件
        input_file = TYPE_TO_FILE[args.type]
        type_name = TYPE_TO_NAME[args.type]
        
        # 如果没有指定输出文件，则自动生成
        if args.output:
            output_file = args.output
        else:
            output_file = f"./{args.type}-{type_name}-results.json"
        
        print(f"\n{'='*60}")
        print(f"📋 批量处理模式")
        print(f"🔢 类型: {args.type}")
        print(f"📁 输入文件: {input_file}")
        print(f"📝 输出文件: {output_file}")
        print(f"{'='*60}\n")
        
        # 创建运行器
        runner = AttackRunner(
            output_file=output_file,
            server_url=args.server,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 执行批量处理
        runner.run_batch(
            jsonl_path=input_file,
            start_index=args.start_index
        )
    
    else:
        # 单个处理模式
        if not args.text:
            print("❌ 错误: 单个处理模式需要 --text 参数")
            parser.print_help()
            return
        
        if not args.image_path:
            print("❌ 错误: 单个处理模式需要 --image_path 参数")
            parser.print_help()
            return
        
        # 如果没有指定输出文件，使用默认值
        output_file = args.output if args.output else './eomol.json'
        
        print(f"\n{'='*60}")
        print(f"📋 单个处理模式")
        print(f"{'='*60}\n")
        
        # 创建运行器
        runner = AttackRunner(
            output_file=output_file,
            server_url=args.server,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # 执行单个攻击
        runner.run(
            text=args.text,
            image_path=args.image_path
        )


if __name__ == '__main__':
    main()

