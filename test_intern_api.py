#!/usr/bin/env python3
"""
测试 InternVL3-78B API 的脚本
用于诊断请求格式问题
"""

import requests
import base64
import json

def test_api_formats():
    """测试不同的 API 格式"""
    
    # 准备测试数据
    server_url = "http://localhost:8000"
    test_text = "请描述这张图片"
    
    # 创建一个简单的测试图片（1x1像素的PNG）
    # 这是一个base64编码的最小PNG图片
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    print("=" * 60)
    print("测试 InternVL3-78B API 格式")
    print("=" * 60)
    
    # 测试可能的模型名称
    model_names = ["internVL", "InternVL3-78B"]
    
    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"测试模型名称: {model_name}")
        print(f"{'='*60}")
        
        # 测试格式1: vLLM Chat Completion API (标准格式)
        print(f"\n[测试 1] vLLM Chat Completion API - /v1/chat/completions")
        payload1 = {
            'model': model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': {
                                'url': f'data:image/png;base64,{tiny_png_base64}'
                            }
                        },
                        {
                            'type': 'text',
                            'text': test_text
                        }
                    ]
                }
            ],
            'max_tokens': 512,
            'temperature': 0.7
        }
        test_endpoint(server_url + "/v1/chat/completions", payload1, model_name)
        
        # 测试格式2: 简化的 messages 格式
        print(f"\n[测试 2] 简化 messages 格式")
        payload2 = {
            'model': model_name,
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': test_text},
                        {'type': 'image', 'image': tiny_png_base64}
                    ]
                }
            ]
        }
        test_endpoint(server_url + "/v1/chat/completions", payload2, model_name)
    
    print(f"\n{'='*60}")
    print("测试 /invocations 端点的各种格式")
    print(f"{'='*60}")
    
    # 测试格式3: 标准格式 {text, image}
    print("\n[测试 3] 标准格式: {text, image}")
    payload3 = {
        'text': test_text,
        'image': tiny_png_base64
    }
    test_endpoint(server_url + "/invocations", payload3, "N/A")
    
    # 测试格式4: {prompt, image_data}
    print("\n[测试 4] 格式: {prompt, image_data}")
    payload4 = {
        'prompt': test_text,
        'image_data': tiny_png_base64
    }
    test_endpoint(server_url + "/invocations", payload4, "N/A")
    
    # 测试格式5: vLLM 多模态格式
    print("\n[测试 5] vLLM 多模态格式: {prompt, multi_modal_data}")
    payload5 = {
        'prompt': test_text,
        'multi_modal_data': {
            'image': tiny_png_base64
        }
    }
    test_endpoint(server_url + "/invocations", payload5, "N/A")
    
    # 测试格式6: Data URL 格式
    print("\n[测试 6] Data URL 格式")
    payload6 = {
        'text': test_text,
        'image': f"data:image/png;base64,{tiny_png_base64}"
    }
    test_endpoint(server_url + "/invocations", payload6, "N/A")
    
    # 测试格式7: MLflow 格式
    print("\n[测试 7] MLflow 格式: {inputs: {text, image}}")
    payload7 = {
        'inputs': {
            'text': test_text,
            'image': tiny_png_base64
        }
    }
    test_endpoint(server_url + "/invocations", payload7, "N/A")
    
    # 测试格式8: 列表格式
    print("\n[测试 8] 列表格式: {instances: [...]}")
    payload8 = {
        'instances': [
            {
                'text': test_text,
                'image': tiny_png_base64
            }
        ]
    }
    test_endpoint(server_url + "/invocations", payload8, "N/A")
    
    # 测试格式9: OpenAI 风格的 messages
    print("\n[测试 9] OpenAI 风格 messages (无 model 字段)")
    payload9 = {
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': test_text},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{tiny_png_base64}'}}
                ]
            }
        ]
    }
    test_endpoint(server_url + "/invocations", payload9, "N/A")
    
    # 测试格式10: 简单的 prompt + image
    print("\n[测试 10] 简单格式: {prompt, image}")
    payload10 = {
        'prompt': test_text,
        'image': tiny_png_base64
    }
    test_endpoint(server_url + "/invocations", payload10, "N/A")

def test_endpoint(url, payload, model_name):
    """测试单个端点和格式"""
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"  模型名称: {model_name}")
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ 成功!")
            try:
                result = response.json()
                # 美化输出
                result_str = json.dumps(result, ensure_ascii=False, indent=2)
                if len(result_str) > 500:
                    print(f"  响应预览: {result_str[:500]}...")
                else:
                    print(f"  完整响应: {result_str}")
            except:
                print(f"  响应文本: {response.text[:500]}")
        else:
            print(f"  ❌ 失败")
            try:
                error_json = response.json()
                print(f"  错误详情: {json.dumps(error_json, ensure_ascii=False, indent=2)[:500]}")
            except:
                print(f"  错误文本: {response.text[:500]}")
            
    except requests.exceptions.Timeout:
        print(f"  ❌ 请求超时（30秒）")
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 连接失败 - 请确保服务器正在运行")
    except Exception as e:
        print(f"  ❌ 异常: {e}")

def test_model_list():
    """测试获取模型列表"""
    server_url = "http://localhost:8000"
    
    print("\n" + "=" * 60)
    print("获取可用模型列表")
    print("=" * 60)
    
    try:
        response = requests.get(f"{server_url}/v1/models", timeout=10)
        if response.status_code == 200:
            models = response.json()
            print("✅ 可用模型:")
            print(json.dumps(models, ensure_ascii=False, indent=2))
        else:
            print(f"❌ 获取模型列表失败: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ 异常: {e}")

if __name__ == '__main__':
    # 先获取模型列表
    test_model_list()
    
    # 然后测试各种格式
    print("\n")
    test_api_formats()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

