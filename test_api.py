#!/usr/bin/env python3
"""
测试 Momol API 的脚本
用于诊断请求格式问题
"""

import requests
import base64
import json

def test_api_formats():
    """测试不同的 API 格式"""
    
    # 准备测试数据
    server_url = "http://localhost:8000"
    test_text = "描述这张图片"
    
    # 创建一个简单的测试图片（1x1像素的PNG）
    # 这是一个base64编码的最小PNG图片
    tiny_png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    print("=" * 60)
    print("测试 Momol API 格式")
    print("=" * 60)
    
    # 测试格式1: 标准格式
    print("\n[测试 1] 标准格式: {text, image}")
    payload1 = {
        'text': test_text,
        'image': tiny_png_base64
    }
    test_endpoint(server_url + "/invocations", payload1)
    
    # 测试格式2: inputs 包装
    print("\n[测试 2] MLflow 格式: {inputs: {text, image}}")
    payload2 = {
        'inputs': {
            'text': test_text,
            'image': tiny_png_base64
        }
    }
    test_endpoint(server_url + "/invocations", payload2)
    
    # 测试格式3: vLLM 多模态格式（推荐）
    print("\n[测试 3] vLLM 多模态格式: {prompt, multi_modal_data: {image}}")
    payload3 = {
        'prompt': test_text,
        'multi_modal_data': {
            'image': tiny_png_base64
        }
    }
    test_endpoint(server_url + "/invocations", payload3)
    
    # 测试格式3b: 旧的 image_data 格式
    print("\n[测试 3b] 备选格式: {prompt, image_data}")
    payload3b = {
        'prompt': test_text,
        'image_data': tiny_png_base64
    }
    test_endpoint(server_url + "/invocations", payload3b)
    
    # 测试格式4: 数据URL格式
    print("\n[测试 4] Data URL 格式")
    payload4 = {
        'text': test_text,
        'image': f"data:image/png;base64,{tiny_png_base64}"
    }
    test_endpoint(server_url + "/invocations", payload4)
    
    # 测试格式5: messages 格式（类似 OpenAI）
    print("\n[测试 5] Messages 格式")
    payload5 = {
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
    test_endpoint(server_url + "/invocations", payload5)
    
    # 测试格式6: 列表格式
    print("\n[测试 6] 列表格式")
    payload6 = {
        'instances': [
            {
                'text': test_text,
                'image': tiny_png_base64
            }
        ]
    }
    test_endpoint(server_url + "/invocations", payload6)

def test_endpoint(url, payload):
    """测试单个端点和格式"""
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=10,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"  状态码: {response.status_code}")
        
        if response.status_code == 200:
            print(f"  ✅ 成功!")
            try:
                result = response.json()
                print(f"  响应: {json.dumps(result, ensure_ascii=False)[:200]}")
            except:
                print(f"  响应文本: {response.text[:200]}")
        else:
            print(f"  ❌ 失败")
            print(f"  错误: {response.text[:300]}")
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 连接失败")
    except Exception as e:
        print(f"  ❌ 异常: {e}")

if __name__ == '__main__':
    test_api_formats()

