# InternVL3-78B 测试说明

## 🚀 快速开始

### 1. 启动 vLLM 服务器

```bash
CUDA_VISIBLE_DEVICES=1,3 \
vllm serve /mnt/nfs/lq/internVL \
    --tensor-parallel-size 2 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.98 \
    --trust-remote-code \
    --disable-custom-all-reduce
```

### 2. 运行测试脚本

```bash
python test_intern_api.py
```

## 📋 测试内容

脚本会自动测试：

1. **获取模型列表** - 确定 vLLM 注册的确切模型名称
2. **测试两种模型名称**:
   - `internVL` (目录名)
   - `InternVL3-78B` (完整模型名)

3. **测试 10+ 种 API 格式**:
   - ✅ vLLM Chat Completion API (`/v1/chat/completions`)
   - ✅ `/invocations` 端点的各种格式
   - ✅ 不同的图片传递方式 (base64, data URL)

## 🔍 查看测试结果

脚本会输出每种格式的测试结果：
- ✅ **成功** - 显示完整响应
- ❌ **失败** - 显示错误详情

找到标记为 "✅ 成功!" 的格式，那就是正确的调用方式。

## 📝 后续步骤

测试成功后：

1. 记录成功的格式
2. 更新 `run.py` 使用正确的格式
3. 使用 `attack.py --type XX` 进行批量处理

## 💡 常见问题

### Q: 所有格式都失败怎么办？
A: 检查：
1. vLLM 服务器是否正常运行
2. 端口 8000 是否可访问
3. 查看 vLLM 服务器日志获取更多信息

### Q: 模型名称不对怎么办？
A: 脚本会先调用 `/v1/models` 获取正确的模型名称

### Q: 超时怎么办？
A: 超时设置为 30 秒，如果模型加载慢，可以增加超时时间

