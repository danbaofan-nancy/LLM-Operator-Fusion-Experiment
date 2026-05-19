# LLM-Operator-Fusion-Experiment

# 算子融合如何缓解大语言模型中的"内存墙"问题？

## 项目简介

本项目基于Decoder-Only Transformer架构，通过PyTorch的`torch.compile`工具，实验验证算子融合对缓解LLM推理中“内存墙”问题的效果。实验在NVIDIA Tesla T4 GPU上进行，对比了算子融合前后的推理延迟和显存占用。

## 核心发现

- **最高加速比**：2.89倍（B=2, T=64）
- **平均加速比**：1.96倍
- **平均显存节省**：45.8%
- 显存节省最多的配置（61.5%）反而加速比最低（1.38倍），表明融合的核心收益来自减少内核调度开销，而非单纯节省显存

## 实验设置

| 实验变量 | Batch Size | Sequence Length | 加速比 | 显存节省 |
|------|-----------|-----------------|--------|----------|
| A | 2 | 64 | 2.89x | 36.0% |
| B | 2 | 128 | 2.09x | 40.1% |
| C | 4 | 128 | 1.46x | 45.5% |
| D | 4 | 256 | 1.38x | 61.5% |

## 环境要求

- PyTorch >= 2.0（需要`torch.compile`支持）
- NVIDIA GPU（测试使用Tesla T4）
- Kaggle Notebook 或 本地CUDA环境

## 运行方法

```python
# 启用算子融合
model = DecoderOnlyTransformer(...).to(device)
model = torch.compile(model, mode="reduce-overhead")
```

