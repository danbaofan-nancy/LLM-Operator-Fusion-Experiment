# 算子融合如何缓解大语言模型中的“内存墙”问题？：基于 Decoder-Only Transformer 

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")


# 1. 定义模型

class SelfAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, _ = x.shape
        
        # (B, T, d_model) -> (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        # 因果掩码
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        
        # (B, num_heads, T, head_dim) -> (B, T, d_model)
        out = (attn @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model=256, d_ff=1024):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

class DecoderBlock(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, num_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size=65, d_model=256, d_ff=1024, 
                 num_layers=4, max_len=512, num_heads=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_ff, num_heads) 
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# 2. 验证模型定义正确，能够在GPU上运行，并且输出形状正确

print("\n验证模型...")
test_model = DecoderOnlyTransformer(vocab_size=65, num_layers=2, d_model=128, d_ff=512).to(device)
test_input = torch.randint(0, 65, (2, 64)).to(device)
with torch.no_grad():
    test_output = test_model(test_input)
print(f"验证通过! 输出shape: {test_output.shape}")
del test_model, test_input
torch.cuda.empty_cache()


# 3. 实验配置

configs = [
    {"name": "Config A (B=2,T=64)", "batch_size": 2, "seq_len": 64},
    {"name": "Config B (B=2,T=128)", "batch_size": 2, "seq_len": 128},
    {"name": "Config C (B=4,T=128)", "batch_size": 4, "seq_len": 128},
    {"name": "Config D (B=4,T=256)", "batch_size": 4, "seq_len": 256},
]

# 模型超参数 (减小以加快实验)
VOCAB_SIZE = 65
D_MODEL = 128 
D_FF = 512   
NUM_LAYERS = 4     
NUM_HEADS = 8

NUM_ITERATIONS = 15  # 每个配置测试次数

results = []

print(f"\n模型参数: d_model={D_MODEL}, d_ff={D_FF}, num_layers={NUM_LAYERS}")


# 4. 运行实验


for cfg in configs:
    B, T = cfg["batch_size"], cfg["seq_len"]
    print(f"\n{'='*50}")
    print(f"Testing: {cfg['name']}")
    print(f"Batch={B}, SeqLen={T}")
    print(f"{'='*50}")
    
    # 创建随机输入
    input_ids = torch.randint(0, VOCAB_SIZE, (B, T)).to(device)
    
    # ---------- Baseline (原生PyTorch) ----------
    print("  Running baseline...")
    model_baseline = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE, 
        d_model=D_MODEL, 
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=T+10,
        num_heads=NUM_HEADS
    ).to(device)
    model_baseline.eval()
    
    # 预热
    with torch.no_grad():
        for _ in range(2):
            _ = model_baseline(input_ids)
    torch.cuda.synchronize()
    
    # 测量延迟
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_baseline(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    lat_baseline = sum(times) / len(times) * 1000
    
    # 测量显存
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model_baseline(input_ids)
    mem_baseline = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # 清理
    del model_baseline
    torch.cuda.empty_cache()
    gc.collect()
    
    # ---------- Compiled (torch.compile) ----------
    print("  Running compiled...")
    model_compiled = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=T+10,
        num_heads=NUM_HEADS
    ).to(device)
    
    # 使用reduce-overhead模式，更适合小模型
    model_compiled = torch.compile(model_compiled, mode="reduce-overhead")
    model_compiled.eval()
    
    # 编译预热 (需要几次forward来触发编译)
    print("    Warming up compiler...")
    with torch.no_grad():
        for i in range(3):
            _ = model_compiled(input_ids)
            torch.cuda.synchronize()
    
    # 测量延迟
    times = []
    for _ in range(NUM_ITERATIONS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model_compiled(input_ids)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    lat_compiled = sum(times) / len(times) * 1000
    
    # 测量显存
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model_compiled(input_ids)
    mem_compiled = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    del model_compiled
    torch.cuda.empty_cache()
    gc.collect()
    
    # 记录结果
    speedup = lat_baseline / lat_compiled
    mem_reduction = (1 - mem_compiled / mem_baseline) * 100
    
    results.append({
        "name": cfg["name"],
        "B": B, 
        "T": T,
        "lat_baseline_ms": lat_baseline,
        "lat_compiled_ms": lat_compiled,
        "speedup": speedup,
        "mem_baseline_mb": mem_baseline,
        "mem_compiled_mb": mem_compiled,
        "mem_reduction_pct": mem_reduction
    })
    
    print(f"    Baseline: {lat_baseline:.2f} ms, Compiled: {lat_compiled:.2f} ms")
    print(f"    Speedup: {speedup:.2f}x, Memory reduction: {mem_reduction:.1f}%")


# 5. 打印结果


print("\n" + "="*80)
print("实验结果汇总")
print("="*80)
print()

# 表格头
print(f"{'Config':<20} {'B':<4} {'T':<5} {'Baseline(ms)':<12} {'Compiled(ms)':<12} {'Speedup':<8} {'Mem Base':<9} {'Mem Comp':<9}")
print("-"*90)

for r in results:
    print(f"{r['name']:<20} {r['B']:<4} {r['T']:<5} {r['lat_baseline_ms']:<12.2f} {r['lat_compiled_ms']:<12.2f} {r['speedup']:<8.2f} {r['mem_baseline_mb']:<9.1f} {r['mem_compiled_mb']:<9.1f}")

print()
print("="*80)
print("分析结论:")
print("="*80)

# 计算平均加速比
avg_speedup = sum(r['speedup'] for r in results) / len(results)
avg_mem_reduction = sum(r['mem_reduction_pct'] for r in results) / len(results)

print(f"平均加速比: {avg_speedup:.2f}x")
print(f"平均显存节省: {avg_mem_reduction:.1f}%")

# 找出最佳和最差
best = max(results, key=lambda x: x['speedup'])
worst = min(results, key=lambda x: x['speedup'])
print(f"最佳加速: {best['name']} -> {best['speedup']:.2f}x")
print(f"最差加速: {worst['name']} -> {worst['speedup']:.2f}x")