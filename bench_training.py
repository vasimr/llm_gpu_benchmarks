import os
import time
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler

from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

# FSDP Imports
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# 1. FSDP2 (Composable)
try:
    from torch.distributed.fsdp import fully_shard, FSDPModule
    #from torch.distributed.fsdp.api import MixedPrecisionPolicy # Sometimes located here
    from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy
    FSDPV2_AVAILABLE = True
except ImportError:
    # Fallback for slightly older 2.x versions
    try:
        from torch.distributed._composable.fsdp import fully_shard
        from torch.distributed.fsdp import MixedPrecision as MixedPrecisionPolicy # Compatibility alias
        FSDPV2_AVAILABLE = True
    except ImportError:
        FSDPV2_AVAILABLE = False


# Fairscale Imports (Conditional to avoid crashing if not installed)
try:
    from fairscale.nn.model_parallel.initialize import initialize_model_parallel
    from fairscale.nn.model_parallel.layers import (
        ColumnParallelLinear,
        ParallelEmbedding,
        RowParallelLinear,
    )
    FAIRSCALE_AVAILABLE = True
except ImportError:
    FAIRSCALE_AVAILABLE = False

# ---------------- CONFIG ----------------
# A "Reasonable" Transformer Block Size (similar to Llama 7B layers)
VOCAB = 8*1024
DIM = 8*1024
HEADS = DIM//64
LAYERS = 3  # Keep deep enough to cause memory transfer, shallow enough for quick bench
DEFAULT_SEQ_LEN = 256

# ---------------- CUSTOM COMPONENTS ----------------

class ChunkedCrossEntropyLoss(nn.Module):
    """
    Computes Cross Entropy Loss by chunking the vocabulary to avoid 
    materializing the massive [Batch*Seq, Vocab] logits matrix.
    Reduces peak VRAM usage by ~4x-10x for large vocabularies.
    """
    def __init__(self, chunk_size=4096):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, hidden_states, targets, weight_matrix):
        # hidden_states: [Batch*Seq, Dim]
        # targets: [Batch*Seq]
        # weight_matrix: [Vocab, Dim]
        
        # 1. Compute the numerator (Logits for the correct classes)
        # We only need the dot product of h[i] and W[target[i]]
        # Gather the specific weights for the targets
        target_weights = F.embedding(targets, weight_matrix) # [B*S, Dim]
        target_logits = torch.sum(hidden_states * target_weights, dim=-1) # [B*S]
        
        # 2. Compute the denominator (LogSumExp over all classes) in chunks
        # We use the Online Softmax trick for numerical stability
        
        # Initialize running max and sum
        max_logits = torch.full_like(target_logits, float('-inf'))
        sum_exp = torch.zeros_like(target_logits)
        
        vocab_size = weight_matrix.size(0)
        
        for i in range(0, vocab_size, self.chunk_size):
            end = min(i + self.chunk_size, vocab_size)
            chunk_W = weight_matrix[i:end] # [Chunk, Dim]
            
            # Compute logits for this chunk: [B*S, Chunk]
            chunk_logits = F.linear(hidden_states, chunk_W)
            
            # Update Online Softmax Statistics
            chunk_max = chunk_logits.max(dim=-1).values
            new_max = torch.maximum(max_logits, chunk_max)
            
            # Update sum_exp: sum_exp * exp(old_max - new_max) + exp(chunk_logits - new_max).sum
            # We do this carefully to avoid overflow
            sum_exp = sum_exp * torch.exp(max_logits - new_max) + \
                      torch.exp(chunk_logits - new_max.unsqueeze(-1)).sum(dim=-1)
            
            max_logits = new_max
            
        # 3. Finalize LogSumExp
        log_sum_exp = max_logits + torch.log(sum_exp)
        
        # 4. Cross Entropy = -log(p_target) = -target_logits + log_sum_exp
        loss = log_sum_exp - target_logits
        return loss.mean()

class FakeDataset(Dataset):
    def __init__(self, size=1000, seq_len=DEFAULT_SEQ_LEN):
        self.size = size
        self.seq_len = seq_len
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # Random integers for tokens
        return torch.randint(0, VOCAB, (self.seq_len,)), torch.randint(0, VOCAB, (self.seq_len,))


# =========================================================================
# MODEL 1: STANDARD (Used for DDP and FSDP)
# =========================================================================

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(DIM, HEADS, batch_first=True)
        self.norm1 = nn.LayerNorm(DIM)
        self.ff = nn.Sequential(
            nn.Linear(DIM, 4 * DIM),
            nn.GELU(),
            nn.Linear(4 * DIM, DIM)
        )
        self.norm2 = nn.LayerNorm(DIM)

    def forward(self, x):
        res = x
        x, _ = self.attn(x, x, x)
        x = self.norm1(x + res)
        res = x
        x = self.ff(x)
        return self.norm2(x + res)

    def reset_parameters(self):
        # Explicit init helper for Meta device materialization
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        for layer in self.ff:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        if hasattr(self.attn, '_reset_parameters'):
            self.attn._reset_parameters()

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, DIM)
        self.layers = nn.Sequential(*[TransformerBlock() for _ in range(LAYERS)])
        self.head = nn.Linear(DIM, VOCAB)

    def forward(self, x):
        x = self.embed(x)
        x = self.layers(x)
        #return x
        return self.head(x)

    def reset_parameters(self):
        self.embed.reset_parameters()
        self.head.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()


# =========================================================================
# MODEL 2: TENSOR PARALLEL (Llama Implementation)
# =========================================================================

class LlamaTPAttention(nn.Module):
    """
    Tensor Parallel Multi-Head Attention (Llama Style).
    Stripped of RoPE, GQA, and KV Cache for strict Bandwidth/FLOPs benchmarking.
    """
    def __init__(self):
        super().__init__()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.n_heads = HEADS
        self.head_dim = DIM // HEADS
        
        # Calculate local heads per GPU
        self.n_local_heads = self.n_heads // world_size
        
        # ColumnParallel: Splits the Output Dim. 
        # Each GPU gets a slice of Heads.
        # [Dim] -> [Dim, Local_Heads * Head_Dim]
        self.wq = ColumnParallelLinear(DIM, HEADS * self.head_dim, bias=False, gather_output=False)
        self.wk = ColumnParallelLinear(DIM, HEADS * self.head_dim, bias=False, gather_output=False)
        self.wv = ColumnParallelLinear(DIM, HEADS * self.head_dim, bias=False, gather_output=False)
        
        # RowParallel: Splits the Input Dim. 
        # Accepts split heads, computes partial dot product, then AllReduce.
        # [Local_Heads * Head_Dim] -> [Dim] (Synced)
        self.wo = RowParallelLinear(HEADS * self.head_dim, DIM, bias=False, input_is_parallel=True)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        bsz, seqlen, _ = x.shape
        
        # 1. Projections (Local Compute)
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # 2. Reshape for Matrix Mult
        # [Batch, Seq, Local_Heads * Head_Dim] -> [Batch, Local_Heads, Seq, Head_Dim]
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim).transpose(1, 2)

        # 3. Attention Scores (Math Bound)
        # We purposely use standard MatMul here to account for pure FLOPs.
        # scores: [Batch, Local_Heads, Seq, Seq]
        scores = torch.matmul(xq, xk.transpose(2, 3)) * self.head_dim**-0.5
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # 4. Weighted Sum
        # output: [Batch, Local_Heads, Seq, Head_Dim]
        output = torch.matmul(scores, xv)

        # 5. Reshape for Output Projection
        # [Batch, Seq, Local_Heads * Head_Dim]
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 6. Output Projection (Interconnect Bound)
        # Triggers AllReduce to combine heads from all GPUs
        return self.wo(output)

class TPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM)
        self.attn = LlamaTPAttention()
        self.norm2 = nn.LayerNorm(DIM)
        
        # Feed Forward (Standard Llama/Transformer MLP)
        # ColumnParallel (Up) -> GELU -> RowParallel (Down + AllReduce)
        self.ff1 = ColumnParallelLinear(DIM, 4 * DIM, bias=False, gather_output=False)
        self.act = nn.GELU()
        self.ff2 = RowParallelLinear(4 * DIM, DIM, bias=False, input_is_parallel=True)

    def forward(self, x):
        # Attention Residual
        h = x + self.attn(self.norm1(x))
        
        # MLP Residual
        # Note: ff1 output is split, passed to act, passed to ff2 (which accepts split input)
        # ff2 performs the AllReduce internally.
        out = self.ff2(self.act(self.ff1(self.norm2(h))))
        
        return h + out

class TPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = ParallelEmbedding(VOCAB, DIM)
        self.layers = nn.Sequential(*[TPBlock() for _ in range(LAYERS)])
        self.head = ColumnParallelLinear(DIM, VOCAB, gather_output=True)

    def forward(self, x):
        x = self.embed(x)
        x = self.layers(x)
        return self.head(x)

# =========================================================================
# MAIN
# =========================================================================

def setup(args):
    
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=local_rank,
            )
    torch.cuda.set_device(local_rank)
    
    if args.mode == "tp":
        if not FAIRSCALE_AVAILABLE:
            raise ImportError("Fairscale not installed. Run 'uv pip install fairscale'")
        world_size = dist.get_world_size()
        initialize_model_parallel(model_parallel_size_=world_size)

    # --- VRAM SIMULATION ---
    if args.limit_vram_gb > 0:
        total_mem = torch.cuda.get_device_properties(local_rank).total_memory
        limit_bytes = int(args.limit_vram_gb * 1024**3)
        fraction = limit_bytes / total_mem
        
        # Safety check to ensure we don't set > 1.0
        if fraction < 1.0:
            torch.cuda.set_per_process_memory_fraction(fraction, local_rank)
            if local_rank == 0:
                print(f"⚠️  SIMULATING HARDWARE LIMIT: Restricted to {args.limit_vram_gb} GB ({fraction:.2f}%)")

def cleanup():
    dist.destroy_process_group()

def format_metrics(tokens, time_sec):
    return f"{tokens/time_sec:.2f} tokens/s"

def run_bench(args):
    setup(args)
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    local_batch_size = args.batch_size // world_size

    if rank == 0:
        print(f"--- BENCHMARK CONFIG ---")
        print(f"GPUs: {world_size} | Mode: {args.mode}")
        print(f"Batch/GPU: {local_batch_size} | Seq Len: {args.seq_len}")
        print(f"------------------------")

    # 1. Model Init
    if args.mode == "tp":
        model = TPModel().to(torch.float16).to(device)
    elif args.mode == "fsdp":
        # FSDP MAGIC: Initialize on META device (0 Memory)
        #with torch.device("meta"):
        model = SimpleModel().to(torch.float16)

    elif args.mode == "fsdp2":
        # FSDP2: Init on Meta Device (No VRAM usage yet)
        if not FSDPV2_AVAILABLE: raise ImportError("FSDPv2 (fully_shard) not available in this PyTorch version.")
        with torch.device("meta"):
            model = SimpleModel().to(torch.float16)

    elif args.mode == "ddp":
        model = SimpleModel().to(torch.float16).to(device)

    local_param_count = sum(p.numel() for p in model.parameters())
    total_params = local_param_count
    if args.mode == "tp":
        # In TP, local params are roughly 1/N of total (ignoring LayerNorms which are replicated)
        # For a clearer benchmark comparison, we estimate the global size
        global_param_count = local_param_count * world_size
        total_params = global_param_count
        if rank == 0:
            print(f"Initializing TP Model:")
            print(f"  > Per-GPU Params: {local_param_count:,} ({local_param_count*2/1e6:.1f} MB)")
            print(f"  > Global Params:  ~{global_param_count:,} (Estimated)")
    else:
        # In DDP/FSDP (before wrapping), the model is fully instantiated locally
        if rank == 0:
            print(f"Initializing Standard Model:")
            print(f"  > Global Params: {local_param_count:,} ({local_param_count*2/1e6:.1f} MB)")


    # 2. Parallelism Strategy
    if args.mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])
    elif args.mode == "fsdp":
        fp16_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16
        )
        # Auto-wrap ensures we shard at the TransformerBlock level
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                TransformerBlock, 
                #torch.nn.Embedding,
                #torch.nn.Linear,
                },
        )
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD, # The "Heavy" FSDP
            device_id=device,
            mixed_precision=fp16_policy,
        )
    elif args.mode == "fsdp2":
        # Composable FSDP
        mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16, 
                reduce_dtype=torch.float16,
                output_dtype=torch.float16,
                )
        
        # 1. Apply Sharding Policy to Layers
        for layer in model.layers:
            fully_shard(layer, mp_policy=mp_policy)
            
        # 2. Apply to Root
        fully_shard(model, mp_policy=mp_policy)
        
        # 3. Materialize on Device (Allocates ONLY local shard)
        model.to_empty(device=device)
        
        # 4. Initialize Weights (Since to_empty leaves garbage)
        model.reset_parameters()

        print(model)

    optimizer = torch.optim.AdamW(model.parameters(), fused=True)
    criterion = nn.CrossEntropyLoss()
    #loss_fn = ChunkedCrossEntropyLoss(chunk_size=1024)

    # Initialize Scaler for DDP (FSDP does its own scaling via MixedPrecision)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mode == "ddp"))

    # 3. Data
    if args.use_static_batch:
        if rank == 0: print(f">> STATIC BATCH MODE: Generating {local_batch_size}x{args.seq_len} on GPU to bypass CPU...")
        # Generate ONCE on GPU
        static_input = torch.randint(0, VOCAB, (local_batch_size, args.seq_len), device=device)
        static_target = torch.randint(0, VOCAB, (local_batch_size, args.seq_len), device=device)
        # Create a dummy loader that just yields None (to keep loop structure)
        loader = [(None,None)] * (args.batch_size * 500) # Mock length
    else:
        dataset = FakeDataset(size=500*args.batch_size, seq_len=args.seq_len)
        if args.mode == "tp":
            sampler = torch.utils.data.SequentialSampler(dataset)
        else:
            sampler = DistributedSampler(dataset)

        loader = DataLoader(dataset, batch_size=local_batch_size, sampler=sampler)

    # 4. Training Step Helper
    def train_step(input_ids, targets):
        optimizer.zero_grad()
        
        #embed_weight = model.module.embed.weight

        with sdpa_kernel(SDPBackend.MATH):
            # DDP Path: Manual Autocast + Scaler
            if args.mode == "ddp":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model(input_ids)
                    loss = criterion(out.view(-1, VOCAB), targets.view(-1))
                    #hidden_states = model(input_ids)
                    #hidden_states = hidden_states.view(-1, DIM)
                    #targets = targets.view(-1)
                    #loss = loss_fn(hidden_states, targets, embed_weight)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            # FSDP Path: Handles casting internally
            elif args.mode in ["fsdp", "fsdp2"]:
                out = model(input_ids)
                loss = criterion(out.view(-1, VOCAB), targets.view(-1))
                #hidden_states = model(input_ids)
                #hidden_states = hidden_states.view(-1, DIM)
                #targets = targets.view(-1)
                #loss = loss_fn(hidden_states, targets, embed_weight)

                loss.backward()
                optimizer.step()
            elif args.mode == "tp":
                out = model(input_ids)
                loss = criterion(out.view(-1, VOCAB), targets.view(-1))
                loss.backward()
                optimizer.step()


        return loss

    # 5. Bench Loop
    model.train()
    
    if rank == 0: print("Warming up...")
    for i, (inputs, targets) in enumerate(loader):
        if i >= 5: break
        if inputs is None:
            inputs, targets = static_inputs, static_targets
        else:
            inputs, targets = inputs.to(device), targets.to(device)
        train_step(inputs, targets)

    dist.barrier()
   
    # --- RESET PEAK MEMORY BEFORE BENCH ---
    torch.cuda.reset_peak_memory_stats()

    if rank == 0: print("Benchmarking...")
    start_t = time.time()
    total_tokens = 0
    steps = 0

    try:
        for i, (inputs, targets) in enumerate(loader):
            if i >= args.steps: break
            if inputs is None:
                inputs, targets = static_inputs, static_targets
            else:
                inputs, targets = inputs.to(device), targets.to(device)
            
            train_step(inputs, targets)

            if args.mode == "tp":
                total_tokens += inputs.numel() 
            else:
                total_tokens += inputs.numel() * world_size

            steps += 1
            
        torch.cuda.synchronize()
        end_t = time.time()
        duration = time.time() - start_t

        if rank == 0:
            tps = total_tokens / duration
            latency_ms = (duration / steps) * 1000
            peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            # --- BANDWIDTH ESTIMATION ---
            # 1. DDP: 2 * Params (FP16) per step
            if args.mode == "ddp":
                data_per_step = 2 * total_params * 2 # 2 bytes
                bw_type = "Gradient AllReduce"
            
            # 2. FSDP: ~3 * Params (FP16) per step (Gather Fwd, Gather Bwd, Scatter Grad)
            elif args.mode in ["fsdp", "fsdp2"]:
                data_per_step = 3 * total_params * 2
                bw_type = "Params+Grads Sharding"

            # 3. TP: Activations (Batch*Seq*Dim) per layer per step
            # 4 All-Reduces per layer (Attn Fwd/Bwd, MLP Fwd/Bwd)
            elif args.mode == "tp":
                # Data = 4 * Layers * (GlobalBatch * Seq * Dim) * 2 bytes
                activation_volume = args.batch_size * args.seq_len * DIM * 2
                data_per_step = 4 * LAYERS * activation_volume
                bw_type = "Tensor Parallel Activations"

            avg_bw = (data_per_step * steps) / duration / 1e9

            print(f"--- RESULTS ({args.mode}) ---")
            print(f"Throughput: {tps:.2f} tokens/sec")
            print(f"Latency: {latency_ms:.2f} ms/step")
            print(f"Peak VRAM: {peak_mem_gb:.2f} GB")
            print(f"Est. Interconnect BW: {avg_bw:.2f} GB/s ({bw_type})")

    except torch.cuda.OutOfMemoryError:
        if rank == 0:
            print(f"\n❌ OOM ERROR: Batch Size {args.batch_size} is too large for memory limit.")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["ddp", "fsdp", "fsdp2", "tp"], default="ddp")
    parser.add_argument("--batch_size", type=int, default=8) # Lowered default
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--limit_vram_gb", type=float, default=0, help="Simulate V100 by capping memory (e.g. 31.0)")
    parser.add_argument("--use_static_batch", action="store_true", help="Use pre-allocated GPU data to bypass CPU bottleneck")
    args = parser.parse_args()
    run_bench(args) 
