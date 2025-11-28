#!/bin/bash
# Usage: ./bench_inference.sh [TP_SIZE] [BATCH_SIZE] [USE_P2P]

TP=${1:-1}
BS=${2:-128}
USE_P2P=${3:-1}  # Default to 1 (Enabled)

# 1. Configure P2P (The Switch)
if [ "$USE_P2P" -eq "0" ]; then
    echo "⚠️  DISABLING P2P (Simulating PCIe-only/System RAM routing)"
    export NCCL_P2P_DISABLE=1
    export VLLM_NCCL_P2P_DISABLE=1 
else
    echo "✅ P2P ENABLED (Using NVLink)"
    unset NCCL_P2P_DISABLE
    unset VLLM_NCCL_P2P_DISABLE
fi

# 2. Model Config
# We use the Unsloth variant to avoid the Meta gate mechanism.
# It is architecturally identical to Llama-3.1-8B.
MODEL="unsloth/Meta-Llama-3.1-8B-Instruct"

echo "------------------------------------------------"
echo "Running Bench: TP=$TP | Batch=$BS | P2P=$USE_P2P"
echo "------------------------------------------------"

# 3. Run vLLM Benchmark
# --load-format dummy: Skips downloading weights (uses random initialization)
# --enforce-eager: Disables CUDA Graphs to expose raw kernel/interconnect latency
vllm bench throughput \
    --model $MODEL \
    --tensor-parallel-size $TP \
    --load-format dummy \
    --dtype float16 \
    --input-len 512 \
    --output-len 256 \
    --num-prompts 200 \
    --enforce-eager \
    --max-model-len 4096
