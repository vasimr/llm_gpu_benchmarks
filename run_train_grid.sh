#!/bin/bash
# Usage: ./run_grid.sh [GPU_NAME] [MAX_GPUS] [TEST_P2P_TOGGLE]
# Example: ./run_grid.sh a6000 2 1
# Example: ./run_grid.sh v100 8 0

GPU_NAME=${1:-"generic"}
MAX_GPUS=${2:-2}
TEST_P2P=${3:-0} # 1 = Test both ON/OFF, 0 = Just ON

#EXTRA_ARGS="--limit_vram_gb 31"
EXTRA_ARGS=""

START_TIME=$(date +%s)
echo "=== Benchmark Started at: $(date) ==="

OUTPUT_FILE="training_benchmark_${GPU_NAME}.csv"
echo "GPU,Mode,Num_GPUs,Global_Batch,Seq_Len,P2P_Enabled,Throughput,Latency,Peak_VRAM,BW_Est" > $OUTPUT_FILE

run_test() {
    local MODE=$1
    local GPUS=$2
    local G_BATCH=$3
    local SEQ=$4
    local P2P=$5

    # Skip invalid configs
    if [ "$G_BATCH" -lt "$GPUS" ]; then return; fi
    
    # Skip Single GPU for TP/FSDP (Not useful for comparison)
    if [ "$GPUS" -eq 1 ] && [[ "$MODE" == "tp" || "$MODE" == "fsdp" || "$MODE" == "fsdp2" ]]; then return; fi

    # Set P2P Environment
    if [ "$P2P" -eq 1 ]; then
        unset NCCL_P2P_DISABLE
        P2P_STR="YES"
    else
        export NCCL_P2P_DISABLE=1
        P2P_STR="NO"
    fi

    echo ">> Running: $MODE | GPUs: $GPUS | Batch: $G_BATCH | Seq: $SEQ | P2P: $P2P_STR"

    # Run Python Script
    # Capture output to temp file to parse metrics
    torchrun --nproc_per_node=$GPUS bench_training.py $EXTRA_ARGS \
        --mode $MODE \
        --batch_size $G_BATCH \
        --seq_len $SEQ \
        --steps 20 2>&1 | tee temp_res.txt

    # Parse Results
    TPS=$(grep "Throughput:" temp_res.txt | awk '{print $2}')
    LAT=$(grep "Latency:" temp_res.txt | awk '{print $2}')
    VRAM=$(grep "Peak VRAM:" temp_res.txt | awk '{print $3}')
    BW=$(grep "Est. Interconnect BW:" temp_res.txt | awk '{print $4}')

    if [ -z "$TPS" ]; then TPS="OOM/Fail"; fi

    echo "$GPU_NAME,$MODE,$GPUS,$G_BATCH,$SEQ,$P2P_STR,$TPS,$LAT,$VRAM,$BW" >> $OUTPUT_FILE
    echo "   Result: $TPS tokens/s | $BW GB/s"
    rm temp_res.txt
}

# ===============================================================
# 1. SCENARIO A: V100 LIMITS (Strong & Weak Scaling)
#    Seq=256, Batch=16 vs Batch=16*N
# ===============================================================
echo "=== SCENARIO A: BASELINE SCALING ==="

current_gpu=1
while [ $current_gpu -le $MAX_GPUS ]; do
    
    # Define Batch Sizes for this GPU count
    # 1. Strong Scaling (Fixed 16)
    # 2. Weak Scaling (16 * N)
    BATCHES=(16 $((16 * current_gpu)))
    # Remove duplicates if 1 GPU
    if [ "$current_gpu" -eq 1 ]; then
        # Single GPU: Baseline only
        BATCHES=(16)
        MODES=("ddp")
    else
        # Multi GPU: Test Scaling
        # Strong Scaling: 16 (Global) -> 8 per GPU (N=2) -> 2 per GPU (N=8)
        # Weak Scaling:   16*N (Global) -> 16 per GPU
        BATCHES=(16 $((16 * current_gpu)))
        MODES=("ddp" "fsdp" "tp")
    fi

    for G_BATCH in "${BATCHES[@]}"; do
        for MODE in "${MODES[@]}"; do
            
            # P2P Toggle Logic
            if [ "$TEST_P2P" -eq 1 ] && [ "$current_gpu" -gt 1 ]; then
                run_test $MODE $current_gpu $G_BATCH 256 1 # ON
                run_test $MODE $current_gpu $G_BATCH 256 0 # OFF
            else
                run_test $MODE $current_gpu $G_BATCH 256 1
            fi
        done
    done
    current_gpu=$((current_gpu * 2))
done

# ===============================================================
# 2. SCENARIO B: MODERN LOAD (High Batch) - GPU >= 2 Only
#    Seq=256, Global Batch 64/128/256
# ===============================================================
if [ "$MAX_GPUS" -ge 2 ]; then
    echo "=== SCENARIO B: HIGH LOAD (VRAM ADVANTAGE) ==="
    # Only test on Max GPUs available (e.g. 2 for A6000, 8 for V100)
    # Testing DDP and FSDP2 (Most efficient)
    
    for G_BATCH in 64 128 256; do
        for MODE in "ddp" "fsdp" "tp"; do
            if [ "$TEST_P2P" -eq 1 ]; then
                run_test $MODE $MAX_GPUS $G_BATCH 256 1
                run_test $MODE $MAX_GPUS $G_BATCH 256 0
            else
                run_test $MODE $MAX_GPUS $G_BATCH 256 1
            fi
        done
    done
fi

# ===============================================================
# 3. SCENARIO C: TP BANDWIDTH TORTURE
#    Batch=16, Seq=1024, 2048, 4096 (TP Only)
# ===============================================================
if [ "$MAX_GPUS" -ge 2 ]; then
    echo "=== SCENARIO C: TP BANDWIDTH STRESS TEST ==="
    
    for SEQ in 1024 2048 4096; do
        if [ "$TEST_P2P" -eq 1 ]; then
            run_test "tp" $MAX_GPUS 16 $SEQ 1
            run_test "tp" $MAX_GPUS 16 $SEQ 0
        else
            run_test "tp" $MAX_GPUS 16 $SEQ 1
        fi
    done
fi

# Calculate Duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo ""
echo "================================================================"
echo "Benchmark Finished at: $(date)"
echo "Total Execution Time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results saved to $OUTPUT_FILE"
echo "================================================================"
cat $OUTPUT_FILE
