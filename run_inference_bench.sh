#!/bin/bash
# Usage: ./run_inference_bench.sh [GPU_NAME] [MAX_TP] [TEST_P2P_TOGGLE]
# Example: ./run_inference_bench.sh a6000 2 1

GPU_NAME=${1:-"unknown_gpu"}
MAX_TP=${2:-1}
TEST_P2P_TOGGLE=${3:-0}

OUTPUT_FILE="inference_benchmark_${GPU_NAME}.csv"
TEMP_FILE="temp_infer_output.txt"

START_TIME=$(date +%s)
echo "=== INFERENCE BENCHMARK STARTED: $(date) ==="
echo "Target: $GPU_NAME | Max TP: $MAX_TP | P2P Toggle: $TEST_P2P_TOGGLE"

# Create CSV Header if not exists
if [ ! -f "$OUTPUT_FILE" ]; then
    echo "GPU,TP,Batch,P2P_Enabled,Req_per_Sec,Tokens_per_Sec,Output_Tokens_per_Sec" > $OUTPUT_FILE
fi

current_tp=1
while [ $current_tp -le $MAX_TP ]; do
    # Testing Latency (1), Sweet Spot (32), and Saturation (64)
    for BS in 1 32 64; do

        # Logic to determine if we run P2P ON/OFF or just DEFAULT
        if [ "$TEST_P2P_TOGGLE" -eq "1" ] && [ "$current_tp" -gt 1 ]; then
            P2P_MODES=("1" "0")
        else
            P2P_MODES=("1")
        fi

        for P2P_SETTING in "${P2P_MODES[@]}"; do

            # Determine Label for CSV and Environment
            if [ "$TEST_P2P_TOGGLE" -eq "1" ] && [ "$current_tp" -gt 1 ]; then
                if [ "$P2P_SETTING" -eq "1" ]; then 
                    P2P_LABEL="YES"
                else 
                    P2P_LABEL="NO"
                fi
            else
                P2P_LABEL="DEFAULT"
            fi

            echo "----------------------------------------------------------------"
            echo ">> RUNNING: TP=$current_tp | Batch=$BS | P2P: $P2P_LABEL"
            echo "----------------------------------------------------------------"

            # --- EXECUTION ---
            # Pipe to tee to show progress and save for parsing
            ./bench_inference.sh $current_tp $BS $P2P_SETTING 2>&1 | tee $TEMP_FILE

            # --- PARSING ---
            # vLLM Output format: 
            # "Throughput: 6.33 requests/s, 4863.79 total tokens/s, 1621.26 output tokens/s"
            
            # Extract specific metrics using awk based on column position
            RPS=$(grep "^Throughput:" $TEMP_FILE | awk '{print $2}' | tail -n 1)
            TPS=$(grep "^Throughput:" $TEMP_FILE | awk '{print $4}' | tail -n 1)
            OUT_TPS=$(grep "^Throughput:" $TEMP_FILE | awk '{print $7}' | tail -n 1)

            # Check for failure
            if [ -z "$RPS" ]; then
                RPS="0"; TPS="0"; OUT_TPS="0"
                echo "   >>> DETECTED FAILURE (OOM or Error) <<<"
            else
                echo "   >>> SUCCESS: $RPS req/s | $TPS tok/s | $OUT_TPS out_tok/s"
            fi

            # Append to CSV
            echo "$GPU_NAME,$current_tp,$BS,$P2P_LABEL,$RPS,$TPS,$OUT_TPS" >> $OUTPUT_FILE

            # Clean up
            rm -f $TEMP_FILE
        done
    done
    
    # Increment TP (Powers of 2: 1, 2, 4...)
    current_tp=$((current_tp * 2))
done

# Calculate Duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINS=$(((DURATION % 3600) / 60))
SECS=$((DURATION % 60))

echo ""
echo "================================================================"
echo "Inference Benchmark Finished at: $(date)"
echo "Total Execution Time: ${HOURS}h ${MINS}m ${SECS}s"
echo "Results saved to $OUTPUT_FILE"
echo "================================================================"
cat $OUTPUT_FILE
