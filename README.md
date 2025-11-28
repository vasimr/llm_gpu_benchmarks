# Distributed GPU Benchmarking Tools

This repository provides lightweight benchmarking utilities for evaluating distributed GPU inference and training performance for language models. Tools are designed to measure throughput under varying tensor parallelism and communication settings, intended mainly for systems with up to 32 GB of GPU memory.

## Contents

| File | Purpose |
|------|----------|
| bench_inference.sh       | Helper script for running vLLM inference benchmarks. |
| bench_training.py        | PyTorch training benchmark for large transformer models. |
| run_inference_bench.sh   | Main inference benchmarking script. |
| run_train_grid.sh        | Main training benchmarking driver. |

---

## Inference Benchmark

Inference is implemented using **vLLM**. The scripts sweep over different batch sizes and tensor-parallel sizes, with and without NVLink enabled, to measure throughput.

Usage:

```

./run_inference_bench.sh GPU_NAME MAX_TP TEST_P2P_TOGGLE

```

Example:

```

./run_inference_bench.sh a6000 2 1

```

Output is written to:

```

inference_benchmark_GPU_NAME.csv

```

Model configuration defaults to **Llama 3.1-8B FP16**. Only the config is fetched (no weight download required) to keep VRAM usage below 32 GB.

---

## Training Benchmark

Training uses a PyTorch benchmark that constructs an extremely wide transformer to stress inter-GPU communication. The script supports:

- DDP (`--mode ddp`)
- FSDP (`--mode fsdp`)
- FSDP2 (`--mode fsdp2`)
- Megatron-style tensor parallelism (`--mode tp`)
and sweeps over batch size, GPU count, sequence length, and NVLink enabled/disabled.

Usage:

```

./run_train_grid.sh GPU_NAME MAX_GPUS TEST_P2P_TOGGLE

```

Example:

```

./run_train_grid.sh a6000 2 1

```

Results are saved to:

```

training_benchmark_GPU_NAME.csv

```

Default model configuration: **2.5 B parameters**, batch size **16**, sequence length **256**, fitting within 32 GB VRAM. The underlying training script also provides a hardware emulation parameter `--limit_vram_gb` which can reduce the VRAM when testing to validate OOM behavior on lower VRAM devices.

---

## Requirements

```

pip install vllm transformers accelerate datasets fairscale

```

---

This project is intended as a simple reference for benchmarking and comparing distributed GPU throughput across inference and training configurations.

