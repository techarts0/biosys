# BIOSYS: A Hierarchical Memory Service for Long-Horizon LLM Agents

Official evaluation code, trace generators, vLLM prompt-replay harnesses, and experimental result logs for:

> **BIOSYS: A Hierarchical Memory Service for Long-Horizon LLM Agents**  
> Pengfei Wang, Xiaolei Guo, and Mingyuan Zhao

---

## Repository Contents

| File | Description |
| :--- | :--- |
| `eval_locomo.py` | Layer 1 trace-driven simulation and ablations on official LoCoMo (10 conversations, 1,536 evidence-bearing QAs) |
| `kill_test.py` | Layer 1 synthetic Coding Agent (240 turns) and Mutable-State consolidation-lag sweep (`L = 5, 10, 20`) |
| `run_supplementary_checks.py` | Layer 2 on-disk `os.fsync` WAL durability & crash-recovery microbenchmark, $(\alpha, E)$ sensitivity sweep, and truncation audit |
| `bench_vllm_serving.py` | Layer 3 sequential live vLLM serving verification (`Qwen2.5-7B-Instruct`, 10,992 requests) |
| `bench_vllm_concurrent.py` | Layer 3 multi-tenant concurrent shared-GPU sweep (`C = 1..32`, 8,540 requests) and `--verify-r1` strictly isolated 3-repeat verification (`10,560` requests) |
| `draw_architecture_figs.py` | Generates vector architecture diagrams (Figures 1–4) |
| `draw_eval_figs.py` | Generates experimental result figures (Figures 5–9) directly from the JSON logs |
| `*_results.json` | Recorded experimental results underlying Tables II–V and Figures 5–9 |

---

## Setup

### 1. Dependencies
```bash
pip install numpy scikit-learn matplotlib
# For Layer 3 live GPU serving (NVIDIA GPU with >=24 GB VRAM, e.g., A10):
pip install vllm
```

### 2. LoCoMo Dataset
Download the third-party `locomo10.json` dataset from the official [Snap Research LoCoMo repository](https://github.com/snap-research/locomo) and place it at:
```text
data/locomo10.json
```

---

## Reproducing Experiments

### Layer 1: Trace Simulation & Ablations (CPU)
```bash
python eval_locomo.py
python kill_test.py
```

### Layer 2: On-Disk SSD WAL Recovery & Sensitivity Sweep (CPU/SSD)
```bash
python run_supplementary_checks.py
```

### Layer 3: Live Serving & Multi-Tenant Scaling on Unmodified vLLM (GPU)
Start an unmodified vLLM server hosting `Qwen/Qwen2.5-7B-Instruct`:
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen2.5-7b \
  --enable-prefix-caching --dtype auto \
  --max-model-len 16384 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000
```

Run sequential serving, multi-tenant sweep, and strictly isolated 3-repeat verification:
```bash
# Sequential accuracy & TTFT verification (10,992 requests)
python bench_vllm_serving.py --max-convs 10 --base-url http://127.0.0.1:8000 --model qwen2.5-7b --resume

# Multi-tenant concurrent sweep across C in {1, 4, 8, 16, 32} (8,540 requests)
python bench_vllm_concurrent.py --base-url http://127.0.0.1:8000 --model qwen2.5-7b --concurrencies 1,4,8,16,32 --turns 20 --resume

# Strictly isolated 3-repeat verification + Sync-KG-Sorted+Offload strong baseline (10,560 requests)
python bench_vllm_concurrent.py --verify-r1 --repeats 3 --turns 20 --base-url http://127.0.0.1:8000 --model qwen2.5-7b --out vllm_r1_verify_results.json --resume
```
*(Add `--mock --smoke` to either serving script for a fast offline dry-run without a GPU.)*

### Regenerating Figures
```bash
python draw_architecture_figs.py
python draw_eval_figs.py
```

---

## Citation

```bibtex
@article{wang2026biosys,
  title   = {{BIOSYS}: A Hierarchical Memory Service for Long-Horizon {LLM} Agents},
  author  = {Wang, Pengfei and Guo, Xiaolei and Zhao, Mingyuan},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```
