## Clinical Error GEPA Experiments

This repo is set up to run DSPy GEPA optimization experiments on the clinical error task using:

1. `Vrda/medgemma-27b-clinical-error-sft` (served via vLLM OpenAI-compatible API)
2. GPT-5.2-class OpenAI model (configurable model id)

### What Is Included

- Dataset fetch script: `main.py` (already pulls both source datasets)
- Unified dataset/split builder: `scripts/prepare_experiment_data.py`
- GEPA experiment runner (baseline vs optimized): `scripts/run_gepa_experiment.py`
- PBS job for MedGemma on GPU cluster: `pbs/gepa_medgemma.pbs`
- PBS job for GPT-5.2 experiment: `pbs/gepa_gpt52.pbs`

### 1) Prepare Data Splits

```bash
source .venv/bin/activate
python scripts/prepare_experiment_data.py
```

This writes:

- `data/experiments/unified_dataset.jsonl`
- `data/experiments/splits/train.jsonl`
- `data/experiments/splits/val.jsonl`
- `data/experiments/splits/test.jsonl`

### 2) Run GEPA For MedGemma (local/OpenAI-compatible endpoint)

Start your MedGemma + LoRA server (example):

```bash
python -m vllm.entrypoints.openai.api_server \
  --model google/medgemma-27b-it \
  --port 8000 \
  --enable-lora \
  --lora-modules "sft_adapter=Vrda/medgemma-27b-clinical-error-sft" \
  --max-lora-rank 64 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16
```

Then run experiment:

```bash
export MEDGEMMA_API_BASE="http://127.0.0.1:8000/v1"
export MEDGEMMA_MODEL_NAME="sft_adapter"
export MEDGEMMA_API_KEY="EMPTY"

# Optional but recommended as reflection LM for GEPA:
export OPENAI_API_KEY="..."
export OPENAI_GPT52_MODEL="gpt-5.2"

python scripts/prepare_experiment_data.py
python scripts/run_gepa_experiment.py --target medgemma --auto medium --run-name medgemma_local
```

### 3) Run GEPA For GPT-5.2

```bash
export OPENAI_API_KEY="..."
export OPENAI_GPT52_MODEL="gpt-5.2"

python scripts/prepare_experiment_data.py
python scripts/run_gepa_experiment.py --target gpt52 --auto medium --run-name gpt52_local
```

If your account uses a different model id, set `OPENAI_GPT52_MODEL` to that id.

### 4) PBS Jobs On HPC

Submit:

```bash
qsub pbs/gepa_medgemma.pbs
qsub pbs/gepa_gpt52.pbs
```

Before `qsub`:

- Ensure `.venv` and dependencies exist on the shared filesystem.
- Ensure `vllm` is installed in that environment for the MedGemma job.
- For MedGemma job, set `HF_TOKEN` with access to `google/medgemma-27b-it`.
- For GPT job, set `OPENAI_API_KEY`.
- Adjust PBS resource lines (`select`, `ngpus`, `mem`, `walltime`) for your cluster.

### Outputs

Each run writes to `artifacts/<run_name>/`:

- `run_summary.json` (baseline vs optimized metrics, deltas)
- `baseline_*_metrics.json`, `optimized_*_metrics.json`
- `baseline_test_predictions.jsonl`, `optimized_test_predictions.jsonl`
- `baseline_instruction.txt`, `optimized_instruction.txt`
- `baseline_program.json`, `optimized_program.json`
- `gepa_logs/` (optimizer traces/checkpoints)
