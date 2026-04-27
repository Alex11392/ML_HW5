# ML2026 HW5 Final Submission

This folder contains the code and output file for the final version selected for submission.

## Selected Version

- Best checkpoint: `artifacts/checkpoints/strong_run/checkpoint-1800`
- Public GSM8K accuracy: `0.3712` (`49/132`)
- Final submission file: `r14922132.txt`

## Files

- `train.py`: LoRA training and checkpoint evaluation entrypoint
- `infer.py`: final inference entrypoint
- `src/hw5_common.py`: shared config, prompting, model loading, and generation utilities
- `scripts/download_data.py`: helper script to download homework data
- `scripts/run_strong_baseline.sh`: example training script
- `requirements.txt`: Python dependencies
- `r14922132.txt`: final JudgeBoi submission file

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN='your_huggingface_token'
```

## Data

Place these files under `data/`:

- `gsm8k_train_self-instruct.jsonl`
- `gsm8k_test_public.jsonl`
- `gsm8k_test_private.jsonl`
- `ailuminate_test.csv`

You can use:

```bash
python scripts/download_data.py
```

## Reproduce Training

The selected submission came from the `strong_run` training path using:

```bash
bash scripts/run_strong_baseline.sh
```

or equivalently:

```bash
python train.py
```

The chosen checkpoint was:

```bash
artifacts/checkpoints/strong_run/checkpoint-1800
```

## Reproduce Final Inference

```bash
python infer.py \
  --checkpoint artifacts/checkpoints/strong_run/checkpoint-1800 \
  --student-id r14922132
```

This writes:

```bash
artifacts/submissions/r14922132.txt
```

## Submission

- Upload `r14922132.txt` to JudgeBoi
- Upload the code files in this folder to NTU COOL according to the homework packaging rule
