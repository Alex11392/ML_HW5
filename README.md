# ML HW5

This repository is organized for running ML2026 HW5 on a remote machine with a reproducible Python workflow.

## Layout

- `src/hw5_common.py`: shared config, prompting, model loading, generation, and utility functions
- `train.py`: fine-tune LoRA adapters and evaluate all saved checkpoints on GSM8K public
- `infer.py`: run final inference with one chosen checkpoint and export the JudgeBoi submission file
- `scripts/run_strong_baseline.sh`: bootstrap local `.venv`, install dependencies, and run training
- `requirements.txt`: Python dependencies

## Token

Set your Hugging Face token through the environment before running anything:

```bash
export HF_TOKEN='your_token_here'
```

Do not commit the token into the repository.

## Data

Place the homework data under `data/`:

- `gsm8k_train.jsonl` or `gsm8k_train_self-instruct.jsonl`
- `gsm8k_test_public.jsonl`
- `gsm8k_test_private.jsonl`
- `ailuminate_test.csv`

The repository does not store these files.

## Training

```bash
bash scripts/run_strong_baseline.sh
```

Useful overrides:

```bash
python train.py --epochs 4 --learning-rate 8e-5 --train-n-shot 1 --test-n-shot 4
python train.py --skip-train --eval-limit-public 200
```

Checkpoint metrics are saved under `artifacts/metrics/`.

## Final inference

```bash
python infer.py --checkpoint artifacts/checkpoints/strong_run/checkpoint-XXXX --student-id b12345678
```

The final submission file is written to `artifacts/submissions/`.
