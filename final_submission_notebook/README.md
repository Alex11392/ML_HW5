# ML2026 HW5 Notebook Submission

This folder is the notebook-only submission version.

## Files

- `r14922132_hw5.ipynb`: the single notebook version of the training, checkpoint evaluation, and final inference workflow
- `README.md`: this file

## Selected Final Result

- Best checkpoint: `artifacts/checkpoints/strong_run/checkpoint-1800`
- Public GSM8K accuracy: `0.3712` (`49/132`)
- Final submission file used for JudgeBoi: `r14922132.txt`

## Notes

- The notebook keeps the workflow in a single file for submission.
- The final selected submission came from `strong_run/checkpoint-1800`.
- Testing used greedy decoding with `max_new_tokens_gsm8k = 384`.

## Running

Set your Hugging Face token before execution:

```bash
export HF_TOKEN='your_huggingface_token'
```

Then open and run:

```bash
r14922132_hw5.ipynb
```
