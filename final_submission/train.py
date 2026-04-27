import argparse
import os

import pandas as pd
import torch
from tqdm.auto import tqdm
from trl import SFTConfig, SFTTrainer

from src.hw5_common import (
    build_fewshot_pool,
    build_gsm8k_messages,
    build_lora_config,
    clear_memory,
    config_to_json,
    ensure_dirs,
    extract_numeric_answer,
    format_training_dataset,
    generate_response,
    huggingface_login_if_needed,
    iter_checkpoints,
    load_base_model,
    load_config,
    load_jsonlines,
    load_model_with_adapter,
    load_tokenizer,
    save_json,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--train-n-shot", type=int, default=None)
    parser.add_argument("--test-n-shot", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--eval-limit-public", type=int, default=None)
    parser.add_argument("--checkpoints", nargs="+", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def evaluate_gsm8k_public(config, tokenizer, fewshot_pool, checkpoint_paths=None):
    gsm8k_public = load_jsonlines(config.gsm8k_public_file)
    rows = []
    checkpoints = checkpoint_paths or iter_checkpoints(config.output_dir)
    for ckpt in checkpoints:
        model = load_model_with_adapter(config, ckpt)
        total = len(gsm8k_public)
        if config.eval_limit_public is not None:
            total = min(total, config.eval_limit_public)
        correct = 0
        predictions = []
        for qna in tqdm(gsm8k_public[:total], desc=f"eval {os.path.basename(ckpt)}"):
            response = generate_response(
                model=model,
                tokenizer=tokenizer,
                messages=build_gsm8k_messages(
                    question=qna["question"],
                    answer=None,
                    mode="test",
                    fewshot_pool=fewshot_pool,
                    n_shot=config.test_n_shot,
                ),
                max_new_tokens=config.max_new_tokens_gsm8k,
                do_sample=config.do_sample_gsm8k,
                temperature=config.gsm8k_temperature,
                top_p=config.gsm8k_top_p,
            )
            pred = extract_numeric_answer(response)
            gold = extract_numeric_answer(qna["answer"])
            predictions.append(pred)
            correct += int(pred == gold)
        acc = correct / total
        rows.append(
            {
                "checkpoint": ckpt,
                "gsm8k_public_acc": acc,
                "num_eval_examples": total,
                "predictions": predictions,
            }
        )
        del model
        clear_memory()
    return rows


def main():
    args = parse_args()
    config = load_config()
    if args.train_file:
        config.train_file = args.train_file
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs is not None:
        config.num_train_epochs = args.epochs
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.train_n_shot is not None:
        config.train_n_shot = args.train_n_shot
    if args.test_n_shot is not None:
        config.test_n_shot = args.test_n_shot
    if args.save_steps is not None:
        config.save_steps = args.save_steps
    if args.eval_limit_public is not None:
        config.eval_limit_public = args.eval_limit_public

    ensure_dirs(config)
    seed_everything(config.seed)
    huggingface_login_if_needed(config)

    print(config_to_json(config))

    train_data = load_jsonlines(config.train_file)
    fewshot_pool = build_fewshot_pool(
        train_data,
        max(config.train_n_shot, config.test_n_shot),
        config.seed,
    )
    tokenizer = load_tokenizer(config)

    if not args.skip_train:
        train_dataset, max_token_len = format_training_dataset(
            config=config,
            tokenizer=tokenizer,
            train_data=train_data,
            fewshot_pool=fewshot_pool,
        )
        model = load_base_model(config)
        training_args = SFTConfig(
            seed=config.seed,
            data_seed=config.seed,
            output_dir=config.output_dir,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            optim="paged_adamw_32bit",
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            lr_scheduler_type="cosine",
            warmup_ratio=config.warmup_ratio,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            bf16=torch.cuda.is_available(),
            max_length=max_token_len,
            dataset_text_field="text",
            report_to="none",
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            peft_config=build_lora_config(config),
            processing_class=tokenizer,
            args=training_args,
        )
        trainer.train(resume_from_checkpoint=False)
        del trainer
        del model
        clear_memory()

    if not args.skip_eval:
        checkpoint_paths = None
        if args.checkpoints:
            checkpoint_paths = []
            for ckpt in args.checkpoints:
                if os.path.isdir(ckpt):
                    checkpoint_paths.append(ckpt)
                else:
                    checkpoint_paths.append(os.path.join(config.output_dir, ckpt))
        rows = evaluate_gsm8k_public(config, tokenizer, fewshot_pool, checkpoint_paths)
        metrics_path = os.path.join(config.metrics_dir, "checkpoint_metrics.json")
        save_json(metrics_path, rows)
        df = pd.DataFrame(
            [{k: v for k, v in row.items() if k != "predictions"} for row in rows]
        ).sort_values("gsm8k_public_acc", ascending=False)
        df.to_csv(os.path.join(config.metrics_dir, "checkpoint_metrics.csv"), index=False)
        print(df.head())
        print(f"saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
