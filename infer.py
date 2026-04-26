import argparse
import ast
import os

from tqdm.auto import tqdm

from src.hw5_common import (
    build_ailuminate_messages,
    build_fewshot_pool,
    build_gsm8k_messages,
    clear_memory,
    config_to_json,
    ensure_dirs,
    extract_numeric_answer,
    generate_response,
    huggingface_login_if_needed,
    load_ailuminate_prompts,
    load_config,
    load_jsonlines,
    load_model_with_adapter,
    load_tokenizer,
    seed_everything,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student-id", default=None)
    parser.add_argument("--test-n-shot", type=int, default=None)
    parser.add_argument("--gsm8k-public-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config()
    if args.student_id:
        config.student_id = args.student_id
    if args.test_n_shot is not None:
        config.test_n_shot = args.test_n_shot

    ensure_dirs(config)
    seed_everything(config.seed)
    huggingface_login_if_needed(config)
    print(config_to_json(config))

    train_data = load_jsonlines(config.train_file)
    fewshot_pool = build_fewshot_pool(train_data, config.test_n_shot, config.seed)
    gsm8k_public = load_jsonlines(config.gsm8k_public_file)
    gsm8k_private = [] if args.gsm8k_public_only else load_jsonlines(config.gsm8k_private_file)
    ailuminate = [] if args.gsm8k_public_only else load_ailuminate_prompts(config.ailuminate_file)

    tokenizer = load_tokenizer(config)
    model = load_model_with_adapter(config, args.checkpoint)
    outputs = []

    for qna in tqdm(gsm8k_public, desc="gsm8k public"):
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
        outputs.append(extract_numeric_answer(response))

    for qna in tqdm(gsm8k_private, desc="gsm8k private"):
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
        outputs.append(extract_numeric_answer(response))

    for prompt in tqdm(ailuminate, desc="ailuminate"):
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages=build_ailuminate_messages(prompt),
            max_new_tokens=config.max_new_tokens_ailuminate,
            do_sample=config.do_sample_ailuminate,
            temperature=config.ailuminate_temperature,
            top_p=config.ailuminate_top_p,
        )
        outputs.append(response)

    output_path = os.path.join(config.submission_dir, f"{config.student_id}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        print(outputs, file=f)
    print(output_path)

    del model
    clear_memory()


if __name__ == "__main__":
    main()
