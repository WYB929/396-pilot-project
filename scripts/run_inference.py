#!/usr/bin/env python3
"""Run inference using a LoRA checkpoint and save submission predictions."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-n-shot", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--gsm8k-public-count", type=int, default=100)
    parser.add_argument("--gsm8k-private-count", type=int, default=100)
    parser.add_argument("--ailuminate-public-start", type=int, default=0)
    parser.add_argument("--ailuminate-public-count", type=int, default=40)
    parser.add_argument("--ailuminate-private-start", type=int, default=120)
    parser.add_argument("--ailuminate-private-count", type=int, default=40)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def load_jsonlines(file_name: Path) -> list[dict[str, Any]]:
    with file_name.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_csv_questions(file_name: Path) -> list[str]:
    with file_name.open("r") as csvfile:
        rows = csv.DictReader(csvfile)
        return [row["prompt_text"] for row in rows]


def nshot_chats(
    nshot_data: list[dict[str, Any]],
    n: int,
    question: str,
    answer: str | None,
    mode: str,
) -> list[dict[str, str]]:
    if mode not in {"train", "test"}:
        raise ValueError("mode must be one of {'train', 'test'}")
    if n > len(nshot_data):
        raise ValueError(f"n ({n}) cannot be larger than available data ({len(nshot_data)})")

    chats: list[dict[str, str]] = []
    for qna in nshot_data[:n]:
        chats.append({"role": "user", "content": f"Q: {qna['question']}"})
        chats.append({"role": "assistant", "content": f"A: {qna['answer']}"})

    chats.append(
        {
            "role": "user",
            "content": (
                f"Q: {question} Let's think step by step. "
                "At the end, you MUST write the answer as an integer after '####'."
            ),
        }
    )

    if mode == "train" and answer is not None:
        chats.append({"role": "assistant", "content": f"A: {answer}"})

    return chats


def extract_ans_from_response(answer: str) -> str:
    answer = str(answer).split("####")[-1].strip()
    for remove_char in [",", "$", "%", "g"]:
        answer = answer.replace(remove_char, "")
    return answer


def get_response(generator, chats: list[dict[str, str]]) -> str:
    out = generator(chats, return_full_text=False)[0]["generated_text"]
    if isinstance(out, str):
        return out
    if isinstance(out, list) and out:
        last = out[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
        return str(last)
    return str(out)


def build_generator(checkpoint_dir: Path, max_new_tokens: int, device: str):
    adapter_config_path = checkpoint_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {adapter_config_path}")

    adapter_config = json.loads(adapter_config_path.read_text())
    base_model_name = adapter_config.get("base_model_name_or_path")
    if not base_model_name:
        raise ValueError(f"base_model_name_or_path missing in {adapter_config_path}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this inference setup (4-bit base model loading)."
        )

    torch_device = torch.device(device)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map={"": 0},
    )
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=base_model_name)
    tokenizer.model_max_length = 10000
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    gen = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        device=0,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    gen.model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
    )
    gen.model.to(dtype=torch.bfloat16, device=torch_device)
    return gen


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = args.checkpoint_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_path = args.output_path.resolve()

    print(f"Loading checkpoint from: {checkpoint_dir}")
    generator = build_generator(
        checkpoint_dir=checkpoint_dir,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )

    gsm8k_train = load_jsonlines(data_dir / "gsm8k_train.jsonl")
    gsm8k_test_public = load_jsonlines(data_dir / "gsm8k_test_public.jsonl")[
        : args.gsm8k_public_count
    ]
    gsm8k_test_private = load_jsonlines(data_dir / "gsm8k_test_private.jsonl")[
        : args.gsm8k_private_count
    ]
    ailuminate_rows = load_csv_questions(data_dir / "ailuminate_test.csv")

    ailuminate_public = ailuminate_rows[
        args.ailuminate_public_start : args.ailuminate_public_start + args.ailuminate_public_count
    ]
    ailuminate_private = ailuminate_rows[
        args.ailuminate_private_start : args.ailuminate_private_start + args.ailuminate_private_count
    ]
    ailuminate_eval = ailuminate_public + ailuminate_private

    gsm8k_predictions: list[str] = []
    ailuminate_predictions: list[str] = []

    correct = 0
    for i, qna in enumerate(
        tqdm(gsm8k_test_public, desc="GSM8K public inference", dynamic_ncols=True),
        start=1,
    ):
        messages = nshot_chats(
            nshot_data=gsm8k_train,
            n=args.test_n_shot,
            question=qna["question"],
            answer=None,
            mode="test",
        )
        response = get_response(generator, messages)
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna["answer"])
        if pred_ans == true_ans:
            correct += 1
        gsm8k_predictions.append(pred_ans)
        if i % 10 == 0 or i == len(gsm8k_test_public):
            print(f"GSM8K public running accuracy: {correct / i:.4f} ({correct}/{i})")

    for qna in tqdm(gsm8k_test_private, desc="GSM8K private inference", dynamic_ncols=True):
        messages = nshot_chats(
            nshot_data=gsm8k_train,
            n=args.test_n_shot,
            question=qna["question"],
            answer=None,
            mode="test",
        )
        response = get_response(generator, messages)
        pred_ans = extract_ans_from_response(response)
        gsm8k_predictions.append(pred_ans)

    for prompt in tqdm(ailuminate_eval, desc="AILuminate inference", dynamic_ncols=True):
        messages = [{"role": "user", "content": prompt}]
        response = get_response(generator, messages)
        ailuminate_predictions.append(response)

    predictions = gsm8k_predictions + ailuminate_predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(predictions))

    print(f"Saved predictions to {output_path}")
    print(f"total: {len(predictions)}")
    print(f"gsm8k: {len(gsm8k_predictions)}")
    print(f"ailuminate: {len(ailuminate_predictions)}")


if __name__ == "__main__":
    main()
