#!/usr/bin/env python3
"""One-stop reproducibility pipeline for download, inference, and evaluation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_DATA_FILES = [
    "gsm8k_train.jsonl",
    "gsm8k_train_self-instruct.jsonl",
    "gsm8k_test_public.jsonl",
    "gsm8k_test_private.jsonl",
    "ailuminate_test.csv",
]


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    project_root = script_root.parent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["checkpoint", "frozen"], default="checkpoint")
    parser.add_argument("--run-name", required=True, help="Example: run_qwen_1, run_3")
    parser.add_argument("--checkpoint", type=int, required=True)
    parser.add_argument("--student-id", default="bfu3205")

    parser.add_argument("--data-repo", required=True, help="HF dataset repo id")
    parser.add_argument("--model-repo", default=None, help="HF model repo id")
    parser.add_argument(
        "--artifact-repo",
        default=None,
        help="HF model/dataset repo containing frozen predictions (optional)",
    )
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--revision", default=None, help="Optional HF revision/commit")

    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--cache-dir", type=Path, default=project_root / ".hf_cache")
    parser.add_argument("--downloads-dir", type=Path, default=project_root / "downloads")

    parser.add_argument("--skip-safety", action="store_true")
    parser.add_argument("--safeguard-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--print-safeguard-outputs", action="store_true")
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def hf_snapshot(
    *,
    repo_id: str,
    repo_type: str,
    local_dir: Path,
    cache_dir: Path,
    token: str | None,
    revision: str | None,
    allow_patterns: list[str],
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        cache_dir=str(cache_dir),
        token=token,
        revision=revision,
        allow_patterns=allow_patterns,
    )
    return Path(snapshot_path)


def ensure_data(data_repo: str, args: argparse.Namespace) -> Path:
    print(f"Downloading dataset snapshot from: {data_repo}")
    local_dir = args.downloads_dir / "data_repo"
    data_root = hf_snapshot(
        repo_id=data_repo,
        repo_type="dataset",
        local_dir=local_dir,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        revision=args.revision,
        allow_patterns=REQUIRED_DATA_FILES,
    )

    missing = [name for name in REQUIRED_DATA_FILES if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Data repo missing required files: {missing}. Downloaded path: {data_root}"
        )
    return data_root


def ensure_checkpoint(model_repo: str, args: argparse.Namespace) -> Path:
    print(f"Downloading model checkpoint from: {model_repo}")
    local_dir = args.downloads_dir / "model_repo"
    run_prefix = f"{args.run_name}/**"
    checkpoint_prefix = f"{args.run_name}/checkpoint-{args.checkpoint}/**"
    model_root = hf_snapshot(
        repo_id=model_repo,
        repo_type="model",
        local_dir=local_dir,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        revision=args.revision,
        allow_patterns=[run_prefix, checkpoint_prefix],
    )
    checkpoint_dir = model_root / args.run_name / f"checkpoint-{args.checkpoint}"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    if not (checkpoint_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing adapter_config.json in {checkpoint_dir}")
    return checkpoint_dir


def ensure_frozen_submission(repo_id: str, args: argparse.Namespace, output_path: Path) -> Path:
    print(f"Downloading frozen predictions from: {repo_id}")
    local_dir = args.downloads_dir / "artifact_repo"
    target_name = f"{args.student_id}_{args.checkpoint}.txt"
    allow_patterns = [f"{args.run_name}/{target_name}"]
    root = hf_snapshot(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        cache_dir=args.cache_dir,
        token=args.hf_token,
        revision=args.revision,
        allow_patterns=allow_patterns,
    )
    source = root / args.run_name / target_name
    if not source.exists():
        raise FileNotFoundError(f"Frozen submission not found: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output_path)
    print(f"Copied frozen submission to {output_path}")
    return output_path


def infer_model_family(run_name: str) -> str:
    return "qwen" if run_name.startswith("run_qwen_") else "llama"


def main() -> None:
    args = parse_args()
    args.project_root = args.project_root.resolve()
    args.cache_dir = args.cache_dir.resolve()
    args.downloads_dir = args.downloads_dir.resolve()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.downloads_dir.mkdir(parents=True, exist_ok=True)

    data_dir = ensure_data(args.data_repo, args)

    run_dir = args.project_root / "runs" / args.run_name
    submission_path = run_dir / f"{args.student_id}_{args.checkpoint}.txt"
    eval_json_path = run_dir / f"eval_result_{args.checkpoint}.json"

    if args.mode == "checkpoint":
        if not args.model_repo:
            raise ValueError("--model-repo is required in checkpoint mode.")
        checkpoint_dir = ensure_checkpoint(args.model_repo, args)

        infer_cmd = [
            sys.executable,
            str(args.project_root / "scripts" / "run_inference.py"),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--data-dir",
            str(data_dir),
            "--output-path",
            str(submission_path),
        ]
        run_cmd(infer_cmd)
    else:
        artifact_repo = args.artifact_repo or args.model_repo
        if not artifact_repo:
            raise ValueError(
                "--artifact-repo (or --model-repo) is required in frozen mode."
            )
        ensure_frozen_submission(artifact_repo, args, submission_path)

    eval_cmd = [
        sys.executable,
        str(args.project_root / "evaluate_metrics.py"),
        "--submission-path",
        str(submission_path),
        "--data-dir",
        str(data_dir),
        "--output-json-path",
        str(eval_json_path),
        "--model-family",
        infer_model_family(args.run_name),
        "--safeguard-model",
        args.safeguard_model,
        "--torch-dtype",
        args.torch_dtype,
        "--device-map",
        args.device_map,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.skip_safety:
        eval_cmd.append("--skip-safety")
    if args.trust_remote_code:
        eval_cmd.append("--trust-remote-code")
    if args.print_safeguard_outputs:
        eval_cmd.append("--print-safeguard-outputs")

    run_cmd(eval_cmd)
    print(f"Reproduction finished. Metrics JSON: {eval_json_path}")


if __name__ == "__main__":
    main()
