# Hugging Face Setup

This guide shows how to upload your datasets/checkpoints and verify download paths used by `reproduce.sh`.

## 1. Prerequisites

1. Create a Hugging Face account: `https://huggingface.co/join`
2. Create an access token: `https://huggingface.co/settings/tokens`
3. Install CLI tools:

```bash
pip install --upgrade huggingface_hub
```

4. Login once:

```bash
hf auth login
```

If you use private repos in automation, also export:

```bash
export HF_TOKEN=hf_xxx
```

## 2. Create Repositories

Replace `<username>` with your HF username.

```bash
hf repo create <username>/396-pilot-data --repo-type dataset
hf repo create <username>/396-pilot-checkpoints --repo-type model
hf repo create <username>/396-pilot-artifacts --repo-type model
```

You can add `--private` when creating repos.

## 3. Upload Dataset Files

Required files:
- `gsm8k_train.jsonl`
- `gsm8k_train_self-instruct.jsonl`
- `gsm8k_test_public.jsonl`
- `gsm8k_test_private.jsonl`
- `ailuminate_test.csv`

Upload:

```bash
hf upload <username>/396-pilot-data data/gsm8k_train.jsonl gsm8k_train.jsonl --repo-type dataset
hf upload <username>/396-pilot-data data/gsm8k_train_self-instruct.jsonl gsm8k_train_self-instruct.jsonl --repo-type dataset
hf upload <username>/396-pilot-data data/gsm8k_test_public.jsonl gsm8k_test_public.jsonl --repo-type dataset
hf upload <username>/396-pilot-data data/gsm8k_test_private.jsonl gsm8k_test_private.jsonl --repo-type dataset
hf upload <username>/396-pilot-data data/ailuminate_test.csv ailuminate_test.csv --repo-type dataset
```

## 4. Upload Checkpoint Folders

Upload full checkpoint folder contents, preserving run/checkpoint paths.

Example for your current best runs:

```bash
hf upload <username>/396-pilot-checkpoints runs/run_qwen_1 run_qwen_1 --repo-type model
hf upload <username>/396-pilot-checkpoints runs/run_3 run_3 --repo-type model
hf upload <username>/396-pilot-checkpoints runs/run_2 run_2 --repo-type model
```

The downloader expects paths like:
- `run_qwen_1/checkpoint-1869/adapter_config.json`
- `run_3/checkpoint-1869/adapter_config.json`
- `run_2/checkpoint-1246/adapter_config.json`

## 5. Upload Frozen Prediction + Metrics Artifacts (recommended)

This enables exact-number reproduction without regeneration noise.

```bash
hf upload <username>/396-pilot-artifacts runs/run_qwen_1/bfu3205_1869.txt run_qwen_1/bfu3205_1869.txt --repo-type model
hf upload <username>/396-pilot-artifacts runs/run_qwen_1/eval_result_1869.json run_qwen_1/eval_result_1869.json --repo-type model

hf upload <username>/396-pilot-artifacts runs/run_3/bfu3205_1869.txt run_3/bfu3205_1869.txt --repo-type model
hf upload <username>/396-pilot-artifacts runs/run_3/eval_result_1869.json run_3/eval_result_1869.json --repo-type model

hf upload <username>/396-pilot-artifacts runs/run_2/bfu3205_1246.txt run_2/bfu3205_1246.txt --repo-type model
hf upload <username>/396-pilot-artifacts runs/run_2/eval_result_1246.json run_2/eval_result_1246.json --repo-type model
```

## 6. Verify Download Paths Before Submission

Test one dataset file download:

```bash
hf download <username>/396-pilot-data gsm8k_test_public.jsonl --repo-type dataset
```

Test one checkpoint file download:

```bash
hf download <username>/396-pilot-checkpoints run_qwen_1/checkpoint-1869/adapter_config.json --repo-type model
```

Test one artifact file download:

```bash
hf download <username>/396-pilot-artifacts run_qwen_1/bfu3205_1869.txt --repo-type model
```

## 7. Run Reproduction

Checkpoint mode:

```bash
bash reproduce.sh \
  --mode checkpoint \
  --run-name run_qwen_1 \
  --checkpoint 1869 \
  --data-repo <username>/396-pilot-data \
  --model-repo <username>/396-pilot-checkpoints
```

Frozen mode:

```bash
bash reproduce.sh \
  --mode frozen \
  --run-name run_qwen_1 \
  --checkpoint 1869 \
  --data-repo <username>/396-pilot-data \
  --artifact-repo <username>/396-pilot-artifacts
```

## 8. Notes / Common Issues

- If you see `401 Unauthorized`, login again or export `HF_TOKEN`.
- If `run_inference.py` fails with CUDA/bitsandbytes errors, run on a GPU machine with CUDA-compatible PyTorch/bitsandbytes.
- For grader stability, keep both modes available:
  - `checkpoint` mode for full rerun from weights
  - `frozen` mode for exact reported numbers
