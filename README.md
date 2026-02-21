# 396-pilot-project

Project member: Yibin Wang, Hongyuan Xu, Zheyu Fan

## Environment Setup (manual)

```bash
conda create -n 396-pilot-project python=3.12 -y
conda activate 396-pilot-project
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduce Report Results (single command; ready-to-run script)

Use `reproduce.sh` to automatically:
1. create/use a local venv,
2. install dependencies,
3. download model checkpoint + datasets from Hugging Face,
4. run inference,
5. run evaluation and save metrics JSON.

Checkpoint mode (re-generate predictions from adapter checkpoint):

```bash
bash reproduce.sh \
  --mode checkpoint \
  --run-name run_qwen_1 \
  --checkpoint 1869 \
  --data-repo YBW929/396-pilot-data \
  --model-repo YBW929/396-pilot-checkpoints
```

Frozen mode (evaluate uploaded prediction file directly for exact-number replay):
We recommend to use this commend to reproduce exact results in our report. 

```bash
bash reproduce.sh \
  --mode frozen \
  --run-name run_qwen_1 \
  --checkpoint 1869 \
  --data-repo YBW929/396-pilot-data \
  --artifact-repo YBW929/396-pilot-artifacts
```

Output files:
- submission txt: `runs/<run-name>/bfu3205_<checkpoint>.txt`
- metrics json: `runs/<run-name>/eval_result_<checkpoint>.json`

## Evaluate Existing Local Submission

You can evaluate a local prediction file directly:

```bash
python evaluate_metrics.py \
  --submission-path runs/run_qwen_1/bfu3205_1869.txt \
  --data-dir data \
  --output-json-path runs/run_qwen_1/eval_result_1869.json
```

Skip safety model if you only want GSM8K accuracy:

```bash
python evaluate_metrics.py \
  --submission-path runs/run_qwen_1/bfu3205_1869.txt \
  --data-dir data \
  --skip-safety
```

## Notebook Training

```bash
jupyter notebook Pilot_Project_Fine_tuning_Leads_to_Forgetting.ipynb
```

## Hugging Face Setup Guide

See detailed first-time instructions:

`HUGGINGFACE_SETUP.md`
