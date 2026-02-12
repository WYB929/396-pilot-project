# 396-pilot-project

Project member: Yibin Wang, Hongyuan Xu, Zheyu Fan

## Environment Setup (conda + pip)

```bash
conda create -n 396-pilot-project python=3.12 -y
conda activate 396-pilot-project
pip install --upgrade pip
pip install -r requirements.txt
```

## Fine-tuning

Run the notebook:

```bash
jupyter notebook Pilot_Project_Fine_tuning_Leads_to_Forgetting.ipynb
```

Essential variables to check before running:

- `run_idx`: selects output folder `runs/run_{run_idx}`.
- `checkpoint_idx`: used to build `ADAPTER_PATH = checkpoint-{checkpoint_idx}`.
- `OUTPUT_DIR`: where checkpoints/predictions are written.
- `ADAPTER_PATH`: which checkpoint is loaded for inference in notebook.
- `CHECKPOINT_STEP`: optional override to test a different checkpoint during inference (`None` means use `ADAPTER_PATH`).
- `STUDENT_ID`: output filename prefix for submission export.

## Evaluation

Run:

```bash
python evaluate_metrics.py
```

Essential variables in `evaluate_metrics.py`:

- `run_id`: selects run folder under `runs/run_{run_id}`.
- `checkpoint_idx`: selects prediction file like `bfu3205_{checkpoint_idx}.txt`.
- `print_safeguard_outputs`: set `True` to print all safeguard model raw outputs for sanity check, `False` to reduce logs.
- `skip_safety`: set `True` to skip safety evaluation (math-only check), `False` to run full evaluation.
