# Repository Guidelines

## Project Structure & Module Organization
Core code lives under `code/`. Use `code/train/` for training entry points (`train_method_acdc.py`, `train_method_mscmr.py`, `run.sh`), `code/test/` for evaluation scripts, `code/dataloader/` for dataset loaders, `code/networks/` for model definitions, and `code/utils/` for losses and training helpers. Validation logic is centralized in `code/val.py`. Dataset files are stored under `data/ACDC` and `data/MSCMR`; generated weights and logs are written to `checkpoints/`.

## Build, Test, and Development Commands
Create the environment and install dependencies:
```bash
conda create -n sdt python=3.10.18
conda activate sdt
pip install -r requirements.txt
```
Run both training jobs from the repo root:
```bash
cd code/train
bash run.sh
```
Run evaluation scripts individually:
```bash
cd code/test
python test_acdc.py
python test_mscmr.py
```
Edit paths and hyperparameters in `code/train/run.sh` before long runs; checkpoints are saved to `checkpoints/<DATASET>_DualTeacher/`.

## Coding Style & Naming Conventions
Follow the existing Python style: 4-space indentation, `snake_case` for functions and variables, `UPPER_CASE` only for true constants, and `CamelCase` for classes such as `ACDCDataSets`. Keep CLI arguments explicit through `argparse`. Match the current module naming pattern (`train_method_acdc.py`, `test_mscmr.py`) when adding dataset-specific code. No formatter is configured in-repo, so keep imports grouped, avoid unrelated reformatting, and preserve readable argument blocks.

## Testing Guidelines
This repository uses script-based evaluation rather than a unit test suite. Treat `code/test/test_acdc.py` and `code/test/test_mscmr.py` as regression checks for model and data pipeline changes. Name new evaluation scripts `test_<dataset>.py`. Before opening a PR, run the relevant evaluation script and confirm required checkpoint files exist, for example `checkpoints/ACDC_DualTeacher/unet_hl_best_model.pth`.

## Commit & Pull Request Guidelines
Recent history uses short messages like `update final code`, but contributors should use concise, imperative summaries such as `fix mscmr inference path` or `add acdc loader guard`. Keep each commit focused on one change. PRs should describe the affected dataset or training path, list the commands you ran, note any data/checkpoint prerequisites, and include metrics or screenshots when behavior or results change.

## Data & Configuration Notes
Do not commit raw dataset changes or large model artifacts unless explicitly required. Keep dataset roots under `data/` and prefer configurable paths through CLI flags or `run.sh` rather than hardcoding machine-specific locations.
