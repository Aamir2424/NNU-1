# Project Name

## Overview
ECG digitization pipeline that converts scanned paper ECG images into digital signals using **nnU-Net (nnUNetv2)** segmentation and post-processing. Includes a simple **Flask web app** (served from `web/`) that lets you upload an image and download the digitized output.

## Dataset
- Input is **paper ECG scans/photos** (PNG/JPG/etc.).  
- If you’re using a Kaggle dataset, describe the source + split here (train/val/test) and any preprocessing (rotation, cropping, contrast normalization).

## Model
- **Architecture**: nnU-Net 2D segmentation (nnUNetv2) for tracing ECG waveforms from the image.
- **Inference**: ensemble across folds (e.g. folds 0–4) when available.
- **Important**: the nnU-Net generated folders are intentionally ignored by git:
  - `nnUNet_raw/`
  - `nnUNet_preprocessed/`
  - `nnUNet_results/`
- **Model weights download**: see `models/README.md` (download separately and place under `working/nnUNet_results/`).

## Installation

```bash
pip install -r requirements.txt
```

Then install nnU-Net (run separately):

```bash
pip install git+https://github.com/FelixKrones/nnUNet.git
```

## Running

```bash
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"
python app.py
```

## Results
- Outputs are written under `working/` (ignored by git), typically including:
  - digitized preview images
  - `signals.csv` / `12_lead_signals.csv` (depending on pipeline path)
  - `metadata.json`
- Add your qualitative examples (before/after) and metrics (SNR, per-lead quality) here.

## Future Work
- Replace placeholders in `frontend/` if/when you add a full React UI.
- Add reproducible training instructions for nnU-Net and a small “toy sample” dataset for quick CI tests.
