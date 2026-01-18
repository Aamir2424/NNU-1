# ECG Digitization Pipeline - Phase 1

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![nnU-Net](https://img.shields.io/badge/nnU--Net-Felix%20Krones%20Fork-green.svg)](https://github.com/FelixKrones/nnUNet)

Digitize paper ECG scans into digital signals using nnU-Net ensemble prediction with 5 trained folds.

**Two Ways to Use:**

1. 🌐 **Web Interface** - Easy drag-and-drop interface (recommended)
2. 💻 **Terminal** - Command-line interface for batch processing

---

## 🚀 Quick Start for New Users (Web Interface)

### Step 1: Install Dependencies

```bash
# Navigate to the project folder
cd "path/to/kaggle copy 2"

# Run the setup script
bash setup.sh

# Install Flask for the web server
pip install flask flask-cors
```

### Step 2: Start the Web Server

```bash
# Set up environment variables and start the server
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"
python app.py
```

### Step 3: Use the Web Interface

1. Open your browser and go to: **http://localhost:8000**
2. Drag and drop your ECG image or click to browse
3. Click "Process ECG" button
4. Wait 20-60 seconds for processing
5. View and download your digitized ECG!

**That's it!** 🎉

---

## Overview

This pipeline converts scanned paper ECGs into digital signals through three stages:

1. **Model Inference** - Run nnU-Net binary segmentation with all 5 folds
2. **Ensemble Prediction** - Combine predictions for improved accuracy
3. **Vectorization & Calibration** - Convert masks to time-series with physical units

### Calibration Standards

- **Paper Speed**: 25 mm/s → 1 mm = 40 ms
- **Gain**: 10 mm/mV → 1 mm = 0.1 mV

---

## Quick Start

### 1. Installation

```bash
# Run the setup script
bash setup.sh
```

Or install manually:

```bash
pip install git+https://github.com/FelixKrones/nnUNet.git
pip install opencv-python numpy scipy matplotlib pandas pillow nibabel
```

### 2. Set Environment Variables

```bash
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"
```

### 3. Run Digitization

**Single Image:**

```bash
python ecg_digitize.py --input ./test_images/ecg_001.jpg
```

**Batch Processing:**

```bash
python ecg_digitize.py --input ./test_images/
```

**With Verbose Output:**

```bash
python ecg_digitize.py --input ./test_images/ecg.jpg --verbose
```

---

## Running in Terminal

### Scenario 1: Starting Fresh in a New Terminal

**Every time you open a new terminal, run these commands:**

```bash
# Navigate to the project directory
cd "/Users/aamiribrahim/Downloads/ECG STUFF/kaggle copy 2"

# Set up environment variables (required for nnU-Net)
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"

# Run the digitization
python ecg_digitize.py --input ./test_images/your_image.jpg
```

**Or as a single command (copy and paste this entire line):**

```bash
cd "/Users/aamiribrahim/Downloads/ECG STUFF/kaggle copy 2" && export nnUNet_results="$(pwd)/working/nnUNet_results" && export nnUNet_raw="$(pwd)/working/nnUNet_raw" && export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed" && python ecg_digitize.py --input ./test_images/your_image.jpg
```

### Scenario 2: Already in the Project Folder

**If you're already in the project directory:**

```bash
# Just run the digitization (if exports are still active)
python ecg_digitize.py --input ./test_images/your_image.jpg
```

**If you get an error, reset the environment variables:**

```bash
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"

python ecg_digitize.py --input ./test_images/your_image.jpg
```

### Common Use Cases

**Process a single image:**

```bash
python ecg_digitize.py --input ./test_images/2.png
```

**Process all images in folder:**

```bash
python ecg_digitize.py --input ./test_images/
```

**Run with verbose output for debugging:**

```bash
python ecg_digitize.py --input ./test_images/2.png --verbose
```

**Check available images:**

```bash
ls test_images/
```

**View results:**

```bash
ls working/results/
```

### Pro Tip: Easy Setup Alias

**Add this to your `~/.zshrc` to make life easier:**

```bash
echo 'alias ecg-setup="cd \"/Users/aamiribrahim/Downloads/ECG STUFF/kaggle copy 2\" && export nnUNet_results=\"\$(pwd)/working/nnUNet_results\" && export nnUNet_raw=\"\$(pwd)/working/nnUNet_raw\" && export nnUNet_preprocessed=\"\$(pwd)/working/nnUNet_preprocessed\""' >> ~/.zshrc
source ~/.zshrc
```

**Then in any new terminal, just type:**

```bash
ecg-setup
python ecg_digitize.py --input ./test_images/your_image.jpg
```

This saves you from typing the long setup commands every time!

---

## Directory Structure

```
./
├── setup.sh                 # Installation script
├── ecg_digitize.py          # Main digitization script
├── test_pipeline.sh         # Test script
├── README.md                # This file
├── test_images/             # Place test images here
└── working/
    ├── nnUNet_results/      # Trained model weights
    │   └── Dataset001_ECG/
    │       └── nnUNetTrainer__nnUNetPlans__2d/
    │           ├── fold_0/  # checkpoint_final.pth
    │           ├── fold_1/
    │           ├── fold_2/
    │           ├── fold_3/
    │           └── fold_4/
    ├── input/               # Temporary input directory
    ├── output-fold0/        # Fold 0 predictions
    ├── output-fold1/        # Fold 1 predictions
    ├── output-fold2/        # Fold 2 predictions
    ├── output-fold3/        # Fold 3 predictions
    ├── output-fold4/        # Fold 4 predictions
    ├── output-ensemble/     # Combined ensemble output
    └── results/             # Final digitized signals
```

---

## Usage Examples

### Basic Usage

```bash
# Process a single ECG image
python ecg_digitize.py --input path/to/ecg.jpg

# Process all images in a directory
python ecg_digitize.py --input ./ecg_images/

# Specify custom output directory
python ecg_digitize.py --input ecg.jpg --output ./my_results/
```

### Advanced Options

```bash
# Skip inference (use existing ensemble output)
python ecg_digitize.py --input ecg.jpg --skip_inference

# Disable plot generation
python ecg_digitize.py --input ecg.jpg --no_plot

# Full verbose mode
python ecg_digitize.py --input ecg.jpg --verbose
```

### Testing

```bash
# Run the full test suite on an image
bash test_pipeline.sh ./test_images/ecg_001.jpg
```

---

## Output Files

After processing, results are saved in `./working/results/<image_name>/`:

| File                  | Description                                                   |
| --------------------- | ------------------------------------------------------------- |
| `12_lead_signals.csv` | Time-series data for all 12 leads (Time_ms, I_mV, II_mV, ...) |
| `metadata.json`       | SNR values, calibration parameters, model info                |
| `waveforms.png`       | Visualization of all 12 leads with quality indicators         |
| `individual_leads/`   | Separate CSV files for each lead                              |

### CSV Format

```csv
Time_ms,I_mV,II_mV,III_mV,aVR_mV,aVL_mV,aVF_mV,V1_mV,V2_mV,V3_mV,V4_mV,V5_mV,V6_mV
0.0000,0.0523,-0.0124,0.0899,...
4.7059,0.0618,-0.0089,0.0912,...
```

### Metadata JSON

```json
{
  "input_file": "ecg_001.jpg",
  "timestamp": "2024-01-15T10:30:45.123456",
  "leads": {
    "I": {"snr_db": 18.5, "quality": "Good", "num_samples": 2500},
    "II": {"snr_db": 19.2, "quality": "Good", "num_samples": 2500},
    ...
  },
  "summary": {
    "total_leads": 12,
    "average_snr_db": 17.8,
    "overall_quality": "Good"
  },
  "calibration": {
    "S_x_pixels_per_mm": 8.5,
    "S_y_pixels_per_mm": 8.3,
    "paper_speed_mm_per_s": 25,
    "gain_mm_per_mv": 10
  }
}
```

---

## Signal Quality (SNR)

The pipeline calculates Signal-to-Noise Ratio (SNR) for each lead:

| SNR (dB) | Quality Rating | Interpretation             |
| -------- | -------------- | -------------------------- |
| > 20     | Excellent      | Clinical-grade signal      |
| 15-20    | Good           | Suitable for most analyses |
| 10-15    | Fair           | May need filtering         |
| 5-10     | Poor           | Review original image      |
| < 5      | Very Poor      | Likely digitization issue  |

---

## Troubleshooting

### "Only found X/5 folds"

Ensure all 5 trained model checkpoints exist:

```bash
ls -la ./working/nnUNet_results/Dataset001_ECG/nnUNetTrainer__nnUNetPlans__2d/fold_*/
```

Each fold should contain `checkpoint_final.pth` or `checkpoint_latest.pth`.

### "nnUNet not found"

Install the correct nnU-Net fork:

```bash
pip uninstall nnunet nnunetv2  # Remove any existing installations
pip install git+https://github.com/FelixKrones/nnUNet.git
```

### "nnUNetv2_predict command not found"

Set environment variables and ensure pip scripts are in PATH:

```bash
export nnUNet_results="$(pwd)/working/nnUNet_results"
export PATH="$HOME/.local/bin:$PATH"  # or wherever pip installs scripts
```

### Low SNR Warnings

If you see poor signal quality:

- Check original image quality (resolution, contrast)
- Ensure image is not skewed or rotated
- Remove any wrinkles, creases, or artifacts
- Verify the image contains actual ECG signal (not blank areas)

### Memory Issues

For large images or batch processing:

```bash
# Process one image at a time with cleanup
for img in ./test_images/*.jpg; do
    python ecg_digitize.py --input "$img"
done
```

---

## Pipeline Stages

### Stage 1: Model Inference

Runs nnU-Net prediction independently on all 5 folds:

```
[Fold 0/4] Running nnUNetv2_predict... ✓ Complete (15.2s)
[Fold 1/4] Running nnUNetv2_predict... ✓ Complete (14.8s)
[Fold 2/4] Running nnUNetv2_predict... ✓ Complete (15.0s)
[Fold 3/4] Running nnUNetv2_predict... ✓ Complete (14.9s)
[Fold 4/4] Running nnUNetv2_predict... ✓ Complete (15.1s)
```

### Stage 2: Ensemble

Combines predictions using probability averaging:

```
Combining predictions from 5 folds...
✓ Ensemble complete → output-ensemble/
```

### Stage 3: Vectorization & Calibration

Converts binary masks to physical signals:

```
Calibration Parameters:
  Grid spacing: S_x = 8.5 px/mm, S_y = 8.3 px/mm
  Time: 40 ms/mm (paper speed: 25 mm/s)
  Amplitude: 0.10 mV/mm (gain: 10 mm/mV)

Processing Lead Signals:
  ✓ I    | SNR: 18.5 dB | Quality: Good      | 2500 samples
  ✓ II   | SNR: 19.2 dB | Quality: Good      | 2500 samples
  ...
```

---

## Phase 2 (Coming Next)

After confirming the terminal version works correctly, Phase 2 will add a web-based frontend with:

- 🖼️ Drag-and-drop image upload
- 📊 Real-time processing progress
- 📈 Interactive waveform visualization
- 💾 Download options (CSV, JSON, PNG)
- 🔄 Batch processing queue

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ecg_digitization_nnunet,
  title = {ECG Digitization Pipeline using nnU-Net Ensemble},
  year = {2024},
  note = {Based on Felix Krones' nnU-Net fork}
}
```

---

## License

This project is for research purposes. The nnU-Net model and weights may have separate licensing requirements.
