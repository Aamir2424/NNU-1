#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
ECG Digitization Pipeline - Phase 1 (Terminal Version)
═══════════════════════════════════════════════════════════════════════════════════

Digitize paper ECG scans into 12-lead time-series signals using nnU-Net ensemble.

Pipeline Flow:
  Input Image → nnU-Net (all 5 folds ensemble) → Digitized ECG Signal

Output:
  - Digitized trace image (clean connected signal line)
  - Time-series data (CSV)
  - Waveform visualization

Usage:
  python ecg_digitize.py --input path/to/ecg_image.jpg
  python ecg_digitize.py --input ./folder_with_ecgs/ --verbose

Author: ECG Digitization Research Team
Model: nnU-Net (Felix Krones fork) - 5-Fold Ensemble
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
import glob
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
import json

# PyTorch for vectorization
import torch
import torch.nn.functional as F

# Optional imports with fallbacks
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plotting disabled.")

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False
    print("Warning: nibabel not available. Will use PNG masks only.")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Default paths (relative to script location)
SCRIPT_DIR = Path(__file__).parent.absolute()
DEFAULT_WORKING_DIR = SCRIPT_DIR / "working"
DEFAULT_NNUNET_RESULTS = DEFAULT_WORKING_DIR / "nnUNet_results"

# nnU-Net model configuration
DATASET_NAME = "Dataset001_ECG"
DATASET_ID = 1
TRAINER = "nnUNetTrainer"
PLANS = "nnUNetPlans"
CONFIG = "2d"
NUM_FOLDS = 5

# ECG calibration constants (standard ECG paper)
# Paper speed: 25 mm/s → 1 mm = 40 ms
# Gain: 10 mm/mV → 1 mm = 0.1 mV
PAPER_SPEED_MM_PER_S = 25
GAIN_MM_PER_MV = 10
TIME_PER_MM_MS = 1000 / PAPER_SPEED_MM_PER_S  # 40 ms/mm
AMPLITUDE_PER_MM_MV = 1 / GAIN_MM_PER_MV       # 0.1 mV/mm

# ECG signal parameters for vectorization
FREQUENCY = 500  # Hz - standard ECG sampling frequency
LONG_SIGNAL_LENGTH_SEC = 10.0   # Full 12-lead ECG strip length
SHORT_SIGNAL_LENGTH_SEC = 2.5   # Individual lead length

# Default y_shift_ratio for each lead (vertical position calibration)
# These values may need adjustment based on your specific ECG format
Y_SHIFT_RATIO = {
    "full": 0.5,
    "I": 0.5, "II": 0.5, "III": 0.5,
    "aVR": 0.5, "aVL": 0.5, "aVF": 0.5,
    "V1": 0.5, "V2": 0.5, "V3": 0.5, "V4": 0.5, "V5": 0.5, "V6": 0.5
}

# 12-lead ECG names
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_setup(working_dir: Path, verbose: bool = False) -> dict:
    """
    Verify all required components are present.
    
    Returns:
        dict with model_path, folds_present, dataset_id
    """
    print("Running pre-flight checks...")
    
    working_dir = Path(working_dir)
    model_info = {}
    
    # Check for nnU-Net results directory
    nnunet_results = working_dir / "nnUNet_results"
    if not nnunet_results.exists():
        raise FileNotFoundError(
            f"Error: nnUNet_results not found at {nnunet_results}\n"
            "Please ensure trained models are in ./working/nnUNet_results/"
        )
    print(f"  ✓ Found nnUNet_results directory")
    
    # Find the model path
    model_path = nnunet_results / DATASET_NAME / f"{TRAINER}__{PLANS}__{CONFIG}"
    if not model_path.exists():
        # Try to find any dataset
        datasets = list(nnunet_results.glob("Dataset*"))
        if datasets:
            dataset_dir = datasets[0]
            trainers = list(dataset_dir.glob("*"))
            if trainers:
                model_path = trainers[0]
                print(f"  ℹ Using model at: {model_path.relative_to(working_dir)}")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Error: Model not found at expected path:\n"
            f"  {model_path}\n"
            "Please check your nnUNet_results folder structure."
        )
    
    model_info['model_path'] = model_path
    
    # Check for all 5 folds
    folds_present = []
    folds_missing = []
    
    for i in range(NUM_FOLDS):
        fold_path = model_path / f"fold_{i}"
        checkpoint_final = fold_path / "checkpoint_final.pth"
        checkpoint_latest = fold_path / "checkpoint_latest.pth"
        
        if fold_path.exists() and (checkpoint_final.exists() or checkpoint_latest.exists()):
            folds_present.append(i)
            checkpoint_name = "checkpoint_final.pth" if checkpoint_final.exists() else "checkpoint_latest.pth"
            print(f"  ✓ Found fold_{i} ({checkpoint_name})")
        else:
            folds_missing.append(i)
            print(f"  ✗ Missing fold_{i}")
    
    model_info['folds_present'] = folds_present
    
    if len(folds_present) != NUM_FOLDS:
        raise FileNotFoundError(
            f"Error: Only found {len(folds_present)}/{NUM_FOLDS} folds.\n"
            f"Missing folds: {folds_missing}\n"
            "All 5 folds are required for ensemble prediction!"
        )
    
    # Check dataset.json for dataset ID
    dataset_json = model_path / "dataset.json"
    if dataset_json.exists():
        with open(dataset_json) as f:
            dataset_info = json.load(f)
        if verbose:
            print(f"  ℹ Dataset: {dataset_info.get('numTraining', '?')} training samples")
    
    # Extract dataset ID from folder name
    dataset_name = model_path.parent.name  # e.g., "Dataset001_ECG"
    try:
        dataset_id = int(dataset_name.split('_')[0].replace('Dataset', ''))
    except:
        dataset_id = DATASET_ID
    
    model_info['dataset_id'] = dataset_id
    model_info['dataset_name'] = dataset_name
    
    # Check nnUNet installation
    try:
        import nnunetv2
        print(f"  ✓ nnU-Net installed (nnunetv2)")
    except ImportError:
        raise ImportError(
            "Error: nnU-Net not found!\n"
            "Please run: pip install git+https://github.com/FelixKrones/nnUNet.git"
        )
    
    # Check nnUNetv2_predict command
    result = subprocess.run(['which', 'nnUNetv2_predict'], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  ✓ nnUNetv2_predict command available")
    else:
        print(f"  ⚠ nnUNetv2_predict not in PATH - will try anyway")
    
    print("✓ All pre-flight checks passed!\n")
    return model_info


def setup_directories(working_dir: Path) -> dict:
    """Create necessary directories for pipeline."""
    working_dir = Path(working_dir)
    
    dirs = {
        'input': working_dir / 'input',
        'output': working_dir / 'output',  # Single output directory
        'results': working_dir / 'results',
    }
    
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure ready")
    return dirs


def setup_environment(working_dir: Path) -> dict:
    """Set up nnU-Net environment variables."""
    working_dir = Path(working_dir).absolute()
    
    env_vars = {
        'nnUNet_results': str(working_dir / 'nnUNet_results'),
        'nnUNet_raw': str(working_dir / 'nnUNet_raw'),
        'nnUNet_preprocessed': str(working_dir / 'nnUNet_preprocessed'),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
    
    return env_vars


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: MODEL INFERENCE (5-FOLD ENSEMBLE)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_input_image(input_image: Path, input_dir: Path) -> Path:
    """
    Prepare input image for nnU-Net inference.
    nnU-Net expects images named like: case_0000.png (with _0000 suffix for channel)
    """
    input_image = Path(input_image)
    input_dir = Path(input_dir)
    
    # Clear input directory
    for f in input_dir.glob('*'):
        f.unlink()
    
    # Generate nnU-Net compatible filename
    # Format: {case_id}_0000.{ext}
    case_id = input_image.stem
    # Remove any existing _0000 suffix
    if case_id.endswith('_0000'):
        case_id = case_id[:-5]
    
    ext = input_image.suffix.lower()
    if ext not in ['.png', '.jpg', '.jpeg']:
        ext = '.png'
    
    # nnU-Net expects _0000 suffix for single-channel images
    output_filename = f"{case_id}_0000{ext}"
    output_path = input_dir / output_filename
    
    # Copy and potentially convert image
    img = cv2.imread(str(input_image))
    if img is None:
        raise ValueError(f"Could not read image: {input_image}")
    
    # Convert to grayscale if needed (for single-channel model)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    
    cv2.imwrite(str(output_path), img_gray)
    
    return output_path


def run_ensemble_inference(
    input_image: Path,
    working_dir: Path,
    dataset_id: int,
    config: str,
    verbose: bool = False
) -> Path:
    """
    Run nnU-Net inference using ALL 5 folds together (ensemble).
    
    Uses -f all flag to combine all folds in a single pass.
    This is more efficient and produces a single clean output.
    """
    print("\n" + "─" * 63)
    print("STAGE 1: Running nnU-Net Ensemble Inference (5 Folds)")
    print("─" * 63)
    
    working_dir = Path(working_dir)
    input_dir = working_dir / 'input'
    output_dir = working_dir / 'output'
    
    # Prepare input image
    print(f"\nPreparing input image...")
    prepared_image = prepare_input_image(input_image, input_dir)
    print(f"  ✓ Image prepared: {prepared_image.name}")
    
    # Clear output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob('*'):
        if f.is_file():
            f.unlink()
    
    # Build nnUNetv2_predict command with ALL folds
    # Specify all fold numbers to use them as ensemble
    cmd = [
        'nnUNetv2_predict',
        '-i', str(input_dir),
        '-o', str(output_dir),
        '-d', str(dataset_id),
        '-c', config,
        '-f', '0', '1', '2', '3', '4',  # Use all 5 folds for ensemble
        '-device', 'cpu',  # Force CPU to avoid MPS tensor issues on macOS
    ]
    
    if verbose:
        print(f"\n  Command: {' '.join(cmd)}")
    
    print(f"\n  Running nnU-Net with 5-fold ensemble...")
    start_time = time.time()
    
    # Run inference
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=os.environ.copy()
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(
            f"Inference failed!\n"
            f"Command: {' '.join(cmd)}\n"
            f"Error: {error_msg}"
        )
    
    if verbose and result.stdout:
        # Print only summary lines
        for line in result.stdout.split('\n'):
            if any(x in line.lower() for x in ['predicting', 'done', 'time']):
                print(f"    {line}")
    
    # Check output
    outputs = list(output_dir.glob('*.png')) + list(output_dir.glob('*.nii.gz'))
    
    if outputs:
        print(f"\n✓ Ensemble inference complete ({elapsed:.1f}s)")
        print(f"  Output: {outputs[0].name}")
    else:
        print(f"  ⚠ Complete ({elapsed:.1f}s) but no output files found")
    
    return output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: VECTORIZATION & CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_mask(mask_path: Path, convert_to_255: bool = True) -> np.ndarray:
    """
    Load segmentation mask from various formats.
    
    nnU-Net outputs masks with values 0 (background) and 1 (signal).
    This function handles both 0/1 and 0/255 formats.
    
    Args:
        mask_path: Path to the mask file
        convert_to_255: If True, convert signal pixels from 1 to 255
        
    Returns:
        Binary mask with values 0 and 255 (or 0 and 1 if convert_to_255=False)
    """
    mask_path = Path(mask_path)
    
    if mask_path.suffix == '.gz' or mask_path.suffix == '.nii':
        # NIfTI format
        if HAS_NIBABEL:
            nii = nib.load(str(mask_path))
            mask = np.array(nii.get_fdata())
            # Squeeze extra dimensions
            mask = np.squeeze(mask)
            # nnU-Net outputs 0 and 1
            binary_mask = (mask > 0).astype(np.uint8)
        else:
            raise ImportError("nibabel required to read .nii.gz files")
    else:
        # PNG/JPG format
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        
        # Handle both 0/1 (nnU-Net raw) and 0/255 (standard) formats
        # If max value is 1, it's nnU-Net format; if max is 255, it's standard
        if mask.max() <= 1:
            # nnU-Net format: 0 = background, 1 = signal
            binary_mask = (mask > 0).astype(np.uint8)
        else:
            # Standard format: 0 = background, 255 = signal
            binary_mask = (mask > 127).astype(np.uint8)
    
    # Convert 1 to 255 for visualization/processing
    if convert_to_255:
        return binary_mask * 255
    else:
        return binary_mask


def vectorize_mask(mask: np.ndarray) -> list:
    """
    Convert binary mask to 1D signal using median-trace algorithm.
    
    For each column (x), find the median y-coordinate of all signal pixels.
    This produces a clean single-valued trace even with thick or noisy lines.
    
    Args:
        mask: Binary mask (H, W) where 1 = signal, 0 = background
        
    Returns:
        List of (x, y) coordinates representing the signal trace
    """
    if mask is None or mask.size == 0:
        return []
    
    signal_coords = []
    
    for x in range(mask.shape[1]):
        # Find all y-coordinates where signal is present
        y_pixels = np.where(mask[:, x] > 0)[0]
        
        if len(y_pixels) > 0:
            # Use median for robustness to thick lines and noise
            median_y = np.median(y_pixels)
            signal_coords.append((x, median_y))
    
    return signal_coords


def vectorise_signal(
    image_rotated: torch.Tensor,
    mask: torch.Tensor,
    signal_cropped: torch.Tensor,
    sec_per_pixel: float,
    mV_per_pixel: float,
    y_shift_ratio: dict,
    lead: str
) -> torch.Tensor:
    """
    Vectorise the signal from mask using research paper algorithm.
    
    This function aligns and scales a signal based on a mask's non-zero regions 
    and a vertical shift ratio. It computes the mean vertical position of non-zero 
    elements in the mask, adjusts the signal's vertical position, and scales the 
    result into physical units (millivolts).
    
    Args:
        image_rotated: Original image tensor (for dimensions)
        mask: Binary mask tensor [1, H, W]
        signal_cropped: Cropped signal y-coordinates tensor
        sec_per_pixel: Seconds per pixel (horizontal scaling)
        mV_per_pixel: Millivolts per pixel (vertical scaling)
        y_shift_ratio: Dict with shift ratios per lead
        lead: Lead name (I, II, III, aVR, aVL, aVF, V1-V6)
        
    Returns:
        Resampled signal tensor at standard frequency
    """
    # Get scaling info
    total_seconds_from_mask = round(torch.tensor(sec_per_pixel).item() * mask.shape[2], 1)
    
    if total_seconds_from_mask > (LONG_SIGNAL_LENGTH_SEC / 2):
        total_seconds = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio.get("full", 0.5)
    else:
        total_seconds = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio.get(lead, 0.5)
    
    values_needed = int(total_seconds * FREQUENCY)

    # Scale y
    # Compute the mean vertical position of non-zero elements in the mask
    non_zero_mean = torch.tensor(
        [
            torch.mean(torch.nonzero(mask[0, :, i]).type(torch.float32))
            if torch.any(mask[0, :, i] > 0) else torch.tensor(0.0)
            for i in range(mask.shape[2])
        ]
    )
    
    # Adjust signal's vertical position using y_shift_ratio
    signal_cropped_shifted = (1 - y_shift_ratio_) * image_rotated.shape[1] - signal_cropped
    
    # Scale into physical units (millivolts)
    predicted_signal = (signal_cropped_shifted - non_zero_mean) * mV_per_pixel

    # Scale x - resample to standard frequency
    n = predicted_signal.shape[0]
    data_reshaped = predicted_signal.view(1, 1, n)
    resampled_data = F.interpolate(
        data_reshaped, size=values_needed, mode="linear", align_corners=False
    )
    predicted_signal_sampled = resampled_data.view(-1)

    return predicted_signal_sampled


def vectorise_from_mask(
    mask: np.ndarray,
    original_image: np.ndarray = None,
    sec_per_pixel: float = None,
    mV_per_pixel: float = None,
    lead: str = "I",
    y_shift_ratio: dict = None
) -> tuple:
    """
    Convert binary mask to time-series signal using research paper vectorization.
    
    This is a wrapper that handles numpy arrays and provides defaults.
    
    Args:
        mask: Binary mask (H, W) numpy array
        original_image: Original image for dimensions (optional)
        sec_per_pixel: Seconds per pixel (auto-calculated if None)
        mV_per_pixel: mV per pixel (auto-calculated if None)
        lead: Lead name
        y_shift_ratio: Vertical shift ratios (uses defaults if None)
        
    Returns:
        (time_ms, amplitude_mv) as numpy arrays
    """
    if mask is None or mask.size == 0:
        return np.array([]), np.array([])
    
    # Use defaults if not provided
    if y_shift_ratio is None:
        y_shift_ratio = Y_SHIFT_RATIO
    
    # Calculate scaling factors if not provided
    # Standard ECG: 25 mm/s paper speed, assume ~8.5 pixels/mm
    if sec_per_pixel is None:
        pixels_per_mm = 8.5  # typical value
        mm_per_sec = PAPER_SPEED_MM_PER_S  # 25 mm/s
        sec_per_pixel = 1.0 / (pixels_per_mm * mm_per_sec)
    
    if mV_per_pixel is None:
        pixels_per_mm = 8.3  # typical value
        mm_per_mV = GAIN_MM_PER_MV  # 10 mm/mV
        mV_per_pixel = 1.0 / (pixels_per_mm * mm_per_mV)
    
    # Get signal trace using median algorithm - get y for each x
    # Create full-width signal array (for all columns in mask)
    signal_y_full = []
    for x in range(mask.shape[1]):
        y_pixels = np.where(mask[:, x] > 0)[0]
        if len(y_pixels) > 0:
            signal_y_full.append(np.median(y_pixels))
        else:
            # Interpolate or use NaN for missing columns
            signal_y_full.append(np.nan)
    
    signal_y_full = np.array(signal_y_full)
    
    # Interpolate NaN values
    valid_mask = ~np.isnan(signal_y_full)
    if not np.any(valid_mask):
        return np.array([]), np.array([])
    
    # Linear interpolation for missing values
    x_valid = np.where(valid_mask)[0]
    y_valid = signal_y_full[valid_mask]
    signal_y_interp = np.interp(np.arange(len(signal_y_full)), x_valid, y_valid)
    
    # Convert to tensor format for vectorise_signal
    signal_y = torch.tensor(signal_y_interp, dtype=torch.float32)
    
    # Prepare mask tensor [1, H, W]
    mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    
    # Create dummy image tensor for dimensions
    if original_image is not None:
        img_tensor = torch.tensor(original_image, dtype=torch.float32)
    else:
        img_tensor = torch.zeros((mask.shape[0], mask.shape[1]), dtype=torch.float32)
    
    # Run vectorization
    try:
        predicted_signal = vectorise_signal(
            image_rotated=img_tensor,
            mask=mask_tensor,
            signal_cropped=signal_y,
            sec_per_pixel=sec_per_pixel,
            mV_per_pixel=mV_per_pixel,
            y_shift_ratio=y_shift_ratio,
            lead=lead
        )
        
        # Convert to numpy
        amplitude_mv = predicted_signal.numpy()
        
        # Generate time array
        n_samples = len(amplitude_mv)
        total_seconds = n_samples / FREQUENCY
        time_ms = np.linspace(0, total_seconds * 1000, n_samples)
        
        return time_ms, amplitude_mv
        
    except Exception as e:
        print(f"    Warning: Advanced vectorization failed ({e}), using simple method")
        # Fallback to simple calibration
        return None, None


def detect_grid_spacing(image: np.ndarray = None) -> tuple:
    """
    Detect 1mm grid spacing in ECG image.
    
    Standard ECG paper has 1mm small squares and 5mm large squares.
    Returns (S_x, S_y) in pixels per mm.
    
    For now, uses typical values. Future: implement FFT-based detection.
    """
    # Typical values for scanned ECG at ~300 DPI
    # 300 DPI = ~11.8 pixels/mm, but varies with scan resolution
    # These values work well for most digitized ECGs
    S_x = 8.5  # pixels per mm (horizontal)
    S_y = 8.3  # pixels per mm (vertical)
    
    # TODO: Implement automatic grid detection using FFT
    # 1. Compute 2D FFT of grayscale image
    # 2. Find peak frequencies corresponding to grid
    # 3. Convert to pixels/mm
    
    return S_x, S_y


def calibrate_signal(
    signal_coords: list,
    S_x: float,
    S_y: float,
    baseline_y: float = None
) -> tuple:
    """
    Convert pixel coordinates to physical units (ms, mV).
    
    Calibration:
        - Paper speed: 25 mm/s → 1 mm = 40 ms
        - Gain: 10 mm/mV → 1 mm = 0.1 mV
    
    Args:
        signal_coords: List of (x, y) pixel coordinates
        S_x: Horizontal scale (pixels per mm)
        S_y: Vertical scale (pixels per mm)
        baseline_y: Y-coordinate of baseline (isoelectric line)
        
    Returns:
        (time_ms, amplitude_mv) as numpy arrays
    """
    if not signal_coords:
        return np.array([]), np.array([])
    
    # Reference point
    px0 = signal_coords[0][0]  # First x coordinate
    
    # Use first point as baseline if not provided
    if baseline_y is None:
        baseline_y = signal_coords[0][1]
    
    time_ms = []
    amplitude_mv = []
    
    for (px, py) in signal_coords:
        # Convert horizontal pixels to time (ms)
        # Δx pixels → (Δx / S_x) mm → (Δx / S_x) * 40 ms
        t = ((px - px0) / S_x) * TIME_PER_MM_MS
        
        # Convert vertical pixels to amplitude (mV)
        # Δy pixels → (Δy / S_y) mm → (Δy / S_y) * 0.1 mV
        # Note: y increases downward in image, so we negate
        a = ((baseline_y - py) / S_y) * AMPLITUDE_PER_MM_MV
        
        time_ms.append(t)
        amplitude_mv.append(a)
    
    return np.array(time_ms), np.array(amplitude_mv)


def smooth_signal(amplitude: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to reduce noise."""
    if len(amplitude) < window_size:
        return amplitude
    
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(amplitude, kernel, mode='same')
    
    # Fix edge effects
    half_window = window_size // 2
    smoothed[:half_window] = amplitude[:half_window]
    smoothed[-half_window:] = amplitude[-half_window:]
    
    return smoothed


def calculate_snr(signal: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio (SNR) in dB.
    
    Uses high-pass filtered signal as noise estimate.
    SNR = 10 * log10(signal_power / noise_power)
    """
    if len(signal) < 2:
        return 0.0
    
    # Signal power (RMS squared)
    signal_power = np.mean(signal ** 2)
    
    if signal_power == 0:
        return 0.0
    
    # Estimate noise as high-frequency component (first difference)
    noise_estimate = np.diff(signal)
    noise_power = np.mean(noise_estimate ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    # SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db


def quality_rating(snr_db: float) -> str:
    """Convert SNR to quality rating."""
    if snr_db > 20:
        return "Excellent"
    elif snr_db > 15:
        return "Good"
    elif snr_db > 10:
        return "Fair"
    elif snr_db > 5:
        return "Poor"
    else:
        return "Very Poor"


def create_connected_trace_image(mask: np.ndarray, output_path: Path = None, thickness: int = 2) -> np.ndarray:
    """
    Create a clean black and white image with connected signal trace.
    
    Instead of showing raw mask pixels (which have gaps), this function:
    1. Extracts the median y-coordinate for each x column
    2. Interpolates ALL missing values to fill gaps
    3. Draws a connected polyline
    
    Args:
        mask: Binary mask (H, W) with signal pixels (0/1 or 0/255)
        output_path: Optional path to save the image
        thickness: Line thickness (default 2 for visibility)
        
    Returns:
        Clean B&W image with connected trace
    """
    height, width = mask.shape
    
    # Create black background
    output_img = np.zeros((height, width), dtype=np.uint8)
    
    # Normalize mask to binary
    if mask.max() > 1:
        binary_mask = (mask > 127).astype(np.uint8)
    else:
        binary_mask = mask
    
    # Extract signal trace - get median y for each x
    y_values = []
    x_with_signal = []
    
    for x in range(width):
        y_pixels = np.where(binary_mask[:, x] > 0)[0]
        if len(y_pixels) > 0:
            median_y = np.median(y_pixels)
            y_values.append(median_y)
            x_with_signal.append(x)
    
    if len(x_with_signal) < 2:
        # Not enough points to draw a line
        if output_path is not None:
            cv2.imwrite(str(output_path), output_img)
        return output_img
    
    # Convert to numpy arrays
    x_with_signal = np.array(x_with_signal)
    y_values = np.array(y_values)
    
    # Create full x range from first to last signal point
    x_full = np.arange(x_with_signal[0], x_with_signal[-1] + 1)
    
    # Interpolate y values for ALL x positions (fills gaps)
    y_interp = np.interp(x_full, x_with_signal, y_values)
    
    # Round to integers for drawing
    y_interp = np.round(y_interp).astype(np.int32)
    
    # Clip y values to valid range
    y_interp = np.clip(y_interp, 0, height - 1)
    
    # Create points array for polyline
    points = np.column_stack((x_full, y_interp)).astype(np.int32)
    
    # Draw connected polyline (white on black)
    cv2.polylines(output_img, [points], isClosed=False, color=255, thickness=thickness)
    
    # Save if path provided
    if output_path is not None:
        cv2.imwrite(str(output_path), output_img)
    
    return output_img


def convert_mask_to_bw(input_path: Path, output_path: Path = None) -> Path:
    """
    Convert nnU-Net mask to clean connected B/W trace image.
    
    Uses interpolation to create a smooth connected signal line
    without gaps or breakages.
    
    Args:
        input_path: Path to mask file from nnU-Net
        output_path: Output path (default: {stem}_digitized.png)
        
    Returns:
        Path to saved B/W image
    """
    input_path = Path(input_path)
    
    # Load the mask in grayscale
    img = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    
    # Determine output path
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_digitized.png"
    
    # Create connected trace image (handles gaps via interpolation)
    create_connected_trace_image(img, output_path, thickness=2)
    
    return output_path


def process_output(
    output_dir: Path,
    original_image: Path = None,
    verbose: bool = False
) -> tuple:
    """
    Process nnU-Net output mask to extract digitized signals.
    
    Returns:
        (results_dict, snr_values, S_x, S_y)
    """
    print("\n" + "─" * 63)
    print("STAGE 2: Vectorization & Calibration")
    print("─" * 63)
    
    output_dir = Path(output_dir)
    
    # Find mask files (exclude already-converted files)
    all_png_files = list(output_dir.glob('*.png'))
    mask_files = sorted([f for f in all_png_files if '_digitized' not in f.stem])
    mask_files += sorted(list(output_dir.glob('*.nii.gz')))
    
    if not mask_files:
        raise FileNotFoundError(
            f"No mask files found in {output_dir}\n"
            "Expected .png or .nii.gz files from nnU-Net prediction."
        )
    
    # Convert mask to B/W image (0/1 → 0/255)
    print(f"\nConverting mask to B/W image...")
    for mask_file in mask_files:
        if mask_file.suffix == '.png':
            bw_path = convert_mask_to_bw(mask_file)
            print(f"  ✓ Created: {bw_path.name}")
    
    print(f"\nFound {len(mask_files)} mask file(s)")
    
    # Detect grid spacing
    original_img = None
    if original_image and Path(original_image).exists():
        original_img = cv2.imread(str(original_image), cv2.IMREAD_GRAYSCALE)
    
    S_x, S_y = detect_grid_spacing(original_img)
    
    print(f"\nCalibration Parameters:")
    print(f"  Grid spacing: S_x = {S_x:.1f} px/mm, S_y = {S_y:.1f} px/mm")
    print(f"  Time: {TIME_PER_MM_MS:.0f} ms/mm (paper speed: {PAPER_SPEED_MM_PER_S} mm/s)")
    print(f"  Amplitude: {AMPLITUDE_PER_MM_MV:.2f} mV/mm (gain: {GAIN_MM_PER_MV} mm/mV)")
    print(f"  Target frequency: {FREQUENCY} Hz")
    
    # Calculate scaling factors for vectorization
    sec_per_pixel = 1.0 / (S_x * PAPER_SPEED_MM_PER_S)
    mV_per_pixel = 1.0 / (S_y * GAIN_MM_PER_MV)
    
    # Process each mask
    results = {}
    snr_values = {}
    
    print(f"\nProcessing Lead Signals:")
    print("-" * 50)
    
    # For single image, the mask represents the full ECG
    # For lead-cropped images, each mask is one lead
    
    for idx, mask_file in enumerate(mask_files[:12]):  # Max 12 leads
        lead_name = LEAD_NAMES[idx] if idx < len(LEAD_NAMES) else f"Lead_{idx}"
        
        try:
            # Load mask
            mask = load_mask(mask_file)
            
            if verbose:
                print(f"  {lead_name}: mask shape = {mask.shape}, "
                      f"signal pixels = {np.sum(mask > 0)}")
            
            # Check if mask has signal
            if np.sum(mask > 0) == 0:
                print(f"  ✗ {lead_name:4s} | No signal detected in mask")
                continue
            
            # Try advanced vectorization first
            time_ms, amplitude_mv = vectorise_from_mask(
                mask=mask,
                original_image=original_img,
                sec_per_pixel=sec_per_pixel,
                mV_per_pixel=mV_per_pixel,
                lead=lead_name,
                y_shift_ratio=Y_SHIFT_RATIO
            )
            
            # Fallback to simple method if advanced fails
            if time_ms is None or len(time_ms) == 0:
                coords = vectorize_mask(mask)
                if not coords:
                    print(f"  ✗ {lead_name:4s} | No signal detected in mask")
                    continue
                time_ms, amplitude_mv = calibrate_signal(coords, S_x, S_y)
            
            # Optional smoothing
            amplitude_mv_smooth = smooth_signal(amplitude_mv, window_size=3)
            
            # Calculate SNR
            snr = calculate_snr(amplitude_mv_smooth)
            snr_values[lead_name] = snr
            quality = quality_rating(snr)
            
            # Store results
            results[lead_name] = {
                'time_ms': time_ms,
                'amplitude_mv': amplitude_mv_smooth,
                'amplitude_raw': amplitude_mv,
                'snr_db': snr,
                'quality': quality,
                'num_samples': len(time_ms),
                'duration_ms': time_ms[-1] - time_ms[0] if len(time_ms) > 1 else 0,
                'sampling_freq_hz': FREQUENCY
            }
            
            # Print summary
            duration = results[lead_name]['duration_ms']
            n_samples = results[lead_name]['num_samples']
            print(f"  ✓ {lead_name:4s} | SNR: {snr:5.1f} dB | "
                  f"Quality: {quality:9s} | {n_samples:4d} samples @ {FREQUENCY}Hz | {duration:.0f} ms")
            
        except Exception as e:
            print(f"  ✗ {lead_name:4s} | Error: {str(e)}")
            if verbose:
                import traceback
                traceback.print_exc()
    
    print("-" * 50)
    
    if not results:
        raise ValueError("No valid signals extracted from masks!")
    
    # Summary statistics
    avg_snr = np.mean(list(snr_values.values()))
    print(f"\nSummary:")
    print(f"  Leads processed: {len(results)}/{len(mask_files)}")
    print(f"  Average SNR: {avg_snr:.1f} dB ({quality_rating(avg_snr)})")
    
    return results, snr_values, S_x, S_y


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    results: dict,
    snr_values: dict,
    S_x: float,
    S_y: float,
    input_filename: str,
    output_dir: Path,
    generate_plot: bool = True
) -> tuple:
    """Save digitized signals and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n" + "─" * 63)
    print("OUTPUT FILES")
    print("─" * 63)
    print(f"\nOutput directory: {output_dir}/")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 1. Save CSV with all leads
    # ─────────────────────────────────────────────────────────────────────────
    
    # Find max length for padding
    max_length = max(
        len(results[lead]['time_ms']) 
        for lead in results
    ) if results else 0
    
    csv_data = {}
    
    # Add time column (from first lead)
    first_lead = list(results.keys())[0]
    time_ms = results[first_lead]['time_ms']
    
    # Pad if needed
    if len(time_ms) < max_length:
        time_ms = np.pad(time_ms, (0, max_length - len(time_ms)), 
                         constant_values=np.nan)
    csv_data['Time_ms'] = time_ms
    
    # Add amplitude columns
    for lead in results:
        amplitude_mv = results[lead]['amplitude_mv']
        if len(amplitude_mv) < max_length:
            amplitude_mv = np.pad(amplitude_mv, (0, max_length - len(amplitude_mv)),
                                  constant_values=np.nan)
        csv_data[f'{lead}_mV'] = amplitude_mv
    
    df = pd.DataFrame(csv_data)
    csv_path = output_dir / '12_lead_signals.csv'
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"  ✓ 12_lead_signals.csv       ({len(df)} samples, {len(results)} leads)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 2. Save metadata JSON
    # ─────────────────────────────────────────────────────────────────────────
    
    metadata = {
        'input_file': input_filename,
        'timestamp': datetime.now().isoformat(),
        'pipeline_version': '1.0.0',
        
        'leads': {
            lead: {
                'snr_db': float(results[lead]['snr_db']),
                'quality': results[lead]['quality'],
                'num_samples': results[lead]['num_samples'],
                'duration_ms': float(results[lead]['duration_ms'])
            }
            for lead in results
        },
        
        'summary': {
            'total_leads': len(results),
            'average_snr_db': float(np.mean(list(snr_values.values()))),
            'overall_quality': quality_rating(np.mean(list(snr_values.values())))
        },
        
        'calibration': {
            'S_x_pixels_per_mm': S_x,
            'S_y_pixels_per_mm': S_y,
            'paper_speed_mm_per_s': PAPER_SPEED_MM_PER_S,
            'gain_mm_per_mv': GAIN_MM_PER_MV,
            'time_conversion': f'1mm = {TIME_PER_MM_MS:.0f}ms',
            'amplitude_conversion': f'1mm = {AMPLITUDE_PER_MM_MV:.2f}mV'
        },
        
        'model_info': {
            'folds_used': NUM_FOLDS,
            'ensemble': True,
            'framework': 'nnU-Net (Felix Krones fork)',
            'dataset': DATASET_NAME,
            'config': CONFIG
        }
    }
    
    json_path = output_dir / 'metadata.json'
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json             (SNR, calibration, model info)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 3. Generate visualization plot
    # ─────────────────────────────────────────────────────────────────────────
    
    plot_path = None
    if generate_plot and HAS_MATPLOTLIB:
        try:
            fig, axes = plt.subplots(len(results), 1, figsize=(15, 2*len(results)))
            
            if len(results) == 1:
                axes = [axes]
            
            lead_names = list(results.keys())
            
            # Color based on quality
            quality_colors = {
                'Excellent': '#2ecc71',  # Green
                'Good': '#27ae60',       # Dark green
                'Fair': '#f39c12',       # Orange
                'Poor': '#e74c3c',       # Red
                'Very Poor': '#c0392b'   # Dark red
            }
            
            for idx, (ax, lead_name) in enumerate(zip(axes, lead_names)):
                time_ms = results[lead_name]['time_ms']
                amplitude_mv = results[lead_name]['amplitude_mv']
                snr = results[lead_name]['snr_db']
                quality = results[lead_name]['quality']
                
                color = quality_colors.get(quality, '#3498db')
                
                ax.plot(time_ms, amplitude_mv, color=color, linewidth=0.8)
                ax.set_ylabel(f'{lead_name}\n(mV)', fontsize=10)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                
                if len(time_ms) > 0:
                    ax.set_xlim([0, max(time_ms)])
                
                # SNR badge
                ax.text(0.02, 0.95, f'SNR: {snr:.1f} dB ({quality})',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
            
            axes[-1].set_xlabel('Time (ms)', fontsize=12)
            
            plt.suptitle(
                f'12-Lead ECG Digitization - {input_filename}\n'
                f'Average SNR: {np.mean(list(snr_values.values())):.1f} dB',
                fontsize=14, fontweight='bold'
            )
            
            plt.tight_layout()
            
            plot_path = output_dir / 'waveforms.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"  ✓ waveforms.png             (12-lead visualization)")
            
        except Exception as e:
            print(f"  ⚠ Could not generate plot: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # 4. Save individual lead files (optional)
    # ─────────────────────────────────────────────────────────────────────────
    
    leads_dir = output_dir / 'individual_leads'
    leads_dir.mkdir(exist_ok=True)
    
    for lead_name in results:
        lead_df = pd.DataFrame({
            'Time_ms': results[lead_name]['time_ms'],
            'Amplitude_mV': results[lead_name]['amplitude_mv']
        })
        lead_csv = leads_dir / f'{lead_name}.csv'
        lead_df.to_csv(lead_csv, index=False, float_format='%.4f')
    
    print(f"  ✓ individual_leads/         ({len(results)} individual CSV files)")
    
    return csv_path, json_path, plot_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point for ECG digitization pipeline."""
    
    parser = argparse.ArgumentParser(
        description='ECG Digitization Pipeline using nnU-Net (Terminal Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:    python ecg_digitize.py --input ./test_images/ecg.jpg
  Batch process:   python ecg_digitize.py --input ./test_images/
  Custom output:   python ecg_digitize.py --input ecg.jpg --output ./my_results/
  Verbose mode:    python ecg_digitize.py --input ecg.jpg --verbose
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to ECG image or directory containing images'
    )
    parser.add_argument(
        '--working_dir', '-w',
        default='./working',
        help='Path to working directory with trained models (default: ./working)'
    )
    parser.add_argument(
        '--output', '-o',
        default='./working/results',
        help='Output directory for results (default: ./working/results)'
    )
    parser.add_argument(
        '--dataset_id', '-d',
        type=int,
        default=None,
        help='nnUNet dataset ID (auto-detected if not specified)'
    )
    parser.add_argument(
        '--config', '-c',
        default='2d',
        help='nnUNet configuration (default: 2d)'
    )
    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='Skip inference, use existing output'
    )
    parser.add_argument(
        '--no_plot',
        action='store_true',
        help='Disable plot generation'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with detailed logging'
    )
    
    args = parser.parse_args()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Header
    # ─────────────────────────────────────────────────────────────────────────
    
    print()
    print("═" * 63)
    print("  ECG DIGITIZATION PIPELINE")
    print("  nnU-Net Ensemble (5 Folds) - Terminal Version")
    print("═" * 63)
    print()
    
    working_dir = Path(args.working_dir).absolute()
    output_dir = Path(args.output).absolute()
    
    print(f"Working directory: {working_dir}")
    print(f"Output directory:  {output_dir}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Setup
    # ─────────────────────────────────────────────────────────────────────────
    
    # Set up environment
    setup_environment(working_dir)
    
    # Verify setup and get model info
    model_info = verify_setup(working_dir, args.verbose)
    
    # Use auto-detected dataset ID if not specified
    dataset_id = args.dataset_id or model_info['dataset_id']
    config = args.config
    
    print(f"Model Configuration:")
    print(f"  Dataset ID: {dataset_id}")
    print(f"  Config: {config}")
    print(f"  Folds: {model_info['folds_present']}")
    
    # Setup directories
    setup_directories(working_dir)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Find input images
    # ─────────────────────────────────────────────────────────────────────────
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            list(input_path.glob('*.jpg')) +
            list(input_path.glob('*.jpeg')) +
            list(input_path.glob('*.png')) +
            list(input_path.glob('*.JPG')) +
            list(input_path.glob('*.JPEG')) +
            list(input_path.glob('*.PNG'))
        )
        if not image_files:
            raise FileNotFoundError(f"No image files found in {input_path}")
        print(f"\n✓ Found {len(image_files)} ECG image(s) to process")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Process each image
    # ─────────────────────────────────────────────────────────────────────────
    
    successful = 0
    failed = 0
    
    for img_idx, img_file in enumerate(image_files):
        print()
        print("═" * 63)
        print(f"Processing [{img_idx + 1}/{len(image_files)}]: {img_file.name}")
        print("═" * 63)
        
        start_time = time.time()
        
        try:
            # Stage 1: Run nnU-Net ensemble inference (all 5 folds at once)
            if not args.skip_inference:
                output_dir_inference = run_ensemble_inference(
                    input_image=img_file,
                    working_dir=working_dir,
                    dataset_id=dataset_id,
                    config=config,
                    verbose=args.verbose
                )
            else:
                output_dir_inference = working_dir / 'output'
                print(f"\nSkipping inference (using existing output: {output_dir_inference})")
            
            # Stage 2: Vectorization & Calibration
            results, snr_values, S_x, S_y = process_output(
                output_dir=output_dir_inference,
                original_image=img_file,
                verbose=args.verbose
            )
            
            # Save results
            output_subdir = output_dir / img_file.stem
            csv_path, json_path, plot_path = save_results(
                results=results,
                snr_values=snr_values,
                S_x=S_x,
                S_y=S_y,
                input_filename=img_file.name,
                output_dir=output_subdir,
                generate_plot=not args.no_plot
            )
            
            elapsed = time.time() - start_time
            
            # Summary
            print()
            print("─" * 63)
            print("✓ PROCESSING COMPLETE")
            print("─" * 63)
            print(f"  Time: {elapsed:.1f} seconds")
            print(f"  Output: {output_subdir}/")
            print(f"  Leads: {len(results)}")
            print(f"  Avg SNR: {np.mean(list(snr_values.values())):.1f} dB")
            
            successful += 1
            
        except Exception as e:
            print()
            print("─" * 63)
            print(f"✗ ERROR: {str(e)}")
            print("─" * 63)
            
            if args.verbose:
                import traceback
                traceback.print_exc()
            
            failed += 1
            continue
    
    # ─────────────────────────────────────────────────────────────────────────
    # Final Summary
    # ─────────────────────────────────────────────────────────────────────────
    
    print()
    print("═" * 63)
    print("  PIPELINE COMPLETE")
    print("═" * 63)
    print(f"  Processed: {successful}/{len(image_files)} images")
    if failed > 0:
        print(f"  Failed: {failed}")
    print(f"  Results saved to: {output_dir}/")
    print("═" * 63)
    print()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
