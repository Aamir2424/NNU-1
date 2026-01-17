#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ECG Digitization Pipeline - Setup Script
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error

echo "═══════════════════════════════════════════════════════════════"
echo "  ECG Digitization Pipeline - Installation"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "  ✓ Python $PYTHON_VERSION detected"
echo ""

# Install nnU-Net (Felix Krones fork)
echo "Installing nnU-Net (Felix Krones fork)..."
pip install git+https://github.com/FelixKrones/nnUNet.git
echo "  ✓ nnU-Net installed"
echo ""

# Install additional dependencies
echo "Installing additional dependencies..."
pip install opencv-python numpy scipy matplotlib pandas pillow nibabel scikit-image
echo "  ✓ Dependencies installed"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import nnunetv2; print('  ✓ nnunetv2 imported successfully')"
python3 -c "import cv2; print('  ✓ OpenCV imported successfully')"
python3 -c "import numpy; print('  ✓ NumPy imported successfully')"
python3 -c "import pandas; print('  ✓ Pandas imported successfully')"
python3 -c "import matplotlib; print('  ✓ Matplotlib imported successfully')"
python3 -c "import nibabel; print('  ✓ NiBabel imported successfully')"
echo ""

# Create test_images directory
mkdir -p ./test_images
echo "  ✓ Created ./test_images/ directory"
echo ""

# Set up environment variables for nnU-Net
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export nnUNet_results="${SCRIPT_DIR}/working/nnUNet_results"
export nnUNet_raw="${SCRIPT_DIR}/working/nnUNet_raw"
export nnUNet_preprocessed="${SCRIPT_DIR}/working/nnUNet_preprocessed"

echo "Environment variables set:"
echo "  nnUNet_results = ${nnUNet_results}"
echo "  nnUNet_raw     = ${nnUNet_raw}"
echo "  nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo ""

echo "═══════════════════════════════════════════════════════════════"
echo "  ✓ Installation Complete!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To test the pipeline, run:"
echo "  python ecg_digitize.py --input path/to/ecg_image.jpg"
echo ""
echo "For verbose output:"
echo "  python ecg_digitize.py --input path/to/ecg_image.jpg --verbose"
echo ""
echo "NOTE: Before running, set environment variables:"
echo "  export nnUNet_results=\"${SCRIPT_DIR}/working/nnUNet_results\""
echo ""
