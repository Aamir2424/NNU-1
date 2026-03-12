#!/bin/bash

# Script to run the ECG Digitizer backend with proper environment settings

# Force CPU usage (disable MPS on Mac and CUDA)
export PYTORCH_ENABLE_MPS_FALLBACK=1
export CUDA_VISIBLE_DEVICES=""
export nnUNet_USE_CUDA=0

# Set nnUNet environment variables
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"
export nnUNet_results="$(pwd)/working/nnUNet_results"

# Activate virtual environment
source venv/bin/activate

# Run the Flask app
python app.py