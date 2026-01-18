#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════
ECG Digitization - Web Application Backend
═══════════════════════════════════════════════════════════════════════════════════

Simple Flask backend that serves the frontend and handles ECG image processing.

Usage:
  python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import sys
import json
import uuid
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)

# Get the directory where this script is located
BASE_DIR = Path(__file__).parent.resolve()
WORKING_DIR = BASE_DIR / "working"
UPLOAD_FOLDER = WORKING_DIR / "uploads"
RESULTS_FOLDER = WORKING_DIR / "web_results"

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

# Create directories
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Set environment variables for nnU-Net
os.environ['nnUNet_results'] = str(WORKING_DIR / "nnUNet_results")
os.environ['nnUNet_raw'] = str(WORKING_DIR / "nnUNet_raw")
os.environ['nnUNet_preprocessed'] = str(WORKING_DIR / "nnUNet_preprocessed")

# ═══════════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    """Check if file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_ecg_digitization(image_path, output_dir):
    """Run the ECG digitization pipeline on an image."""
    script_path = BASE_DIR / "ecg_digitize.py"
    
    cmd = [
        sys.executable,
        str(script_path),
        "--input", str(image_path),
        "--output", str(output_dir),
        "--verbose"
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(BASE_DIR),
        env={**os.environ}
    )
    
    return result.returncode == 0, result.stdout, result.stderr


def generate_plot_from_csv(csv_path, output_path):
    """Generate a plot from the CSV data that looks like the original ECG image."""
    try:
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        # Get all lead columns (exclude Time_ms)
        lead_columns = [col for col in df.columns if col != 'Time_ms' and '_mV' in col]
        
        if not lead_columns:
            return False
        
        # Create figure matching ECG strip proportions (very wide, short height like ECG paper)
        num_leads = len(lead_columns)
        # ECG strips are typically ~20:4 ratio (wide and short)
        fig, axes = plt.subplots(num_leads, 1, figsize=(25, 5), sharex=True)
        
        # Handle single lead case
        if num_leads == 1:
            axes = [axes]
        
        # Plot each lead with proper ECG appearance
        for idx, lead in enumerate(lead_columns):
            # Plot the signal normally (positive up, like standard ECG) - THICKER line
            axes[idx].plot(df['Time_ms'], df[lead], 'k-', linewidth=2.5)
            
            # Set ylabel with lead name - BIGGER font
            axes[idx].set_ylabel(lead.replace('_mV', ''), fontsize=20, fontweight='bold')
            
            # Add grid like ECG paper - THICKER lines
            axes[idx].grid(True, which='major', alpha=0.4, linestyle='-', linewidth=1.0, color='#d4a373')
            axes[idx].grid(True, which='minor', alpha=0.2, linestyle='-', linewidth=0.6, color='#d4a373')
            axes[idx].minorticks_on()
            
            # Auto-scale Y-axis to show waveform detail
            axes[idx].set_xlim(df['Time_ms'].min(), df['Time_ms'].max())
            axes[idx].margins(y=0.2)  # Add 20% margin for better visibility
            
            # Style the axes - THICKER borders
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)
            axes[idx].spines['left'].set_linewidth(1.5)
            axes[idx].spines['bottom'].set_linewidth(1.5)
            axes[idx].set_facecolor('#fffef8')  # Cream color like ECG paper
            
            # Show Y-axis values - BIGGER font
            axes[idx].tick_params(axis='y', labelsize=18, width=1.5, length=6)
            axes[idx].tick_params(axis='x', labelsize=18, width=1.5, length=6)
            axes[idx].yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
        
        # Set common xlabel - BIGGER font
        axes[-1].set_xlabel('Time (ms)', fontsize=18, fontweight='bold')
        
        # Add title - BIGGER font
        fig.suptitle('Digitized ECG Signals (from CSV data)', fontsize=20, fontweight='bold', y=0.98)
        
        # Adjust layout and save with minimal padding to match ECG strip
        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white', pad_inches=0.05)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error generating plot: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve the main page."""
    return send_from_directory('web', 'index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and process the ECG image."""
    
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    upload_path = UPLOAD_FOLDER / f"{job_id}_{filename}"
    file.save(str(upload_path))
    
    # Create output directory for this job
    output_dir = RESULTS_FOLDER / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run digitization
    success, stdout, stderr = run_ecg_digitization(upload_path, output_dir)
    
    if not success:
        return jsonify({
            'error': 'Processing failed',
            'details': stderr or stdout
        }), 500
    
    # Prepare response
    response = {
        'success': True,
        'job_id': job_id,
        'message': 'ECG digitization complete!',
        'files': {}
    }
    
    # Copy original uploaded image for display
    original_dest = output_dir / "original.png"
    shutil.copy(str(upload_path), str(original_dest))
    response['files']['original'] = f'/results/{job_id}/original.png'
    
    # Find the black & white digitized image in the working/output folder
    # Try multiple patterns to find the digitized file
    output_folder = WORKING_DIR / "output"
    digitized_path = None
    
    if output_folder.exists():
        # Look for any file with job_id and "digitized" in the name
        for file in output_folder.glob(f"{job_id}*digitized*.png"):
            digitized_path = file
            break
        
        # If not found, try without job_id (some images might not include it)
        if not digitized_path:
            image_stem = Path(filename).stem
            for file in output_folder.glob(f"*{image_stem}*digitized*.png"):
                digitized_path = file
                break
    
    if digitized_path and digitized_path.exists():
        # Copy the digitized image to our results folder for serving
        dest_path = output_dir / "digitized.png"
        shutil.copy(str(digitized_path), str(dest_path))
        response['files']['digitized'] = f'/results/{job_id}/digitized.png'
    
    # Find CSV and metadata in the results subdirectories
    for item in output_dir.iterdir():
        if item.is_dir():
            csv_file = item / "12_lead_signals.csv"
            metadata_file = item / "metadata.json"
            
            if csv_file.exists():
                # Copy CSV to results folder
                csv_dest = output_dir / "signals.csv"
                shutil.copy(str(csv_file), str(csv_dest))
                response['files']['csv'] = f'/results/{job_id}/signals.csv'
                
                # Generate plot from CSV
                plot_dest = output_dir / "plot.png"
                if generate_plot_from_csv(csv_file, plot_dest):
                    response['files']['plot'] = f'/results/{job_id}/plot.png'
                
                # Read CSV preview (first 10 rows)
                import csv as csv_module
                with open(csv_file, 'r') as f:
                    reader = csv_module.reader(f)
                    rows = list(reader)
                    response['csv_preview'] = {
                        'headers': rows[0] if rows else [],
                        'data': rows[1:11] if len(rows) > 1 else [],  # First 10 data rows
                        'total_rows': len(rows) - 1
                    }
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    response['metadata'] = json.load(f)
            
            break
    
    return jsonify(response)


@app.route('/results/<path:filepath>')
def serve_results(filepath):
    """Serve result files."""
    return send_from_directory(str(RESULTS_FOLDER), filepath)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat()
    })


# ═══════════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("═" * 60)
    print("  ECG Digitization Web Application")
    print("═" * 60)
    print(f"\n  Open http://localhost:8000 in your browser\n")
    print("═" * 60)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
