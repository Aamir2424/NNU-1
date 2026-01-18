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
    
    # Find the black & white digitized image in the working/output folder
    # The digitized file is named: {job_id}_{image_name}_digitized.png
    image_stem = Path(filename).stem
    digitized_filename = f"{job_id}_{image_stem}_digitized.png"
    digitized_path = WORKING_DIR / "output" / digitized_filename
    
    if digitized_path.exists():
        # Copy the digitized image to our results folder for serving
        dest_path = output_dir / "digitized.png"
        shutil.copy(str(digitized_path), str(dest_path))
        response['files']['digitized'] = f'/results/{job_id}/digitized.png'
    
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
