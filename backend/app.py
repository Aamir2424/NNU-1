

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import base64
import subprocess
import json
import shutil
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Force CPU usage to avoid MPS device issues on Mac
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)


app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})

# Configuration
WORKING_DIR = os.path.join(os.path.dirname(__file__), 'working')
UPLOAD_FOLDER = os.path.join(WORKING_DIR, 'uploads')
INPUT_FOLDER = os.path.join(WORKING_DIR, 'input')
OUTPUT_FOLDER = os.path.join(WORKING_DIR, 'output-ensemble')
RESULTS_FOLDER = os.path.join(WORKING_DIR, 'results')

# Constants for vectorization
FREQUENCY = 500  # Hz
LONG_SIGNAL_LENGTH_SEC = 10
SHORT_SIGNAL_LENGTH_SEC = 2.5

# Ensure directories exist
for folder in [UPLOAD_FOLDER, INPUT_FOLDER, OUTPUT_FOLDER, RESULTS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Set nnUNet environment variables
os.environ["nnUNet_raw"] = os.path.join(WORKING_DIR, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(WORKING_DIR, "nnUNet_preprocessed")
os.environ["nnUNet_results"] = os.path.join(WORKING_DIR, "nnUNet_results")


def convert_to_grayscale(image_path, output_path):
    """Convert image to grayscale if it's colored."""
    img = Image.open(image_path)
    
    # Check if image is already grayscale
    if img.mode == 'L':
        img.save(output_path)
        return False
    
    # Convert to grayscale
    gray_img = img.convert('L')
    gray_img.save(output_path)
    return True


def run_nnunet_inference(input_folder, output_folder):
    """Run nnU-Net inference using all folds (ensemble)."""
    # Set environment to force CPU usage
    env = os.environ.copy()
    env['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = ''
    # Force nnUNet to use CPU
    env['nnUNet_USE_CUDA'] = '0'
    
    print(f"Running inference...")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Files in input: {os.listdir(input_folder) if os.path.exists(input_folder) else 'folder not found'}")
    
    # Try with -device flag first
    cmd = [
        'nnUNetv2_predict',
        '-i', input_folder,
        '-o', output_folder,
        '-d', '1',
        '-c', '2d',
        '-f', '0', '1', '2', '3', '4',
        '-device', 'cpu'
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        print(f"Inference successful!")
        print(f"Output: {result.stdout[:500]}")  # Print first 500 chars
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        # If -device flag fails, try without it (rely on env vars)
        if 'unrecognized arguments' in e.stderr:
            print("Retrying without -device flag...")
            cmd_no_device = [
                'nnUNetv2_predict',
                '-i', input_folder,
                '-o', output_folder,
                '-d', '1',
                '-c', '2d',
                '-f', '0', '1', '2', '3', '4'
            ]
            try:
                print(f"Running command: {' '.join(cmd_no_device)}")
                result = subprocess.run(cmd_no_device, check=True, capture_output=True, text=True, env=env)
                print(f"Inference successful!")
                print(f"Output: {result.stdout[:500]}")
                return True, result.stdout
            except subprocess.CalledProcessError as e2:
                print(f"Inference failed: {e2.stderr}")
                return False, e2.stderr
        print(f"Inference failed: {e.stderr}")
        return False, e.stderr


def load_mask(mask_path):
    """Load the segmentation mask from nnU-Net output."""
    print(f"Loading mask from: {mask_path}")
    
    # nnU-Net typically outputs .nii.gz or .npz files
    # Adjust based on your actual output format
    if mask_path.endswith('.npy'):
        mask = np.load(mask_path)
    elif mask_path.endswith('.npz'):
        mask = np.load(mask_path)['arr_0']
    elif mask_path.endswith('.nii.gz') or mask_path.endswith('.nii'):
        # For NIfTI files
        import nibabel as nib
        mask = nib.load(mask_path).get_fdata()
    else:
        # Try loading as image (PNG output from nnUNet)
        mask = np.array(Image.open(mask_path))
    
    print(f"Loaded mask shape (numpy): {mask.shape}")
    print(f"Mask dtype: {mask.dtype}")
    print(f"Mask min/max: {mask.min()}/{mask.max()}")
    
    # Convert to torch tensor
    mask_tensor = torch.from_numpy(mask).float()
    
    # Add batch and channel dimensions if needed
    # Expected final shape: [batch, channel, height, width]
    if mask_tensor.dim() == 2:
        # [H, W] -> [1, 1, H, W]
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
    elif mask_tensor.dim() == 3:
        # Could be [C, H, W] or [H, W, C]
        # Check if it's likely [H, W, C] (e.g., grayscale with channel last)
        if mask_tensor.shape[2] < mask_tensor.shape[0] and mask_tensor.shape[2] < mask_tensor.shape[1]:
            # [H, W, C] -> [C, H, W] -> [1, C, H, W]
            mask_tensor = mask_tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            # [C, H, W] -> [1, C, H, W]
            mask_tensor = mask_tensor.unsqueeze(0)
    
    # If there are multiple channels, take the first one (or max across channels)
    if mask_tensor.shape[1] > 1:
        # Take argmax across channel dimension to get single channel
        mask_tensor = torch.argmax(mask_tensor, dim=1, keepdim=True).float()
    
    print(f"Final mask tensor shape: {mask_tensor.shape}")
    
    return mask_tensor


def vectorise(
    image_rotated,
    mask,
    signal_cropped,
    sec_per_pixel,
    mV_per_pixel,
    y_shift_ratio,
):
    """
    Notebook-faithful single-lead vectorisation
    """

    # mask: [1, 1, H, W]
    H, W = mask.shape[2], mask.shape[3]

    # ---- TIME SCALING ----
    total_seconds_from_mask = round(sec_per_pixel * W, 1)

    if total_seconds_from_mask > (LONG_SIGNAL_LENGTH_SEC / 2):
        total_seconds = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio
    else:
        total_seconds = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio

    values_needed = int(total_seconds * FREQUENCY)

    # ---- Y SCALING (NOTEBOOK LOGIC) ----
    non_zero_mean = torch.tensor([
        torch.mean(torch.nonzero(mask[0, 0, :, x]).float())
        if torch.count_nonzero(mask[0, 0, :, x]) > 0
        else 0.0
        for x in range(W)
    ])

    signal_cropped_shifted = (
        (1 - y_shift_ratio_) * H - signal_cropped
    )

    predicted_signal = (
        signal_cropped_shifted - non_zero_mean
    ) * mV_per_pixel

    # ---- BASELINE CORRECTION (GOOD & NECESSARY) ----
    predicted_signal = predicted_signal - torch.median(predicted_signal)

    # ---- X RESAMPLING ----
    resampled = F.interpolate(
        predicted_signal.view(1, 1, -1),
        size=values_needed,
        mode="linear",
        align_corners=False,
    )

    return resampled.view(-1)



def extract_signal_from_mask(image, mask, calibration, lead_name=None):
    """Extract single-lead signal from mask and image."""
    # Default calibration parameters
    sec_per_pixel = calibration.get('sec_per_pixel', 0.004)
    mV_per_pixel = calibration.get('mV_per_pixel', 0.01)
    y_shift_ratio = calibration.get('y_shift_ratio', 0.5)
    
    # Convert image to tensor
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
    else:
        image_tensor = torch.from_numpy(np.array(image)).float()
    
    # Ensure image has proper dimensions [batch, channel, height, width]
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Mask tensor shape: {mask.shape}")
    
    # Extract signal trace from mask using argmax along height dimension
    # mask is [batch, channel, height, width]
    signal_trace = torch.tensor([
    torch.mean(torch.nonzero(mask[0, 0, :, i]).float())
    if torch.count_nonzero(mask[0, 0, :, i]) > 0
    else 0.0
    for i in range(mask.shape[3])
])
 # [width]
    
    print(f"Signal trace shape: {signal_trace.shape}")
    
    # Vectorize the single signal
    signal = vectorise(
        image_tensor,
        mask,
        signal_trace,
        sec_per_pixel,
        mV_per_pixel,
        y_shift_ratio
    )
    
    return signal.numpy().tolist()


def plot_signal(signal, lead_name, output_path, img_width_px, img_height_px):

    """Plot single ECG signal."""
    DPI = 100  # keep fixed
    fig_width_in = img_width_px / DPI
    fig_height_in = img_height_px / DPI
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=DPI)
    
    time = np.arange(len(signal)) / FREQUENCY
    ax.plot(time, signal, 'g-', linewidth=1.2)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Amplitude (mV)', fontsize=12)
    ax.set_title(f'Digitized ECG Signal{" - " + lead_name if lead_name else ""}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, max(time))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def mask_to_base64(mask_tensor):
    """Convert mask tensor to base64 encoded image."""
    # Convert to numpy and normalize
    mask_np = mask_tensor.squeeze().numpy()
    if mask_np.max() > mask_np.min():
        mask_np = (mask_np - mask_np.min()) / (mask_np.max() - mask_np.min()) * 255
    else:
        mask_np = mask_np * 255
    mask_np = mask_np.astype(np.uint8)
    
    # Convert to PIL Image
    mask_img = Image.fromarray(mask_np)
    
    # Convert to base64
    buffer = io.BytesIO()
    mask_img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()


@app.route('/api/digitize', methods=['POST'])
def digitize_ecg():
    """Main endpoint to digitize single-lead ECG image."""
    if 'ecg_image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['ecg_image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get optional lead name from form data
    lead_name = request.form.get('lead_name', None)
    
    # Get optional calibration parameters
    try:
        sec_per_pixel = float(request.form.get('sec_per_pixel', 0.004))
        mV_per_pixel = float(request.form.get('mV_per_pixel', 0.01))
        y_shift_ratio = float(request.form.get('y_shift_ratio', 0.5))
    except ValueError:
        return jsonify({'error': 'Invalid calibration parameters'}), 400
    
    try:
        # Save uploaded file
        timestamp = str(int(os.times().elapsed * 1000))
        filename = f"ecg_{timestamp}.png"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        # Convert to grayscale if needed
        input_filename = f"ecg_{timestamp}_0000.png"
        input_path = os.path.join(INPUT_FOLDER, input_filename)
        was_converted = convert_to_grayscale(upload_path, input_path)
        
        # Run nnU-Net inference
        success, message = run_nnunet_inference(INPUT_FOLDER, OUTPUT_FOLDER)
        if not success:
            return jsonify({'error': f'Inference failed: {message}'}), 500
        
        # Load the output mask - check what files were actually created
        print(f"Looking for mask in: {OUTPUT_FOLDER}")
        print(f"Expected filename: {input_filename}")
        
        # List all files in output folder for debugging
        if os.path.exists(OUTPUT_FOLDER):
            output_files = os.listdir(OUTPUT_FOLDER)
            print(f"Files in output folder: {output_files}")
        else:
            return jsonify({'error': f'Output folder not found: {OUTPUT_FOLDER}'}), 500
        
        # Try to find the mask file with various naming conventions
        mask_output_path = None
        base_name = input_filename.replace('_0000.png', '')
        
        # Try different possible output names
        possible_names = [
            input_filename,  # ecg_timestamp_0000.png
            base_name + '.png',  # ecg_timestamp.png
            base_name + '.npy',  # ecg_timestamp.npy
            base_name + '.npz',  # ecg_timestamp.npz
            base_name + '.nii.gz',  # ecg_timestamp.nii.gz
            base_name + '.nii',  # ecg_timestamp.nii
            input_filename.replace('.png', '.npy'),
            input_filename.replace('.png', '.npz'),
            input_filename.replace('.png', '.nii.gz'),
            input_filename.replace('.png', '.nii'),
        ]
        
        for possible_name in possible_names:
            test_path = os.path.join(OUTPUT_FOLDER, possible_name)
            if os.path.exists(test_path):
                mask_output_path = test_path
                print(f"Found mask at: {mask_output_path}")
                break
        
        # If still not found, check if there's any file in the output folder
        if mask_output_path is None and output_files:
            # Just use the first file found (assuming it's the prediction)
            mask_output_path = os.path.join(OUTPUT_FOLDER, output_files[0])
            print(f"Using first available file: {mask_output_path}")
        
        if mask_output_path is None or not os.path.exists(mask_output_path):
            return jsonify({
                'error': 'Mask output not found',
                'debug_info': {
                    'output_folder': OUTPUT_FOLDER,
                    'expected_name': input_filename,
                    'files_found': output_files if os.path.exists(OUTPUT_FOLDER) else []
                }
            }), 500
        
        mask = load_mask(mask_output_path)
        
        # Load original image
        original_image = Image.open(input_path)
        img_width_px, img_height_px = original_image.size
        
        # Extract single signal
        calibration = {
            'sec_per_pixel': sec_per_pixel,
            'mV_per_pixel': mV_per_pixel,
            'y_shift_ratio': y_shift_ratio
        }
        
        signal = extract_signal_from_mask(original_image, mask, calibration, lead_name)
        
        # Create result folder
        result_folder = os.path.join(RESULTS_FOLDER, f"ecg_{timestamp}")
        os.makedirs(result_folder, exist_ok=True)
        
        # Save signal to CSV
        csv_path = os.path.join(result_folder, 'signal.csv')
        with open(csv_path, 'w') as f:
            f.write('Time(s),Amplitude(mV)\n')
            for i, value in enumerate(signal):
                time = i / FREQUENCY
                f.write(f'{time:.4f},{value}\n')
        
        # Plot signal
        plot_path = os.path.join(result_folder, 'waveform.png')
        plot_signal(
    signal,
    lead_name,
    plot_path,
    img_width_px,
    img_height_px
)

        
        # Convert plot to base64
        with open(plot_path, 'rb') as f:
            plot_base64 = base64.b64encode(f.read()).decode()
        
        # Convert mask to base64
        mask_base64 = mask_to_base64(mask)
        
        # Prepare metadata
        metadata = {
            'sampling_rate': FREQUENCY,
            'duration': len(signal) / FREQUENCY,
            'signal_length': len(signal),
            'was_grayscaled': was_converted,
            'mV_per_pixel': calibration['mV_per_pixel'],
            'sec_per_pixel': calibration['sec_per_pixel'],
            'y_shift_ratio': calibration['y_shift_ratio']
        }
        
        if lead_name:
            metadata['lead_name'] = lead_name
        
        # Save metadata
        with open(os.path.join(result_folder, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Cleanup temporary files
        os.remove(upload_path)
        os.remove(input_path)
        if os.path.exists(mask_output_path):
            os.remove(mask_output_path)
        
        return jsonify({
            'success': True,
            'signal': signal,
            'metadata': metadata,
            'plot': plot_base64,
            'mask_preview': mask_base64
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    model_path = os.path.join(os.environ.get('nnUNet_results', ''), 'Dataset001_ECG')
    return jsonify({
        'status': 'healthy',
        'model_available': os.path.exists(model_path)
    })


if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=False)