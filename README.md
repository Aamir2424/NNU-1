# ECG Digitizer

A full-stack ECG digitization system using **nnU-Net** for segmentation and a **React** frontend for visualization and interaction.

---

## 📁 Project Structure

```
ecg-digitizer/
│
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── run_backend.sh
│   ├── run_backend.bat
│   ├── venv/                          # Virtual environment (created locally)
│   └── working/
│       ├── nnUNet_raw/
│       ├── nnUNet_preprocessed/
│       ├── nnUNet_results/
│       │   └── Dataset001_ECG/
│       │       └── nnUNetTrainer__nnUNetPlans__2d/
│       │           ├── fold_0/
│       │           │   └── checkpoint_final.pth
│       │           ├── fold_1/
│       │           │   └── checkpoint_final.pth
│       │           ├── fold_2/
│       │           │   └── checkpoint_final.pth
│       │           ├── fold_3/
│       │           │   └── checkpoint_final.pth
│       │           └── fold_4/
│       │               └── checkpoint_final.pth
│       ├── uploads/                   # Temporary uploads
│       ├── input/                     # Preprocessed inputs
│       ├── output-ensemble/           # Model outputs
│       └── results/                   # Final digitized ECG signals
│
└── frontend/
    ├── public/
    │   ├── index.html
    │   ├── frames/                    # Scroll animation frames
    │   │   ├── frame-001.png
    │   │   ├── frame-002.png
    │   │   ├── ...
    │   │   └── frame-240.png
    │   └── favicon.ico
    ├── src/
    │   ├── App.jsx
    │   ├── App.css
    │   ├── index.jsx
    │   └── index.css
    ├── package.json
    ├── package-lock.json
    └── node_modules/                  # Installed locally (ignored in git)
```

---

## 📄 File Descriptions

### Backend Files

| File | Description |
|------|-------------|
| `app.py` | Flask backend server handling nnU-Net inference, ECG image preprocessing, and signal extraction |
| `requirements.txt` | Python dependencies for backend and nnU-Net inference |
| `run_backend.sh` | Startup script for macOS / Linux |
| `run_backend.bat` | Startup script for Windows |
| `working/` | Workspace containing nnU-Net data, trained models, intermediate files, and results |

### Frontend Files

| File | Description |
|------|-------------|
| `public/index.html` | Landing page with scroll-based animation |
| `public/frames/` | PNG image sequence used for scroll animation (typically 240 frames) |
| `src/App.jsx` | Main React component for ECG upload, processing, and visualization |
| `src/App.css` | Styling for the React application |
| `src/index.jsx` | React entry point |
| `src/index.css` | Global styles |
| `package.json` | Node.js dependencies and scripts |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 14+
- Git

### 1️⃣ Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install git+https://github.com/FelixKrones/nnUNet.git
```

**Copy your trained nnU-Net models into:**
```
backend/working/nnUNet_results/
```

**Start backend server:**
```bash
./run_backend.sh
# OR
python app.py
```

### 2️⃣ Frontend Setup

```bash
cd frontend
npm install
npm start
```

### 3️⃣ Access the Application

- **Frontend (UI):** http://localhost:3000
- **Backend API:** http://localhost:5001

---

## 📝 Important Notes

### Frames Folder
If you don't have all 240 frames, you can:
- Create placeholder images
- Reduce frame count and update the frontend logic
- Disable the scroll animation entirely

### Model Files
All 5 nnU-Net folds must be present for ensemble inference.

### Ignored Files
`venv/`, `node_modules/`, and large nnU-Net artifacts are intentionally excluded from Git.

### Port Conflicts
- **Frontend:** 3000
- **Backend:** 5001

Update ports in configuration files if already in use.

---

## 🛠️ Technologies Used

- **Backend:** Flask, nnU-Net, PyTorch
- **Frontend:** React, HTML5, CSS3
- **Image Processing:** OpenCV, NumPy
- **Deep Learning:** nnU-Net (medical image segmentation)

---

## 📧 Support

For issues or questions, please open an issue in the repository.

---

## 📄 License

This project is licensed under the MIT License.
