# 🫀 ECG Digitization - Quick Start Guide

Welcome! This guide will help you get the ECG Digitization tool up and running in minutes.

---

## ✅ What You'll Need

- **Python 3.8+** installed on your computer
- **macOS** (if you're on Windows/Linux, some commands might differ slightly)
- **Terminal/Command Line** access

---

## 🚀 Setup Instructions (First Time Only)

### Step 1: Open Terminal

1. Press `Cmd + Space` and type "Terminal"
2. Press Enter to open Terminal

### Step 2: Navigate to the Project

```bash
cd "/path/to/your/kaggle copy 2"
```

**Replace `/path/to/your/` with the actual folder location!**

For example:

```bash
cd "/Users/yourname/Downloads/kaggle copy 2"
```

### Step 3: Install Everything

Copy and paste this entire command into Terminal:

```bash
bash setup.sh && pip install flask flask-cors
```

Press Enter and wait for it to finish (may take a few minutes).

✅ You only need to do this ONCE!

---

## 🌐 Starting the Web Application

Every time you want to use the tool:

### Copy and paste this command:

```bash
cd "/path/to/your/kaggle copy 2" && export nnUNet_results="$(pwd)/working/nnUNet_results" && export nnUNet_raw="$(pwd)/working/nnUNet_raw" && export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed" && python app.py
```

**Remember to replace `/path/to/your/` with your actual path!**

You should see:

```
════════════════════════════════════════════════════════════
  ECG Digitization Web Application
════════════════════════════════════════════════════════════

  Open http://localhost:8000 in your browser

════════════════════════════════════════════════════════════
```

---

## 🎯 Using the Application

1. **Open your web browser**
   - Go to: `http://localhost:8000`

2. **Upload your ECG image**
   - Drag and drop the image onto the page
   - OR click the upload area to browse files

3. **Process it**
   - Click the "⚡ Process ECG" button
   - Wait 20-60 seconds (you'll see a loading spinner)

4. **Download your result**
   - The digitized black & white ECG will appear
   - Click "⬇️ Download Digitized ECG" to save it

5. **Process another?**
   - Click "↩️ Process Another ECG" button

---

## 🛑 Stopping the Server

When you're done:

- Go back to Terminal
- Press `Ctrl + C` to stop the server

---

## 📁 Supported File Formats

- PNG
- JPG/JPEG
- BMP
- TIFF

---

## ❓ Troubleshooting

### "Port already in use" error?

```bash
lsof -ti:8000 | xargs kill -9 2>/dev/null
```

Then try starting the server again.

### "Command not found: python"?

Try using `python3` instead:

```bash
python3 app.py
```

### Need help?

Check the full README.md for more detailed instructions.

---

## 📝 Quick Reference

**Start the server:**

```bash
cd "/path/to/kaggle copy 2"
export nnUNet_results="$(pwd)/working/nnUNet_results"
export nnUNet_raw="$(pwd)/working/nnUNet_raw"
export nnUNet_preprocessed="$(pwd)/working/nnUNet_preprocessed"
python app.py
```

**Then open:** http://localhost:8000

**Stop the server:** Press `Ctrl + C` in Terminal

---

Happy digitizing! 🎉
