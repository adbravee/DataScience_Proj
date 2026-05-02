# 🚀 Complete Setup Guide - Smart Classroom Attendance System

## ⚡ FASTEST Setup (Recommended - Run NOW!)

Copy and paste these exact commands into your terminal:

```bash
cd /workspaces/DataScience_Proj
bash quick_install.sh
source venv/bin/activate
streamlit run App.py
```

That's it! The app opens at `http://localhost:8501`

---

## 🔧 If quick_install.sh doesn't work

### Step 1: Install System Libraries (one-time, requires sudo)
```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgfortran5 build-essential
```

### Step 2: Activate Virtual Environment
```bash
cd /workspaces/DataScience_Proj
source venv/bin/activate
```

### Step 3: Install Python Packages
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Run the App
```bash
streamlit run App.py
```

---

## 📋 One-Line Installation (If Comfortable)

Just run this single command in the project directory:
```bash
bash quick_install.sh && source venv/bin/activate && streamlit run App.py
```

---

## ❌ Troubleshooting

| Error | Solution |
|-------|----------|
| `libGL.so.1: cannot open shared object file` | `sudo apt-get install libgl1 libglib2.0-0` |
| `No module named 'cv2'` | `pip install opencv-python --no-cache-dir` |
| `No module named 'tensorflow'` | `pip install tensorflow` |
| `streamlit: command not found` | `source venv/bin/activate` (make sure venv is active) |
| `Permission denied` (with apt-get) | Add `sudo` at the start: `sudo apt-get ...` |

### Diagnostic Check
Run this to see what's missing:
```bash
bash diagnose.sh
```

---

## 🎓 How to Use After Installation

1. **Enroll Students:**
   - Click "Enroll Students" tab
   - Upload student photos
   - Click "Enroll" - system learns their faces

2. **Mark Attendance:**
   - Click "Mark Attendance" tab
   - Take a classroom group photo
   - System detects all faces and matches them
   - Shows attendance results

3. **View Records:**
   - Click "View/Download Attendance"
   - Download Excel file with all attendance records

---

## 📐 System Requirements

- **OS:** Linux (Ubuntu 20.04+)
- **Python:** 3.10, 3.11, or 3.12
- **RAM:** 1GB minimum, 2GB+ recommended
- **Disk:** 500MB for dependencies

---

## 🧠 Model Info

- **Face Detector:** MTCNN (more accurate than traditional methods)
- **Face Recognition:** ArcFace CNN via DeepFace (99.4% LFW benchmark)
- **Matching:** Cosine similarity with adjustable threshold
- **Model Size:** ~350MB (downloaded on first run)

---

## ✅ Verification Commands

After installation, verify everything works:
```bash
source venv/bin/activate
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
python -c "import streamlit; print('Streamlit OK')"
```

---

Still having issues? Run:
```bash
bash diagnose.sh
```

This will identify exactly what's missing.
