#!/bin/bash

# Quick Install Script (No sudo required)
# Use this if system libraries are already installed

set -e  # Exit on any error

cd /workspaces/DataScience_Proj

echo "🚀 Quick Setup - Smart Classroom Attendance System"
echo ""

# Step 1: Create venv if needed
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Step 2: Activate venv
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Step 3: Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet 2>&1 | grep -v "already satisfied" || true

# Step 4: Install packages one by one with error handling
echo "📦 Installing dependencies..."

# Core packages
echo "  → Installing streamlit..."
pip install streamlit>=1.32.0 --quiet

echo "  → Installing OpenCV..."
pip install opencv-python>=4.8.0 --quiet --no-cache-dir

echo "  → Installing numpy..."
pip install numpy>=1.24.0 --quiet

echo "  → Installing pandas..."
pip install pandas>=2.2.0 --quiet

echo "  → Installing scipy..."
pip install scipy>=1.10.0 --quiet

echo "  → Installing Pillow..."
pip install Pillow>=10.0.0 --quiet

echo "  → Installing TensorFlow..."
pip install tensorflow>=2.16.0 --quiet

echo "  → Installing TF-Keras..."
pip install tf-keras>=2.16.0 --quiet

echo "  → Installing DeepFace..."
pip install deepface>=0.0.91 --quiet

echo "  → Installing face detection..."
pip install mtcnn>=0.1.1 retina-face>=0.0.14 --quiet

echo "  → Installing Excel support..."
pip install openpyxl>=3.1.2 --quiet

echo ""
echo "✅ Installation complete!"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "import cv2; print('  ✅ cv2 (OpenCV) OK')" || echo "  ❌ cv2 FAILED"
python3 -c "import streamlit; print('  ✅ streamlit OK')" || echo "  ❌ streamlit FAILED"
python3 -c "import tensorflow; print('  ✅ tensorflow OK')" || echo "  ❌ tensorflow FAILED"

echo ""
echo "🎓 To run the application:"
echo "   source venv/bin/activate"
echo "   streamlit run App.py"
echo ""
