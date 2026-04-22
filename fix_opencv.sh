#!/bin/bash

# Force OpenCV Installation Fix

echo "🔧 Fixing OpenCV installation..."
echo ""

cd /workspaces/DataScience_Proj

# Activate venv
source venv/bin/activate

# Remove any corrupted cv2
echo "🗑️  Cleaning up old OpenCV files..."
pip uninstall -y opencv-python 2>/dev/null || true

# Install with no cache and build from source if needed
echo "📦 Installing OpenCV (no cache)..."
pip install --no-cache-dir --force-reinstall opencv-python>=4.8.0

# Verify
echo ""
echo "✅ Verifying OpenCV..."
python3 -c "import cv2; print(f'✅ OpenCV {cv2.__version__} installed successfully')" && \
echo "✅ Success! cv2 is now working" || \
echo "❌ Still failed - try: pip install opencv-contrib-python"

echo ""
echo "Now run:"
echo "  source venv/bin/activate"
echo "  streamlit run App.py"
