#!/bin/bash

# Diagnostic Script - Smart Classroom Attendance System

echo "🔍 Running diagnostics for Smart Classroom Attendance System..."
echo ""

# Check Python version
echo "📌 Python Version:"
python3 --version
echo ""

# Check if venv exists
echo "📌 Virtual Environment:"
if [ -d "venv" ]; then
    echo "✅ venv directory found"
else
    echo "❌ venv directory NOT found - run: python3 -m venv venv"
fi
echo ""

# Check if venv is activated
echo "📌 Virtual Environment Activation Status:"
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  venv is NOT activated - run: source venv/bin/activate"
else
    echo "✅ venv is activated: $VIRTUAL_ENV"
fi
echo ""

# Check pip
echo "📌 Pip Version:"
pip --version
echo ""

# Check system libraries
echo "📌 System Libraries (OpenCV dependencies):"
if ldconfig -p | grep libGL.so.1 > /dev/null; then
    echo "✅ libGL.so.1 found"
else
    echo "❌ libGL.so.1 NOT found - install with: sudo apt-get install libgl1"
fi

if ldconfig -p | grep libglib-2.0 > /dev/null; then
    echo "✅ libglib-2.0 found"
else
    echo "❌ libglib-2.0 NOT found - install with: sudo apt-get install libglib2.0-0"
fi
echo ""

# Check Python packages
echo "📌 Python Packages:"
pip list | grep -E "streamlit|opencv|tensorflow|deepface|numpy|pandas"
echo ""

# Try importing key packages
echo "📌 Testing Package Imports:"
python3 -c "import streamlit; print('✅ streamlit OK')" 2>&1 || echo "❌ streamlit FAILED"
python3 -c "import cv2; print('✅ cv2 OK')" 2>&1 || echo "❌ cv2 FAILED"
python3 -c "import tensorflow; print('✅ tensorflow OK')" 2>&1 || echo "❌ tensorflow FAILED"
python3 -c "import numpy; print('✅ numpy OK')" 2>&1 || echo "❌ numpy FAILED"
python3 -c "import pandas; print('✅ pandas OK')" 2>&1 || echo "❌ pandas FAILED"
echo ""

# Check App.py
echo "📌 App.py Syntax Check:"
python3 -m py_compile App.py 2>&1 && echo "✅ App.py syntax OK" || echo "❌ App.py has syntax errors"
echo ""

echo "📋 Diagnostic complete!"
echo ""
echo "If you see ❌ marks, refer to SETUP_GUIDE.md for fixes."
