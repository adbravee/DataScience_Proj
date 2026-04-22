#!/bin/bash

# Smart Classroom Attendance System - Setup Script
# This script installs all required system and Python dependencies

echo "🚀 Setting up Smart Classroom Attendance System..."
echo ""

# Check for root/sudo for system packages
if [ "$EUID" -eq 0 ]; then 
    # Running as root - install system dependencies
    echo "📚 Installing system libraries (as root)..."
    apt-get update -qq
    apt-get install -y -qq \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgfortran5 \
        python3-dev \
        build-essential
    echo "✅ System libraries installed"
else
    # Not running as root - just warn
    echo "⚠️  To install system libraries (optional), run:"
    echo "   sudo apt-get install -y libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 libgfortran5 build-essential"
    echo ""
fi

echo ""

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating Python virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip, setuptools, wheel
echo "📦 Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel --quiet

# Install Python dependencies
echo "📦 Installing Python dependencies (this may take 5-10 minutes)..."
pip install -r requirements.txt -q

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎓 To run the application:"
echo "   source venv/bin/activate"
echo "   streamlit run App.py"
echo ""
