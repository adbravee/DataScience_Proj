#!/bin/bash

# Startup Fix & Run Script

cd /workspaces/DataScience_Proj

echo "🚀 Smart Classroom Attendance System - STARTUP"
echo "================================================"
echo ""

# Verify all packages
echo "📦 Verifying packages..."
source venv/bin/activate 2>/dev/null

python3 -c "
import sys
packages = ['streamlit', 'cv2', 'tensorflow', 'pandas', 'numpy']
missing = []
for pkg in packages:
    try:
        __import__(pkg)
        print(f'  ✅ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'  ❌ {pkg}')

if missing:
    print(f'\n❌ Missing packages: {', '.join(missing)}')
    sys.exit(1)
" || exit 1

echo ""
echo "✅ All packages ready!"
echo ""

# Run the app
echo "🎓 Starting Streamlit app..."
echo "Access it at: http://localhost:8501"
echo ""
echo "NOTE: First run will download the AI model (~350MB)"
echo "Wait for 'You can now view your Streamlit app' message"
echo ""

streamlit run App.py --logger.level=warning
