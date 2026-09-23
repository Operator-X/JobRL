#!/usr/bin/env bash
# ==============================================================================
# JobRL Visual Scenario Designer Launcher (macOS)
# Double-click this file in Finder to start the Visual Scenario Designer.
# ==============================================================================

# Move to the script's directory
cd "$(dirname "$0")" || exit 1

echo "============================================================"
echo "         🏭 JobRL Visual Scenario Designer (macOS)"
echo "============================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 was not found on your system."
    echo "Please install Python 3 (from https://www.python.org/downloads/) and try again."
    read -n 1 -s -r -p "Press any key to close..."
    exit 1
fi

# Step 1: Check or create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 [1/2] Creating Python virtual environment (venv)..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment."
        read -n 1 -s -r -p "Press any key to close..."
        exit 1
    fi
fi

# Activate virtual environment
source venv/bin/activate

# Step 2: Check if Streamlit and dependencies are installed
python -c "import streamlit" &> /dev/null
if [ $? -ne 0 ]; then
    echo "⚙️ [2/2] First-time setup: Installing required libraries..."
    echo "This may take about 30-60 seconds. Please wait..."
    pip install --upgrade pip
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Dependency installation encountered an error."
        read -n 1 -s -r -p "Press any key to close..."
        exit 1
    fi
fi

echo "============================================================"
echo "🚀 Launching Visual Designer in your default browser..."
echo "Press Ctrl+C or close this window when you are finished."
echo "============================================================"

# Launch Streamlit web app
streamlit run app.py --server.headless=false