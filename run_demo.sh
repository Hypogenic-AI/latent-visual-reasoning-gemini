#!/bin/bash
# run_demo.sh
# Simple script to run the visualization demo

# Ensure script stops on error
set -e

echo "Initializing Demo..."

# Check if venv exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found! Please run 'uv venv' and install dependencies first."
    exit 1
fi

echo "Running GIF Visualization..."
python -m src.visualize_gif

echo "Done! Check results/plots/comparison.gif"
