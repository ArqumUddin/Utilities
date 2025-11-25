#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Create Conda Environment
echo "Creating conda environment..."
conda env create -f environment.yml

# Step 2: Install BasicSR
echo "Installing BasicSR..."
cd "$SCRIPT_DIR/.."
git clone https://github.com/ArqumUddin/BasicSR.git
cd BasicSR
eval "$(conda shell.bash hook)"
conda activate utilities
pip install -e .

# Step 3: Install vision_utils
echo "Installing vision_utils..."
cd "$SCRIPT_DIR"
pip install -e .

echo "Installation complete! Run 'conda activate utilities' to use."
