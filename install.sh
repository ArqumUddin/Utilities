#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Step 1: Create Conda Environment
echo "Creating conda environment..."
conda env create -f environment.yml

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Step 2: Install BasicSR
echo "Installing BasicSR..."
cd "$SCRIPT_DIR/.."
git clone https://github.com/ArqumUddin/BasicSR.git
cd BasicSR
conda run -n utilities pip install .

# Step 3: Install dinov3
echo "Installing DINOv3..."
cd "$SCRIPT_DIR/.."
git clone https://github.com/facebookresearch/dinov3.git
cd dinov3
conda run -n utilities pip install .

# Step 4: Install vision_utils
echo "Installing vision_utils..."
cd "$SCRIPT_DIR"
conda run -n utilities pip install -e .

# Step 5: Remove dinov3 and BasicSR directories
cd "$SCRIPT_DIR/.."
rm -rf BasicSR/ dinov3/

echo "Installation complete! Run 'conda activate utilities' to use."
