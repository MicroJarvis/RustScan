#!/bin/bash
# Spann3R Setup Script
# Run this when you have stable network connection to github.com
#
# Usage: bash scripts/setup_spann3r.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SPAN3R_DIR="$PROJECT_DIR/spann3r"
CHECKPOINTS_DIR="$SPAN3R_DIR/checkpoints"

echo "=== Spann3R Setup ==="
echo "Project dir: $PROJECT_DIR"

# 1. Clone Spann3R
if [ ! -d "$SPAN3R_DIR" ]; then
    echo "[1/5] Cloning Spann3R..."
    git clone --depth 1 https://github.com/HengyiWang/spann3r.git "$SPAN3R_DIR"
else
    echo "[1/5] Spann3R already cloned, skipping."
fi

# 2. Download DUSt3R checkpoint
mkdir -p "$CHECKPOINTS_DIR"
DUST3R_CKPT="$CHECKPOINTS_DIR/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
if [ ! -f "$DUST3R_CKPT" ]; then
    echo "[2/5] Downloading DUSt3R checkpoint (~1.2GB)..."
    wget -q --show-progress -O "$DUST3R_CKPT" \
        "https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
else
    echo "[2/5] DUSt3R checkpoint exists, skipping."
fi

# 3. Download Spann3R v1.01 checkpoint
SPAN3R_CKPT="$CHECKPOINTS_DIR/spann3r_v1.01.pth"
if [ ! -f "$SPAN3R_CKPT" ]; then
    echo "[3/5] Downloading Spann3R v1.01 checkpoint..."
    echo "      Please download manually from:"
    echo "      https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy"
    echo "      and place it at: $SPAN3R_CKPT"
    echo ""
    read -p "Press Enter after downloading (or Ctrl+C to skip)..."
else
    echo "[3/5] Spann3R checkpoint exists, skipping."
fi

# 4. Install Python dependencies
echo "[4/5] Installing Python dependencies..."
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
if [ ! -f "$VENV_PYTHON" ]; then
    echo "ERROR: venv not found at $PROJECT_DIR/venv"
    echo "Run: /opt/homebrew/bin/python3.12 -m venv $PROJECT_DIR/venv"
    exit 1
fi

"$VENV_PYTHON" -m pip install -q -r "$SPAN3R_DIR/requirements.txt" 2>&1 | grep -v "already satisfied" || true

# 5. Compile CUDA kernels for RoPE (optional, CPU fallback available)
echo "[5/5] Compiling RoPE CUDA kernels..."
cd "$SPAN3R_DIR/croco/models/curope/"
"$VENV_PYTHON" setup.py build_ext --inplace 2>&1 || echo "Warning: CUDA kernel compilation failed (CPU fallback will be used)"
cd "$PROJECT_DIR"

# 6. Download example data
EXAMPLES_DIR="$SPAN3R_DIR/examples"
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo "[+] Downloading example data..."
    mkdir -p "$EXAMPLES_DIR"
    echo "    Please download example data from:"
    echo "    https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy"
    echo "    and unzip as: $EXAMPLES_DIR/"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Run Spann3R demo:"
echo "     cd $SPAN3R_DIR && $VENV_PYTHON demo.py --demo_path ./examples/s00567 --kf_every 10"
echo ""
echo "  2. Export to ONNX:"
echo "     $VENV_PYTHON $PROJECT_DIR/scripts/export_onnx.py \\"
echo "         --checkpoint $SPAN3R_CKPT \\"
echo "         --dust3r_checkpoint $DUST3R_CKPT \\"
echo "         --output $PROJECT_DIR/models/"
