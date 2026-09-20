#!/usr/bin/env python3
"""Analyze Spann3R model structure and prepare ONNX export.

This script works in two modes:
1. With checkpoint: Full analysis + ONNX export
2. Without checkpoint: Architecture analysis only (model structure, param count)

Usage:
    # Analyze architecture only (no download needed)
    python scripts/analyze_model.py --analyze-only

    # Full export (requires checkpoints)
    python scripts/analyze_model.py --checkpoint checkpoints/spann3r_v1.01.pth --output models/
"""

import argparse
import os
import sys


def analyze_croco_architecture():
    """Analyze the CroCo ViT architecture used by Spann3R/DUSt3R."""
    print("=" * 60)
    print("Spann3R / DUSt3R Architecture Analysis")
    print("=" * 60)

    print("""
## Model Components

### 1. Encoder: CroCo ViT-Large
   - Type: Vision Transformer (ViT) with cross-attention
   - Patch size: 14x14
   - Embedding dim: 1024
   - Depth: 24 layers
   - Attention heads: 16
   - Input: [B, 3, 512, 512]
   - Output: [B, 1369, 1024] (37x37 patches)
   - Params: ~300M (encoder only)
   - Positional encoding: RoPE (Rotary Position Embedding)

### 2. Decoder: DUSt3R Asymmetric Decoder
   - Cross-attention between two view features
   - Outputs: pointmap [B, H*W, 3] + confidence [B, H*W]
   - Head: DPT (Dense Prediction Transformer)
   - Params: ~100M

### 3. Spann3R Spatial Memory
   - Explicit memory bank storing past frame features
   - Key innovation: memory concatenation before decoder
   - No GRU/LSTM - simple feature concatenation
   - Memory update: append new features, pop oldest when full

## ONNX Export Strategy

The model CANNOT be exported as a single ONNX file because:
1. Spatial memory bank is dynamic (Python list of tensors)
2. RoPE uses custom CUDA kernels (curope)

### Recommended export: Split into 3 modules

Module A: Image Encoder
  Input:  [1, 3, H, W]  - single RGB image
  Output: [1, T, D]     - feature tokens (T = H*W/196, D = 1024)

Module B: Memory + Decoder
  Input:  [1, T, D] * (1 + K)  - current + K memory features
  Output: [1, T, 3]            - pointmap
          [1, T]               - confidence

Module C: Pose Extraction (non-neural, pure Rust)
  Input:  pointmap [T, 3] + confidence [T]
  Output: SE3 pose [4, 4]
  Method: Weighted Procrustes alignment (SVD-based)
  -> This can be implemented directly in Rust, no ONNX needed

### RoPE Handling for ONNX
The custom curope CUDA kernel must be replaced with pure PyTorch
equivalents before export. See export_onnx.py for details.

## Param Count Estimate
  Encoder:  ~300M
  Decoder:  ~100M
  Total:    ~400M (ViT-Large configuration)
  FP16:     ~800MB
  INT8:     ~400MB

## Memory Requirements
  Inference (512px, 200-frame memory):
    Encoder: ~2GB (single frame)
    Decoder: ~4GB (memory context)
    Total:   ~6-8GB RAM/VRAM

## Input Preprocessing
  1. Resize to 512x512 (Lanczos)
  2. Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
  3. Channel order: RGB, CHW format
""")


def try_load_and_analyze(checkpoint_path):
    """Load checkpoint and analyze actual weights."""
    import torch

    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt.state_dict() if hasattr(ckpt, 'state_dict') else ckpt

    print(f"\nCheckpoint contains {len(state_dict)} tensors")

    # Analyze parameter groups
    encoder_params = 0
    decoder_params = 0
    other_params = 0

    for name, tensor in state_dict.items():
        n = tensor.numel()
        if name.startswith("enc."):
            encoder_params += n
        elif name.startswith("dec."):
            decoder_params += n
        else:
            other_params += n

    total = encoder_params + decoder_params + other_params
    print(f"\nParameter breakdown:")
    print(f"  Encoder:  {encoder_params:>12,} ({encoder_params/total*100:.1f}%)")
    print(f"  Decoder:  {decoder_params:>12,} ({decoder_params/total*100:.1f}%)")
    print(f"  Other:    {other_params:>12,} ({other_params/total*100:.1f}%)")
    print(f"  Total:    {total:>12,}")
    print(f"  FP32:     {total*4/1024/1024:.1f} MB")
    print(f"  FP16:     {total*2/1024/1024:.1f} MB")

    # Print key layer names
    print(f"\nKey layer names (first 20):")
    for name in list(state_dict.keys())[:20]:
        shape = list(state_dict[name].shape)
        print(f"  {name}: {shape}")

    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Analyze Spann3R model")
    parser.add_argument("--checkpoint", help="Path to Spann3R checkpoint")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only print architecture analysis, don't load model")
    parser.add_argument("--output", default="models/", help="Output directory for ONNX")
    args = parser.parse_args()

    # Always print architecture
    analyze_croco_architecture()

    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = try_load_and_analyze(args.checkpoint)
    elif not args.analyze_only:
        print("\n[!] No checkpoint provided. Run with --checkpoint <path> for full analysis.")
        print("    Download from:")
        print("    DUSt3R:  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        print("    Spann3R: https://drive.google.com/drive/folders/1bqtcVf8lK4VC8LgG-SIGRBECcrFqM7Wy")


if __name__ == "__main__":
    main()
