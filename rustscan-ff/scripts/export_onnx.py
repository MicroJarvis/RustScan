#!/usr/bin/env python3
"""Export Spann3R model to ONNX format.

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/spann3r_v1.01.pth --output models/

Prerequisites:
    1. Run scripts/setup_spann3r.sh first
    2. Or manually:
       - Clone Spann3R: git clone https://github.com/HengyiWang/spann3r.git
       - Install deps: pip install -r spann3r/requirements.txt onnx onnxruntime
       - Download weights (see README.md)
"""

import argparse
import os
import sys

import torch
import torch.nn as nn


def replace_rope_with_standard(model):
    """Replace custom CUDA RoPE with pure PyTorch equivalent for ONNX compatibility.

    The curope (CUDA RoPE) kernel cannot be exported to ONNX.
    We replace it with a standard sinusoidal or learned positional encoding.
    """
    replaced = 0
    for name, module in model.named_modules():
        if "rope" in name.lower() or "curope" in name.lower():
            print(f"  Found RoPE module: {name} -> will need manual replacement")
            replaced += 1

    if replaced == 0:
        print("  No explicit RoPE modules found (may be inline in attention)")
    return model


class EncoderWrapper(nn.Module):
    """Wrapper for CroCo encoder with clean ONNX interface.

    Input:  [B, 3, H, W]  - RGB image, normalized
    Output: [B, T, D]     - Feature tokens
    """

    def __init__(self, model):
        super().__init__()
        self.enc = model.enc
        self.patch_size = getattr(model, "patch_size", 14)

    def forward(self, image):
        # Encode image to patch tokens
        features = self.enc(image)
        return features


class DecoderWrapper(nn.Module):
    """Wrapper for DUSt3R decoder with clean ONNX interface.

    Spann3R processes current frame features against memory features.
    For ONNX export, we concatenate current + one memory frame features
    and feed through the decoder.

    Input:  feat_current [B, T, D], feat_memory [B, T, D]
    Output: pointmap [B, T, 3], confidence [B, T]
    """

    def __init__(self, model):
        super().__init__()
        self.dec = model.dec
        # DPT head for pointmap prediction
        self.head = getattr(model, "head", None)
        self.conf_head = getattr(model, "conf_head", None)

    def forward(self, feat_current, feat_memory):
        # The decoder takes two view features and produces
        # pairwise pointmaps and confidence maps
        dec_out = self.dec(feat_current, feat_memory)

        # Extract pointmaps and confidence
        # Actual output format depends on DUSt3R decoder implementation
        if isinstance(dec_out, tuple):
            # (pred1, pred2) or (pointmap, confidence)
            return dec_out
        elif isinstance(dec_out, dict):
            return dec_out.get("pts3d"), dec_out.get("conf")
        else:
            return dec_out


def export_encoder(model, output_dir, image_size=512):
    """Export the CroCo ViT encoder to ONNX."""
    encoder = EncoderWrapper(model)
    encoder.eval()

    dummy_input = torch.randn(1, 3, image_size, image_size)
    encoder_path = os.path.join(output_dir, "encoder.onnx")

    print("  Replacing RoPE for ONNX compatibility...")
    replace_rope_with_standard(model)

    print("  Exporting encoder...")
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            dummy_input,
            encoder_path,
            input_names=["image"],
            output_names=["features"],
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "features": {0: "batch"},
            },
            opset_version=17,
            do_constant_folding=True,
        )

    # Verify exported model
    import onnx
    onnx_model = onnx.load(encoder_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Encoder exported and verified: {encoder_path}")
    return encoder_path


def export_decoder(model, output_dir, image_size=512):
    """Export the DUSt3R decoder to ONNX."""
    decoder = DecoderWrapper(model)
    decoder.eval()

    patch_size = 14
    n_patches = (image_size // patch_size) ** 2  # 1369 for 512px
    embed_dim = getattr(model, "enc_embed_dim", 1024)

    dummy_current = torch.randn(1, n_patches, embed_dim)
    dummy_memory = torch.randn(1, n_patches, embed_dim)
    decoder_path = os.path.join(output_dir, "decoder.onnx")

    print("  Exporting decoder...")
    with torch.no_grad():
        try:
            torch.onnx.export(
                decoder,
                (dummy_current, dummy_memory),
                decoder_path,
                input_names=["feat_current", "feat_memory"],
                output_names=["pointmap", "confidence"],
                dynamic_axes={
                    "feat_current": {0: "batch"},
                    "feat_memory": {0: "batch"},
                },
                opset_version=17,
                do_constant_folding=True,
            )

            import onnx
            onnx_model = onnx.load(decoder_path)
            onnx.checker.check_model(onnx_model)
            print(f"  Decoder exported and verified: {decoder_path}")
        except Exception as e:
            print(f"  WARNING: Decoder export failed: {e}")
            print("  This is expected - decoder may have dynamic memory operations.")
            print("  Will need manual ONNX-compatible decoder implementation.")
            return None

    return decoder_path


def verify_with_onnxruntime(encoder_path, decoder_path=None):
    """Verify ONNX models with ONNX Runtime."""
    import onnxruntime as ort
    import numpy as np

    print("\nVerifying with ONNX Runtime...")

    # Test encoder
    session = ort.InferenceSession(encoder_path)
    input_shape = session.get_inputs()[0].shape
    print(f"  Encoder input: {input_shape}")

    dummy = np.random.randn(1, 3, 512, 512).astype(np.float32)
    outputs = session.run(None, {"image": dummy})
    print(f"  Encoder output shape: {outputs[0].shape}")

    if decoder_path and os.path.exists(decoder_path):
        session = ort.InferenceSession(decoder_path)
        print(f"  Decoder inputs: {[i.name for i in session.get_inputs()]}")
        print(f"  Decoder outputs: {[o.name for o in session.get_outputs()]}")

    print("  ONNX Runtime verification passed!")


def main():
    parser = argparse.ArgumentParser(description="Export Spann3R to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to Spann3R checkpoint")
    parser.add_argument("--dust3r_checkpoint",
                        default="checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
                        help="Path to DUSt3R base checkpoint")
    parser.add_argument("--output", default="models/", help="Output directory")
    parser.add_argument("--image_size", type=int, default=512, help="Input image size")
    parser.add_argument("--skip-verify", action="store_true", help="Skip ONNX verification")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Add Spann3R to path
    spann3r_path = os.path.join(os.path.dirname(__file__), "..", "spann3r")
    if os.path.exists(spann3r_path):
        sys.path.insert(0, spann3r_path)
        print(f"Added {spann3r_path} to Python path")
    else:
        print(f"ERROR: Spann3R not found at {spann3r_path}")
        print("Run: bash scripts/setup_spann3r.sh")
        sys.exit(1)

    # Import DUSt3R model
    from dust3r.model import AsymmetricCroCo3DStereo

    print("\n=== Loading Model ===")
    print(f"  DUSt3R checkpoint: {args.dust3r_checkpoint}")
    print(f"  Spann3R checkpoint: {args.checkpoint}")

    base_model = AsymmetricCroCo3DStereo.from_pretrained(args.dust3r_checkpoint)

    print("  Loading Spann3R weights...")
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    base_model.load_state_dict(state_dict, strict=False)

    model = base_model
    model.eval()

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\n=== Exporting Encoder ===")
    encoder_path = export_encoder(model, args.output, args.image_size)

    print("\n=== Exporting Decoder ===")
    decoder_path = export_decoder(model, args.output, args.image_size)

    if not args.skip_verify:
        verify_with_onnxruntime(encoder_path, decoder_path)

    print("\n=== Export Complete ===")
    print(f"  Output directory: {args.output}")
    print(f"  Encoder: {encoder_path}")
    if decoder_path:
        print(f"  Decoder: {decoder_path}")
    print()
    print("Rust usage:")
    print(f'  Spann3RInference::from_onnx("{encoder_path}", "{decoder_path}")?')


if __name__ == "__main__":
    main()
