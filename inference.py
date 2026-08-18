"""Standalone batch inference entry point for Evidence-DAR SEM-SR.

Restores 2D .npy arrays (128x128 or 256x256) to 2x resolution (256x256 or 512x512) float32 arrays clipped to [0.0, 1.0].
Supports arguments:
    --input-dir <DIR> --output-dir <DIR> --checkpoint <PATH> [--device <cuda|cpu>] [--precision <fp32|bf16>] [--benchmark-json <PATH>]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restoration.inference import (
    main,
    run_inference,
    load_model_from_checkpoint,
    batched_d4_tta_inference,
)

__all__ = [
    "main",
    "run_inference",
    "load_model_from_checkpoint",
    "batched_d4_tta_inference",
]

if __name__ == "__main__":
    main()
