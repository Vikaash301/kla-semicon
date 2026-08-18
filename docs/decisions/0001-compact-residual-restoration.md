# ADR-001: Use compact bicubic-anchored residual restoration

## Status

Accepted for the five-hour submission build.

## Context

The released data contains 3,200 paired float32 samples at `128x128 -> 256x256`. Quality, OOD behavior, inference speed, and a runnable evaluator all matter. Input values may exceed the target range, while targets are normalized to `[0,1]`.

## Decision

Use a compact low-resolution gated residual trunk with late PixelShuffle upsampling and an exact bicubic skip. Train with reconstruction and edge losses only. Select the checkpoint by untouched full-image validation PSNR and report PSNR, SSIM, LPIPS, model size, and latency.

## Alternatives considered

- Lightweight SwinIR: strong public 2x weights, but greater integration and kernel-launch complexity; reconsider only if the compact model plateaus materially below it.
- Real-ESRGAN: mature blind-SR pipeline, but GAN texture synthesis risks inventing semiconductor detail and reducing PSNR/SSIM.
- Full NAFNet encoder-decoder: efficient for same-resolution restoration, but full-resolution processing after upsampling spends compute before the judging latency is known.

## Consequences

The submitted graph is small, dependency-light, and starts from a safe bicubic identity. It trades transformer-scale capacity for faster training, simpler reproduction, and a lower hallucination risk. Hidden-test superiority remains an empirical question, not a claimed property.
