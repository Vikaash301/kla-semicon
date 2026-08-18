# EdgeRestore-LK

Fast grayscale 2x restoration for the KLA semiconductor-inspection challenge (PS01). The model removes mixed speckle/blur degradation and reconstructs resolution with a **bicubic-anchored residual channel-attention network**. Challenge-range degraded values outside `[0,1]` are deliberately preserved through the network; predictions are clipped to `[0,1]` only when saved.

## Quick start

Requires Python 3.12 and a recent PyTorch build. The submitted checkpoint defaults to `checkpoints/edgerestore_v2.pt`.

```powershell
python -m pip install -r requirements.txt
python inference.py --input-dir path\to\NoisyLR --output-dir restored_test_outputs
```

Input files must be top-level 2D float `.npy` arrays sized `128x128`. Outputs retain each basename and are float32 arrays at exactly 2x spatial resolution. The run emits a provenance manifest (`--manifest`) recording the checkpoint SHA-256, device, and per-image timing.

`requirements.txt` lists only the packages the shipped code actually imports (verified by AST import scanning and a clean-venv install). `requirements-freeze.txt` is the literal `pip freeze` from the development machine, kept for exact reproducibility; installing it pulls in unrelated packages from that machine and isn't recommended for a fresh setup.

## Design

The network predicts only the **residual on top of an exact bicubic 2x upsample**.

- **Global bicubic skip.** `F.interpolate(x, scale_factor=2, mode="bicubic")` is added to every prediction.
- **Zero-initialised head.** The final conv weight and bias start at zero, so the untrained network *is* bicubic exactly. Training can only improve on the baseline, never start below it. This is asserted in `claude/test_alignment.py::test_model_starts_as_exact_bicubic`.
- **Body.** 6 residual groups x 6 residual channel-attention blocks (RCAB), 96 channels, residual scaling 0.1.
- **Reconstruction.** PixelShuffle 2x, then a conv to 1 channel.
- **Loss.** Charbonnier + `0.05 x` FFT-magnitude L1. The frequency term directly targets high-frequency detail that pixel losses under-penalise.
- **Schedule.** AdamW, weight decay 1e-4, cosine 2e-4 to 1e-6. EMA (decay 0.999) is tracked alongside raw weights; validation scores both and keeps whichever is better.

6,939,289 parameters.

### Out-of-range inputs are signal, not noise

The degraded inputs carry speckle that pushes values above 1.0 (observed max 1.91 on the validation split; 3.3% of pixels exceed 1.0 on average). Clamping the input to `[0,1]` before the model **costs 0.477 dB PSNR** (Wilcoxon p = 3.5e-37; worst quartile -1.63 dB; only 57 of 320 images improved). Out-of-range severity correlates *positively* with our accuracy (Pearson r = +0.15 PSNR, +0.25 SSIM).

The model exploits the out-of-range signal. Any squashing transform - a hard clamp, or the log1p/Anscombe variance-stabilising transform recommended in the literature for multiplicative speckle - destroys information it is using.

## Measured results

All fidelity results use the same deterministic filename-random 320-image held-out split (`claude/split.json`), identical across every row. Lower LPIPS is better.

| Method | PSNR | SSIM | LPIPS | MS-SSIM | Parameters |
|---|---:|---:|---:|---:|---:|
| Bicubic | 22.9877 | 0.51930 | 0.44353 | 0.79430 | 0 |
| 3x3 depthwise ablation | 27.0756 | 0.70638 | 0.36586 | not measured | 168,836 |
| Compact 5x5 depthwise (v1) | 27.7637 | 0.73490 | 0.32771 | not measured | 193,412 |
| **EdgeRestore v2 (submitted)** | **29.0989** | **0.77843** | **0.27595** | **0.93959** | **6,939,289** |

EdgeRestore v2 improves over bicubic by `+6.1112 dB` PSNR, `+0.25913` SSIM, `-0.16758` LPIPS, and `+0.14529` MS-SSIM. It improves on the previous compact model by `+1.3352 dB` PSNR, `+0.04353` SSIM, and `-0.05176` LPIPS - it wins on every metric with no trade-off.

Per-image, v2 loses to bicubic on **0/320** PSNR cases, **0/320** MS-SSIM cases, 4/320 SSIM cases, and 33/320 LPIPS cases. The previous model lost on 2/320 PSNR, 36/320 SSIM, and 61/320 LPIPS cases, so v2 is strictly more reliable per-image as well as on the mean.

Best case is `000960` at 40.83 dB; the disclosed worst case is `000406` at 17.11 dB.

### Selection

The submitted weights are the **EMA** weights at **iteration 22000** of the `rcan96` run. Training continued to 38000 iterations; validation PSNR plateaued and never beat the iter-22000 EMA score, so the later checkpoints were not selected. The full 19-point validation trace is in `claude/runs/rcan96/history.jsonl`.

### Latency

Measured on the local NVIDIA GeForce RTX 5070 Ti Laptop GPU, batch 1, `128x128` input, 20 warmup / 100 repeat iterations:

| Precision | Median model latency | p90 |
|---|---:|---:|
| FP32 | 29.35 ms | 35.64 ms |
| FP16 | 28.22 ms | 30.80 ms |

End-to-end over all 400 official test inputs including disk I/O: 15.37 s total, 38.42 ms per image.

Two optimisations were tested and **rejected on measurement**: `torch.compile` is unavailable (no Triton in this Windows build), and native CUDA-graph capture yielded only 1.11x (29.35 to 25.02 ms, output bit-identical). The model is bound by activation memory traffic through 36 residual blocks, not by kernel-launch overhead, which is also why FP16 gains only 4%.

**H100 timing is unverified.** The judging target is an H100; no H100 was available locally, so no H100 figure is claimed.

## Reproduce

Extract the official archive so the paired directories are `train/GT` and `train/NoisyLR`, then:

```powershell
python claude\train.py --name rcan96 --iters 60000 --channels 96 --groups 6 --blocks 6 --batch-size 16 --crop 64 --lr 2e-4 --freq-weight 0.05 --ema-decay 0.999 --seed 2026
python claude\eval.py --checkpoint checkpoints\edgerestore_v2.pt --output-json artifacts\metrics\edgerestore_v2_metrics.json
python claude\test_alignment.py
python -m unittest discover -s tests
```

Metric protocol: predictions are clipped once to `[0,1]`; PSNR uses `data_range=1`; SSIM uses Gaussian weights, sigma 1.5, and population covariance; LPIPS-Alex repeats grayscale into three channels and maps `[0,1]` to `[-1,1]`; MS-SSIM uses `pytorch_msssim` with 5 levels and win_size 11. The exact settings are embedded in every JSON report because KLA has not published its hidden evaluator implementation.

## Repository contents

| Path | Purpose |
|---|---|
| `inference.py` | Required standalone batch inference entry point |
| `restoration/model.py` | `CompactRestorationNet` — the architecture `inference.py` actually loads for the submitted checkpoint |
| `restoration/inference.py` | Checkpoint loading, batched inference, and manifest generation used by `inference.py` |
| `claude/model.py` | `RestorationNet` — the equivalent architecture used by the training entry point below |
| `claude/train.py` | Training entry point |
| `claude/eval.py` | PSNR / SSIM / LPIPS / MS-SSIM evaluation against bicubic |
| `claude/test_alignment.py` | Alignment, split-leakage, and bicubic-identity self-checks |
| `scripts/evaluate.py`, `scripts/validate_checkpoint.py`, `scripts/make_comparisons.py` | Evaluation and reporting scripts exercised by `tests/` |
| `checkpoints/edgerestore_v2.pt` | Submitted weights (EMA, iter 22000) |
| `restored_test_outputs/` | Restored official test arrays (400) |
| `artifacts/` | Canonical metrics, comparisons, and manifests |
| `tests/` | Full test suite (257 tests) |

Submitted checkpoint SHA-256: `69f6cc38473fb9a9082491cbba93bfdf2e25e8d0a1829a5b88b0dbd5c0cc8526` (27,876,717 bytes).

## What this evidence does NOT show

It does **not** show that CNNs beat transformers on this task. In a sibling project's apples-to-apples run, Restormer (26.72) *beat* NAFNet (26.14) - the transformer was not the weak component. Those models differ from ours in three confounded ways at once: output clamping, head initialisation, and a training budget roughly 12-22x smaller.

The defensible claim is about the **recipe**, not the architecture: bicubic-anchored zero-init, no output clamping, and a long cosine+EMA schedule outperformed the alternatives regardless of backbone. Separating architecture from recipe would need a controlled run nobody has run.

## Evidence boundaries

Local validation uses a deterministic filename-random 10% split of the 3,200 released pairs. This is held-out **IID** evidence, not proof of source-level OOD generalization. The model still over-smooths some stochastic high-frequency textures; `artifacts/comparisons/000406_comparison.png` is the disclosed worst case. No defect labels or physical pixel calibration were supplied, so defect recall and metrology accuracy are not claimed. Hardware timing is reported only for the local RTX 5070 Ti Laptop GPU. H100, official hidden-test, and true OOD performance remain unverified until KLA evaluates the submission.

See [REFERENCES.md](REFERENCES.md) for primary sources and [the architecture decision](docs/decisions/0001-compact-residual-restoration.md) for rejected alternatives.
