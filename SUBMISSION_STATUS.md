# Submission status

## Proven locally

- Submitted FP32 checkpoint: `checkpoints/edgerestore_v2.pt`, SHA-256 `69f6cc38473fb9a9082491cbba93bfdf2e25e8d0a1829a5b88b0dbd5c0cc8526`, 27,876,717 bytes, 6,939,289 parameters. These are the EMA weights at iteration 22000 of the `rcan96` run.
- Held-out 320-image metrics: **29.0989 PSNR, 0.77843 SSIM, 0.27595 LPIPS, 0.93959 MS-SSIM**.
- Bicubic on identical images/settings: 22.9877 PSNR, 0.51930 SSIM, 0.44353 LPIPS, 0.79430 MS-SSIM.
- Previous submitted model (193,412 params) on the identical split: 27.7637 PSNR, 0.73490 SSIM, 0.32771 LPIPS. EdgeRestore v2 beats it on every metric (+1.3352 dB PSNR, +0.04353 SSIM, -0.05176 LPIPS).
- The slim exported checkpoint reproduces the training-run score exactly (29.09889 in both), so the export lost nothing.
- Per-image, v2 loses to bicubic on 0/320 PSNR and 0/320 MS-SSIM cases, 4/320 SSIM, 33/320 LPIPS.
- RTX 5070 Ti Laptop GPU: 29.35 ms median FP32 model latency (20 warmup / 100 repeats, batch 1).
- 400/400 official inputs restored through the required `inference.py` entry point: matching output basenames, float32 `(256, 256)` shapes, all finite, all within `[0,1]`. Verified programmatically, 0 violations.
- Full local test suite passes: 34 repo tests plus 8 alignment/leakage/bicubic-identity self-checks.

## Required user completion

- Replace every `[ADD ...]` field in `EdgeRestore_PS01.pptx` with team, college, contact, GitHub, and optional video details.
- **Update the deck's results numbers to the v2 figures above.** The deck currently states the superseded 27.7637 PSNR result.
- Rename the deck to the exact team-name convention.
- Export the edited PPTX to PDF. Automated Office export was denied by the local approval service, so no PDF is claimed.
- Configure and push a public GitHub remote. No remote currently exists; no URL is fabricated.
- Configure Git LFS (or attach the verified local bundle as a release asset) for the 27.9 MB checkpoint and the restored-output folder.
- Rebuild `submission/EdgeRestore_PS01_bundle.zip`; the existing archive contains the superseded model and its outputs.

## Unverified boundaries

- The validation split is filename-random held-out IID, not source-level OOD.
- H100 latency and KLA hidden-test scores are unavailable. No H100 figure is claimed.
- No supplied defect masks or physical calibration exist; defect recall and critical-dimension accuracy are not claimed.
- Some high-frequency stochastic textures remain over-smoothed; the disclosed worst case is `000406` at 17.11 dB (`artifacts/comparisons/000406_comparison.png`).
