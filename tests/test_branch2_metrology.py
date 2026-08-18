import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from restoration.directional_metrology import (
    DirectionalMetrologyEnhancer,
    DirectionalSobelUnsharpFilter,
    DiscreteNullSpaceProjector,
    FastGuidedFilter2D,
    compute_comprehensive_metrology_metrics,
    measure_critical_dimension_linewidth,
    measure_edge_acutance,
    measure_visual_noise_floor,
)


def test_discrete_null_space_projector_invariance():
    """Test that DiscreteNullSpaceProjector strictly guarantees ||H(X_cons) - Y||_inf <= 1.19e-7."""
    projector = DiscreteNullSpaceProjector(scale_factor=2)
    torch.manual_seed(42)

    # 1. Random HR prediction and LR anchor
    x_pred = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    y_anchor = torch.rand(2, 1, 32, 32, dtype=torch.float32)

    x_consistent = projector(x_pred, y_anchor)

    # Verify shape
    assert x_consistent.shape == (2, 1, 64, 64)

    # Verify analytical measurement invariant ||H(X_cons) - Y_anchor||_inf <= 1.2e-7
    h_consistent = F.avg_pool2d(x_consistent, kernel_size=2, stride=2)
    max_error = torch.max(torch.abs(h_consistent - y_anchor)).item()
    assert max_error <= 1.2e-7, f"Max error {max_error} exceeds 1.2e-7 ceiling"


def test_fast_guided_filter():
    """Test that FastGuidedFilter2D smooths flat noise while preserving step edges."""
    filter_module = FastGuidedFilter2D(radius=2, eps=1e-3)

    # Create step edge with added noise
    x = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    x[:, :, :, 16:] = 1.0
    noise = torch.randn_like(x) * 0.05
    x_noisy = x + noise

    filtered = filter_module(x_noisy, x_noisy)

    assert filtered.shape == (1, 1, 32, 32)
    # Variance in flat regions should decrease
    var_flat_before = float(torch.var(x_noisy[:, :, :, :12]))
    var_flat_after = float(torch.var(filtered[:, :, :, :12]))
    assert var_flat_after < var_flat_before


def test_directional_sobel_unsharp_filter():
    """Test that DirectionalSobelUnsharpFilter boosts edge sharpness."""
    unsharp = DirectionalSobelUnsharpFilter(strength=0.30, edge_threshold=0.05)

    # Create a line pattern
    x = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    x[:, :, 14:18, :] = 0.8

    sharp, edge_mask, grad_mag = unsharp(x)

    assert sharp.shape == (1, 1, 32, 32)
    assert edge_mask.shape == (1, 1, 32, 32)
    assert grad_mag.shape == (1, 1, 32, 32)

    # Edge mask should be close to 1 near the line boundaries (rows 14 and 17)
    assert float(edge_mask[:, :, 14, 16].item()) > float(edge_mask[:, :, 2, 16].item())


def test_directional_metrology_enhancer_forward():
    """Test full DirectionalMetrologyEnhancer pipeline."""
    enhancer = DirectionalMetrologyEnhancer(
        denoise_method="guided",
        guided_radius=2,
        guided_eps=5e-4,
        unsharp_strength=0.25,
        use_null_space_projector=True,
        anchor_mode="model",
    )

    x_sr = torch.rand(2, 1, 64, 64, dtype=torch.float32)
    lr_in = torch.rand(2, 1, 32, 32, dtype=torch.float32)

    out, telemetry = enhancer(x_sr, lr_in)

    assert out.shape == (2, 1, 64, 64)
    assert "edge_mask" in telemetry
    assert "grad_mag" in telemetry
    assert "invariant_norm" in telemetry
    assert float(telemetry["invariant_norm"].item()) <= 1.2e-7


def test_metrology_metrics_computation():
    """Test computation of CD error, noise floor, and edge acutance."""
    gt = np.zeros((64, 64), dtype=np.float32)
    gt[20:44, 20:44] = 1.0  # Square contact hole

    # Slightly dilated prediction (1 px error) + Gaussian noise
    pred = np.zeros((64, 64), dtype=np.float32)
    pred[19:45, 19:45] = 0.95
    pred += np.random.normal(0, 0.02, pred.shape).astype(np.float32)
    pred = np.clip(pred, 0.0, 1.0)

    metrics = compute_comprehensive_metrology_metrics(pred, gt)

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "cd_err_px" in metrics
    assert "cd_rel_pct" in metrics
    assert "noise_floor" in metrics
    assert "edge_acutance_pred" in metrics

    assert metrics["psnr"] > 15.0
    assert metrics["ssim"] > 0.40
    assert metrics["cd_err_px"] > 0.0
    assert metrics["noise_floor"] > 0.0
