"""Branch 2: High-Clarity Metrology Enhancement & Directional Denoising Module.

Implements:
1. Edge-Preserving Denoising (GPU-native Guided / Bilateral Filter)
2. Directional Sobel Unsharp Masking (High-frequency sidewall boost with soft edge gating)
3. Discrete Null-Space Consistency Projector (P_N) ensuring ||H(x_hat) - y||_inf <= 1.19e-7
4. Comprehensive Semiconductor Metrology Metrics (CD Line-Width Error, Visual Noise Floor, Edge Acutance)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class DiscreteNullSpaceProjector(nn.Module):
    """Discrete Null-Space Consistency Projector (P_N).
    
    Given forward decimation operator H: R^{2H x 2W} -> R^{H x W} (2x2 average pooling)
    and its pseudo-inverse H_dagger: R^{H x W} -> R^{2H x 2W} (2x nearest upsampling),
    decomposes any super-resolved image X into:
        X = P_R(X) + P_N(X) = H_dagger(H(X)) + (X - H_dagger(H(X)))
    
    Enforces exact physical measurement consistency with anchor Y_anchor:
        X_consistent = H_dagger(Y_anchor) + P_N(X)
    
    Guarantees ||H(X_consistent) - Y_anchor||_inf <= 1.19e-7.
    """

    def __init__(self, scale_factor: int = 2) -> None:
        super().__init__()
        self.scale_factor = scale_factor

    def forward(
        self, x: torch.Tensor, y_anchor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Projects x into the null-space consistent manifold of y_anchor.
        
        Args:
            x: High-resolution tensor of shape (B, 1, 2H, 2W)
            y_anchor: Optional low-resolution measurement anchor (B, 1, H, W).
                      If None, uses the range component H(x) as the clean anchor.
        Returns:
            Null-space consistent tensor of shape (B, 1, 2H, 2W).
        """
        # H(X): 2x2 average pooling
        h_x = F.avg_pool2d(
            x, kernel_size=self.scale_factor, stride=self.scale_factor
        )
        # H_dagger(H(X)): 2x nearest-neighbor upsampling
        h_dagger_hx = F.interpolate(h_x, scale_factor=self.scale_factor, mode="nearest")
        # Null-space component P_N(X)
        p_n = x - h_dagger_hx

        if y_anchor is None:
            # Reconstruct with original range
            return h_dagger_hx + p_n

        h_dagger_y = F.interpolate(y_anchor, scale_factor=self.scale_factor, mode="nearest")
        return h_dagger_y + p_n

    @staticmethod
    def verify_measurement_invariant(
        x_consistent: torch.Tensor, y_anchor: torch.Tensor
    ) -> float:
        """Computes ||H(X_consistent) - Y_anchor||_inf."""
        h_x = F.avg_pool2d(x_consistent, kernel_size=2, stride=2)
        inf_norm = float(torch.max(torch.abs(h_x - y_anchor)).item())
        return inf_norm


class FastGuidedFilter2D(nn.Module):
    """Fast GPU-native Edge-Preserving Guided Filter in PyTorch."""

    def __init__(self, radius: int = 2, eps: float = 1e-3) -> None:
        super().__init__()
        self.radius = radius
        self.eps = eps
        self.kernel_size = 2 * radius + 1
        # 2D box filter kernel
        box = torch.ones(1, 1, self.kernel_size, self.kernel_size, dtype=torch.float32)
        box = box / (self.kernel_size * self.kernel_size)
        self.register_buffer("box_kernel", box)

    def _box_filter(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.box_kernel, padding=self.radius)

    def forward(self, p: torch.Tensor, i_guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Filters input tensor p guided by i_guide (default: p itself)."""
        if i_guide is None:
            i_guide = p

        mean_i = self._box_filter(i_guide)
        mean_p = self._box_filter(p)
        mean_ip = self._box_filter(i_guide * p)
        cov_ip = mean_ip - mean_i * mean_p

        mean_ii = self._box_filter(i_guide * i_guide)
        var_i = mean_ii - mean_i * mean_i

        a = cov_ip / (var_i + self.eps)
        b = mean_p - a * mean_i

        mean_a = self._box_filter(a)
        mean_b = self._box_filter(b)

        q = mean_a * i_guide + mean_b
        return q


class DirectionalSobelUnsharpFilter(nn.Module):
    """GPU-native Directional Sobel Unsharp Masking module.
    
    Computes horizontal and vertical Sobel gradients, forms an adaptive soft
    edge gating mask, and boosts directional high frequencies along line sidewalls.
    """

    def __init__(
        self,
        strength: float = 0.25,
        edge_threshold: float = 0.05,
        edge_slope: float = 25.0,
        blur_sigma: float = 0.6,
    ) -> None:
        super().__init__()
        self.strength = strength
        self.edge_threshold = edge_threshold
        self.edge_slope = edge_slope
        self.blur_sigma = blur_sigma

        # Sobel kernels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # Gaussian blur kernel (3x3 or 5x5)
        k_size = 5
        radius = k_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(coords, coords, indexing="ij")
        gauss = torch.exp(-(grid_x**2 + grid_y**2) / (2.0 * blur_sigma**2))
        gauss = (gauss / gauss.sum()).view(1, 1, k_size, k_size)
        self.register_buffer("gauss_kernel", gauss)
        self.gauss_pad = radius

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies directional Sobel unsharp edge enhancement.
        
        Args:
            x: Input tensor of shape (B, 1, H, W)
        Returns:
            Tuple of (sharp_image, edge_mask, gradient_magnitude)
        """
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-12)

        # Soft sigmoid edge confidence mask
        edge_mask = torch.sigmoid(self.edge_slope * (grad_mag - self.edge_threshold))

        # High-frequency detail
        blurred = F.conv2d(x, self.gauss_kernel, padding=self.gauss_pad)
        high_freq = x - blurred

        # Directional unsharp boost
        sharp = x + self.strength * edge_mask * high_freq
        return sharp, edge_mask, grad_mag


class DirectionalMetrologyEnhancer(nn.Module):
    """Complete Branch 2: High-Clarity Metrology Enhancement Pipeline.
    
    Combines:
    1. Fast GPU-native Edge-Preserving Denoising
    2. Directional Sobel Unsharp Edge Boosting
    3. Discrete Null-Space Consistency Projector (P_N)
    """

    def __init__(
        self,
        denoise_method: str = "guided",
        guided_radius: int = 2,
        guided_eps: float = 5e-4,
        unsharp_strength: float = 0.25,
        edge_threshold: float = 0.05,
        edge_slope: float = 25.0,
        blur_sigma: float = 0.6,
        use_null_space_projector: bool = True,
        anchor_mode: str = "model",  # 'model', 'lr', 'blend', 'none'
        blend_alpha: float = 0.05,
    ) -> None:
        super().__init__()
        self.denoise_method = denoise_method
        self.use_null_space_projector = use_null_space_projector
        self.anchor_mode = anchor_mode
        self.blend_alpha = blend_alpha

        if denoise_method == "guided":
            self.denoiser = FastGuidedFilter2D(radius=guided_radius, eps=guided_eps)
        else:
            self.denoiser = nn.Identity()

        self.unsharp = DirectionalSobelUnsharpFilter(
            strength=unsharp_strength,
            edge_threshold=edge_threshold,
            edge_slope=edge_slope,
            blur_sigma=blur_sigma,
        )

        self.null_space_projector = DiscreteNullSpaceProjector(scale_factor=2)

    def forward(
        self,
        x_sr: torch.Tensor,
        lr_input: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Enhances super-resolved output with edge-preserving clarity and metrology consistency.
        
        Args:
            x_sr: Super-resolved tensor of shape (B, 1, 2H, 2W) in [0, 1]
            lr_input: Low-resolution input tensor of shape (B, 1, H, W)
        Returns:
            Tuple of (enhanced_tensor, telemetry_dict)
        """
        # 1. Edge-preserving denoising
        if self.denoise_method == "guided":
            x_denoised = self.denoiser(x_sr, x_sr)
        else:
            x_denoised = x_sr

        # 2. Directional Sobel unsharp masking
        x_sharp, edge_mask, grad_mag = self.unsharp(x_denoised)
        x_sharp = torch.clamp(x_sharp, 0.0, 1.0)

        # 3. Discrete Null-Space Consistency Projector (P_N)
        if self.use_null_space_projector and self.anchor_mode != "none":
            # Determine low-frequency anchor
            h_model = F.avg_pool2d(x_sr, kernel_size=2, stride=2)
            if self.anchor_mode == "model":
                y_anchor = h_model
            elif self.anchor_mode == "lr" and lr_input is not None:
                y_anchor = torch.clamp(lr_input, 0.0, 1.0)
            elif self.anchor_mode == "blend" and lr_input is not None:
                y_anchor = (1.0 - self.blend_alpha) * h_model + self.blend_alpha * torch.clamp(lr_input, 0.0, 1.0)
            else:
                y_anchor = h_model

            # Project null space
            x_consistent = self.null_space_projector(x_sharp, y_anchor)
            invariant_norm = self.null_space_projector.verify_measurement_invariant(x_consistent, y_anchor)
            x_out = x_consistent
        else:
            x_out = x_sharp
            y_anchor = F.avg_pool2d(x_out, kernel_size=2, stride=2)
            invariant_norm = 0.0

        telemetry = {
            "edge_mask": edge_mask,
            "grad_mag": grad_mag,
            "invariant_norm": torch.tensor(invariant_norm, device=x_sr.device),
        }
        return x_out, telemetry


# ---------------------------------------------------------------------------
# Comprehensive Semiconductor Metrology Evaluation Metrics
# ---------------------------------------------------------------------------

def measure_critical_dimension_linewidth(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, float]:
    """Computes Critical Dimension (CD) Line-Width Error across 1D scanlines.
    
    Standard semiconductor metrology method:
    Evaluates top-down line profiles along horizontal and vertical scanlines
    at 50% height threshold.
    """
    pred_bin = (pred > threshold).astype(np.float32)
    gt_bin = (gt > threshold).astype(np.float32)

    # Horizontal scanlines
    h_gt = np.sum(gt_bin, axis=1)
    h_pred = np.sum(pred_bin, axis=1)
    valid_h = (h_gt >= 4) & (h_gt <= gt.shape[1] - 4)

    # Vertical scanlines
    v_gt = np.sum(gt_bin, axis=0)
    v_pred = np.sum(pred_bin, axis=0)
    valid_v = (v_gt >= 4) & (v_gt <= gt.shape[0] - 4)

    cd_errors = []
    gt_cds = []
    if np.sum(valid_h) > 0:
        cd_errors.extend(np.abs(h_pred[valid_h] - h_gt[valid_h]))
        gt_cds.extend(h_gt[valid_h])
    if np.sum(valid_v) > 0:
        cd_errors.extend(np.abs(v_pred[valid_v] - v_gt[valid_v]))
        gt_cds.extend(v_gt[valid_v])

    if not cd_errors:
        return {"cd_err_px": 0.0, "cd_rel_pct": 0.0, "ler_3sigma": 0.0}

    mean_cd_err = float(np.mean(cd_errors))
    mean_gt_cd = float(np.mean(gt_cds)) if gt_cds else 1.0
    rel_pct = float((mean_cd_err / max(1e-5, mean_gt_cd)) * 100.0)
    ler_3sigma = float(3.0 * np.std(cd_errors))

    return {
        "cd_err_px": mean_cd_err,
        "cd_rel_pct": rel_pct,
        "ler_3sigma": ler_3sigma,
    }


def measure_visual_noise_floor(
    pred: np.ndarray,
    gt: np.ndarray,
    flat_grad_threshold: float = 0.035,
) -> Dict[str, float]:
    """Measures visual noise floor standard deviation in homogeneous flat-field regions."""
    gx = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    grad_gt = np.sqrt(gx**2 + gy**2)

    flat_mask = grad_gt < flat_grad_threshold
    num_flat_pixels = int(np.sum(flat_mask))

    if num_flat_pixels < 50:
        return {"noise_floor": 0.0, "flat_mae": 0.0, "flat_pixel_ratio": 0.0}

    pred_flat = pred[flat_mask]
    gt_flat = gt[flat_mask]

    noise_floor_std = float(np.std(pred_flat))
    flat_mae = float(np.mean(np.abs(pred_flat - gt_flat)))
    flat_ratio = float(num_flat_pixels / (gt.shape[0] * gt.shape[1]))

    return {
        "noise_floor": noise_floor_std,
        "flat_mae": flat_mae,
        "flat_pixel_ratio": flat_ratio,
    }


def measure_edge_acutance(
    pred: np.ndarray,
    gt: np.ndarray,
    edge_grad_threshold: float = 0.08,
) -> Dict[str, float]:
    """Measures edge sharpness / acutance and edge sidewall gradient magnitude."""
    gx_gt = cv2.Sobel(gt, cv2.CV_32F, 1, 0, ksize=3)
    gy_gt = cv2.Sobel(gt, cv2.CV_32F, 0, 1, ksize=3)
    grad_gt = np.sqrt(gx_gt**2 + gy_gt**2)

    edge_mask = grad_gt > edge_grad_threshold
    if np.sum(edge_mask) < 50:
        return {"edge_acutance_pred": 0.0, "edge_acutance_gt": 0.0, "edge_contrast_ratio": 1.0}

    gx_pred = cv2.Sobel(pred, cv2.CV_32F, 1, 0, ksize=3)
    gy_pred = cv2.Sobel(pred, cv2.CV_32F, 0, 1, ksize=3)
    grad_pred = np.sqrt(gx_pred**2 + gy_pred**2)

    pred_acutance = float(np.mean(grad_pred[edge_mask]))
    gt_acutance = float(np.mean(grad_gt[edge_mask]))
    contrast_ratio = float(pred_acutance / max(1e-5, gt_acutance))

    return {
        "edge_acutance_pred": pred_acutance,
        "edge_acutance_gt": gt_acutance,
        "edge_contrast_ratio": contrast_ratio,
    }


def compute_comprehensive_metrology_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    data_range: float = 1.0,
) -> Dict[str, float]:
    """Computes full set of PSNR, SSIM, CD error, noise floor, and edge acutance."""
    pred = np.clip(pred.astype(np.float32), 0.0, 1.0)
    gt = np.clip(gt.astype(np.float32), 0.0, 1.0)

    # 1. Standard reconstruction metrics
    psnr_val = float(peak_signal_noise_ratio(gt, pred, data_range=data_range))
    ssim_val = float(
        structural_similarity(
            gt,
            pred,
            data_range=data_range,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )

    # 2. Metrology metrics
    cd_metrics = measure_critical_dimension_linewidth(pred, gt)
    noise_metrics = measure_visual_noise_floor(pred, gt)
    edge_metrics = measure_edge_acutance(pred, gt)

    return {
        "psnr": psnr_val,
        "ssim": ssim_val,
        **cd_metrics,
        **noise_metrics,
        **edge_metrics,
    }
