"""Compound Augmentation Stack for Evidence-DAR SEM-SR.

Implements physics-grounded and microscopy-native data augmentations:
1. MultiViewConsistencyAugmenter: Generates multiple stochastic degradation realizations
   y_1, y_2 ~ p_SEM(theta | x) of the same GT to enforce content invariance F_c(y_1) ~ F_c(y_2).
2. NegativeRestorationSampler: Generates clean / near-clean identity pairs (x_clean -> x)
   to regularize the network against hallucination and over-smoothing on clean structures.
3. SpectralHighFrequencyAugmenter: Downscales GT structures by factor s in [1.2, 2.0] before
   cropping, synthesizing high-frequency line pitch to bridge the 1.7x-1.8x test spectral shift.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from restoration.simulator.operator import SEMForwardOperator


class MultiViewConsistencyAugmenter:
    """Generates multi-view stochastic degradation realizations for self-supervised consistency.

    Produces paired or N-ary degraded views (y_1, y_2, ...) from a single high-resolution
    ground truth tensor with randomized physical degradation parameters (PSF blur, sensor noise,
    speckle). This enables enforcing content representation consistency F_c(y_1) ~ F_c(y_2) while
    degradation representations F_d(y_1) and F_d(y_2) capture distinct noise states.
    """

    def __init__(
        self,
        operator: Optional[SEMForwardOperator] = None,
        psf_range: Tuple[float, float] = (0.4, 1.4),
        noise_scale_range: Tuple[float, float] = (0.015, 0.035),
        speckle_std_range: Tuple[float, float] = (0.0, 0.20),
    ) -> None:
        self.operator = operator if operator is not None else SEMForwardOperator()
        self.psf_range = psf_range
        self.noise_scale_range = noise_scale_range
        self.speckle_std_range = speckle_std_range

    def sample_view_parameters(
        self,
        severity: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Samples degradation parameter dictionary for a specific view or severity level."""
        _rng = rng if rng is not None else np.random.default_rng()

        if severity == "mild":
            psf_sigma = float(_rng.uniform(self.psf_range[0], (self.psf_range[0] + self.psf_range[1]) / 2))
            noise_scale = float(_rng.uniform(self.noise_scale_range[0], 0.0233))
            speckle_std = float(_rng.uniform(0.0, 0.05))
        elif severity == "severe":
            psf_sigma = float(_rng.uniform((self.psf_range[0] + self.psf_range[1]) / 2, self.psf_range[1]))
            noise_scale = float(_rng.uniform(0.0233, self.noise_scale_range[1]))
            speckle_std = float(_rng.uniform(0.08, self.speckle_std_range[1]))
        else:
            psf_sigma = float(_rng.uniform(self.psf_range[0], self.psf_range[1]))
            noise_scale = float(_rng.uniform(self.noise_scale_range[0], self.noise_scale_range[1]))
            speckle_std = float(_rng.uniform(self.speckle_std_range[0], self.speckle_std_range[1]))

        return {
            "psf_sigma": psf_sigma,
            "noise_scale": noise_scale,
            "noise_exponent": 0.836,
            "speckle_std": speckle_std,
            "s_floor": 0.02,
        }

    def generate_views(
        self,
        hr: torch.Tensor,
        num_views: int = 2,
        randomize_params: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> List[torch.Tensor]:
        """Generates a list of num_views distinct degraded LR views from a single HR tensor."""
        views: List[torch.Tensor] = []
        for v_idx in range(num_views):
            if randomize_params:
                theta = self.sample_view_parameters()
            else:
                theta = None
            view_lr = self.operator.degrade(hr, theta=theta, generator=generator)
            views.append(view_lr)
        return views

    def generate_paired_views(
        self,
        hr: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], Dict[str, Any]]:
        """Generates a paired mild vs severe stochastic degradation realization of the same HR.

        Returns:
            (view_1, view_2, theta_1, theta_2)
        """
        theta_1 = self.sample_view_parameters(severity="mild")
        theta_2 = self.sample_view_parameters(severity="severe")

        view_1 = self.operator.degrade(hr, theta=theta_1, generator=generator)
        view_2 = self.operator.degrade(hr, theta=theta_2, generator=generator)

        return view_1, view_2, theta_1, theta_2

    @staticmethod
    def compute_consistency_loss(
        f_c_1: torch.Tensor,
        f_c_2: torch.Tensor,
        loss_type: str = "mse",
    ) -> torch.Tensor:
        """Computes content representation invariance loss between two degradation views."""
        if loss_type == "mse":
            return F.mse_loss(f_c_1, f_c_2)
        elif loss_type == "l1":
            return F.l1_loss(f_c_1, f_c_2)
        elif loss_type == "cosine":
            # Normalized cosine distance across channel dimensions
            c1_norm = F.normalize(f_c_1, p=2, dim=1)
            c2_norm = F.normalize(f_c_2, p=2, dim=1)
            return (1.0 - (c1_norm * c2_norm).sum(dim=1)).mean()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")


class NegativeRestorationSampler:
    """Generates clean / near-clean negative samples to regularize identity mapping.

    Provides negative samples (identity mapping on clean or low-noise inputs) to prevent
    the network from hallucinating artificial textures or over-smoothing pristine structures.
    """

    def __init__(
        self,
        operator: Optional[SEMForwardOperator] = None,
        clean_prob: float = 0.25,
        max_noise_std: float = 0.005,
    ) -> None:
        self.operator = operator if operator is not None else SEMForwardOperator()
        self.clean_prob = clean_prob
        self.max_noise_std = max_noise_std

    def create_negative_sample(
        self,
        hr: torch.Tensor,
        add_minimal_noise: bool = False,
        noise_std: float = 0.002,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Creates a pristine/near-clean downscaled LR image without instrument degradation."""
        # Pure polyphase decimation with no PSF blur
        lr_clean = self.operator.clean(hr, theta={"psf_sigma": 0.0})
        if add_minimal_noise and noise_std > 0:
            noise = torch.randn(lr_clean.shape, generator=generator, device=hr.device, dtype=hr.dtype) * min(noise_std, self.max_noise_std)
            return lr_clean + noise
        return lr_clean

    def sample_batch(
        self,
        hr_batch: torch.Tensor,
        negative_prob: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mixes standard degraded pairs and clean negative samples in a training batch.

        Args:
            hr_batch: (B, 1, 2H, 2W) ground truth batch.
            negative_prob: Probability of each item being a negative sample.
            generator: Optional torch.Generator.

        Returns:
            Tuple of:
                - lr_batch: (B, 1, H, W) LR inputs for restoration network.
                - hr_targets: (B, 1, 2H, 2W) HR targets.
                - is_negative_mask: (B,) boolean tensor (True = negative sample).
        """
        prob = self.clean_prob if negative_prob is None else negative_prob
        batch_size = hr_batch.shape[0]

        # Determine negative samples
        rand_draws = torch.rand(batch_size, generator=generator, device=hr_batch.device)
        is_negative = rand_draws < prob

        lr_list = []
        for i in range(batch_size):
            hr_item = hr_batch[i : i + 1]
            if is_negative[i]:
                # Negative sample: noise-free or minimal noise LR
                lr_item = self.create_negative_sample(hr_item, add_minimal_noise=True, noise_std=0.001, generator=generator)
            else:
                # Standard physical degradation
                lr_item = self.operator.degrade(hr_item, generator=generator)
            lr_list.append(lr_item)

        lr_batch = torch.cat(lr_list, dim=0)
        return lr_batch, hr_batch, is_negative

    @staticmethod
    def compute_identity_loss(
        pred_hr: torch.Tensor,
        target_hr: torch.Tensor,
        is_negative_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes identity supervision loss on negative sample items."""
        if is_negative_mask is None or is_negative_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_hr.device, dtype=pred_hr.dtype)

        neg_preds = pred_hr[is_negative_mask]
        neg_targets = target_hr[is_negative_mask]
        # Charbonnier distance on negative items
        eps = 1e-3
        return torch.sqrt((neg_preds - neg_targets).pow(2) + eps**2).mean()


class SpectralHighFrequencyAugmenter:
    """High-frequency spectral augmentation bridging the 1.7x - 1.8x test spectral shift.

    Pre-downscales GT structures by factor s in [1.2, 2.0] before cropping, effectively
    increasing spatial frequency and line pitch density by 1.2x-2.0x to match test distribution.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (1.2, 2.0),
        p: float = 0.5,
        interpolation_mode: str = "bicubic",
    ) -> None:
        self.scale_range = scale_range
        self.p = p
        self.interpolation_mode = interpolation_mode

    def augment(
        self,
        hr: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
        return_scale: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """Applies spectral downscale-crop augmentation to an HR tensor.

        Args:
            hr: (B, 1, H, W) high-resolution image tensor.
            target_size: Desired output crop size (out_h, out_w). Defaults to input shape.
            return_scale: If True, returns (augmented_hr, scale_factor).
            rng: Optional NumPy random generator.

        Returns:
            Augmented tensor of shape (B, 1, out_h, out_w), or tuple with scale factor.
        """
        _rng = rng if rng is not None else np.random.default_rng()
        b, c, h, w = hr.shape
        out_h, out_w = target_size if target_size is not None else (h, w)

        if _rng.uniform(0.0, 1.0) > self.p:
            # Identity / Center crop if necessary
            if (h, w) != (out_h, out_w):
                top = max(0, (h - out_h) // 2)
                left = max(0, (w - out_w) // 2)
                cropped = hr[:, :, top : top + out_h, left : left + out_w]
            else:
                cropped = hr
            return (cropped, 1.0) if return_scale else cropped

        # Sample compression scale factor
        scale = float(_rng.uniform(self.scale_range[0], self.scale_range[1]))
        scaled_h = max(out_h + 4, int(math.ceil(out_h * scale)))
        scaled_w = max(out_w + 4, int(math.ceil(out_w * scale)))

        # 1. Upsample canvas if input is smaller than scaled_h/scaled_w
        if h < scaled_h or w < scaled_w:
            canvas = F.interpolate(
                hr,
                size=(max(h, scaled_h), max(w, scaled_w)),
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode != "nearest" else None,
            )
        else:
            canvas = hr

        # 2. Downscale to compress pitch
        downscaled = F.interpolate(
            canvas,
            scale_factor=1.0 / scale,
            mode=self.interpolation_mode,
            align_corners=False if self.interpolation_mode != "nearest" else None,
        )

        # 3. Random crop to target size
        dh, dw = downscaled.shape[2], downscaled.shape[3]
        if dh >= out_h and dw >= out_w:
            max_top = dh - out_h
            max_left = dw - out_w
            top = int(_rng.integers(0, max_top + 1)) if max_top > 0 else 0
            left = int(_rng.integers(0, max_left + 1)) if max_left > 0 else 0
            out = downscaled[:, :, top : top + out_h, left : left + out_w]
        else:
            out = F.interpolate(
                downscaled,
                size=(out_h, out_w),
                mode=self.interpolation_mode,
                align_corners=False if self.interpolation_mode != "nearest" else None,
            )

        return (out, scale) if return_scale else out

    @staticmethod
    def measure_radial_spectral_energy(
        img: torch.Tensor,
        num_bands: int = 10,
    ) -> torch.Tensor:
        """Computes normalized radial power spectral energy across concentric frequency rings.

        Args:
            img: (B, 1, H, W) or (1, H, W) image tensor.
            num_bands: Number of radial frequency bands above DC.

        Returns:
            Tensor of shape (B, num_bands) containing normalized power in each radial band.
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)

        b, c, h, w = img.shape
        fft2 = torch.fft.fftshift(torch.fft.fft2(img, norm="ortho"), dim=(-2, -1))
        power = (fft2.real.pow(2) + fft2.imag.pow(2)).squeeze(1)  # (B, H, W)

        # Radial distance map from center (DC)
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(
            torch.arange(h, device=img.device, dtype=torch.float32) - cy,
            torch.arange(w, device=img.device, dtype=torch.float32) - cx,
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        max_r = min(cy, cx)

        bands = []
        band_edges = torch.linspace(0, max_r, num_bands + 1, device=img.device)
        total_energy = power.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

        for k in range(num_bands):
            r_lo = band_edges[k]
            r_hi = band_edges[k + 1]
            mask = (r >= r_lo) & (r < r_hi)
            if mask.sum() > 0:
                band_power = (power * mask.unsqueeze(0)).sum(dim=(-2, -1)) / total_energy.view(b)
                bands.append(band_power.view(b, 1))
            else:
                bands.append(torch.zeros((b, 1), device=img.device, dtype=img.dtype))

        return torch.cat(bands, dim=1)  # (B, num_bands)
