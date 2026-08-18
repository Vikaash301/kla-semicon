"""Calibrated Physical Forward Operator for Scanning Electron Microscopy (SEM).

Implements the calibrated acquisition forward model p_SEM(theta):
    LR = W4 * (PSF * GT)  +  sigma(s, rho) * (h * eps)

Physics components:
1. Electron Beam Point Spread Function (PSF) blur: 2D Gaussian/Airy kernel with
   sigma in [0.4, 1.4] pixels.
2. 2x Polyphase Decimation: 4x4 separable kernel W_4 with exact partition of unity
   (sum = 1.0) and replication boundary padding.
3. Multiplicative Gamma/Gaussian Speckle: variance sigma_speckle^2 <= 0.25.
4. Signal-Dependent Heteroscedastic Sensor Noise: power-law model
   sigma(s) = 0.0233 * (s / 0.1)^0.836, with local signal floor s_floor = 0.02.
5. Spatial Noise Autocorrelation: 3x3 plus-kernel h reproducing measured lag-1 ACF
   (bx = -0.052, by = -0.056).
6. Differentiable forward pass for physical re-degradation consistency loss L_phys.
7. Unclamped output dynamic range (preserving observed excursions in [-0.28, 2.16]).
"""

from __future__ import annotations

import math
import pathlib
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Calibrated 4x4 polyphase decimation kernel W_4 (least-squares fit on 2,880 train stems)
# Separable taps closely matching bicubic a=-0.65 with partition of unity (sum = 1.0)
_RAW_W4 = np.array(
    [
        [0.01490366, -0.04459596, -0.04336780, 0.01228557],
        [-0.04037773, 0.32057613, 0.32235768, -0.03752689],
        [-0.04750104, 0.31955266, 0.32538718, -0.04424888],
        [0.01359543, -0.04130475, -0.04048958, 0.01059944],
    ],
    dtype=np.float32,
)
CALIBRATED_W4 = (_RAW_W4 / _RAW_W4.sum()).astype(np.float32)

# Calibrated 3x3 spatial ACF plus-kernel h
CALIBRATED_BX = -0.02630766
CALIBRATED_BY = -0.02821796

# Calibrated power-law noise parameters: sigma(s) = noise_scale * (s / 0.1)^noise_exponent
CALIBRATED_NOISE_SCALE = 0.0233
CALIBRATED_NOISE_EXPONENT = 0.836
CALIBRATED_S_FLOOR = 0.02


def _build_gaussian_kernel_2d(
    sigma_x: float,
    sigma_y: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Builds a normalized 2D Gaussian blur kernel with odd spatial dimensions."""
    if sigma_x <= 1e-4 and sigma_y <= 1e-4:
        return torch.ones((1, 1, 1, 1), device=device, dtype=dtype)

    rad_x = max(1, int(math.ceil(3.0 * max(sigma_x, 1e-4))))
    rad_y = max(1, int(math.ceil(3.0 * max(sigma_y, 1e-4))))

    x = torch.arange(-rad_x, rad_x + 1, device=device, dtype=dtype)
    y = torch.arange(-rad_y, rad_y + 1, device=device, dtype=dtype)

    kx = torch.exp(-0.5 * (x / max(sigma_x, 1e-4)) ** 2) if sigma_x > 1e-4 else torch.zeros_like(x)
    ky = torch.exp(-0.5 * (y / max(sigma_y, 1e-4)) ** 2) if sigma_y > 1e-4 else torch.zeros_like(y)
    if sigma_x <= 1e-4:
        kx[rad_x] = 1.0
    if sigma_y <= 1e-4:
        ky[rad_y] = 1.0

    kx = kx / kx.sum()
    ky = ky / ky.sum()

    k2d = torch.outer(ky, kx)
    k2d = k2d / k2d.sum()
    return k2d.view(1, 1, 2 * rad_y + 1, 2 * rad_x + 1)


class SEMForwardOperator(nn.Module):
    """Calibrated physical degradation forward operator for semiconductor SEM imaging.

    Maps high-resolution ground truth GT (B, 1, 2H, 2W) to degraded low-resolution
    measurements NoisyLR (B, 1, H, W) under physical electron-beam blur, polyphase
    decimation, heteroscedastic power-law sensor noise, and multiplicative speckle.
    """

    def __init__(
        self,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        psf_sigma_range: Tuple[float, float] = (0.4, 1.4),
        noise_scale: float = CALIBRATED_NOISE_SCALE,
        noise_exponent: float = CALIBRATED_NOISE_EXPONENT,
        max_speckle_var: float = 0.25,
        s_floor: float = CALIBRATED_S_FLOOR,
        params_path: Optional[Union[str, pathlib.Path]] = None,
    ) -> None:
        super().__init__()
        self.target_dtype = dtype
        if device is None:
            self.target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.target_device = torch.device(device)

        self.psf_sigma_range = psf_sigma_range
        self.noise_scale = float(noise_scale)
        self.noise_exponent = float(noise_exponent)
        self.max_speckle_var = float(max_speckle_var)
        self.s_floor = float(s_floor)

        # Initialize polyphase kernel W4 buffer
        w4_tensor = torch.from_numpy(CALIBRATED_W4).to(dtype=dtype, device=self.target_device)
        self.register_buffer("w4", w4_tensor.view(1, 1, 4, 4))

        # Initialize spatial ACF filter h buffer
        h_matrix = torch.tensor(
            [
                [0.0, CALIBRATED_BY, 0.0],
                [CALIBRATED_BX, 1.0, CALIBRATED_BX],
                [0.0, CALIBRATED_BY, 0.0],
            ],
            dtype=dtype,
            device=self.target_device,
        )
        h_norm = h_matrix / h_matrix.pow(2).sum().sqrt()
        self.register_buffer("h_acf", h_norm.view(1, 1, 3, 3))

        # Optional cache parameters if available
        self._lut_params: Optional[Dict[str, Any]] = None
        if params_path is not None and pathlib.Path(params_path).exists():
            self._load_cache_params(pathlib.Path(params_path))
        else:
            default_cache = pathlib.Path(__file__).resolve().parent.parent.parent / "claude" / "operator" / "cache" / "params.npz"
            if default_cache.exists():
                self._load_cache_params(default_cache)

    def _load_cache_params(self, cache_file: pathlib.Path) -> None:
        """Loads empirical noise LUT tables from calibrated cache if present."""
        try:
            z = np.load(cache_file)
            t = lambda k: torch.as_tensor(z[k], device=self.target_device, dtype=self.target_dtype)
            bx, by = float(z["bx"]), float(z["by"])
            h = torch.tensor([[0.0, by, 0.0], [bx, 1.0, bx], [0.0, by, 0.0]], device=self.target_device, dtype=self.target_dtype)
            self._lut_params = dict(
                W4=t("W4").view(1, 1, 4, 4),
                SIG=t("SIG").T.contiguous().view(1, 1, z["SIG"].shape[1], z["SIG"].shape[0]),
                RE=t("RE"),
                LUT=t("LUT").reshape(-1),
                h=(h / h.pow(2).sum().sqrt()).view(1, 1, 3, 3),
                sfloor=float(z["sfloor"]),
                nq=int(z["LUT"].shape[1]),
                ng=int(z["ng"]),
            )
        except Exception:
            self._lut_params = None

    def apply_psf_blur(
        self,
        hr: torch.Tensor,
        sigma_x: float = 0.0,
        sigma_y: float = 0.0,
    ) -> torch.Tensor:
        """Applies 2D Gaussian PSF electron beam blur differentiably."""
        if sigma_x <= 1e-4 and sigma_y <= 1e-4:
            return hr

        kernel = _build_gaussian_kernel_2d(sigma_x, sigma_y, device=hr.device, dtype=hr.dtype)
        kh, kw = kernel.shape[2], kernel.shape[3]
        pad_h = kh // 2
        pad_w = kw // 2

        hr_padded = F.pad(hr, (pad_w, pad_w, pad_h, pad_h), mode="replicate")
        return F.conv2d(hr_padded, kernel)

    def apply_decimation(self, hr: torch.Tensor) -> torch.Tensor:
        """Applies 2x polyphase decimation with W_4 kernel and replicate padding."""
        w4 = self.w4.to(device=hr.device, dtype=hr.dtype)
        # Pad 1 top/left, 2 bottom/right for 4x4 kernel with stride 2
        hr_padded = F.pad(hr, (1, 2, 1, 2), mode="replicate")
        return F.conv2d(hr_padded, w4, stride=2)

    def clean(
        self,
        hr: torch.Tensor,
        theta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Computes the noise-free LR observation: s = W_4 * (PSF * GT).

        Args:
            hr: High-resolution tensor of shape (B, 1, 2H, 2W).
            theta: Optional parameter dictionary with keys:
                - 'psf_sigma': float or (sigma_x, sigma_y). Default 0.0 for pure decimation.

        Returns:
            Noise-free downscaled tensor of shape (B, 1, H, W).
        """
        if hr.dim() != 4 or hr.shape[1] != 1:
            raise ValueError(f"Expected 4D single-channel tensor (B, 1, H, W), got shape {tuple(hr.shape)}")
        if hr.shape[2] % 2 != 0 or hr.shape[3] % 2 != 0:
            raise ValueError(f"Spatial dimensions must be even for 2x decimation, got {tuple(hr.shape[2:])}")

        sigma_x, sigma_y = 0.0, 0.0
        if theta is not None and "psf_sigma" in theta:
            psf = theta["psf_sigma"]
            if isinstance(psf, (tuple, list)):
                sigma_x, sigma_y = float(psf[0]), float(psf[1])
            elif isinstance(psf, (int, float)):
                sigma_x = sigma_y = float(psf)

        blurred = self.apply_psf_blur(hr, sigma_x=sigma_x, sigma_y=sigma_y)
        return self.apply_decimation(blurred)

    def compute_noise_sigma(
        self,
        signal: torch.Tensor,
        noise_scale: Optional[float] = None,
        noise_exponent: Optional[float] = None,
        s_floor: Optional[float] = None,
    ) -> torch.Tensor:
        """Computes heteroscedastic standard deviation sigma(s) = scale * (max(s, s_floor) / 0.1)^exponent.

        Args:
            signal: Local noise-free signal tensor s.
            noise_scale: Optional override for noise scale constant (default 0.0233).
            noise_exponent: Optional override for power-law exponent (default 0.836).
            s_floor: Optional override for signal clamping floor (default 0.02).

        Returns:
            Tensor of per-pixel standard deviations sigma(s).
        """
        scale = float(self.noise_scale if noise_scale is None else noise_scale)
        exponent = float(self.noise_exponent if noise_exponent is None else noise_exponent)
        floor = float(self.s_floor if s_floor is None else s_floor)

        s_clamped = signal.clamp_min(floor)
        return scale * (s_clamped / 0.1) ** exponent

    def degrade(
        self,
        hr: torch.Tensor,
        theta: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Synthesizes a realistic NoisyLR measurement from clean ground truth HR.

        Physics steps:
        1. PSF blur (sigma in [0.4, 1.4]).
        2. 2x polyphase decimation W_4 -> noise-free LR s.
        3. Multiplicative Gamma speckle: s_speckle = s * (1 + delta_speckle).
        4. Heteroscedastic noise: sigma(s) = 0.0233 * (s / 0.1)^0.836.
        5. Spatial autocorrelation filtering: eps = h * noise.
        6. Combined measurement: NoisyLR = s_speckle + sigma(s) * eps.

        Args:
            hr: (B, 1, 2H, 2W) float tensor.
            theta: Optional parameter dictionary overriding physical constants:
                - 'psf_sigma': float or (sigma_x, sigma_y) in [0.4, 1.4].
                - 'noise_scale': float, default 0.0233.
                - 'noise_exponent': float, default 0.836.
                - 'speckle_var' or 'speckle_std': float, variance <= 0.25.
                - 's_floor': float, default 0.02.
                - 'use_lut': bool, whether to use empirical LUT if available.
            generator: Optional torch.Generator for reproducible random draws.

        Returns:
            Unclamped degraded measurement (B, 1, H, W) on same device and dtype.
        """
        if hr.dim() != 4 or hr.shape[1] != 1:
            raise ValueError(f"Expected 4D single-channel tensor (B, 1, H, W), got shape {tuple(hr.shape)}")
        if hr.shape[2] % 2 != 0 or hr.shape[3] % 2 != 0:
            raise ValueError(f"Spatial dimensions must be even, got {tuple(hr.shape[2:])}")

        # Parse theta parameters
        psf_val = None
        n_scale = self.noise_scale
        n_exp = self.noise_exponent
        s_floor = self.s_floor
        speckle_std = 0.0
        use_lut = False

        if theta is not None:
            if "psf_sigma" in theta:
                psf_val = theta["psf_sigma"]
            if "noise_scale" in theta:
                n_scale = float(theta["noise_scale"])
            if "noise_exponent" in theta:
                n_exp = float(theta["noise_exponent"])
            if "s_floor" in theta:
                s_floor = float(theta["s_floor"])
            if "speckle_std" in theta:
                speckle_std = float(theta["speckle_std"])
            elif "speckle_var" in theta:
                speckle_std = float(math.sqrt(max(0.0, min(theta["speckle_var"], self.max_speckle_var))))
            if "use_lut" in theta:
                use_lut = bool(theta["use_lut"])

        # 1 & 2. Compute noise-free signal s = W4 * (PSF * GT)
        clean_theta = {"psf_sigma": psf_val} if psf_val is not None else None
        s = self.clean(hr, clean_theta)

        # 3. Multiplicative Gamma speckle
        if speckle_std > 1e-6:
            # Gamma speckle with mean 1.0 and variance sigma^2
            k_shape = 1.0 / (speckle_std**2)
            # Differentiable approximation: 1 + Normal(0, speckle_std^2)
            speckle_noise = torch.randn(s.shape, generator=generator, device=hr.device, dtype=hr.dtype) * speckle_std
            s_speckled = s * (1.0 + speckle_noise)
        else:
            s_speckled = s

        # 4 & 5. Heteroscedastic sensor noise
        if use_lut and self._lut_params is not None and hr.device == self.target_device:
            # Empirical LUT noise generator
            p = self._lut_params
            gp = F.pad(hr, (1, 2, 1, 2), mode="replicate")
            m1 = F.avg_pool2d(gp, 4, stride=2)
            m2 = F.avg_pool2d(gp * gp, 4, stride=2)
            rho = (m2 - m1 * m1).clamp_min(0).sqrt() / s.clamp_min(p["sfloor"])

            RE, nr = p["RE"], p["SIG"].shape[2]
            j = torch.searchsorted(RE, rho.detach().contiguous(), right=True).clamp(1, nr)
            gy = ((j - 0.5) / nr) * 2 - 1
            gx = s.clamp(0, 1).sqrt() * 2 - 1
            grid = torch.stack([gx, gy.to(gx.dtype).expand_as(gx)], dim=-1).squeeze(1)
            sigma = F.grid_sample(
                p["SIG"].expand(hr.shape[0], -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            u = torch.rand(s.shape, generator=generator, device=hr.device, dtype=torch.float32)
            grp = (s.detach().clamp(0, 1).sqrt() * p["ng"]).long().clamp_(0, p["ng"] - 1)
            eps = p["LUT"][grp * p["nq"] + (u * p["nq"]).long().clamp_(0, p["nq"] - 1)].to(hr.dtype)
            eps = F.conv2d(F.pad(eps, (1, 1, 1, 1), mode="replicate"), p["h"])
            noisy_lr = s_speckled + sigma * eps
        else:
            # Analytical power-law heteroscedastic noise
            sigma = self.compute_noise_sigma(
                s_speckled,
                noise_scale=n_scale,
                noise_exponent=n_exp,
                s_floor=s_floor,
            )
            raw_eps = torch.randn(s.shape, generator=generator, device=hr.device, dtype=hr.dtype)
            h = self.h_acf.to(device=hr.device, dtype=hr.dtype)
            eps = F.conv2d(F.pad(raw_eps, (1, 1, 1, 1), mode="replicate"), h)
            noisy_lr = s_speckled + sigma * eps

        return noisy_lr

    def forward(
        self,
        hr: torch.Tensor,
        theta: Optional[Dict[str, Any]] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Forward pass alias for degrade()."""
        return self.degrade(hr, theta=theta, generator=generator)

    def compute_nll(
        self,
        pred_hr: torch.Tensor,
        lr_obs: torch.Tensor,
        theta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Computes probabilistic Gaussian negative log-likelihood for physical consistency:

            -log p(y_LR | A_theta(x_hat)) = 0.5 * ((y_LR - s) / sigma)^2 + log(sigma)

        Args:
            pred_hr: High-resolution restoration prediction x_hat of shape (B, 1, 2H, 2W).
            lr_obs: Observed input measurement y_LR of shape (B, 1, H, W).
            theta: Optional forward operator parameters.

        Returns:
            Scalar NLL loss tensor for gradient backpropagation.
        """
        s = self.clean(pred_hr, theta)
        n_scale = theta.get("noise_scale", self.noise_scale) if theta else self.noise_scale
        n_exp = theta.get("noise_exponent", self.noise_exponent) if theta else self.noise_exponent
        s_floor = theta.get("s_floor", self.s_floor) if theta else self.s_floor

        sigma = self.compute_noise_sigma(s, noise_scale=n_scale, noise_exponent=n_exp, s_floor=s_floor)
        res = lr_obs - s
        nll = 0.5 * (res / sigma).pow(2) + torch.log(sigma)
        return nll.mean()

    def sample_parameters(
        self,
        randomize: bool = True,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Samples degradation parameter dictionary theta within calibrated ranges.

        Ranges:
        - psf_sigma: uniform in [0.4, 1.4] (or 0.0 if not random)
        - noise_scale: uniform in [0.018, 0.030] (calibrated 0.0233)
        - noise_exponent: uniform in [0.78, 0.90] (calibrated 0.836)
        - speckle_std: uniform in [0.0, 0.20] (variance <= 0.04 <= 0.25)
        """
        if not randomize:
            return {
                "psf_sigma": 0.0,
                "noise_scale": self.noise_scale,
                "noise_exponent": self.noise_exponent,
                "speckle_std": 0.0,
                "s_floor": self.s_floor,
            }

        _rng = rng if rng is not None else np.random.default_rng()
        psf_sigma = float(_rng.uniform(self.psf_sigma_range[0], self.psf_sigma_range[1]))
        noise_scale = float(_rng.uniform(0.018, 0.030))
        noise_exponent = float(_rng.uniform(0.78, 0.90))
        speckle_std = float(_rng.uniform(0.0, 0.20))

        return {
            "psf_sigma": psf_sigma,
            "noise_scale": noise_scale,
            "noise_exponent": noise_exponent,
            "speckle_std": speckle_std,
            "s_floor": self.s_floor,
        }
