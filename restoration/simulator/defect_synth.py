"""Synthetic Semiconductor Defect Generator for SEM Metrology & Defect-Invariance Regularization.

Generates realistic semiconductor nanolithography defects:
1. Void Defects: Missing conductive material / line breaks (elliptical / rectangular gaps).
2. Bridge Defects: Undesired conductive connections / shorts bridging adjacent lines.
3. Line-Edge Roughness (LER): Stochastic high-frequency boundary fluctuations localized to edges.
4. Protrusion / Intrusion Defects: Localized outward bumps or inward notches on feature edges.

Provides defect perturbation tensors delta_defect for:
- Defect-Invariance Regularization L_defect: Enforcing F_d(x + delta_defect) ~ F_d(x).
- Robustness evaluation & defect fidelity benchmark testing.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


class SyntheticDefectGenerator:
    """Generates realistic semiconductor nanolithography defect perturbations delta_defect."""

    SUPPORTED_TYPES = ("void", "bridge", "ler", "protrusion", "intrusion")

    def __init__(
        self,
        defect_types: Sequence[str] = ("void", "bridge", "ler", "protrusion", "intrusion"),
        prob: float = 0.5,
        max_defects_per_image: int = 3,
        void_radius_range: Tuple[int, int] = (2, 8),
        bridge_length_range: Tuple[int, int] = (4, 16),
        bridge_width_range: Tuple[int, int] = (1, 4),
        ler_amplitude: float = 0.05,
    ) -> None:
        self.defect_types = [t for t in defect_types if t in self.SUPPORTED_TYPES]
        if not self.defect_types:
            self.defect_types = list(self.SUPPORTED_TYPES)
        self.prob = prob
        self.max_defects_per_image = max_defects_per_image
        self.void_radius_range = void_radius_range
        self.bridge_length_range = bridge_length_range
        self.bridge_width_range = bridge_width_range
        self.ler_amplitude = ler_amplitude

    @staticmethod
    def _compute_edge_map(img: torch.Tensor) -> torch.Tensor:
        """Computes Sobel gradient magnitude edge map of single-channel image tensor."""
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        pad_img = F.pad(img, (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(pad_img, sobel_x)
        gy = F.conv2d(pad_img, sobel_y)
        return torch.sqrt(gx**2 + gy**2)

    def add_void(
        self,
        img: torch.Tensor,
        center: Optional[Tuple[int, int]] = None,
        radius: Optional[Tuple[int, int]] = None,
        intensity_drop: float = 0.6,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Injects a void (missing pattern material / gap) into the image."""
        _rng = rng if rng is not None else np.random.default_rng()
        b, c, h, w = img.shape

        if center is None:
            cy = int(_rng.integers(h // 6, 5 * h // 6))
            cx = int(_rng.integers(w // 6, 5 * w // 6))
        else:
            cy, cx = center

        if radius is None:
            ry = int(_rng.integers(self.void_radius_range[0], self.void_radius_range[1] + 1))
            rx = int(_rng.integers(self.void_radius_range[0], self.void_radius_range[1] + 1))
        else:
            ry, rx = radius

        y_coords = torch.arange(h, device=img.device, dtype=torch.float32).view(1, 1, h, 1)
        x_coords = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, 1, w)

        dist_sq = ((y_coords - cy) / max(ry, 1)) ** 2 + ((x_coords - cx) / max(rx, 1)) ** 2
        mask = torch.exp(-0.5 * dist_sq)  # Smooth Gaussian elliptical mask

        # Void reduces foreground line intensity
        perturbed = img - intensity_drop * mask
        meta = {
            "type": "void",
            "center": (cy, cx),
            "radius": (ry, rx),
            "intensity_drop": intensity_drop,
        }
        return perturbed, meta

    def add_bridge(
        self,
        img: torch.Tensor,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None,
        width: Optional[int] = None,
        intensity_boost: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Injects a conductive bridge defect between two spatial points."""
        _rng = rng if rng is not None else np.random.default_rng()
        b, c, h, w = img.shape

        if start is None:
            sy = int(_rng.integers(h // 6, 5 * h // 6))
            sx = int(_rng.integers(w // 6, 5 * w // 6))
        else:
            sy, sx = start

        if end is None:
            length = int(_rng.integers(self.bridge_length_range[0], self.bridge_length_range[1] + 1))
            angle = float(_rng.uniform(0, 2 * math.pi))
            ey = int(np.clip(sy + length * math.sin(angle), 2, h - 3))
            ex = int(np.clip(sx + length * math.cos(angle), 2, w - 3))
        else:
            ey, ex = end

        bw = width if width is not None else int(_rng.integers(self.bridge_width_range[0], self.bridge_width_range[1] + 1))

        # Compute line distance field
        y_coords = torch.arange(h, device=img.device, dtype=torch.float32).view(1, 1, h, 1)
        x_coords = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, 1, w)

        p1 = np.array([sy, sx], dtype=np.float32)
        p2 = np.array([ey, ex], dtype=np.float32)
        vec = p2 - p1
        length_sq = max(float(np.sum(vec**2)), 1e-4)

        t = ((y_coords - sy) * vec[0] + (x_coords - sx) * vec[1]) / length_sq
        t_clamped = torch.clamp(t, 0.0, 1.0)
        proj_y = sy + t_clamped * vec[0]
        proj_x = sx + t_clamped * vec[1]

        dist_sq = (y_coords - proj_y) ** 2 + (x_coords - proj_x) ** 2
        mask = torch.exp(-0.5 * (dist_sq / max(bw**2, 1)))

        perturbed = img + intensity_boost * mask
        meta = {
            "type": "bridge",
            "start": (sy, sx),
            "end": (ey, ex),
            "width": bw,
            "intensity_boost": intensity_boost,
        }
        return perturbed, meta

    def add_ler(
        self,
        img: torch.Tensor,
        amplitude: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Injects line-edge roughness (LER) perturbations localized to semiconductor edges."""
        _rng = rng if rng is not None else np.random.default_rng()
        amp = self.ler_amplitude if amplitude is None else amplitude
        edge_map = self._compute_edge_map(img)
        edge_mask = (edge_map > 0.1).float()

        # Band-limited Gaussian random field for correlated edge roughness
        noise = torch.randn_like(img)
        # Apply 3x3 Gaussian smoothing to noise
        smooth_kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            device=img.device,
            dtype=img.dtype,
        ).view(1, 1, 3, 3) / 16.0
        smooth_noise = F.conv2d(F.pad(noise, (1, 1, 1, 1), mode="replicate"), smooth_kernel)

        delta = amp * smooth_noise * edge_mask
        perturbed = img + delta
        meta = {"type": "ler", "amplitude": amp}
        return perturbed, meta

    def add_protrusion(
        self,
        img: torch.Tensor,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 3,
        height: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Injects an edge protrusion (bump extending out of a line edge)."""
        _rng = rng if rng is not None else np.random.default_rng()
        b, c, h, w = img.shape

        if center is None:
            edge_map = self._compute_edge_map(img)
            edge_idx = (edge_map[0, 0] > 0.15).nonzero()
            if len(edge_idx) > 0:
                pick = int(_rng.integers(0, len(edge_idx)))
                cy, cx = int(edge_idx[pick, 0].item()), int(edge_idx[pick, 1].item())
            else:
                cy, cx = h // 2, w // 2
        else:
            cy, cx = center

        y_coords = torch.arange(h, device=img.device, dtype=torch.float32).view(1, 1, h, 1)
        x_coords = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, 1, w)

        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        mask = torch.exp(-0.5 * (dist_sq / max(radius**2, 1)))

        perturbed = img + height * mask
        meta = {"type": "protrusion", "center": (cy, cx), "radius": radius, "height": height}
        return perturbed, meta

    def add_intrusion(
        self,
        img: torch.Tensor,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 3,
        depth: float = 0.4,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Injects an edge intrusion (notch / mouse-bite into a line edge)."""
        _rng = rng if rng is not None else np.random.default_rng()
        b, c, h, w = img.shape

        if center is None:
            edge_map = self._compute_edge_map(img)
            edge_idx = (edge_map[0, 0] > 0.15).nonzero()
            if len(edge_idx) > 0:
                pick = int(_rng.integers(0, len(edge_idx)))
                cy, cx = int(edge_idx[pick, 0].item()), int(edge_idx[pick, 1].item())
            else:
                cy, cx = h // 2, w // 2
        else:
            cy, cx = center

        y_coords = torch.arange(h, device=img.device, dtype=torch.float32).view(1, 1, h, 1)
        x_coords = torch.arange(w, device=img.device, dtype=torch.float32).view(1, 1, 1, w)

        dist_sq = (y_coords - cy) ** 2 + (x_coords - cx) ** 2
        mask = torch.exp(-0.5 * (dist_sq / max(radius**2, 1)))

        perturbed = img - depth * mask
        meta = {"type": "intrusion", "center": (cy, cx), "radius": radius, "depth": depth}
        return perturbed, meta

    def generate(
        self,
        img: torch.Tensor,
        num_defects: Optional[int] = None,
        defect_types: Optional[Sequence[str]] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """Generates realistic semiconductor defect perturbations on input image tensor.

        Args:
            img: (B, 1, H, W) or (1, H, W) float image tensor.
            num_defects: Number of defects to inject. Defaults to random in [1, max_defects_per_image].
            defect_types: Allowed defect types. Defaults to self.defect_types.
            rng: Optional NumPy random generator.

        Returns:
            Tuple of:
                - perturbed_img: Image tensor with injected defects.
                - delta_defect: Exact perturbation map (perturbed_img - img).
                - defect_meta: List of metadata dictionaries for each defect injected.
        """
        _rng = rng if rng is not None else np.random.default_rng()
        if img.dim() == 3:
            img = img.unsqueeze(0)

        types = [t for t in (defect_types if defect_types is not None else self.defect_types) if t in self.SUPPORTED_TYPES]
        if not types:
            types = list(self.SUPPORTED_TYPES)

        k = num_defects if num_defects is not None else int(_rng.integers(1, self.max_defects_per_image + 1))
        current = img.clone()
        meta_list = []

        for _ in range(k):
            defect_choice = str(_rng.choice(types))
            if defect_choice == "void":
                current, meta = self.add_void(current, rng=_rng)
            elif defect_choice == "bridge":
                current, meta = self.add_bridge(current, rng=_rng)
            elif defect_choice == "ler":
                current, meta = self.add_ler(current, rng=_rng)
            elif defect_choice == "protrusion":
                current, meta = self.add_protrusion(current, rng=_rng)
            elif defect_choice == "intrusion":
                current, meta = self.add_intrusion(current, rng=_rng)
            else:
                current, meta = self.add_void(current, rng=_rng)
            meta_list.append(meta)

        delta_defect = current - img
        return current, delta_defect, meta_list
