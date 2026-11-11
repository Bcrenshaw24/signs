"""
Edge-detection for the sign classifier.

Every filter is an nn.Module that takes a batch of images in [0, 1]
(shape N x 3 x H x W) and returns an edge map (N x C x H x W), where
C == filter.out_channels. That lets models.py size its first conv layer
from whichever filter it is given.

    LearnedEdgeFilter  - trainable conv kernels (initialised as Sobel/Scharr/Laplacian)
    SobelFilter        - fixed Sobel gradients: gx, gy, magnitude
    CannyFilter        - full Canny pipeline in torch (blur -> Sobel -> NMS ->
                         double threshold -> hysteresis), batched and GPU-friendly
    IdentityFilter     - no edge detection (raw RGB), useful as a baseline

Use build_filter(name) to construct one by name.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]])
SOBEL_Y = SOBEL_X.t().contiguous()

SCHARR_X = torch.tensor([[-3.0, 0.0, 3.0],
                         [-10.0, 0.0, 10.0],
                         [-3.0, 0.0, 3.0]]) / 4.0
SCHARR_Y = SCHARR_X.t().contiguous()

DIAG_45 = torch.tensor([[0.0, 1.0, 2.0],
                        [-1.0, 0.0, 1.0],
                        [-2.0, -1.0, 0.0]])
DIAG_135 = torch.tensor([[-2.0, -1.0, 0.0],
                         [-1.0, 0.0, 1.0],
                         [0.0, 1.0, 2.0]])

LAPLACIAN = torch.tensor([[0.0, 1.0, 0.0],
                          [1.0, -4.0, 1.0],
                          [0.0, 1.0, 0.0]])


def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma. Accepts N x 3 x H x W or N x 1 x H x W."""
    if x.shape[1] == 1:
        return x
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def gaussian_kernel2d(size: int, sigma: float) -> torch.Tensor:
    ax = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
    g = torch.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def _resize_kernel(k: torch.Tensor, size: int) -> torch.Tensor:
    """Centre a 3x3 kernel inside a size x size kernel (zero padded)."""
    if size == 3:
        return k
    out = torch.zeros(size, size)
    o = (size - 3) // 2
    out[o:o + 3, o:o + 3] = k
    return out


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #

class IdentityFilter(nn.Module):
    """No edge detection -- passes RGB straight through. Baseline."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.out_channels = in_channels

    def forward(self, x):
        return x


class LearnedEdgeFilter(nn.Module):
    """
    A bank of trainable convolution kernels that act as a learned edge detector.

    Args:
        in_channels:   3 to look at colour edges (red/blue sign borders matter
                       for GTSRB), 1 to work on grayscale.
        out_channels:  number of learned kernels.
        kernel_size:   odd kernel size (3 or 5 are sensible).
        init:          "sobel"  -> first kernels start as Sobel x/y, diagonals,
                                   Scharr, Laplacian; the rest random.
                       "random" -> kaiming init.
        zero_mean:     if True each kernel is forced to sum to zero (per input
                       channel). A zero-sum kernel gives 0 response on flat
                       regions, i.e. it can *only* respond to edges/texture.
                       This is what keeps it an "edge" filter rather than a
                       generic first conv layer.
        magnitude:     if True, output |response| (orientation-sign invariant).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 8,
                 kernel_size: int = 3, init: str = "sobel",
                 zero_mean: bool = True, magnitude: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.zero_mean = zero_mean
        self.magnitude = magnitude
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels,
                                               kernel_size, kernel_size))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")

        if init == "sobel":
            bank = [SOBEL_X, SOBEL_Y, DIAG_45, DIAG_135,
                    SCHARR_X, SCHARR_Y, LAPLACIAN]
            with torch.no_grad():
                for i, k in enumerate(bank[:out_channels]):
                    k = _resize_kernel(k, kernel_size) / k.abs().sum() * 4.0
                    # same kernel on every input channel -> starts as a
                    # luminance edge detector, free to become colour-specific
                    self.weight[i] = k.unsqueeze(0).repeat(in_channels, 1, 1) / in_channels
        elif init != "random":
            raise ValueError(f"unknown init '{init}'")

    def kernels(self) -> torch.Tensor:
        w = self.weight
        if self.zero_mean:
            w = w - w.mean(dim=(2, 3), keepdim=True)
        return w

    def forward(self, x):
        if self.in_channels == 1:
            x = rgb_to_gray(x)
        y = F.conv2d(x, self.kernels(), padding=self.padding)
        return y.abs() if self.magnitude else y


class SobelFilter(nn.Module):
    """
    Fixed (non-trainable) Sobel operator on the grayscale image.
    Output channels: [gx, gy, |g|].
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("kernel", torch.stack([SOBEL_X, SOBEL_Y]).unsqueeze(1))  # 2x1x3x3
        self.out_channels = 3

    def forward(self, x):
        g = F.conv2d(F.pad(rgb_to_gray(x), (1, 1, 1, 1), mode="replicate"), self.kernel)
        gx, gy = g[:, 0:1], g[:, 1:2]
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        return torch.cat([gx, gy, mag], dim=1)


class CannyFilter(nn.Module):
    """
    Canny edge detector implemented with tensor ops so it runs batched on GPU.

    Pipeline: grayscale -> Gaussian blur -> Sobel -> non-max suppression
              -> double threshold -> hysteresis.

    Thresholds are relative to each image's max gradient magnitude, which
    makes them robust to GTSRB's huge brightness variation.

    Args:
        sigma, blur_size:        Gaussian smoothing.
        low_ratio, high_ratio:   hysteresis thresholds as a fraction of the
                                 per-image max gradient.
        hysteresis_iters:        max number of edge-growing passes.
        return_magnitude:        if True output 2 channels: [binary edges,
                                 thinned (post-NMS) magnitude]. The binary map
                                 alone throws away a lot of signal on 32x32
                                 images, so this is worth experimenting with.

    Not differentiable (thresholding), but it has no parameters so that
    doesn't matter -- gradients simply stop here.
    """

    def __init__(self, sigma: float = 1.0, blur_size: int = 5,
                 low_ratio: float = 0.1, high_ratio: float = 0.25,
                 hysteresis_iters: int = 16, return_magnitude: bool = False):
        super().__init__()
        self.register_buffer("gauss", gaussian_kernel2d(blur_size, sigma).view(1, 1, blur_size, blur_size))
        self.register_buffer("sobel", torch.stack([SOBEL_X, SOBEL_Y]).unsqueeze(1))
        self.blur_pad = blur_size // 2
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.hysteresis_iters = hysteresis_iters
        self.return_magnitude = return_magnitude
        self.out_channels = 2 if return_magnitude else 1

    @staticmethod
    def _shift(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Returns s with s[..., y, x] = t[..., y+dy, x+dx] (zero outside)."""
        H, W = t.shape[-2:]
        p = F.pad(t, (1, 1, 1, 1))
        return p[..., 1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

    @torch.no_grad()
    def forward(self, x):
        g = rgb_to_gray(x)
        g = F.conv2d(F.pad(g, (self.blur_pad,) * 4, mode="replicate"), self.gauss)

        grad = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), self.sobel)
        gx, gy = grad[:, 0:1], grad[:, 1:2]
        mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
        mag = mag / mag.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)  # per-image [0,1]

        # gradient direction quantised to 0/45/90/135 degrees
        ang = torch.rad2deg(torch.atan2(gy, gx)) % 180.0
        d0 = (ang < 22.5) | (ang >= 157.5)
        d45 = (ang >= 22.5) & (ang < 67.5)
        d90 = (ang >= 67.5) & (ang < 112.5)
        d135 = (ang >= 112.5) & (ang < 157.5)

        s = self._shift
        # neighbours along the gradient direction (y axis points down)
        keep = (
            (d0 & (mag >= s(mag, 0, 1)) & (mag >= s(mag, 0, -1)))
            | (d45 & (mag >= s(mag, 1, 1)) & (mag >= s(mag, -1, -1)))
            | (d90 & (mag >= s(mag, 1, 0)) & (mag >= s(mag, -1, 0)))
            | (d135 & (mag >= s(mag, 1, -1)) & (mag >= s(mag, -1, 1)))
        )
        nms = mag * keep

        strong = nms >= self.high_ratio
        weak = nms >= self.low_ratio
        # hysteresis: grow strong edges into connected weak pixels
        edges = strong.float()
        weak_f = weak.float()
        for _ in range(self.hysteresis_iters):
            grown = F.max_pool2d(edges, 3, stride=1, padding=1) * weak_f
            if torch.equal(grown, edges):
                break
            edges = grown

        if self.return_magnitude:
            return torch.cat([edges, nms], dim=1)
        return edges


class FrontEnd(nn.Module):
    """Filter, optionally concatenated with the original image."""

    def __init__(self, filt: nn.Module, concat_input: bool = False, in_channels: int = 3):
        super().__init__()
        self.filt = filt
        self.concat_input = concat_input
        self.out_channels = filt.out_channels + (in_channels if concat_input else 0)

    def forward(self, x):
        e = self.filt(x)
        return torch.cat([e, x], dim=1) if self.concat_input else e


FILTERS = ("none", "sobel", "canny", "learned")


def build_filter(name: str, concat_input: bool = False, **kwargs) -> FrontEnd:
    """
    build_filter("sobel")
    build_filter("canny", return_magnitude=True)
    build_filter("learned", out_channels=16, kernel_size=5)
    build_filter("learned", concat_input=True)   # edges + raw RGB
    """
    name = name.lower()
    if name == "none":
        f = IdentityFilter()
    elif name == "sobel":
        f = SobelFilter(**kwargs)
    elif name == "canny":
        f = CannyFilter(**kwargs)
    elif name == "learned":
        f = LearnedEdgeFilter(**kwargs)
    else:
        raise ValueError(f"unknown filter '{name}', choose from {FILTERS}")
    return FrontEnd(f, concat_input=concat_input)
