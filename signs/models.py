"""
Classifiers for GTSRB (43 classes) that sit on top of an edge-detection front end.

All three models share the *same* backbone topology so the comparison is fair:

    front_end (filters.py) -> BN -> [conv-bn-relu x2, pool] x3 -> avgpool(4) -> FC 256 -> FC 43

    DenseCNN      - full-precision reference.
    QuantizedCNN  - quantization-aware training (QAT). Weights and activations are
                    fake-quantized to `w_bits` / `a_bits` (down to 1-bit binary
                    weights) with a straight-through estimator. Activation clip
                    ranges are learned (PACT). First/last layer default to 8 bit.
    SparseCNN     - weights carry binary masks; global magnitude pruning ramps
                    sparsity up during training. `to_sparse()` exports a model
                    that actually runs on torch sparse CSR matrices
                    (conv via im2col + sparse @ dense matmul, BN folded in).

Note: the learned edge filter (if used) stays full precision / dense in every
model -- it is tiny and is the thing being compared, so it is kept identical.

Use build_model(kind, front_end) to construct one.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_CLASSES = 43


# --------------------------------------------------------------------------- #
# Shared backbone
# --------------------------------------------------------------------------- #

def _backbone(in_ch, num_classes, conv, linear, act, width=32, hidden=256, dropout=0.3):
    """
    conv(cin, cout, first: bool) -> conv module (3x3, pad 1, no bias)
    linear(fin, fout, last: bool) -> linear module
    act() -> activation module
    """
    w1, w2, w3 = width, width * 2, width * 4
    layers = [nn.BatchNorm2d(in_ch)]  # normalises filter outputs (Canny is 0/1, Sobel unbounded, ...)
    cin = in_ch
    first = True
    for cout in (w1, w2, w3):
        for _ in range(2):
            layers += [conv(cin, cout, first), nn.BatchNorm2d(cout), act()]
            cin, first = cout, False
        layers.append(nn.MaxPool2d(2))
    features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(4), nn.Flatten())
    classifier = nn.Sequential(
        nn.Dropout(dropout),
        linear(w3 * 16, hidden, False), nn.BatchNorm1d(hidden), act(),
        nn.Dropout(dropout),
        linear(hidden, num_classes, True),
    )
    return features, classifier


class _SignNet(nn.Module):
    def __init__(self, front_end, features, classifier):
        super().__init__()
        self.front_end = front_end
        self.features = features
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.features(self.front_end(x)))

    def weight_layers(self):
        return [m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]

    def storage_bits(self) -> int:
        """Bits needed to store all trainable parameters (dense fp32)."""
        return 32 * sum(p.numel() for p in self.parameters())


class DenseCNN(_SignNet):
    def __init__(self, front_end, num_classes=NUM_CLASSES, width=32):
        conv = lambda ci, co, first: nn.Conv2d(ci, co, 3, padding=1, bias=False)
        lin = lambda fi, fo, last: nn.Linear(fi, fo)
        feats, cls = _backbone(front_end.out_channels, num_classes, conv, lin, lambda: nn.ReLU(inplace=True), width)
        super().__init__(front_end, feats, cls)


# --------------------------------------------------------------------------- #
# Quantization
# --------------------------------------------------------------------------- #

class _RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


class _SignSTE(torch.autograd.Function):
    """sign() forward, clipped identity backward (BinaryConnect / XNOR-Net)."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1).to(g.dtype)


def quantize_weight(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-output-channel fake quantization."""
    if bits >= 32:
        return w
    dims = tuple(range(1, w.dim()))
    if bits == 1:
        # binary weights scaled by mean |w| per output channel
        alpha = w.abs().mean(dim=dims, keepdim=True)
        return _SignSTE.apply(w) * alpha
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8) / qmax
    return _RoundSTE.apply(torch.clamp(w / scale, -qmax, qmax)) * scale


class QuantConv2d(nn.Conv2d):
    def __init__(self, *args, w_bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits

    def forward(self, x):
        return self._conv_forward(x, quantize_weight(self.weight, self.w_bits), self.bias)


class QuantLinear(nn.Linear):
    def __init__(self, *args, w_bits=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_bits = w_bits

    def forward(self, x):
        return F.linear(x, quantize_weight(self.weight, self.w_bits), self.bias)


class QuantReLU(nn.Module):
    """
    PACT: y = clamp(x, 0, alpha) quantized to `bits` unsigned levels,
    with alpha learned. bits >= 32 -> plain ReLU.
    """

    def __init__(self, bits=4, alpha_init=6.0):
        super().__init__()
        self.bits = bits
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x):
        if self.bits >= 32:
            return F.relu(x)
        y = torch.minimum(F.relu(x), self.alpha)
        scale = self.alpha / (2 ** self.bits - 1)
        return _RoundSTE.apply(y / scale) * scale


class QuantizedCNN(_SignNet):
    def __init__(self, front_end, num_classes=NUM_CLASSES, width=32,
                 w_bits=4, a_bits=4, first_last_bits=8):
        conv = lambda ci, co, first: QuantConv2d(ci, co, 3, padding=1, bias=False,
                                                 w_bits=first_last_bits if first else w_bits)
        lin = lambda fi, fo, last: QuantLinear(fi, fo, w_bits=first_last_bits if last else w_bits)
        act = lambda: QuantReLU(a_bits)
        feats, cls = _backbone(front_end.out_channels, num_classes, conv, lin, act, width)
        super().__init__(front_end, feats, cls)
        self.w_bits, self.a_bits, self.first_last_bits = w_bits, a_bits, first_last_bits

    def storage_bits(self) -> int:
        """Quantized weights at their bit-width + fp32 per-channel scales;
        everything else (BN, biases, filter, PACT alphas) at fp32."""
        bits = 0
        counted = set()
        for m in self.modules():
            if isinstance(m, (QuantConv2d, QuantLinear)):
                bits += m.weight.numel() * m.w_bits + 32 * m.weight.shape[0]
                counted.add(id(m.weight))
        bits += 32 * sum(p.numel() for p in self.parameters() if id(p) not in counted)
        return bits


# --------------------------------------------------------------------------- #
# Sparsity
# --------------------------------------------------------------------------- #

class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, prunable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prunable = prunable
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        return self._conv_forward(x, self.weight * self.mask, self.bias)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, prunable=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.prunable = prunable
        self.register_buffer("mask", torch.ones_like(self.weight))

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class SparseCNN(_SignNet):
    """
    Train with masks, call `set_sparsity(s)` on a schedule (see
    `cubic_sparsity_schedule`), then `to_sparse()` for real sparse inference.
    The first conv is not pruned (it has very few weights and is sensitive).
    """

    def __init__(self, front_end, num_classes=NUM_CLASSES, width=32):
        conv = lambda ci, co, first: MaskedConv2d(ci, co, 3, padding=1, bias=False, prunable=not first)
        lin = lambda fi, fo, last: MaskedLinear(fi, fo)
        feats, cls = _backbone(front_end.out_channels, num_classes, conv, lin, lambda: nn.ReLU(inplace=True), width)
        super().__init__(front_end, feats, cls)

    def masked_layers(self, prunable_only=True):
        return [m for m in self.modules()
                if isinstance(m, (MaskedConv2d, MaskedLinear)) and (m.prunable or not prunable_only)]

    @torch.no_grad()
    def set_sparsity(self, sparsity: float):
        """Global magnitude pruning: zero the smallest |w| across all prunable layers."""
        layers = self.masked_layers()
        if sparsity <= 0:
            for m in layers:
                m.mask.fill_(1.0)
            return
        scores = torch.cat([(m.weight * m.mask).abs().flatten() for m in layers])
        k = int(sparsity * scores.numel())
        if k < 1:
            return
        threshold = torch.kthvalue(scores.cpu(), k).values.item()
        for m in layers:
            m.mask.copy_(((m.weight * m.mask).abs() > threshold).float())
            m.weight.mul_(m.mask)

    @torch.no_grad()
    def apply_masks(self):
        """Keep pruned weights at exactly zero (call after optimizer.step();
        weight decay / momentum can otherwise drift them)."""
        for m in self.masked_layers(prunable_only=False):
            m.weight.mul_(m.mask)

    def sparsity(self, prunable_only=False) -> float:
        layers = self.masked_layers(prunable_only)
        total = sum(m.mask.numel() for m in layers)
        nnz = sum(int(m.mask.sum().item()) for m in layers)
        return 1.0 - nnz / total

    def nnz(self) -> int:
        return sum(int(m.mask.sum().item()) for m in self.masked_layers(False))

    def storage_bits(self, index_bits=16) -> int:
        """CSR storage: fp32 value + 16-bit column index per non-zero, plus
        32-bit row pointers; everything else fp32."""
        bits = 0
        counted = set()
        for m in self.masked_layers(False):
            nnz = int(m.mask.sum().item())
            bits += nnz * (32 + index_bits) + 32 * (m.weight.shape[0] + 1)
            counted.add(id(m.weight))
        bits += 32 * sum(p.numel() for p in self.parameters() if id(p) not in counted)
        return bits

    @torch.no_grad()
    def to_sparse(self) -> "SparseInferenceCNN":
        return SparseInferenceCNN(self)


def cubic_sparsity_schedule(step, total_steps, final_sparsity, start_step=0):
    """Zhu & Gupta (2017): s_t = s_f * (1 - (1 - t)^3), t in [0, 1]."""
    if step < start_step:
        return 0.0
    t = min(1.0, (step - start_step) / max(1, total_steps - start_step))
    return final_sparsity * (1.0 - (1.0 - t) ** 3)


# --------------------------- real sparse inference --------------------------- #

def _fold_bn(weight, bias, bn):
    """Fold an eval-mode BatchNorm into the preceding conv/linear weights."""
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight / std
    shape = (-1,) + (1,) * (weight.dim() - 1)
    w = weight * gamma.view(shape)
    b = bn.bias - bn.running_mean * gamma
    if bias is not None:
        b = b + bias * gamma
    return w, b


class SparseConv2dCSR(nn.Module):
    """3x3 conv as im2col + CSR sparse matmul. Weight is (Cout, Cin*k*k) CSR."""

    def __init__(self, weight, bias, padding, stride=1):
        super().__init__()
        self.cout, cin, kh, kw = weight.shape
        self.k, self.padding, self.stride = (kh, kw), padding, stride
        self.register_buffer("w", weight.reshape(self.cout, -1).to_sparse_csr())
        self.register_buffer("b", bias.clone())
        self.nnz = int((weight != 0).sum())

    def forward(self, x):
        N, _, H, W = x.shape
        cols = F.unfold(x, self.k, padding=self.padding, stride=self.stride)  # N, Ckk, L
        L = cols.shape[-1]
        cols = cols.transpose(0, 1).reshape(cols.shape[1], N * L)            # Ckk, N*L
        out = torch.sparse.mm(self.w, cols)                                    # Cout, N*L
        Ho = (H + 2 * self.padding - self.k[0]) // self.stride + 1
        Wo = (W + 2 * self.padding - self.k[1]) // self.stride + 1
        out = out.view(self.cout, N, Ho, Wo).transpose(0, 1)
        return out + self.b.view(1, -1, 1, 1)


class SparseLinearCSR(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()
        self.register_buffer("w", weight.to_sparse_csr())
        self.register_buffer("b", bias.clone())
        self.nnz = int((weight != 0).sum())

    def forward(self, x):
        return torch.sparse.mm(self.w, x.t().contiguous()).t() + self.b


class SparseInferenceCNN(nn.Module):
    """
    Inference-only copy of a trained SparseCNN where every masked layer is a
    CSR sparse matrix and BatchNorm is folded in. Intended for CPU latency
    measurements; accuracy should match the masked model in eval mode.
    """

    def __init__(self, model: SparseCNN):
        super().__init__()
        model = model.eval()
        self.front_end = model.front_end

        def convert(seq):
            mods = list(seq)
            out, i = [], 0
            while i < len(mods):
                m = mods[i]
                if isinstance(m, (MaskedConv2d, MaskedLinear)):
                    w = (m.weight * m.mask).detach()
                    b = m.bias.detach() if m.bias is not None else None
                    nxt = mods[i + 1] if i + 1 < len(mods) else None
                    if isinstance(nxt, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        w, b = _fold_bn(w, b, nxt)
                        i += 1
                    if b is None:
                        b = torch.zeros(w.shape[0], device=w.device)
                    if isinstance(m, MaskedConv2d):
                        out.append(SparseConv2dCSR(w, b, m.padding[0], m.stride[0]))
                    else:
                        out.append(SparseLinearCSR(w, b))
                elif isinstance(m, nn.Dropout):
                    pass
                else:
                    out.append(m)
                i += 1
            return nn.Sequential(*out)

        with torch.no_grad():
            self.features = convert(model.features)
            self.classifier = convert(model.classifier)

    @torch.no_grad()
    def forward(self, x):
        return self.classifier(self.features(self.front_end(x)))


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

MODELS = ("dense", "quant", "sparse")


def build_model(kind: str, front_end, num_classes=NUM_CLASSES, **kwargs) -> _SignNet:
    """
    build_model("dense", fe)
    build_model("quant", fe, w_bits=2, a_bits=4)
    build_model("sparse", fe)
    """
    kind = kind.lower()
    if kind == "dense":
        return DenseCNN(front_end, num_classes, **kwargs)
    if kind == "quant":
        return QuantizedCNN(front_end, num_classes, **kwargs)
    if kind == "sparse":
        return SparseCNN(front_end, num_classes, **kwargs)
    raise ValueError(f"unknown model '{kind}', choose from {MODELS}")
