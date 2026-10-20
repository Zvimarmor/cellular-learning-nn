"""Perturbed layers for three-factor (node-perturbation) learning.

Each perturbed layer is an independent "agent":

    1. forward:      z = W x + b
    2. perturbation: z~ = z + xi,           xi ~ N(0, sigma^2)        (factor 1: local noise)
    3. trace:        E = xi x^T                                       (factor 2: pre-synaptic input)
    4. reward:       R = L(z~) - baseline                             (factor 3: global, scalar per sample)
    5. update:       dW = -(eta / sigma^2) * R * E

Why the 1/sigma^2: for small sigma, L(z + xi) - L(z) ~= xi . dL/dz, so
E[R xi] = sigma^2 dL/dz and E[R xi x^T] / sigma^2 = dL/dz x^T = dL/dW. With the
normalisation, eta has the same meaning as an SGD learning rate and the update
is an unbiased (for sigma -> 0) estimator of the true gradient. Without it, the
effective learning rate silently changes with sigma.

Memory: materialising the per-sample eligibility trace E_b (shape B x out x in,
or B x O x C x kh x kw for convs) costs B times the parameter count of the
layer, which is prohibitive for ResNets. Since the update only ever needs the
reward-weighted sum over the batch,

    sum_b R_b E_b = sum_b (R_b xi_b) x_b^T,

we store only the factors (the layer input x and the noise xi, i.e. exactly
what autograd would store as activations) and contract them once the reward is
known. ``eligibility_trace()`` still returns the explicit per-sample E for
inspection and testing.

No autograd graph is involved anywhere: the three-factor forward pass runs under
``torch.no_grad()`` and estimates are written directly into buffers.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.grad import conv2d_weight


class PerturbedMixin:
    """State and bookkeeping shared by all perturbed layers."""

    def _init_perturbation(self, sigma: float) -> None:
        self.sigma = sigma
        self.perturb = False          # toggled by ThreeFactorNet.set_perturb()
        # Top-K sparse perturbation: noise only on the topk_ratio most active units
        # of each sample. 1.0 = perturb every unit.
        self.topk_ratio = 1.0
        self.topk_score = "abs"       # "abs": rank by |z|; "pos": rank by z (ReLU-active units)
        self._x: torch.Tensor | None = None    # layer input (pre-synaptic activity)
        self._xi: torch.Tensor | None = None   # injected noise
        # Accumulated gradient estimates, (1/(B sigma^2)) sum_b R_b E_b.
        # Allocated lazily so that a layer used only with backprop costs nothing.
        self.weight_est: torch.Tensor | None = None
        self.bias_est: torch.Tensor | None = None

    def _perturb_output(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.perturb:
            return z
        xi = torch.randn_like(z).mul_(self.sigma)
        if self.topk_ratio < 1.0:
            xi.mul_(self.topk_mask(z))
        self._x = x.detach()
        self._xi = xi
        return z + xi

    def topk_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Per-sample mask of the ceil(topk_ratio * n) most active units, n = units per sample.

        Ranking uses the pre-noise z, so the mask is deterministic given the
        input. The node-perturbation estimator then targets the *masked*
        gradient sum_b (m_b * dL/dz_b) x_b^T: unbiased for the selected units,
        zero elsewhere, with variance set by K units instead of n.

        The masked xi is kept dense. Contracting a zero-padded xi with
        addmm / conv2d_weight gives exactly the sparse sum over the K selected
        units and is faster on GPU than sparse kernels at K/n ~ 5%. What reduces
        the variance is the dimension of the noise, not the storage format.
        """
        score = z.detach().flatten(1)
        if self.topk_score == "abs":
            score = score.abs()
        k = max(1, math.ceil(self.topk_ratio * score.size(1)))
        thresh = score.topk(k, dim=1, sorted=False).values.min(dim=1, keepdim=True).values
        return (score >= thresh).view_as(z).to(z.dtype)

    def clear_trace(self) -> None:
        self._x = None
        self._xi = None

    def zero_estimate(self) -> None:
        if self.weight_est is None:
            self.weight_est = torch.zeros_like(self.weight)
            if self.bias is not None:
                self.bias_est = torch.zeros_like(self.bias)
        else:
            self.weight_est.zero_()
            if self.bias_est is not None:
                self.bias_est.zero_()

    @torch.no_grad()
    def accumulate_estimate(self, reward: torch.Tensor, scale: float) -> None:
        """Add ``scale * sum_b reward_b * E_b`` to the running estimate.

        ``reward`` has shape (B,) for the current (micro-)batch. ``scale`` is
        1 / (B_total * sigma^2), so that calling this once per micro-batch yields
        the full-batch mean estimate.
        """
        if self._xi is None:
            raise RuntimeError(
                f"{type(self).__name__} has no stored trace; run a forward pass with perturb=True first."
            )
        # R_b * xi_b, broadcast over all non-batch dims of xi.
        r_xi = self._xi * reward.view(-1, *([1] * (self._xi.dim() - 1)))
        self._accumulate(r_xi, scale)

    def _accumulate(self, r_xi: torch.Tensor, scale: float) -> None:
        raise NotImplementedError

    def eligibility_trace(self) -> torch.Tensor:
        raise NotImplementedError


class PerturbedLinear(PerturbedMixin, nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sigma: float = 0.05):
        super().__init__(in_features, out_features, bias=bias)
        self._init_perturbation(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._perturb_output(x, F.linear(x, self.weight, self.bias))

    def _accumulate(self, r_xi: torch.Tensor, scale: float) -> None:
        # sum_b (R_b xi_b) x_b^T  ==  (R*xi)^T @ X       -> (out, in)
        r_xi = r_xi.reshape(-1, self.out_features)
        x = self._x.reshape(-1, self.in_features)
        self.weight_est.addmm_(r_xi.t(), x, alpha=scale)
        if self.bias_est is not None:
            # Bias sees a constant input of 1, so its trace is just xi.
            self.bias_est.add_(r_xi.sum(0), alpha=scale)

    def eligibility_trace(self) -> torch.Tensor:
        """Per-sample E_b = xi_b x_b^T, shape (B, out, in). Memory heavy; for inspection only."""
        return torch.einsum("bo,bi->boi", self._xi, self._x)


class PerturbedConv2d(PerturbedMixin, nn.Conv2d):
    """Conv2d with node perturbation on every output unit (channel x spatial location).

    How the conv eligibility trace is flattened
    -------------------------------------------
    A convolution is a linear layer applied at every output location p with
    *shared* weights. Unfold (im2col) the input into patches:

        X_unf[b, k, p]   with k = (c, i, j) indexing in_channels x kh x kw
                         and  p indexing the H_out x W_out output locations,
        W_flat[o, k]     the weight reshaped to (out_channels, C*kh*kw).

    Then the forward pass is a batched matrix product:

        z[b, o, p] = sum_k W_flat[o, k] X_unf[b, k, p] + bias[o].

    Noise is injected independently at every output unit: xi[b, o, p].
    Each location p is its own "neuron" with its own pre-synaptic input
    X_unf[b, :, p], so it contributes its own local trace xi[b,o,p] X_unf[b,k,p].
    Because the weight W_flat[o, k] is *shared* by all locations, the trace of
    the shared synapse is the sum over locations:

        E[b, o, k] = sum_p xi[b, o, p] * X_unf[b, k, p]
                   = (xi_flat[b] @ X_unf[b]^T)[o, k]      xi_flat: (B, O, P)

    i.e. one (O x P) @ (P x C*kh*kw) matmul per sample, reshaped back to
    (O, C, kh, kw). The bias trace is sum_p xi[b, o, p].

    The update needs sum_b R_b E[b], which is the same as the expression above
    with xi replaced by R_b * xi. That contraction -- "correlate an output-shaped
    tensor with input patches" -- is exactly the weight-gradient kernel of a
    convolution, so we call ``torch.nn.grad.conv2d_weight(x, W.shape, R*xi)``.
    This is a plain function (no autograd graph), handles stride / padding /
    dilation / groups, and never materialises X_unf, which for a 64-channel
    3x3 conv on 32x32 inputs at batch 128 would be 128*576*1024 floats ~ 300 MB
    per layer. ``eligibility_trace()`` implements the explicit unfold version.
    """

    def __init__(self, *args, sigma: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_perturbation(sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._perturb_output(x, self._conv_forward(x, self.weight, self.bias))

    def _accumulate(self, r_xi: torch.Tensor, scale: float) -> None:
        if self.padding_mode != "zeros":
            raise NotImplementedError("Only padding_mode='zeros' is supported for the eligibility trace.")
        grad_w = conv2d_weight(
            self._x, self.weight.shape, r_xi,
            stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups,
        )
        self.weight_est.add_(grad_w, alpha=scale)
        if self.bias_est is not None:
            self.bias_est.add_(r_xi.sum(dim=(0, 2, 3)), alpha=scale)

    def eligibility_trace(self) -> torch.Tensor:
        """Explicit per-sample E via unfold, shape (B, O, C, kh, kw). Memory heavy; for inspection only."""
        if self.groups != 1 or self.padding_mode != "zeros":
            raise NotImplementedError("Explicit trace only implemented for groups=1, zero padding.")
        x_unf = F.unfold(self._x, self.kernel_size, dilation=self.dilation,
                         padding=self.padding, stride=self.stride)       # (B, C*kh*kw, P)
        xi_flat = self._xi.flatten(2)                                     # (B, O, P)
        e = torch.bmm(xi_flat, x_unf.transpose(1, 2))                     # (B, O, C*kh*kw)
        return e.view(e.size(0), *self.weight.shape)


PERTURBED_TYPES = (PerturbedLinear, PerturbedConv2d)
