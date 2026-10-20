"""Architectures built from perturbed layers.

The same model object is used for both methods: with ``perturb=False`` (the
default) every perturbed layer behaves exactly like nn.Linear / nn.Conv2d, so
backprop baselines train the identical architecture and initialisation.

Noise is injected on pre-activations (conv/linear outputs, before BN/ReLU).

Every model is a sequence of ``stages`` (forward = stages applied in order).
Stages are the units of the local-reward mode: stage s gets its own reward
from an auxiliary head on its output, and the last stage ends in the real
classifier. Within a stage, each direct child that contains perturbed layers
is one *perturbation group*, the unit of the layer-wise mode: a single layer
in the MLP/CNN, a stem conv, a whole residual block or the readout in ResNets.

Normalisation layers: in three-factor mode only perturbed layers learn. BN/GN
affine parameters stay at their init (gamma=1, beta=0), which is equivalent
to affine=False; running statistics still update in the forward pass.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn

from layers import PERTURBED_TYPES, PerturbedConv2d, PerturbedLinear


def _perturbed_in(module: nn.Module):
    return [m for m in module.modules() if isinstance(m, PERTURBED_TYPES)]


class AuxHead(nn.Module):
    """Local readout for one stage: global-average-pool (if spatial) -> linear -> logits.

    ``fixed``: a frozen random projection (no learning; the stage learns to
    make its features linearly decodable through a fixed readout).
    ``learned``: a PerturbedLinear trained by the same local reward as its
    stage, so no backprop is involved anywhere.
    """

    def __init__(self, in_features: int, num_classes: int, kind: str, sigma: float):
        super().__init__()
        self.kind = kind
        if kind == "learned":
            self.proj = PerturbedLinear(in_features, num_classes, sigma=sigma)
        elif kind == "fixed":
            self.register_buffer("weight", torch.randn(num_classes, in_features) / in_features ** 0.5)
        else:
            raise ValueError(f"unknown aux head {kind!r}")

    def forward(self, h):
        if h.dim() == 4:
            h = h.mean(dim=(2, 3))
        return self.proj(h) if self.kind == "learned" else h @ self.weight.t()


class ThreeFactorNet(nn.Module):
    """Base class: ``self.stages`` (nn.ModuleList) plus perturbation bookkeeping."""

    stages: nn.ModuleList
    # aux_heads (nn.ModuleList) is registered by build_aux_heads(). It must not be
    # a class attribute: that would shadow the registered submodule on lookup.

    def _aux_heads(self):
        return self._modules.get("aux_heads")

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x

    # -- layers ---------------------------------------------------------------
    def perturbed_layers(self):
        """Perturbed layers of the main network (excludes auxiliary heads)."""
        return _perturbed_in(self.stages)

    def named_perturbed_layers(self):
        return [(f"stages.{n}", m) for n, m in self.stages.named_modules() if isinstance(m, PERTURBED_TYPES)]

    def aux_layers(self):
        heads = self._aux_heads()
        return _perturbed_in(heads) if heads is not None else []

    def perturbation_groups(self):
        """Groups for layer-wise perturbation, in forward order."""
        groups = []
        for stage in self.stages:
            for child in stage.children():
                layers = _perturbed_in(child)
                if layers:
                    groups.append(layers)
        return groups

    def stage_index(self) -> dict:
        """Maps every perturbed layer (main and aux) to the stage it belongs to."""
        idx = {m: s for s, stage in enumerate(self.stages) for m in _perturbed_in(stage)}
        if self._aux_heads() is not None:
            idx.update({m: s for s, head in enumerate(self._aux_heads()) for m in _perturbed_in(head)})
        return idx

    # -- configuration --------------------------------------------------------
    def set_perturb(self, enabled: bool, layers=None) -> None:
        """Turn noise on for ``layers`` (default: all, including aux heads) and off for the rest."""
        active = None if layers is None else set(layers)
        for m in self.perturbed_layers() + self.aux_layers():
            m.perturb = enabled and (active is None or m in active)
            if not m.perturb:
                m.clear_trace()

    def set_sigma(self, sigma: float) -> None:
        for m in self.perturbed_layers() + self.aux_layers():
            m.sigma = sigma

    def set_topk(self, ratio: float, score: str = "abs") -> None:
        """Sparse perturbation on all hidden layers.

        The output layer (num_classes units) stays fully perturbed: it adds
        negligible variance, and top-5% of 10 logits would be a single unit.
        """
        for m in self.perturbed_layers()[:-1]:
            m.topk_ratio, m.topk_score = ratio, score

    def clear_traces(self) -> None:
        for m in self.perturbed_layers() + self.aux_layers():
            m.clear_trace()

    def build_aux_heads(self, kind: str, num_classes: int, sigma: float) -> None:
        """One head per non-final stage; the final stage uses the real classifier."""
        dims = self.stage_out_features()
        self.aux_heads = nn.ModuleList(AuxHead(d, num_classes, kind, sigma) for d in dims[:-1])

    def stage_out_features(self) -> list[int]:
        raise NotImplementedError


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "gn":
        return nn.GroupNorm(min(32, channels), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class MLP(ThreeFactorNet):
    """784 -> 256 -> 10 for MNIST (input size inferred for other datasets)."""

    def __init__(self, in_shape=(1, 28, 28), num_classes=10, hidden=256, sigma=0.05):
        super().__init__()
        in_features = 1
        for d in in_shape:
            in_features *= d
        self.hidden = hidden
        self.stages = nn.ModuleList([
            nn.Sequential(OrderedDict(flatten=nn.Flatten(), fc1=PerturbedLinear(in_features, hidden, sigma=sigma),
                                      relu=nn.ReLU())),
            nn.Sequential(OrderedDict(fc2=PerturbedLinear(hidden, num_classes, sigma=sigma))),
        ])

    def stage_out_features(self):
        return [self.hidden, None]


class ShallowCNN(ThreeFactorNet):
    """Three conv blocks + linear readout (CIFAR-10: 32 -> 16 -> 8 -> 4). One stage per conv block."""

    def __init__(self, in_shape=(3, 32, 32), num_classes=10, width=32, sigma=0.05):
        super().__init__()
        c, h, w = in_shape
        self.chans = [c, width, 2 * width, 4 * width]
        stages = []
        for i, (cin, cout) in enumerate(zip(self.chans[:-1], self.chans[1:])):
            block = OrderedDict(conv=PerturbedConv2d(cin, cout, 3, padding=1, sigma=sigma),
                                relu=nn.ReLU(inplace=True), pool=nn.MaxPool2d(2))
            h, w = h // 2, w // 2
            if i == 2:
                block["flatten"] = nn.Flatten()
                block["classifier"] = PerturbedLinear(cout * h * w, num_classes, sigma=sigma)
            stages.append(nn.Sequential(block))
        self.stages = nn.ModuleList(stages)

    def stage_out_features(self):
        return self.chans[1:3] + [None]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride, norm, sigma):
        super().__init__()
        self.conv1 = PerturbedConv2d(cin, cout, 3, stride=stride, padding=1, bias=False, sigma=sigma)
        self.bn1 = _norm(norm, cout)
        self.conv2 = PerturbedConv2d(cout, cout, 3, stride=1, padding=1, bias=False, sigma=sigma)
        self.bn2 = _norm(norm, cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                PerturbedConv2d(cin, cout, 1, stride=stride, bias=False, sigma=sigma),
                _norm(norm, cout),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNet(ThreeFactorNet):
    """CIFAR-style ResNet (3x3 stem, no max-pool). ``blocks=(2,2,2,2)`` is ResNet-18.

    ``width`` scales all channel counts (64 -> standard ResNet-18). Stages are
    the four resolution stages; the stem belongs to stage 0 and the
    pool + classifier to stage 3.
    """

    def __init__(self, in_shape=(3, 32, 32), num_classes=10, blocks=(2, 2, 2, 2), width=64,
                 norm="bn", sigma=0.05):
        super().__init__()
        c = in_shape[0]
        stem = OrderedDict(conv1=PerturbedConv2d(c, width, 3, padding=1, bias=False, sigma=sigma),
                           bn1=_norm(norm, width), relu=nn.ReLU(inplace=True))
        stages, cin = [], width
        self.widths = []
        for i, n in enumerate(blocks):
            cout = width * 2 ** i
            stage = OrderedDict(stem) if i == 0 else OrderedDict()
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                stage[f"block{j}"] = BasicBlock(cin, cout, stride, norm, sigma)
                cin = cout
            if i == len(blocks) - 1:
                stage["pool"] = nn.AdaptiveAvgPool2d(1)
                stage["flatten"] = nn.Flatten()
                stage["fc"] = PerturbedLinear(cin, num_classes, sigma=sigma)
            stages.append(nn.Sequential(stage))
            self.widths.append(cout)
        self.stages = nn.ModuleList(stages)

    def stage_out_features(self):
        return self.widths[:-1] + [None]


def build_model(name: str, in_shape, num_classes: int, sigma: float, width: int | None = None,
                norm: str = "bn") -> ThreeFactorNet:
    name = name.lower()
    if name == "mlp":
        return MLP(in_shape, num_classes, hidden=width or 256, sigma=sigma)
    if name in ("cnn", "shallowcnn"):
        return ShallowCNN(in_shape, num_classes, width=width or 32, sigma=sigma)
    if name in ("resnet", "resnet18"):
        return ResNet(in_shape, num_classes, blocks=(2, 2, 2, 2), width=width or 64, norm=norm, sigma=sigma)
    if name == "resnet10":
        return ResNet(in_shape, num_classes, blocks=(1, 1, 1, 1), width=width or 32, norm=norm, sigma=sigma)
    raise ValueError(f"unknown model {name!r}")
