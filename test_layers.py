"""Sanity checks for the eligibility traces. Run: python -m pytest test_layers.py -q"""

import torch
import torch.nn.functional as F

from layers import PerturbedConv2d, PerturbedLinear
from models import build_model


def _fill(layer, x, reward):
    layer.perturb = True
    layer(x)
    layer.zero_estimate()
    layer.accumulate_estimate(reward, scale=1.0)
    return layer


def test_linear_fused_matches_explicit_trace():
    torch.manual_seed(0)
    layer = PerturbedLinear(7, 5, sigma=0.3)
    r = torch.randn(4)
    _fill(layer, torch.randn(4, 7), r)
    explicit = (r.view(-1, 1, 1) * layer.eligibility_trace()).sum(0)
    torch.testing.assert_close(layer.weight_est, explicit)
    torch.testing.assert_close(layer.bias_est, (r.view(-1, 1) * layer._xi).sum(0))


def test_conv_fused_matches_unfold_trace():
    torch.manual_seed(0)
    for kw in [dict(padding=1), dict(stride=2, padding=1), dict(dilation=2, padding=2)]:
        layer = PerturbedConv2d(3, 6, 3, sigma=0.3, **kw)
        r = torch.randn(4)
        _fill(layer, torch.randn(4, 3, 11, 11), r)
        explicit = (r.view(-1, 1, 1, 1, 1) * layer.eligibility_trace()).sum(0)
        torch.testing.assert_close(layer.weight_est, explicit, rtol=1e-4, atol=1e-4)


def test_estimator_aligns_with_true_gradient():
    """Averaged over many noise draws, R E / sigma^2 should point along dL/dW.

    The estimator is unbiased but its variance grows with the number of
    perturbed units, so this uses a tiny CNN (~230 units/sample).
    """
    torch.manual_seed(0)
    model = build_model("cnn", (1, 8, 8), 10, sigma=1e-2, width=2)
    x, y = torch.randn(32, 1, 8, 8), torch.randint(0, 10, (32,))

    loss = F.cross_entropy(model(x), y)
    loss.backward()
    layers = model.perturbed_layers()
    true = torch.cat([m.weight.grad.flatten() for m in layers])

    sigma, n_rep = 1e-2, 2000
    for m in layers:
        m.zero_estimate()
    with torch.no_grad():
        model.set_perturb(False)
        clean = F.cross_entropy(model(x), y, reduction="none")
        for _ in range(n_rep):
            model.set_perturb(True)
            r = F.cross_entropy(model(x), y, reduction="none") - clean
            for m in layers:
                m.accumulate_estimate(r, 1.0 / (32 * sigma ** 2 * n_rep))
    est = torch.cat([m.weight_est.flatten() for m in layers])
    cos = F.cosine_similarity(true, est, dim=0).item()
    assert cos > 0.5, cos
    # Unbiased with the right scale: the component of est along the true
    # gradient should equal the gradient itself (the residual is pure noise).
    proj = (est @ true / (true @ true)).item()
    assert 0.8 < proj < 1.2, proj
