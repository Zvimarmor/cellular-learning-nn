"""Checks for the layerwise / topk / blockwise_reward perturbation modes on mini networks.

Run: python -m pytest test_new_modes.py -q

Monte-Carlo trick: repeating a mini-batch n times inside one batch makes the
batch-mean estimate an average over n independent noise draws per sample, so a
single ``estimate()`` call converges to the expected update. Each test compares
that expectation with the exact gradient (via autograd) of the objective the
mode is supposed to target.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from layers import PerturbedConv2d
from models import build_model
from train import ThreeFactorTrainer, make_optimizer, parse_args

SIGMA = 1e-2


def make_trainer(model, modes, *extra):
    args = parse_args(["--perturb-mode", modes, "--sigma", str(SIGMA), *extra])
    if "topk" in args.perturb_modes:
        model.set_topk(args.topk_ratio, args.topk_score)
    if "blockwise_reward" in args.perturb_modes:
        model.build_aux_heads(args.aux_head, 10, SIGMA)
    return ThreeFactorTrainer(model, args, make_optimizer(args, model)), args


def repeated(x, y, n):
    return x.repeat(n, *([1] * (x.dim() - 1))), y.repeat(n)


def cos(a, b):
    return F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def proj(est, true):
    """Coefficient of est along true: 1 for an unbiased estimator, the residual being noise."""
    return ((est.flatten() @ true.flatten()) / (true.flatten() @ true.flatten())).item()


# ----------------------------------------------------------------------------- structure

def test_perturbation_groups():
    assert [len(g) for g in build_model("mlp", (1, 28, 28), 10, SIGMA).perturbation_groups()] == [1, 1]
    assert [len(g) for g in build_model("cnn", (3, 32, 32), 10, SIGMA).perturbation_groups()] == [1, 1, 1, 1]
    # ResNet-18: stem conv | 8 residual blocks (3 layers when the block has a 1x1 shortcut) | fc
    groups = build_model("resnet18", (3, 32, 32), 10, SIGMA, width=8).perturbation_groups()
    assert [len(g) for g in groups] == [1, 2, 2, 3, 2, 3, 2, 3, 2, 1]


# ----------------------------------------------------------------------------- top-k

def test_topk_mask_keeps_k_units_and_fused_trace_is_exact():
    torch.manual_seed(0)
    layer = PerturbedConv2d(3, 6, 3, padding=1, sigma=0.3)
    layer.topk_ratio, layer.perturb = 0.1, True
    layer(torch.randn(4, 3, 9, 9))
    n = 6 * 9 * 9
    assert ((layer._xi != 0).flatten(1).sum(1) == math.ceil(0.1 * n)).all()
    r = torch.randn(4)
    layer.zero_estimate()
    layer.accumulate_estimate(r, 1.0)
    explicit = (r.view(-1, 1, 1, 1, 1) * layer.eligibility_trace()).sum(0)
    torch.testing.assert_close(layer.weight_est, explicit, rtol=1e-4, atol=1e-4)


def test_topk_leaves_output_layer_dense():
    model = build_model("cnn", (3, 16, 16), 10, SIGMA, width=4)
    model.set_topk(0.05)
    ratios = [m.topk_ratio for m in model.perturbed_layers()]
    assert ratios == [0.05, 0.05, 0.05, 1.0]


@pytest.mark.parametrize("score", ["abs", "pos"])
def test_topk_estimator_targets_masked_gradient(score):
    """E[update] = sum_b (m_b * dL/dz_b) x_b^T for the top-k mask m_b of the clean z."""
    torch.manual_seed(0)
    model = build_model("mlp", (1, 6, 6), 10, SIGMA, width=64)
    trainer, args = make_trainer(model, "layerwise,topk", "--topk-ratio", "0.1", "--topk-score", score)
    fc1 = model.perturbed_layers()[0]
    x, y = torch.randn(16, 1, 6, 6), torch.randint(0, 10, (16,))

    # exact gradient wrt the pre-activation z of fc1
    z = {}

    def keep(module, inputs, out):
        out.retain_grad()
        z["z"] = out

    h = fc1.register_forward_hook(keep)
    F.cross_entropy(model(x), y).backward()
    h.remove()
    mask = fc1.topk_mask(z["z"])
    x_flat = x.flatten(1)
    full = z["z"].grad.t() @ x_flat
    masked = (mask * z["z"].grad).t() @ x_flat

    trainer.estimate(*repeated(x, y, 3000), [fc1])
    est = fc1.weight_est
    assert cos(est, masked) > 0.9
    assert 0.85 < proj(est, masked) < 1.15
    assert cos(est, masked) > cos(est, full) + 0.1


# ----------------------------------------------------------------------------- layerwise

def test_layerwise_reduces_estimator_variance():
    """Same samples, same number of noise draws. With every layer perturbed, the reward is
    dominated by noise on the readout (largest gradients) and the hidden-layer estimates
    are ~orthogonal to their gradients; perturbing one group at a time recovers them."""
    torch.manual_seed(0)
    model = build_model("cnn", (3, 16, 16), 10, SIGMA, width=8)
    trainer, _ = make_trainer(model, "all")
    x, y = torch.randn(16, 3, 16, 16), torch.randint(0, 10, (16,))
    F.cross_entropy(model(x), y).backward()
    hidden = model.perturbed_layers()[:-1]
    true = {m: m.weight.grad.clone() for m in hidden}

    xr, yr = repeated(x, y, 256)
    trainer.estimate(xr, yr, model.perturbed_layers())
    cos_all = sum(cos(m.weight_est, true[m]) for m in hidden) / len(hidden)
    cos_one = 0.0
    for m in hidden:
        trainer.estimate(xr, yr, [m])
        cos_one += cos(m.weight_est, true[m]) / len(hidden)
    assert cos_one > cos_all + 0.25, (cos_one, cos_all)


@pytest.mark.parametrize("schedule", ["cycle", "random"])
def test_layerwise_step_updates_only_active_group(schedule):
    torch.manual_seed(0)
    model = build_model("resnet10", (3, 16, 16), 10, SIGMA, width=8)
    trainer, _ = make_trainer(model, "layerwise", "--layer-schedule", schedule, "--optimizer", "adam",
                              "--lr", "1e-3")
    x, y = torch.randn(8, 3, 16, 16), torch.randint(0, 10, (8,))
    seen = set()
    for _ in range(3 * len(trainer.groups)):
        before = {m: m.weight.detach().clone() for m in model.perturbed_layers()}
        logs = trainer.step(x, y)
        g = logs["tf/active_group"]
        seen.add(g)
        changed = {m for m in model.perturbed_layers() if not torch.equal(before[m], m.weight)}
        assert changed == set(trainer.groups[g])
    if schedule == "cycle":
        assert seen == set(range(len(trainer.groups)))


# ----------------------------------------------------------------------------- blockwise reward

@pytest.mark.parametrize("head", ["fixed", "learned"])
def test_blockwise_estimator_targets_local_losses(head):
    """Stage s should receive the gradient of its own local loss CE(head_s(stage_s(h_{s-1})), y),
    with h_{s-1} the clean output of the previous stage."""
    torch.manual_seed(0)
    model = build_model("cnn", (3, 8, 8), 10, SIGMA, width=4)
    trainer, _ = make_trainer(model, "blockwise_reward", "--aux-head", head)
    x, y = torch.randn(16, 3, 8, 8), torch.randint(0, 10, (16,))

    h, last = x, len(model.stages) - 1
    for s, stage in enumerate(model.stages):
        out = stage(h.detach())
        logits = out if s == last else model.aux_heads[s](out)
        F.cross_entropy(logits, y).backward()
        h = out
    layers = model.perturbed_layers() + model.aux_layers()
    true = {m: m.weight.grad.clone() for m in layers}

    # 3200 draws: a learned head adds its own (large-gradient) noise to each stage's reward
    trained, _ = trainer.estimate(*repeated(x, y, 3200), model.perturbed_layers())
    assert set(trained) == set(layers)
    for m in layers:
        # Unbiased: the component along the local gradient has the right scale. The cosine
        # floor only rules out noise (a learned head's noise slows stage 0: ~0.57 at 3200
        # draws, ~0.93 at 51200).
        assert 0.85 < proj(m.weight_est, true[m]) < 1.15, (m, proj(m.weight_est, true[m]))
        assert cos(m.weight_est, true[m]) > 0.5, (m, cos(m.weight_est, true[m]))


def test_blockwise_rewards_are_stage_local():
    """Huge noise in stage 0 must not leak into the reward of the last stage, whose input is
    the clean h_{s-1}. Under a global reward it dominates every layer's reward."""
    torch.manual_seed(0)
    model = build_model("cnn", (3, 8, 8), 10, SIGMA, width=4)
    trainer, _ = make_trainer(model, "blockwise_reward")
    x, y = torch.randn(64, 3, 8, 8), torch.randint(0, 10, (64,))
    first, *_, last = model.perturbed_layers()
    first.sigma = 100.0
    with torch.no_grad():
        local, _, _ = trainer._stage_rewards(x, y, [first, last])
        model.clear_traces()
        glob, _, _ = trainer._global_rewards(x, y, [first, last])
    model.set_perturb(False)
    assert local[first].abs().mean() > 1.0          # stage 0 sees its own large noise
    assert local[last].abs().mean() < 0.05          # last stage only sees sigma=1e-2 noise
    assert glob[last].abs().mean() > 1.0            # a global reward would leak it


# ----------------------------------------------------------------------------- smoke

@pytest.mark.parametrize("modes", ["all", "layerwise", "topk", "blockwise_reward", "layerwise,topk",
                                   "topk,blockwise_reward", "layerwise,topk,blockwise_reward"])
def test_all_modes_step_on_resnet(modes):
    torch.manual_seed(0)
    model = build_model("resnet10", (3, 16, 16), 10, SIGMA, width=8)
    trainer, _ = make_trainer(model, modes, "--optimizer", "adam", "--lr", "1e-3", "--micro-batch-size", "4",
                              "--aux-head", "learned")
    x, y = torch.randn(8, 3, 16, 16), torch.randint(0, 10, (8,))
    for _ in range(3):
        logs = trainer.step(x, y, measure_alignment=True)
    assert math.isfinite(logs["train/loss"]) and math.isfinite(logs["align/total"])
    assert logs["tf/perturbed_units"] > 0
