import math

import pytest
import torch
from torch.nn import functional as F

from neural_tree.core import NeuralTree, Baseline, parameter_count, objective, penalty_scale, shrink_tree


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def test_probability_mass_manual_mixture_and_extreme_routes():
    model = NeuralTree(classes=2, width=.125, depth=1)
    z = torch.randn(5, model.trunk.output_dim)
    output = model(z, from_features=True)
    gate = output.route_logits[0].sigmoid()[:, None]
    manual = (1 - gate) * output.class_logits[1].softmax(-1) + gate * output.class_logits[2].softmax(-1)
    torch.testing.assert_close(output.log_probs.exp(), manual)
    torch.testing.assert_close(output.log_probs.exp().sum(-1), torch.ones(5))
    torch.testing.assert_close(sum(output.log_reach[i].exp() for i in output.leaves), torch.ones(5))
    with torch.no_grad():
        model.nodes["0"].router.weight.zero_()
        model.nodes["0"].router.bias.fill_(1000)
    extreme = model(z, from_features=True)
    assert torch.isfinite(extreme.log_probs).all()
    torch.testing.assert_close(extreme.log_probs, F.log_softmax(extreme.class_logits[2], -1))


def test_gradients_reach_trunk_routers_and_leaves():
    model = NeuralTree(classes=3, width=.125, depth=2)
    output = model(torch.randn(6, 3, 32, 32))
    loss, _ = objective(output, torch.tensor([0, 1, 2, 0, 1, 2]), .1, .01)
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.trunk.parameters()) > 0
    for node in model.nodes.values():
        assert node.classifier.weight.grad is not None
        assert torch.isfinite(node.classifier.weight.grad).all()
        assert node.classifier.weight.grad.abs().sum() > 0
        if node.router is not None:
            assert node.router.weight.grad.abs().sum() > 0


def test_depth_zero_and_matched_baseline():
    model = NeuralTree(classes=2, width=.125, depth=0)
    x = torch.randn(3, 3, 32, 32)
    output = model(x)
    torch.testing.assert_close(output.log_probs, F.log_softmax(output.class_logits[0], -1))
    loss, parts = objective(output, torch.tensor([0, 1, 0]), 1, 1)
    torch.testing.assert_close(loss, parts["nll"])
    for classes in [2, 10]:
        tree = NeuralTree(classes=classes, width=.25)
        baseline = Baseline(classes=classes, width=.25)
        assert abs(parameter_count(tree) - parameter_count(baseline)) < tree.trunk.output_dim + classes + 1


def test_hard_routing_skips_unused_nodes_and_pruning_removes_parameters():
    model = NeuralTree(classes=2, width=.125, depth=2)
    for i in model.internal:
        with torch.no_grad():
            model.nodes[str(i)].router.weight.zero_()
            model.nodes[str(i)].router.bias.fill_(-100)
    calls = []
    handles = [node.register_forward_hook(lambda m, x, y, key=key: calls.append(key))
               for key, node in model.nodes.items()]
    x = torch.randn(3, 3, 32, 32)
    hard = model.hard(x)
    assert calls == ["0", "1", "3"]
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(hard, F.log_softmax(model(x).class_logits[3], -1))
    count = parameter_count(model)
    model.prune(1)
    assert parameter_count(model) < count
    assert "3" not in model.nodes and "4" not in model.nodes
    rebuilt = NeuralTree(classes=2, width=.125, depth=2, pruned=[1])
    rebuilt.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(x).log_probs, rebuilt(x).log_probs)
    model.prune(0)
    assert list(model.nodes) == ["0"]


def test_auxiliary_loss_and_uniform_balance():
    model = NeuralTree(classes=2, width=.125, depth=2)
    output = model(torch.randn(4, model.trunk.output_dim), from_features=True, uniform=True)
    labels = torch.tensor([0, 1, 0, 1])
    _, parts = objective(output, labels, .1, .01)
    manual = (F.cross_entropy(output.class_logits[0], labels) +
              .5 * F.cross_entropy(output.class_logits[1], labels) +
              .5 * F.cross_entropy(output.class_logits[2], labels)) / 2
    torch.testing.assert_close(parts["auxiliary"], manual)
    # Actual router logits still exist, although mixture routing is uniform.
    for i in model.internal:
        output.route_logits[i] = torch.zeros(4)
    _, parts = objective(output, labels)
    assert parts["balance"].item() == pytest.approx(math.log(2), abs=1e-6)
    assert penalty_scale(0, 20) == 1 and penalty_scale(20, 20) == 0


def test_uniform_routing_removes_prediction_dependence_on_router():
    model = NeuralTree(classes=2, width=.125, depth=2)
    z = torch.randn(4, model.trunk.output_dim)
    before = model(z, from_features=True, uniform=True).log_probs
    with torch.no_grad():
        for i in model.internal:
            model.nodes[str(i)].router.bias.add_(50)
    after = model(z, from_features=True, uniform=True).log_probs
    torch.testing.assert_close(before, after)


def test_adapters_receive_gradients_and_structural_shrink_roundtrips():
    config = dict(classes=3, width=.25, depth=2, pool_size=4, adapter_rank=4, pruned=[1])
    model = NeuralTree(**config)
    x = torch.randn(5, 3, 32, 32)
    loss, _ = objective(model(x), torch.tensor([0, 1, 2, 0, 1]), .1, .01)
    loss.backward()
    assert model.nodes["0"].adapter[-1].weight.grad.abs().sum() > 0
    small, settings = shrink_tree(model, config, .125, 1)
    assert parameter_count(small.trunk) < parameter_count(model.trunk)
    assert parameter_count(small) < parameter_count(model)
    rebuilt = NeuralTree(**settings)
    rebuilt.load_state_dict(small.state_dict())
    torch.testing.assert_close(small(x).log_probs, rebuilt(x).log_probs)
    torch.testing.assert_close(small(x).log_probs.exp().sum(-1), torch.ones(5))
    assert set(small.nodes) == set(model.nodes)


def test_tree_heads_can_memorize_a_synthetic_feature_batch():
    """Optimization sanity check only; not the real-image memorization milestone."""
    model = NeuralTree(classes=2, width=.125, depth=2, node_width=32)
    z = torch.randn(16, model.trunk.output_dim)
    labels = torch.randint(0, 2, (16,))
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    for _ in range(150):
        optimizer.zero_grad(set_to_none=True)
        output = model(z, from_features=True)
        loss, _ = objective(output, labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        assert (model(z, from_features=True).log_probs.argmax(-1) == labels).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_forward_backward():
    model = NeuralTree(classes=2, width=.125, depth=1).to("mps")
    output = model(torch.randn(4, 3, 32, 32, device="mps"))
    loss, _ = objective(output, torch.tensor([0, 1, 0, 1], device="mps"), .1, .01)
    loss.backward()
    torch.mps.synchronize()
    assert torch.isfinite(loss)
