"""
Node factory: every router and every leaf of the tree is produced by this file.

The whole point of the "generic" requirement is that a node is *any* nn.Module
mapping Tensor[N, ...] -> Tensor[N, out_dim]. The tree (tree.py) never looks
inside a node; it only calls it. Swapping "linear" for "cnn" changes nothing
about the tree's optimization problem, only the capacity of h_i(x) inside the
router sigmoid sigma(beta * h_i(x)) or the leaf softmax(f_l(x)).
"""
import torch
import torch.nn as nn


class ConstantHead(nn.Module):
    """A leaf that ignores x entirely (Hinton's input-independent leaf distribution).
    Initialized with small random noise, not exact zeros: every ConstantHead in a tree
    would otherwise start numerically identical, which is a symmetric saddle point of a
    mixture loss (swapping two identical leaves leaves the loss unchanged) that plain
    gradient descent cannot break out of -- the classic mode-collapse-to-the-marginal
    failure of mixture density training from a symmetric initialization."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(out_dim) * 0.1)

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x.flatten(1))


class CNNHead(nn.Module):
    """A small conv classifier for a node -- used to show a node can be a full
    neural network, not just a linear/MLP probe. Expects x as [N, C, H, W]."""

    def __init__(self, in_shape, out_dim, channels=8):
        super().__init__()
        c, h, w = in_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, channels, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Linear(channels * 4 * 4, out_dim)

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def make_node(kind, in_dim, out_dim, **kwargs):
    """in_dim is an int for linear/mlp/constant, a (C,H,W) tuple for cnn."""
    if kind == "constant":
        return ConstantHead(in_dim, out_dim)
    if kind == "linear":
        return LinearHead(in_dim, out_dim)
    if kind == "mlp":
        return MLPHead(in_dim, out_dim, hidden=kwargs.get("hidden", 64))
    if kind == "cnn":
        return CNNHead(in_dim, out_dim, channels=kwargs.get("channels", 8))
    raise ValueError(f"unknown node kind {kind!r}")


class BatchedLinearRouters(nn.Module):
    """Vectorized equivalent of a ModuleList of `n_nodes` LinearHead(in_dim, 1) routers.

    Holds one weight tensor [n_nodes, in_dim] and bias [n_nodes] and evaluates every
    router for every sample in a single matmul, instead of n_nodes python-level calls.
    Numerically identical to the generic path (checked in tests/test_equivalence.py);
    exists purely so depth-8 sweeps (255 routers) are fast enough to run.
    """

    def __init__(self, n_nodes, in_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_nodes, in_dim) * (1.0 / in_dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def forward(self, x):
        # x: [N, in_dim] -> [N, n_nodes]
        return x.flatten(1) @ self.weight.T + self.bias

    def load_from_generic(self, module_list):
        with torch.no_grad():
            for i, m in enumerate(module_list):
                self.weight[i] = m.fc.weight.squeeze(0)
                self.bias[i] = m.fc.bias.squeeze(0)


class CNNTrunk(nn.Module):
    """Shared feature extractor for CIFAR-10: the tree's routers/leaves see
    trunk(x), not raw pixels. Four conv blocks with BatchNorm, out_dim-d features."""

    def __init__(self, in_channels=3, out_dim=128):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(in_channels, 32),  # 32x32 -> 16x16
            block(32, 64),           # 16x16 -> 8x8
            block(64, 128),          # 8x8 -> 4x4
        )
        self.proj = nn.Linear(128 * 4 * 4, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        return self.proj(self.features(x).flatten(1))


class TrunkModel(nn.Module):
    """Composes a shared feature trunk with any downstream head (a
    SoftDecisionTree, TreeRegressor, or plain nn.Linear baseline). The tree
    never sees raw pixels for CIFAR-10 -- only trunk(x); see the CIFAR-10
    discussion for why routing on raw pixels fails."""

    def __init__(self, trunk, head):
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x):
        return self.head(self.trunk(x))

    def hard_route(self, x):
        return self.head.hard_route(self.trunk(x))

    @property
    def beta(self):
        return self.head.beta

    @beta.setter
    def beta(self, value):
        self.head.beta = value


class BatchedLinearLeaves(nn.Module):
    """Vectorized equivalent of a ModuleList of `n_leaves` LinearHead(in_dim, K) leaves."""

    def __init__(self, n_leaves, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_leaves, out_dim, in_dim) * (1.0 / in_dim ** 0.5))
        self.bias = nn.Parameter(torch.zeros(n_leaves, out_dim))

    def forward(self, x):
        # x: [N, in_dim] -> [N, n_leaves, out_dim]
        xf = x.flatten(1)
        return torch.einsum("nd,lkd->nlk", xf, self.weight) + self.bias

    def load_from_generic(self, module_list):
        with torch.no_grad():
            for i, m in enumerate(module_list):
                self.weight[i] = m.fc.weight
                self.bias[i] = m.fc.bias
