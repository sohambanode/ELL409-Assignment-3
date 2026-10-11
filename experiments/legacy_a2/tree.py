"""
SoftDecisionTree: a perfect binary tree where routers compute the relaxed path
probabilities pi_l(x), and leaves are arbitrary classifiers q_l(y|x).

Node indexing is standard BFS/heap order: root = 0, children of i are 2i+1, 2i+2.
At depth d the tree holds nodes [2^d - 1, 2^{d+1} - 2] (2^d of them). Leaves are
indexed 0..2^D-1 by reading off the sequence of left(0)/right(1) branch choices
from root to leaf, MSB first -- this labeling falls out for free from the
doubling recursion used below (see forward()), no manual bit-twiddling needed.

Everything is computed *densely*: every router and every leaf is evaluated for
every sample on every forward pass (this is what makes the tree differentiable
and is also its main computational cost -- see the CIFAR-10 discussion).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nodes import make_node, BatchedLinearRouters, BatchedLinearLeaves


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, in_dim, out_dim, router_kind="linear", leaf_kind="linear",
                 mode="vectorized", beta=1.0, router_kwargs=None, leaf_kwargs=None):
        super().__init__()
        self.depth = depth
        self.out_dim = out_dim
        self.n_internal = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.mode = mode
        self.beta = beta  # plain python float; annealed externally by the trainer
        router_kwargs = router_kwargs or {}
        leaf_kwargs = leaf_kwargs or {}

        if mode == "generic":
            self.routers = nn.ModuleList(
                [make_node(router_kind, in_dim, 1, **router_kwargs) for _ in range(self.n_internal)]
            )
            self.leaves = nn.ModuleList(
                [make_node(leaf_kind, in_dim, out_dim, **leaf_kwargs) for _ in range(self.n_leaves)]
            )
        elif mode == "vectorized":
            if router_kind != "linear" or leaf_kind != "linear":
                raise ValueError("mode='vectorized' only supports homogeneous linear nodes; "
                                  "use mode='generic' for mlp/cnn/constant/mixed node types")
            if not isinstance(in_dim, int):
                raise ValueError("mode='vectorized' requires a flat int in_dim")
            self.routers = BatchedLinearRouters(self.n_internal, in_dim)
            self.leaves = BatchedLinearLeaves(self.n_leaves, in_dim, out_dim)
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def _all_router_logits(self, x):
        if self.mode == "generic":
            return torch.cat([r(x) for r in self.routers], dim=1)  # [N, n_internal]
        return self.routers(x)

    def _all_leaf_logits(self, x):
        if self.mode == "generic":
            return torch.stack([l(x) for l in self.leaves], dim=1)  # [N, n_leaves, out_dim]
        return self.leaves(x)

    def forward(self, x):
        """Returns (leaf_log_pi [N, n_leaves], leaf_logits [N, n_leaves, out_dim], diagnostics)."""
        N = x.shape[0]
        device = x.device
        router_logits_all = self._all_router_logits(x) * self.beta  # [N, n_internal]
        leaf_logits = self._all_leaf_logits(x)                      # [N, n_leaves, out_dim]

        log_mu = torch.zeros(N, 1, device=device)  # root is reached with prob 1
        offset = 0
        depth_logits, depth_log_reach = [], []
        for d in range(self.depth):
            n_d = 2 ** d
            logits_d = router_logits_all[:, offset:offset + n_d]  # [N, n_d]
            offset += n_d
            depth_logits.append(logits_d)
            depth_log_reach.append(log_mu)
            log_g = F.logsigmoid(logits_d)    # log P(go left | reached node)
            log_1mg = F.logsigmoid(-logits_d)  # log P(go right | reached node)
            left = log_mu + log_g
            right = log_mu + log_1mg
            log_mu = torch.stack([left, right], dim=2).reshape(N, n_d * 2)

        leaf_log_pi = log_mu  # [N, n_leaves], logsumexp over dim=1 is ~0 (checked in tests)
        diagnostics = {"depth_logits": depth_logits, "depth_log_reach": depth_log_reach}
        return leaf_log_pi, leaf_logits, diagnostics

    @torch.no_grad()
    def hard_route(self, x):
        """Greedy single-path inference: at every node take the argmax branch.
        Returns (leaf_idx [N], class_logits [N, out_dim])."""
        N = x.shape[0]
        device = x.device
        router_logits_all = self._all_router_logits(x) * self.beta
        leaf_logits = self._all_leaf_logits(x)

        leaf_idx = torch.zeros(N, dtype=torch.long, device=device)
        offset = 0
        for d in range(self.depth):
            n_d = 2 ** d
            logits_d = router_logits_all[:, offset:offset + n_d]
            offset += n_d
            cur = logits_d.gather(1, leaf_idx.unsqueeze(1)).squeeze(1)
            bit = (cur <= 0).long()  # go right if P(left) <= 0.5
            leaf_idx = leaf_idx * 2 + bit

        class_logits = leaf_logits.gather(
            1, leaf_idx.view(N, 1, 1).expand(-1, 1, self.out_dim)
        ).squeeze(1)
        return leaf_idx, class_logits

    @torch.no_grad()
    def leaf_usage(self, x):
        """Mean path probability per leaf over a batch -- diagnostic for dead leaves."""
        leaf_log_pi, _, _ = self.forward(x)
        return leaf_log_pi.exp().mean(0)  # [n_leaves]

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
