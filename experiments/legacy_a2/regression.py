"""
Regression analogue: leaves output a per-leaf Gaussian (mu, log_sigma) instead of a
softmax over classes. The tree becomes a routed mixture-density network:

    P(y|x) = sum_l pi_l(x) * N(y; mu_l(x), sigma_l(x)^2)

`mean_mse` is included specifically to *demonstrate* the classic mode-averaging
failure of plain regression trees/ensembles on multimodal targets: minimizing
MSE on the mixture mean forces mu_pred = sum_l pi_l*mu_l toward the average of
the modes, whereas the mixture NLL keeps the two branches specialized on the
two modes. See figures/regression_multimodal.png.
"""
import math
import torch
import torch.nn as nn

from .tree import SoftDecisionTree

_LOG_2PI = math.log(2 * math.pi)


class TreeRegressor(nn.Module):
    def __init__(self, depth, in_dim, router_kind="linear", leaf_kind="linear",
                 mode="vectorized", beta=1.0, router_kwargs=None, leaf_kwargs=None):
        super().__init__()
        # out_dim=2 per leaf: (mu, log_sigma) for a scalar target
        self.tree = SoftDecisionTree(depth, in_dim, out_dim=2, router_kind=router_kind,
                                      leaf_kind=leaf_kind, mode=mode, beta=beta,
                                      router_kwargs=router_kwargs, leaf_kwargs=leaf_kwargs)

    @property
    def beta(self):
        return self.tree.beta

    @beta.setter
    def beta(self, value):
        self.tree.beta = value

    def forward(self, x):
        leaf_log_pi, leaf_out, diagnostics = self.tree(x)
        mu, log_sigma = leaf_out[..., 0], leaf_out[..., 1].clamp(-5, 5)
        return leaf_log_pi, mu, log_sigma, diagnostics

    @torch.no_grad()
    def predict(self, x):
        """Posterior mean prediction: sum_l pi_l(x) * mu_l(x)."""
        leaf_log_pi, mu, _, _ = self(x)
        return (leaf_log_pi.exp() * mu).sum(dim=1)


def gaussian_mixture_nll(leaf_log_pi, mu, log_sigma, y, kind="mixture"):
    """y: [N]. Returns per-sample loss [N]."""
    y = y.view(-1, 1)  # broadcast against [N, L]
    sigma = log_sigma.exp()
    log_normal = -0.5 * _LOG_2PI - log_sigma - 0.5 * ((y - mu) / sigma) ** 2  # [N, L]

    if kind == "mixture":
        return -torch.logsumexp(leaf_log_pi + log_normal, dim=1)

    if kind == "expected":
        pi = leaf_log_pi.exp()
        return -(pi * log_normal).sum(dim=1)

    if kind == "mean_mse":
        pi = leaf_log_pi.exp()
        mu_pred = (pi * mu).sum(dim=1)
        return (mu_pred - y.squeeze(1)) ** 2

    raise ValueError(f"unknown loss kind {kind!r}")
