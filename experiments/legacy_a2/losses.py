"""
The three fit terms from the optimization problem, plus the two regularizers.
All operate on the raw output of SoftDecisionTree.forward(x): (leaf_log_pi, leaf_logits, diagnostics).
"""
import torch
import torch.nn.functional as F


def _leaf_log_q_at_y(leaf_logits, y):
    """log q_l(y|x) for every leaf. leaf_logits: [N, L, K], y: [N] -> [N, L]."""
    log_q = F.log_softmax(leaf_logits, dim=-1)  # [N, L, K]
    return log_q.gather(2, y.view(-1, 1, 1).expand(-1, log_q.shape[1], 1)).squeeze(-1)


def tree_loss(leaf_log_pi, leaf_logits, y, kind="mixture"):
    """Per-sample loss, shape [N]. Caller takes .mean()."""
    log_q_y = _leaf_log_q_at_y(leaf_logits, y)  # [N, L]

    if kind == "mixture":
        # exact NLL of the mixture: -log sum_l pi_l(x) q_l(y|x)
        return -torch.logsumexp(leaf_log_pi + log_q_y, dim=1)

    if kind == "expected_ce":
        # Frosst & Hinton SDT loss: -sum_l pi_l(x) log q_l(y|x), a Jensen upper
        # bound on `mixture` (E[log] <= log[E]), fully differentiable through pi.
        pi = leaf_log_pi.exp()
        return -(pi * log_q_y).sum(dim=1)

    if kind == "em":
        # Hard-EM-style: detach the responsibilities (E-step), then maximize the
        # expected complete-data log-likelihood (M-step) w.r.t. both leaves and routers.
        with torch.no_grad():
            log_joint = leaf_log_pi + log_q_y
            r = F.softmax(log_joint, dim=1)  # posterior responsibilities, detached
        return -(r * (leaf_log_pi + log_q_y)).sum(dim=1)

    raise ValueError(f"unknown loss kind {kind!r}")


def balance_regularizer(diagnostics):
    """Encourages every router to send ~half its mass each way, weighted by how much
    of the data reaches that router (path-weighted mean left-probability alpha_i),
    with a 2^-depth decay since deeper nodes see exponentially less effective data.
    Returns a scalar; add lambda_bal * this to the loss."""
    total = 0.0
    for d, (logits_d, log_reach_d) in enumerate(zip(diagnostics["depth_logits"], diagnostics["depth_log_reach"])):
        reach = log_reach_d.exp()          # [N, n_d], P(reach node | x)
        g = torch.sigmoid(logits_d)        # [N, n_d], P(go left | reached)
        w = reach.sum(0).clamp_min(1e-8)   # [n_d], total reach mass per node
        alpha = (reach * g).sum(0) / w     # [n_d], path-weighted mean P(left)
        alpha = alpha.clamp(1e-6, 1 - 1e-6)
        node_term = -(0.5 * torch.log(alpha) + 0.5 * torch.log(1 - alpha))  # min at alpha=0.5
        total = total + (2.0 ** -d) * node_term.mean()
    return total


def purity_regularizer(leaf_log_pi, leaf_logits, y):
    """Pushes each leaf's path-weighted average predicted distribution toward a
    single class (low entropy), so leaves specialize -- used for the multiclass
    discussion. Returns a scalar."""
    pi = leaf_log_pi.exp()                          # [N, L]
    q = F.softmax(leaf_logits, dim=-1)               # [N, L, K]
    w = pi.sum(0).clamp_min(1e-8)                    # [L]
    q_bar = (pi.unsqueeze(-1) * q).sum(0) / w.unsqueeze(-1)  # [L, K], path-weighted avg dist
    pi_bar = pi.mean(0)                              # [L]
    entropy = -(q_bar.clamp_min(1e-8) * q_bar.clamp_min(1e-8).log()).sum(-1)  # [L]
    return (pi_bar * entropy).sum()
