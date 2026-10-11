"""One-vs-rest variant of the leaf loss, for the multiclass-optimization discussion:
each leaf predicts K independent sigmoids (Bernoulli per class) instead of one
softmax; used to compare against softmax leaves on a multi-label-flavoured
metric and to discuss when independent-class assumptions are preferable
(e.g. multi-label targets, very large K, class-imbalanced leaves)."""
import torch
import torch.nn.functional as F


def tree_loss_ovr(leaf_log_pi, leaf_logits, y, n_classes):
    y_onehot = F.one_hot(y, n_classes).float()                   # [N, K]
    log_p1 = F.logsigmoid(leaf_logits)                           # [N, L, K]
    log_p0 = F.logsigmoid(-leaf_logits)
    per_class_logp = y_onehot.unsqueeze(1) * log_p1 + (1 - y_onehot.unsqueeze(1)) * log_p0
    leaf_loglik = per_class_logp.sum(dim=-1)                     # [N, L] naive independence
    pi = leaf_log_pi.exp()
    return -(pi * leaf_loglik).sum(dim=1)
