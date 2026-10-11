"""
Hierarchical softmax over *labels* (not inputs) -- the complementary way a tree
shows up when optimizing a multiclass problem: instead of routing samples to
specialized leaf classifiers, route classes down a fixed binary tree so that
P(y|x) is a product of O(log K) binary decisions instead of one O(K) softmax.
Classic reference: Morin & Bengio 2005.

Also included: recursive spectral bisection of the confusion matrix, used to
*choose* the label tree structure instead of a random/index-order split, so
that confusable classes (e.g. cat/dog, automobile/truck on CIFAR-10) end up
sharing an ancestor and get resolved by a dedicated deeper router.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def spectral_label_order(similarity, indices=None):
    """Recursive Fiedler-vector bisection of a KxK similarity matrix (e.g. a
    symmetrized confusion matrix) into a leaf order for a balanced binary tree,
    such that mutually similar/confusable classes end up nearby (and hence
    share a deeper common ancestor in the resulting tree)."""
    if indices is None:
        indices = list(range(similarity.shape[0]))
    if len(indices) <= 1:
        return list(indices)
    sub = similarity[np.ix_(indices, indices)]
    d = sub.sum(axis=1)
    laplacian = np.diag(d) - sub
    eigvals, eigvecs = np.linalg.eigh(laplacian)
    fiedler = eigvecs[:, 1] if len(indices) > 2 else eigvecs[:, -1]
    order = np.argsort(fiedler)
    mid = len(indices) // 2
    left = [indices[i] for i in order[:mid]]
    right = [indices[i] for i in order[mid:]]
    return spectral_label_order(similarity, left) + spectral_label_order(similarity, right)


def _build_paths(leaf_order):
    """leaf_order[i] = class id placed at leaf position i. Returns, for each
    class id, the sequence of (internal_node_id, direction) from root to leaf,
    using the same recursive-bisection internal node ids as build order."""
    n_leaves = len(leaf_order)
    paths = {c: [] for c in leaf_order}
    counter = [0]

    def recurse(positions):
        if len(positions) == 1:
            return
        node_id = counter[0]
        counter[0] += 1
        mid = len(positions) // 2
        left, right = positions[:mid], positions[mid:]
        for p in left:
            paths[leaf_order[p]].append((node_id, 0))
        for p in right:
            paths[leaf_order[p]].append((node_id, 1))
        recurse(left)
        recurse(right)

    recurse(list(range(n_leaves)))
    n_internal = counter[0]
    return paths, n_internal


class HierarchicalSoftmax(nn.Module):
    """P(y|x) via a fixed binary tree over the K classes. Cost is O(log K)
    router evaluations per sample instead of O(K) softmax logits -- the
    benefit grows with K (irrelevant at K=10, the point at K=10^4+)."""

    def __init__(self, in_dim, n_classes, leaf_order=None):
        super().__init__()
        self.n_classes = n_classes
        leaf_order = list(leaf_order) if leaf_order is not None else list(range(n_classes))
        paths, n_internal = _build_paths(leaf_order)
        self.n_internal = n_internal
        self.router = nn.Linear(in_dim, max(n_internal, 1))

        max_depth = max(len(p) for p in paths.values())
        node_ids = torch.zeros(n_classes, max_depth, dtype=torch.long)
        dirs = torch.zeros(n_classes, max_depth)
        mask = torch.zeros(n_classes, max_depth)
        for c, path in paths.items():
            for t, (nid, d) in enumerate(path):
                node_ids[c, t] = nid
                dirs[c, t] = d
                mask[c, t] = 1.0
        self.register_buffer("node_ids", node_ids)
        self.register_buffer("dirs", dirs)
        self.register_buffer("mask", mask)

    def log_prob(self, x, y):
        """log P(y|x), shape [N]."""
        logits = self.router(x)  # [N, n_internal]
        log_g = F.logsigmoid(logits)
        log_1mg = F.logsigmoid(-logits)

        nid = self.node_ids[y]  # [N, max_depth]
        d = self.dirs[y]        # [N, max_depth]
        m = self.mask[y]        # [N, max_depth]

        lg = log_g.gather(1, nid)
        l1mg = log_1mg.gather(1, nid)
        step_logp = torch.where(d > 0.5, l1mg, lg)
        return (step_logp * m).sum(dim=1)

    def log_prob_all_classes(self, x):
        """log P(y|x) for every class, shape [N, K] -- for computing accuracy."""
        logits = self.router(x)
        log_g = F.logsigmoid(logits)
        log_1mg = F.logsigmoid(-logits)
        N = x.shape[0]
        out = torch.zeros(N, self.n_classes, device=x.device)
        for c in range(self.n_classes):
            nid = self.node_ids[c].unsqueeze(0).expand(N, -1)
            d = self.dirs[c].unsqueeze(0).expand(N, -1)
            m = self.mask[c].unsqueeze(0).expand(N, -1)
            lg = log_g.gather(1, nid)
            l1mg = log_1mg.gather(1, nid)
            step_logp = torch.where(d > 0.5, l1mg, lg)
            out[:, c] = (step_logp * m).sum(dim=1)
        return out
