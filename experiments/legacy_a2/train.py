import time
import torch
import torch.nn.functional as F

from .losses import tree_loss, balance_regularizer
from .regression import gaussian_mixture_nll


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def linear_beta_schedule(beta0, beta1, epochs):
    def sched(epoch):
        t = min(1.0, epoch / max(1, epochs - 1))
        return beta0 + t * (beta1 - beta0)
    return sched


def evaluate_classifier(model, test_dl, device):
    model.eval()
    correct_soft = correct_hard = n = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            leaf_log_pi, leaf_logits, _ = model(xb)
            log_q = F.log_softmax(leaf_logits, dim=-1)
            log_mix = torch.logsumexp(leaf_log_pi.unsqueeze(-1) + log_q, dim=1)  # [N,K]
            pred_soft = log_mix.argmax(1)
            _, class_logits_hard = model.hard_route(xb)
            pred_hard = class_logits_hard.argmax(1)
            correct_soft += (pred_soft == yb).sum().item()
            correct_hard += (pred_hard == yb).sum().item()
            n += xb.size(0)
    return correct_soft / n, correct_hard / n


def train_classifier(model, train_dl, test_dl, epochs, loss_kind="mixture", lr=1e-3,
                      device=None, lambda_bal=0.0, weight_decay=0.0, beta_schedule=None,
                      verbose=True):
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "test_soft_acc": [], "test_hard_acc": [], "epoch_time": [], "beta": []}

    for epoch in range(epochs):
        if beta_schedule is not None:
            model.beta = beta_schedule(epoch)
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            leaf_log_pi, leaf_logits, diag = model(xb)
            loss = tree_loss(leaf_log_pi, leaf_logits, yb, kind=loss_kind).mean()
            if lambda_bal > 0:
                loss = loss + lambda_bal * balance_regularizer(diag)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        soft_acc, hard_acc = evaluate_classifier(model, test_dl, device)
        history["train_loss"].append(total_loss / n)
        history["test_soft_acc"].append(soft_acc)
        history["test_hard_acc"].append(hard_acc)
        history["epoch_time"].append(time.time() - t0)
        history["beta"].append(getattr(model, "beta", None))
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  "
                  f"soft_acc={soft_acc:.4f}  hard_acc={hard_acc:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return history


def evaluate_regressor(model, test_dl, device):
    model.eval()
    se = n = 0.0
    with torch.no_grad():
        for xb, yb in test_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model.predict(xb)
            se += ((pred - yb) ** 2).sum().item()
            n += xb.size(0)
    return (se / n) ** 0.5  # RMSE


def train_regressor(model, train_dl, test_dl, epochs, loss_kind="mixture", lr=1e-3,
                     device=None, lambda_bal=0.0, weight_decay=0.0, beta_schedule=None,
                     verbose=True):
    device = device or get_device()
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss": [], "test_rmse": [], "epoch_time": []}

    for epoch in range(epochs):
        if beta_schedule is not None:
            model.beta = beta_schedule(epoch)
        model.train()
        t0 = time.time()
        total_loss = n = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            leaf_log_pi, mu, log_sigma, diag = model(xb)
            loss = gaussian_mixture_nll(leaf_log_pi, mu, log_sigma, yb, kind=loss_kind).mean()
            if lambda_bal > 0:
                loss = loss + lambda_bal * balance_regularizer(diag)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        rmse = evaluate_regressor(model, test_dl, device)
        history["train_loss"].append(total_loss / n)
        history["test_rmse"].append(rmse)
        history["epoch_time"].append(time.time() - t0)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}  loss={total_loss/n:.4f}  rmse={rmse:.4f}  "
                  f"({time.time()-t0:.1f}s)")
    return history
