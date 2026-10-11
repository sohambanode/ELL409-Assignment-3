import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

DATA_ROOT = "/Users/soham_banode/ELL409/A2/data"


def mnist_loaders(batch_size=256, flatten=True):
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor()])
    train = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True, transform=tfm)
    test = torchvision.datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)

    def to_tensors(ds):
        x = torch.stack([ds[i][0] for i in range(len(ds))])
        y = torch.tensor([ds[i][1] for i in range(len(ds))])
        if flatten:
            x = x.flatten(1)
        return x, y

    xtr, ytr = to_tensors(train)
    xte, yte = to_tensors(test)
    train_dl = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(xte, yte), batch_size=1024, shuffle=False)
    return train_dl, test_dl, xtr.shape[1] if flatten else tuple(xtr.shape[1:])


def cifar10_loaders(batch_size=128, augment=True):
    import torchvision
    import torchvision.transforms as T

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tfm_list = []
    if augment:
        train_tfm_list += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    train_tfm_list += [T.ToTensor(), T.Normalize(mean, std)]
    train_tfm = T.Compose(train_tfm_list)
    test_tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tfm)
    test = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tfm)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test, batch_size=512, shuffle=False, num_workers=0)
    return train_dl, test_dl, (3, 32, 32)


def toy_multimodal_regression(n=4000, seed=0):
    """A single scalar input x, uniform on [-3,3] and INDEPENDENT of which of
    two branches generates y: y = +2+noise with prob 0.5, y = -2+noise with
    prob 0.5, regardless of x. x is a pure distractor -- by construction no
    function of x can predict which mode a given point came from, so this
    isolates the one thing a routed mixture genuinely buys over plain
    regression: since x carries no branch information, ANY point predictor
    (mixture posterior mean or a plain MSE regressor) is driven to the same
    unconditional mean E[y]=0 and gets similar RMSE. The real difference is in
    the predictive DISTRIBUTION -- a mixture-NLL-trained model still learns
    two well-separated, tightly-calibrated Gaussian leaves (mu near +-2, sigma
    near the true noise std) even though pi_l(x) stays uninformatively ~0.5,
    correctly representing that y is genuinely bimodal given x; a plain
    mean-squared-error loss has no such distributional target -- indeed
    log_sigma never receives any gradient under a pure MSE loss (see
    gaussian_mixture_nll, kind='mean_mse') and stays at its random
    initialization, uncalibrated, and the two leaf means are only constrained
    to average to 0, not to individually recover the true modes.
    """
    rng = np.random.default_rng(seed)
    branch = rng.integers(0, 2, size=n)
    x = rng.uniform(-3, 3, size=n).astype(np.float32).reshape(-1, 1)
    y = np.where(branch == 0, 2.0, -2.0) + rng.normal(0, 0.3, n)
    y = y.astype(np.float32)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    n_train = int(0.8 * n)
    train_dl = DataLoader(TensorDataset(xt[:n_train], yt[:n_train]), batch_size=128, shuffle=True)
    test_dl = DataLoader(TensorDataset(xt[n_train:], yt[n_train:]), batch_size=512, shuffle=False)
    return train_dl, test_dl, 1


def california_housing_loaders(batch_size=128, seed=0):
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = fetch_california_housing()
    x, y = data.data.astype(np.float32), data.target.astype(np.float32)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=seed)
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr).astype(np.float32), scaler.transform(xte).astype(np.float32)
    train_dl = DataLoader(TensorDataset(torch.from_numpy(xtr), torch.from_numpy(ytr)),
                           batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(TensorDataset(torch.from_numpy(xte), torch.from_numpy(yte)),
                          batch_size=1024, shuffle=False)
    return train_dl, test_dl, xtr.shape[1]
