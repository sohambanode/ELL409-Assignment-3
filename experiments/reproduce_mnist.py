"""Repeat A2's depth-2 vectorized mixture experiment, seed 0, 20 epochs.

The legacy_a2 package is an unchanged copy of the earlier assignment's Python
modules. The checkpoint and source hashes accompany the new measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, TensorDataset
from legacy_a2.tree import SoftDecisionTree
from legacy_a2.train import train_classifier, evaluate_classifier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="runs/mnist_reproduction")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    loaders = []
    for train in [True, False]:
        data = MNIST(args.root, train=train, download=False)
        tensor_data = TensorDataset(data.data.float().flatten(1) / 255., data.targets)
        loaders.append(DataLoader(tensor_data, batch_size=256 if train else 1024, shuffle=train))
    model = SoftDecisionTree(2, 784, 10, mode="vectorized")
    hist = train_classifier(model, *loaders, epochs=20, loss_kind="mixture", lr=.002, device="cpu")
    training = evaluate_classifier(model, DataLoader(loaders[0].dataset, batch_size=1024), "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).with_name("legacy_a2").glob("*.py")}
    result = dict(history=hist, train_soft_accuracy=training[0], train_hard_accuracy=training[1],
                  train_count=60000, test_count=10000, parameters=model.n_params(), seed=0,
                  epochs=20, lr=.002, depth=2, device="cpu", torch=torch.__version__, source_sha256=source)
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    torch.save(model.state_dict(), out / "model.pt")
