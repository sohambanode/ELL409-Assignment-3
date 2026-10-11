"""Data preparation, neural models, training, and checkpoint persistence.

Read in order: data -> models -> losses -> checkpoint helpers -> training.
Command-line workflows live in __main__.py.
"""
from collections import Counter, defaultdict
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


# Data

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def group_split(records, seed=42, ratios=(.8, .1, .1)):
    """Greedily approximate class proportions without separating duplicate groups."""
    groups = defaultdict(list)
    totals = Counter(r["label"] for r in records)
    for record in records:
        groups[record["group"]].append(record)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    counts = [Counter() for _ in ratios]
    names = ["train", "val", "test"]
    for group in ordered:
        labels = Counter(r["label"] for r in group)
        def increase(i):
            return sum(((counts[i][c] + n - totals[c] * ratios[i]) ** 2
                        - (counts[i][c] - totals[c] * ratios[i]) ** 2)
                       / max(totals[c] * ratios[i], 1) for c, n in labels.items())
        split = min(range(3), key=increase)
        counts[split].update(labels)
        for record in group:
            record["split"] = names[split]
    return records


def stratified_subset(records, size, seed=42):
    """Round-robin classes for a deterministic, approximately balanced subset."""
    if size is None:
        return records
    if not 0 < size <= len(records):
        raise ValueError(f"subset must be in [1, {len(records)}]")
    by_class = defaultdict(list)
    for row in records:
        by_class[row["label"]].append(row)
    rng = random.Random(seed)
    for rows in by_class.values():
        rng.shuffle(rows)
    selected = []
    while len(selected) < size:
        for label in sorted(by_class):
            if by_class[label] and len(selected) < size:
                selected.append(by_class[label].pop())
    return selected


def validate_manifest(manifest, verify_files=False, root=None):
    ids, groups = set(), {}
    root = Path(root or manifest["root"])
    for row in manifest["records"]:
        if row["id"] in ids:
            raise ValueError(f"Repeated sample ID: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid split")
        if not 0 <= row["label"] < len(manifest["classes"]):
            raise ValueError("Invalid label")
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Duplicate group crosses splits: {row['group']}")
        if verify_files and manifest["dataset"] == "cats_dogs":
            if file_hash(root / row["path"]) != row["sha256"]:
                raise ValueError(f"Image changed: {row['path']}")


def prepare_cifar(root, output, seed):
    from torchvision.datasets import CIFAR10
    root = Path(root).resolve()
    train = CIFAR10(str(root), train=True, download=True)
    test = CIFAR10(str(root), train=False, download=True)
    rng, validation = random.Random(seed), set()
    for label in range(10):
        indices = [i for i, y in enumerate(train.targets) if y == label]
        rng.shuffle(indices)
        validation.update(indices[:500])
    records = []
    for source, dataset in [("train", train), ("test", test)]:
        for i, label in enumerate(dataset.targets):
            key = f"{source}:{i}"
            records.append({"id": key, "source": source, "index": i, "label": label,
                            "group": key, "split": "test" if source == "test" else
                            ("val" if i in validation else "train")})
    manifest = {"dataset": "cifar10", "source": "https://www.cs.toronto.edu/~kriz/cifar.html",
                "root": str(root), "classes": train.classes, "image_size": 32,
                "seed": seed, "records": records,
                "audit": {"note": "Official CIFAR split; torchvision verifies archive integrity. Duplicate image audit not performed."}}
    validate_manifest(manifest)
    write_json(output, manifest)


def difference_hash(image):
    thumbnail = image.convert("L").resize((9, 8))
    pixels = [thumbnail.getpixel((x, y)) for y in range(8) for x in range(9)]
    bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1] for y in range(8) for x in range(8)]
    return f"{sum(int(bit) << i for i, bit in enumerate(bits)):016x}"


def prepare_cats(root, output, seed, source, archive=None):
    root = Path(root).resolve()
    if not all((root / folder).is_dir() for folder in ("Cat", "Dog")):
        raise ValueError("--root must contain the PetImages Cat/ and Dog/ directories")
    rows, excluded = [], []
    for label, folder in enumerate(("Cat", "Dog")):
        for path in sorted((root / folder).iterdir()):
            if not path.is_file():
                continue
            relative = str(path.relative_to(root))
            try:
                with Image.open(path) as image:
                    image.load()
                    rgb = ImageOps.exif_transpose(image).convert("RGB")
                    pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    perceptual = difference_hash(rgb)
                rows.append({"id": relative, "path": relative, "label": label,
                             "sha256": file_hash(path), "pixel_sha256": pixels,
                             "dhash": perceptual})
            except (OSError, ValueError, Image.DecompressionBombError) as error:
                excluded.append({"path": relative, "reason": str(error)})
    if not rows:
        raise ValueError("No decodable images found")
    labels_by_pixels = defaultdict(set)
    for row in rows:
        labels_by_pixels[row["pixel_sha256"]].add(row["label"])
    usable = []
    for row in rows:
        if len(labels_by_pixels[row["pixel_sha256"]]) > 1:
            excluded.append({"path": row["path"], "reason": "identical decoded pixels with conflicting labels"})
        else:
            # Identical pixel images have identical dHashes; equal perceptual hashes
            # are grouped conservatively, even when the match may be a false positive.
            row["group"] = row["dhash"]
            usable.append(row)
    if not usable:
        raise ValueError("No images remain after conflict audit")
    duplicates = Counter(r["group"] for r in usable)
    manifest = {"dataset": "cats_dogs", "source": source, "root": str(root),
                "classes": ["cat", "dog"], "image_size": 96, "seed": seed,
                "records": group_split(usable, seed),
                "audit": {"excluded": excluded, "usable": len(usable),
                          "grouping": "Equal 64-bit dHash; catches exact pixels and some near duplicates, not an exhaustive similarity search.",
                          "duplicate_groups": {k: n for k, n in duplicates.items() if n > 1},
                          "archive_sha256": file_hash(archive) if archive else None}}
    validate_manifest(manifest)
    write_json(output, manifest)


class ImageDataset:
    def __init__(self, manifest, records, root=None):
        from torchvision import transforms
        self.manifest, self.records = manifest, records
        self.root = Path(root or manifest["root"])
        self.transform = transforms.Compose([
            transforms.Resize((manifest["image_size"], manifest["image_size"]), antialias=True),
            transforms.ToTensor(), transforms.Normalize((.5,) * 3, (.5,) * 3)])
        self.cifar = {}
        if manifest["dataset"] == "cifar10":
            from torchvision.datasets import CIFAR10
            for source in {r["source"] for r in records}:
                self.cifar[source] = CIFAR10(str(self.root), train=source == "train", download=False)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        if self.cifar:
            image, _ = self.cifar[row["source"]][row["index"]]
        else:
            with Image.open(self.root / row["path"]) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
        return self.transform(image), row["label"]


def datasets(config, include_test=False):
    """Return deterministic train/validation datasets; test is opt-in."""
    data = config["data"]
    manifest = json.loads(Path(data["manifest"]).read_text())
    validate_manifest(manifest)
    if len(manifest["classes"]) != config["model"]["classes"]:
        raise ValueError("Manifest classes do not match model")
    result = {}
    for split in ["train", "val"] + (["test"] if include_test else []):
        selected = [r for r in manifest["records"] if r["split"] == split or
                    (split == "train" and data.get("all_train") and r["split"] == "val")]
        if split == "val" and data.get("all_train"):
            selected = []
        if split == "train":
            selected = stratified_subset(selected, data.get("subset"), data.get("subset_seed", 42))
            selected = corrupt_labels(selected, len(manifest["classes"]),
                                      data.get("label_noise", 0.), data.get("noise_seed", 42))
        elif split == "val" and data.get("validation_subset"):
            selected = stratified_subset(selected, data["validation_subset"], data.get("subset_seed", 42))
        result[split] = ImageDataset(manifest, selected, data.get("root"))
    if not len(result["train"]):
        raise ValueError("Empty training split")
    return result


def corrupt_labels(records, classes, fraction=0., seed=42):
    """Change exactly round(N*fraction) training labels; never mutate a manifest."""
    if not 0 <= fraction <= 1 or classes < 2:
        raise ValueError("Label noise requires fraction in [0, 1] and at least two classes")
    rows = copy.deepcopy(records)
    rng = random.Random(seed)
    for index in rng.sample(range(len(rows)), round(len(rows) * fraction)):
        row = rows[index]
        row["clean_label"] = row["label"]
        row["label"] = (row["label"] + rng.randrange(1, classes)) % classes
    return rows


# Models

def scaled(value, width):
    return max(1, int(math.floor(value * width + 0.5)))


class Trunk(nn.Module):
    def __init__(self, width=1.0, pool_size=4):
        super().__init__()
        if width <= 0 or pool_size not in (1, 4):
            raise ValueError("width must be positive; pool_size must be 1 or 4")
        layers, channels = [], 3
        for out in [scaled(c, width) for c in (32, 64, 128)]:
            layers.extend([nn.Conv2d(channels, out, 3, padding=1), nn.ReLU(),
                           nn.Conv2d(out, out, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
            channels = out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.output_dim = scaled(128, width)
        self.projection = nn.Linear(channels * pool_size ** 2, self.output_dim)

    def forward(self, x):
        maps = self.features(x)
        return self.projection(self.pool(maps).flatten(1)), maps


class Node(nn.Module):
    def __init__(self, features, hidden, classes, internal, adapter_rank=0):
        super().__init__()
        if adapter_rank < 0:
            raise ValueError("adapter_rank must be nonnegative")
        self.adapter = None
        if adapter_rank:
            self.adapter = nn.Sequential(nn.Linear(features, adapter_rank), nn.ReLU(),
                                         nn.Linear(adapter_rank, features))
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)
        self.hidden = nn.Linear(features, hidden)
        self.classifier = nn.Linear(hidden, classes)
        self.router = nn.Linear(hidden, 1) if internal else None

    def forward(self, z):
        if self.adapter is not None:
            z = z + self.adapter(z)
        h = F.relu(self.hidden(z))
        return self.classifier(h), None if self.router is None else self.router(h).squeeze(-1)


@dataclass
class TreeOutput:
    log_probs: torch.Tensor
    class_logits: dict
    route_logits: dict
    log_reach: dict
    leaves: list
    maps: torch.Tensor | None


class NeuralTree(nn.Module):
    def __init__(self, classes=10, width=1.0, depth=3, pool_size=4,
                 node_width=None, pruned=None, adapter_rank=0, **unused):
        super().__init__()
        if not 0 <= depth <= 8:
            raise ValueError("depth must be between 0 and 8")
        self.trunk = Trunk(width, pool_size)
        self.depth, self.classes = depth, classes
        self.nodes = nn.ModuleDict()
        hidden = scaled(64, width) if node_width is None else int(node_width)
        if hidden < 1:
            raise ValueError("node_width must be positive")
        for i in range(2 ** (depth + 1) - 1):
            self.nodes[str(i)] = Node(self.trunk.output_dim, hidden, classes, i < 2 ** depth - 1, adapter_rank)
        for node in sorted(pruned or []):
            self.prune(node)

    @property
    def internal(self):
        return [int(k) for k, node in self.nodes.items() if node.router is not None]

    def prune(self, node_id):
        """Replace a subtree with its node classifier and delete unused parameters."""
        key = str(node_id)
        if key not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist")
        pending = [2 * node_id + 1, 2 * node_id + 2]
        while pending:
            child = pending.pop()
            if str(child) in self.nodes:
                pending.extend([2 * child + 1, 2 * child + 2])
                del self.nodes[str(child)]
        self.nodes[key].router = None

    def forward(self, x, *, from_features=False, uniform=False):
        z, maps = (x, None) if from_features else self.trunk(x)
        reaches = {0: z.new_zeros(z.shape[0])}
        classes, routes, leaves, terms = {}, {}, [], []
        for key, node in self.nodes.items():
            i = int(key)
            logits, route = node(z)
            classes[i] = logits
            if route is None:
                leaves.append(i)
                terms.append(reaches[i][:, None] + F.log_softmax(logits, dim=-1))
            else:
                routes[i] = route
                decision = torch.zeros_like(route) if uniform else route
                reaches[2 * i + 1] = reaches[i] + F.logsigmoid(-decision)
                reaches[2 * i + 2] = reaches[i] + F.logsigmoid(decision)
        return TreeOutput(torch.logsumexp(torch.stack(terms), dim=0), classes, routes,
                          reaches, leaves, maps)

    @torch.no_grad()
    def hard(self, x, *, from_features=False):
        """Evaluate only visited nodes; split a batch into conditional sub-batches."""
        z = x if from_features else self.trunk(x)[0]
        result = z.new_empty((len(z), self.classes))
        pending = [(0, torch.arange(len(z), device=z.device))]
        while pending:
            i, indices = pending.pop()
            if indices.numel() == 0:
                continue
            logits, route = self.nodes[str(i)](z[indices])
            if route is None:
                result[indices] = F.log_softmax(logits, dim=-1)
            else:
                right = route >= 0
                pending.extend([(2 * i + 1, indices[~right]), (2 * i + 2, indices[right])])
        return result


class Baseline(nn.Module):
    def __init__(self, classes=10, width=1., depth=3, pool_size=4, node_width=None,
                 baseline_hidden=None, adapter_rank=0, **unused):
        super().__init__()
        self.trunk = Trunk(width, pool_size)
        features = self.trunk.output_dim
        hidden = scaled(64, width) if node_width is None else int(node_width)
        # Match all tree head parameters, including auxiliary classifiers and routers.
        nodes, routers = 2 ** (depth + 1) - 1, 2 ** depth - 1
        tree_head = nodes * ((features + 1) * hidden + (hidden + 1) * classes)
        tree_head += routers * (hidden + 1)
        if adapter_rank:
            tree_head += nodes * ((features + 1) * adapter_rank + (adapter_rank + 1) * features)
        matched = max(1, round((tree_head - classes) / (features + classes + 1)))
        self.head = Node(features, baseline_hidden or matched, classes, False)

    def forward(self, x, *, from_features=False, uniform=False):
        z, maps = (x, None) if from_features else self.trunk(x)
        logits, _ = self.head(z)
        return TreeOutput(F.log_softmax(logits, -1), {0: logits}, {},
                          {0: z.new_zeros(len(z))}, [0], maps)

    @torch.no_grad()
    def hard(self, x, *, from_features=False):
        return self(x, from_features=from_features).log_probs


def build_model(config):
    if config.get("kind", "tree") not in ("tree", "baseline"):
        raise ValueError("model.kind must be tree or baseline")
    cls = Baseline if config.get("kind") == "baseline" else NeuralTree
    return cls(**config)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def shrink_tree(model, model_config, width, pool_size=1):
    """Structured L1 channel selection, including projection and node dimensions.

    A 4x4-to-1x1 pool conversion sums spatial projection weights. This is an
    initialization approximation, not a prediction-preserving transformation.
    """
    if not isinstance(model, NeuralTree) or not 0 < width < model_config.get("width", 1.):
        raise ValueError("Shrink requires a tree and a smaller positive width")
    config = copy.deepcopy(model_config)
    config.update(width=width, pool_size=pool_size)
    student = build_model(config).to(next(model.parameters()).device)
    def strongest(weight, count):
        return weight.flatten(1).abs().sum(1).topk(count).indices.sort().values
    inputs = torch.arange(3, device=next(model.parameters()).device)
    old_convs = [m for m in model.trunk.features if isinstance(m, nn.Conv2d)]
    new_convs = [m for m in student.trunk.features if isinstance(m, nn.Conv2d)]
    for old, new in zip(old_convs, new_convs):
        outputs = strongest(old.weight, new.out_channels)
        new.weight.copy_(old.weight[outputs][:, inputs])
        new.bias.copy_(old.bias[outputs])
        inputs = outputs
    old, new = model.trunk.projection, student.trunk.projection
    features = strongest(old.weight, new.out_features)
    old_pool = model_config.get("pool_size", 4)
    weights = old.weight[features].reshape(len(features), old_convs[-1].out_channels, old_pool, old_pool)[:, inputs]
    if pool_size == 1:
        weights = weights.sum((-1, -2), keepdim=True)
    elif pool_size != old_pool:
        raise ValueError("Pooling can only be retained or reduced to 1")
    new.weight.copy_(weights.flatten(1))
    new.bias.copy_(old.bias[features])
    for key, new_node in student.nodes.items():
        old_node = model.nodes[key]
        hidden = strongest(old_node.hidden.weight, new_node.hidden.out_features)
        new_node.hidden.weight.copy_(old_node.hidden.weight[hidden][:, features])
        new_node.hidden.bias.copy_(old_node.hidden.bias[hidden])
        new_node.classifier.weight.copy_(old_node.classifier.weight[:, hidden])
        new_node.classifier.bias.copy_(old_node.classifier.bias)
        if new_node.router is not None:
            new_node.router.weight.copy_(old_node.router.weight[:, hidden])
            new_node.router.bias.copy_(old_node.router.bias)
        if new_node.adapter is not None:
            new_node.adapter[0].weight.copy_(old_node.adapter[0].weight[:, features])
            new_node.adapter[0].bias.copy_(old_node.adapter[0].bias)
            new_node.adapter[2].weight.copy_(old_node.adapter[2].weight[features])
            new_node.adapter[2].bias.copy_(old_node.adapter[2].bias[features])
    return student, config


# Losses

def objective(output, labels, auxiliary=0., balance=0.):
    nll = F.nll_loss(output.log_probs, labels)
    aux, bal = nll.new_zeros(()), nll.new_zeros(())
    levels = {}
    for node in output.route_logits:
        levels.setdefault(int(math.log2(node + 1)), []).append(node)
    for nodes in levels.values():
        level_aux, level_bal = nll.new_zeros(()), []
        for node in nodes:
            reach = output.log_reach[node].exp()
            ce = F.cross_entropy(output.class_logits[node], labels, reduction="none")
            level_aux = level_aux + (reach * ce).mean()
            mass = reach.sum()
            # Smooth effectively unvisited nodes toward 0.5 without NaNs.
            alpha = ((reach * output.route_logits[node].sigmoid()).sum() + 5e-7) / (mass + 1e-6)
            alpha = alpha.clamp(1e-6, 1 - 1e-6)
            level_bal.append(-0.5 * (alpha.log() + torch.log1p(-alpha)))
        aux = aux + level_aux
        bal = bal + torch.stack(level_bal).mean()
    if levels:
        aux, bal = aux / len(levels), bal / len(levels)
    return nll + auxiliary * aux + balance * bal, {"nll": nll, "auxiliary": aux, "balance": bal}


def penalty_scale(epoch, decay_epochs):
    """Epochs are zero-based. The fixed schedule reaches zero at decay_epochs."""
    return max(0., 1. - epoch / decay_epochs) if decay_epochs > 0 else 0.


# Checkpoints and reproducibility

def device_for(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this interpreter. Use --device cpu or a supported PyTorch environment.")
    return torch.device(name)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.backends.mps.is_available():
        state["mps"] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "mps" in state and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["mps"].cpu())
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def provenance(config):
    import torchvision
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    source = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source.update(path.name.encode() + path.read_bytes())
    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "commit": commit, "source_sha256": source.hexdigest(),
            "manifest_sha256": file_hash(config["data"]["manifest"]),
            "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")}


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Project checkpoints contain optimizer and Python/NumPy RNG state. Load only
    # checkpoints you created or trust; do not use this on arbitrary downloaded files.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"]["model"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


# Training and evaluation

def load_data(config, include_test=False):
    cache_path = config["data"].get("cache")
    if not cache_path:
        return datasets(config, include_test), None
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache["manifest_sha256"] != file_hash(config["data"]["manifest"]):
        raise ValueError("Feature cache belongs to a different manifest")
    for key, default in [("subset", None), ("subset_seed", 42), ("all_train", False),
                         ("label_noise", 0.), ("noise_seed", 42), ("validation_subset", None)]:
        if config["data"].get(key, default) != cache["data_config"].get(key, default):
            raise ValueError(f"Cache training selection differs: {key}")
    for key, default in [("width", 1.), ("pool_size", 4)]:
        if config["model"].get(key, default) != cache["model_config"].get(key, default):
            raise ValueError(f"Cached trunk is fixed; cannot change {key}. Vary --node-width instead.")
    required = ["train", "val"] + (["test"] if include_test else [])
    if any(split not in cache["splits"] for split in required):
        raise ValueError("Requested split missing from cache. Regenerate with --include-test for final testing.")
    return {s: TensorDataset(cache["splits"][s]["features"], cache["splits"][s]["labels"])
            for s in required}, cache


@torch.no_grad()
def evaluate(model, loader, device, *, from_features=False, mode="soft", diagnostics=False):
    model.eval()
    count, errors, loss_sum = 0, 0, 0.
    stats = {}
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = None
        if mode == "hard":
            log_probs = model.hard(inputs, from_features=from_features)
        else:
            output = model(inputs, from_features=from_features, uniform=mode == "uniform")
            log_probs = output.log_probs
        count += len(labels)
        errors += int((log_probs.argmax(-1) != labels).sum().item())
        loss_sum += float(F.nll_loss(log_probs, labels, reduction="sum").item())
        if diagnostics and output is not None:
            cpu_labels = labels.cpu()
            hard_masks = {0: torch.ones(len(labels), dtype=torch.bool)}
            for node, logits in output.class_logits.items():
                reach = output.log_reach[node].exp().cpu()
                correct = (logits.argmax(-1).cpu() == cpu_labels).float()
                entry = stats.setdefault(str(node), {"mass": 0., "correct_mass": 0., "hard_count": 0,
                                                    "classes": [0.] * log_probs.shape[1],
                                                    "right_mass": 0., "right_squared_mass": 0., "entropy_mass": 0.})
                entry["mass"] += float(reach.sum())
                entry["correct_mass"] += float((correct * reach).sum())
                entry["hard_count"] += int(hard_masks[node].sum())
                class_mass = torch.bincount(cpu_labels, weights=reach, minlength=log_probs.shape[1]).tolist()
                entry["classes"] = [a + b for a, b in zip(entry["classes"], class_mass)]
                if node in output.route_logits:
                    p = output.route_logits[node].sigmoid().cpu()
                    if mode == "uniform":
                        p = torch.full_like(p, .5)
                    safe = p.clamp(1e-6, 1 - 1e-6)
                    entropy = -(safe * safe.log() + (1 - safe) * (1 - safe).log())
                    entry["right_mass"] += float((reach * p).sum())
                    entry["right_squared_mass"] += float((reach * p.square()).sum())
                    entry["entropy_mass"] += float((reach * entropy).sum())
                    hard_masks[2 * node + 1] = hard_masks[node] & (p < .5)
                    hard_masks[2 * node + 2] = hard_masks[node] & (p >= .5)
    if not count:
        return None
    for entry in stats.values():
        mass = max(entry["mass"], 1e-12)
        entry["occupancy"] = entry["mass"] / count
        entry["conditional_accuracy"] = entry["correct_mass"] / mass
        entry["right_mean"] = entry["right_mass"] / mass
        entry["right_std"] = max(0., entry["right_squared_mass"] / mass - entry["right_mean"] ** 2) ** .5
        entry["routing_entropy"] = entry["entropy_mass"] / mass
    return {"count": count, "errors": errors, "accuracy": 1 - errors / count,
            "cross_entropy": loss_sum / count, "nodes": stats}


def gradient_norms(model):
    groups = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            key = "trunk" if name.startswith("trunk.") else ".".join(name.split(".")[:2])
            if ".router." in name:
                key += ".router"
            groups[key] = groups.get(key, 0.) + float(parameter.grad.detach().square().sum().item())
    return {k: v ** .5 for k, v in groups.items()}


def run(config, output_dir, resume=None, initial_model=None):
    config = copy.deepcopy(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    if not resume and (history_path.exists() or (output_dir / "last.pt").exists()):
        raise ValueError("Run already exists. Use --resume or a new --out directory.")
    train_cfg = config["train"]
    seed_everything(train_cfg["seed"])
    device = device_for(train_cfg.get("device", "auto"))
    data, cache = load_data(config)
    metadata = provenance(config)
    if cache is not None:
        metadata["cache_sha256"] = file_hash(config["data"]["cache"])
    checkpoint = None
    if resume:
        model, checkpoint = load_checkpoint(resume, device)
        previous = copy.deepcopy(checkpoint["config"])
        for key in ("epochs", "device", "workers"):
            previous["train"][key] = config["train"][key]
        if previous != config:
            raise ValueError("Resume permits only changes to epochs, device, and workers")
        for key in ("manifest_sha256", "cache_sha256"):
            if checkpoint["metadata"].get(key) != metadata.get(key):
                raise ValueError(f"Resume input changed: {key}")
    else:
        model = (initial_model or build_model(config["model"])).to(device)
    if cache is not None:
        if not resume and initial_model is None:
            model.trunk.load_state_dict(cache["trunk"])
        model.trunk.requires_grad_(False)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=train_cfg["lr"], weight_decay=0.)
    start, best, near_seen, exact_seen = 0, float("inf"), False, False
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_val"]
        near_seen, exact_seen = checkpoint["near_seen"], checkpoint["exact_seen"]
        restore_rng(checkpoint["rng"])
        if history_path.exists():
            rows = [json.loads(line) for line in history_path.read_text().splitlines()]
            if rows and rows[-1]["epoch"] != start:
                raise ValueError("History does not end at the resume checkpoint; resume into a new directory")
    if train_cfg["epochs"] <= start:
        raise ValueError("--epochs is the total epoch target and must exceed the completed epoch")
    loaders = {s: DataLoader(ds, batch_size=train_cfg["batch_size"], shuffle=False,
                            num_workers=train_cfg.get("workers", 0)) for s, ds in data.items()}
    training = DataLoader(data["train"], batch_size=train_cfg["batch_size"], shuffle=True,
                          num_workers=train_cfg.get("workers", 0))
    metadata.update({"device": str(device), "parameters": parameter_count(model),
                     "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                     "train_samples": len(data["train"]), "validation_samples": len(data["val"]),
                     "from_features": cache is not None, "resumed_from": str(resume) if resume else None})
    if initial_model is not None:
        metadata["initialized_from_existing_model"] = True
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "metadata.json", metadata)
    if cache is None:
        write_json(output_dir / "sample_ids.json", {s: [r["id"] for r in ds.records] for s, ds in data.items()})
        write_json(output_dir / "corrupted_labels.json", [
            {k: r[k] for k in ("id", "label", "clean_label")}
            for r in data["train"].records if "clean_label" in r])
    else:
        write_json(output_dir / "sample_ids.json", {s: cache["splits"][s]["ids"] for s in data})
    print(f"device={device}, parameters={parameter_count(model):,}, train={len(data['train'])}, val={len(data['val'])}", flush=True)
    for epoch in range(start, train_cfg["epochs"]):
        model.train()
        synchronize(device)
        begin = time.perf_counter()
        scale = penalty_scale(epoch, train_cfg["penalty_decay_epochs"])
        gradients, running_loss = {}, 0.
        for step, (inputs, labels) in enumerate(training):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs, from_features=cache is not None, uniform=train_cfg.get("uniform", False))
            loss, _ = objective(prediction, labels, train_cfg["auxiliary"] * scale, train_cfg["balance"] * scale)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch + 1}, step {step}")
            loss.backward()
            if step == 0:
                gradients = gradient_norms(model)
            optimizer.step()
            running_loss += float(loss.item()) * len(labels)
        synchronize(device)
        training_seconds = time.perf_counter() - begin
        mode = "uniform" if train_cfg.get("uniform", False) else "soft"
        train_metrics = evaluate(model, loaders["train"], device, from_features=cache is not None, mode=mode, diagnostics=True)
        val_metrics = evaluate(model, loaders["val"], device, from_features=cache is not None, mode=mode)
        synchronize(device)
        row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics,
               "optimization_loss": running_loss / len(data["train"]), "penalty_scale": scale,
               "train_seconds": training_seconds, "epoch_seconds": time.perf_counter() - begin,
               "gradient_norms_first_batch": gradients, "parameters": parameter_count(model)}
        improved = val_metrics is not None and val_metrics["cross_entropy"] < best
        if improved:
            best = val_metrics["cross_entropy"]
        near = train_metrics["errors"] / train_metrics["count"] <= .001
        exact = train_metrics["errors"] == 0
        first_near, first_exact = near and not near_seen, exact and not exact_seen
        near_seen, exact_seen = near_seen or near, exact_seen or exact
        payload = {"config": config, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "epoch": epoch, "best_val": best, "near_seen": near_seen, "exact_seen": exact_seen,
                   "rng": rng_state(), "metadata": metadata, "metrics": row}
        save_checkpoint(output_dir / "last.pt", payload)
        for name, condition in [("best_val", improved), ("first_near", first_near), ("first_exact", first_exact)]:
            if condition:
                save_checkpoint(output_dir / f"{name}.pt", payload)
        with history_path.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(f"epoch={epoch + 1} train_errors={train_metrics['errors']}/{train_metrics['count']} "
              f"val_accuracy={None if val_metrics is None else round(val_metrics['accuracy'], 4)} "
              f"seconds={row['epoch_seconds']:.2f}", flush=True)
    return output_dir / "last.pt"
