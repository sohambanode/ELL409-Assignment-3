"""Run experiments with python -m neural_tree COMMAND --help."""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .core import (
    NeuralTree, build_model, datasets, device_for, evaluate, file_hash,
    load_checkpoint, load_data, objective, parameter_count, prepare_cats,
    prepare_cifar, run, save_checkpoint, seed_everything, synchronize,
    validate_manifest, write_json, shrink_tree,
)


def checked_checkpoint(args):
    """Load a checkpoint and verify its dataset before running an experiment."""
    device = device_for(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = copy.deepcopy(checkpoint["config"])
    if file_hash(config["data"]["manifest"]) != checkpoint["metadata"]["manifest_sha256"]:
        raise ValueError("Manifest changed since checkpoint creation")
    return device, model, checkpoint, config


def plotting():
    """Load the noninteractive plotting backend only for plot commands."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def data_command(args):
    if args.dataset == "cifar10":
        prepare_cifar(args.root, args.output, args.seed)
    elif args.dataset == "cats_dogs":
        prepare_cats(args.root, args.output, args.seed, args.source, args.archive)
    else:
        validate_manifest(json.loads(Path(args.output).read_text()), verify_files=True)
    manifest = json.loads(Path(args.output).read_text())
    print(json.dumps({s: dict(Counter(r["label"] for r in manifest["records"] if r["split"] == s))
                      for s in ("train", "val", "test")}, indent=2))


def train_command(args):
    if args.resume:
        _, checkpoint = load_checkpoint(args.resume)
        config = checkpoint["config"]
    elif args.config:
        config = json.loads(Path(args.config).read_text())
    else:
        raise ValueError("provide --config or --resume")
    overrides = {
        "train": ("epochs", "batch_size", "seed", "workers", "lr", "device"),
        "model": ("width", "node_width", "kind", "adapter_rank", "pool_size"),
        "data": ("subset", "cache", "label_noise", "noise_seed", "validation_subset"),
    }
    for section, keys in overrides.items():
        config[section].update({key: getattr(args, key) for key in keys
                                if getattr(args, key) is not None})
    if args.all_train:
        config["data"]["all_train"] = True
    if args.uniform:
        config["train"]["uniform"] = True
    run(config, args.out, args.resume)


def evaluate_command(args):
    device, model, checkpoint, config = checked_checkpoint(args)
    data, cache = load_data(config, include_test=args.split == "test")
    loader = DataLoader(data[args.split], batch_size=config["train"]["batch_size"])
    metrics = evaluate(model, loader, device, from_features=cache is not None,
                       mode=args.mode, diagnostics=args.mode != "hard")
    if metrics is None:
        raise ValueError("Requested split is empty")
    result = {"checkpoint": args.checkpoint, "epoch": checkpoint["epoch"] + 1,
              "split": args.split, "mode": args.mode, "metrics": metrics}
    write_json(args.out, result)
    print({k: v for k, v in metrics.items() if k != "nodes"})


def cache_command(args):
    device, model, checkpoint, config = checked_checkpoint(args)
    config["data"].pop("cache", None)
    data = datasets(config, include_test=args.include_test)
    splits = {}
    model.eval()
    with torch.no_grad():
        for split, dataset in data.items():
            features, labels = [], []
            for x, y in DataLoader(dataset, batch_size=config["train"]["batch_size"]):
                features.append(model.trunk(x.to(device))[0].cpu())
                labels.append(y)
            splits[split] = {"features": torch.cat(features) if features else torch.empty(0, model.trunk.output_dim),
                             "labels": torch.cat(labels) if labels else torch.empty(0, dtype=torch.long),
                             "ids": [r["id"] for r in dataset.records]}
    save_checkpoint(args.out, {"splits": splits, "trunk": {k: v.cpu() for k, v in model.trunk.state_dict().items()},
                              "manifest_sha256": file_hash(config["data"]["manifest"]),
                              "checkpoint_sha256": file_hash(args.checkpoint),
                              "data_config": config["data"], "model_config": config["model"],
                              "checkpoint": args.checkpoint})
    print(f"Saved features to {args.out}. Trunk remains part of every model's parameter count.")


def sweeps_command(args):
    if args.grid:
        grid = json.loads(Path(args.grid).read_text())
        root = Path(args.grid).parent
    else:
        if not args.out:
            raise ValueError("--out is required with --config")
        root = Path(args.out)
        if (root / "grid.json").exists():
            raise ValueError("Grid already exists; use --grid to execute it")
        config = json.loads(Path(args.config).read_text())
        if config["data"].get("cache") and not args.node_widths:
            raise ValueError("Frozen features require --node-widths; the cached trunk cannot vary")
        grid = []
        key = "node_width" if args.node_widths else "width"
        for value in args.node_widths or args.widths:
            for seed in args.seeds:
                candidate = copy.deepcopy(config)
                candidate["model"][key] = value
                candidate["train"]["seed"] = seed
                grid.append({"name": f"{key}_{value:g}_seed_{seed}", "config": candidate})
        if len({row["name"] for row in grid}) != len(grid):
            raise ValueError("Duplicate widths or seeds")
        write_json(root / "grid.json", grid)
    print(f"Grid: {len(grid)} runs, {sum(r['config']['train']['epochs'] for r in grid)} total target epochs")
    if not args.execute:
        print("Plan saved only. Execute later with --grid PATH --execute.")
        return
    for row in grid:
        out = root / row["name"]
        status_path = out / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
            continue
        resume = out / "last.pt"
        write_json(status_path, {"status": "running"})
        try:
            if resume.exists():
                _, checkpoint = load_checkpoint(resume)
                finished = checkpoint["epoch"] + 1 >= row["config"]["train"]["epochs"]
            else:
                finished = False
            if not finished:
                run(row["config"], out, resume=resume if resume.exists() else None)
            write_json(status_path, {"status": "complete", "meaning": "Reached epoch budget; interpolation is recorded separately."})
        except Exception as error:
            write_json(status_path, {"status": "failed", "reason": str(error)})
            raise


def macs(model, x, mode="soft", from_features=False):
    """Count multiply-accumulate pairs for executed Conv2d/Linear layers only."""
    count, hooks = [0], []
    def hook(module, inputs, output):
        if isinstance(module, nn.Linear):
            count[0] += output.numel() * module.in_features
        elif isinstance(module, nn.Conv2d):
            count[0] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            if mode == "hard":
                model.hard(x, from_features=from_features)
            else:
                model(x, from_features=from_features)
    finally:
        for handle in hooks:
            handle.remove()
    return count[0] / len(x)


def timed(action, device, repeats, warmup=3):
    for _ in range(warmup):
        action()
    timings = []
    for _ in range(repeats):
        synchronize(device)
        begin = time.perf_counter()
        action()
        synchronize(device)
        timings.append(time.perf_counter() - begin)
    return {"median_seconds": statistics.median(timings), "min_seconds": min(timings),
            "max_seconds": max(timings)}


def benchmark_command(args):
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")
    seed_everything(42)
    device = device_for(args.device)
    if args.checkpoint:
        model, checkpoint = load_checkpoint(args.checkpoint, device)
        config = checkpoint["config"]
    else:
        config = json.loads(Path(args.config).read_text())
        model = build_model(config["model"]).to(device)
    data, cache = load_data(config)
    if cache is not None:
        if not args.checkpoint:
            model.trunk.load_state_dict(cache["trunk"])
        model.trunk.requires_grad_(False)
    loader = DataLoader(data["train"], batch_size=args.batch_size, num_workers=config["train"].get("workers", 0))
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    from_features = cache is not None
    model.eval()
    buffer = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buffer)
    result = {"device": str(device), "parameters": parameter_count(model),
              "inference_state_bytes": buffer.tell(), "batch_size": len(x),
              "cached_features": from_features, "inference": {}}
    with torch.no_grad():
        for mode in ("soft", "hard"):
            def inference(batch=x):
                return model.hard(batch, from_features=from_features) if mode == "hard" else model(batch, from_features=from_features)
            batch_time = timed(inference, device, args.steps)
            single_time = timed(lambda: inference(x[:1]), device, args.steps)
            result["inference"][mode] = {"batch": batch_time, "single_image": single_time,
                                         "images_per_second": len(x) / batch_time["median_seconds"],
                                         "linear_conv_macs_per_image": macs(model, x, mode, from_features)}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=config["train"]["lr"])
    model.train()
    def train_step(inputs=x, labels=y):
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, from_features=from_features, uniform=config["train"].get("uniform", False))
        loss, _ = objective(output, labels, config["train"]["auxiliary"], config["train"]["balance"])
        loss.backward()
        optimizer.step()
    compute = timed(train_step, device, args.steps)
    result["training_compute"] = compute
    result["estimated_compute_only_epoch_seconds"] = compute["median_seconds"] * ((len(data["train"]) + len(x) - 1) // len(x))
    if args.full_epoch:
        synchronize(device)
        begin = time.perf_counter()
        for inputs, labels in loader:
            train_step(inputs.to(device), labels.to(device))
        synchronize(device)
        result["full_train_epoch_seconds"] = time.perf_counter() - begin
        begin = time.perf_counter()
        evaluate(model, loader, device, from_features=from_features)
        val_loader = DataLoader(data["val"], batch_size=args.batch_size)
        evaluate(model, val_loader, device, from_features=from_features)
        synchronize(device)
        result["full_train_and_val_evaluation_seconds"] = time.perf_counter() - begin
    if device.type == "mps":
        result["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        result["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    result["note"] = "Temporary model updated during timing; no trained checkpoint saved. MAC count excludes activation, routing, pooling and mixture operations. Cached timing excludes the CNN."
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


def node_score(output, node, target, direction, class_index):
    if node not in output.class_logits:
        raise ValueError(f"Node {node} is not in this model")
    if target == "router":
        if node not in output.route_logits:
            raise ValueError("A leaf has no router; choose --target class")
        return output.route_logits[node] * (1 if direction == "right" else -1)
    if target == "reach":
        if node == 0:
            raise ValueError("Root reach is constant; choose a non-root node")
        return output.log_reach[node]
    return F.log_softmax(output.class_logits[node], -1)[:, class_index]


def normalize(array):
    array = np.asarray(array)
    return (array - array.min()) / max(float(array.max() - array.min()), 1e-12)


def receptive_field():
    size, jump = 1, 1
    for _ in range(3):
        for kernel, stride in [(3, 1), (3, 1), (2, 2)]:
            size += (kernel - 1) * jump
            jump *= stride
    return {"last_shared_map_local_receptive_field": size, "map_stride": jump,
            "node_architectural_receptive_field": "Whole input through spatial pooling and dense projection. Borders include padding."}


def explain_command(args):
    plt = plotting()
    if min(args.count, args.scan_limit, args.patch) < 1:
        raise ValueError("count, scan-limit and patch must be positive")
    device, model, checkpoint, config = checked_checkpoint(args)
    model.eval()
    config["data"].pop("cache", None)  # Run the retained trunk for pixel attribution.
    data = datasets(config, include_test=args.split == "test")[args.split]
    if args.class_index is not None and not 0 <= args.class_index < config["model"]["classes"]:
        raise ValueError("class-index is outside the model's class range")
    if args.patch > data.manifest["image_size"]:
        raise ValueError("patch cannot exceed image size")
    uniform = config["train"].get("uniform", False)
    if uniform and args.target in ("router", "reach"):
        raise ValueError("Uniform-routing runs have constant routing; use --target class")
    rows = []
    counts = {}
    scan_count = min(args.scan_limit, len(data))
    if scan_count == 0:
        raise ValueError("Selected split is empty")
    with torch.no_grad():
        for start in range(0, scan_count, 64):
            batch = [data[i] for i in range(start, min(start + 64, scan_count))]
            x = torch.stack([item[0] for item in batch]).to(device)
            output = model(x, uniform=uniform)
            node_score(output, args.node, args.target, args.direction, args.class_index or 0)
            reach = output.log_reach[args.node].exp().cpu().tolist()
            prediction = output.log_probs.argmax(-1).cpu().tolist()
            route = output.route_logits.get(args.node)
            probs = route.sigmoid().cpu().tolist() if route is not None else [None] * len(batch)
            for j, (_, label) in enumerate(batch):
                rows.append({"index": start + j, "label": label, "prediction": prediction[j],
                             "reach": reach[j], "right_probability": probs[j]})
                # Summarize the full scan before selecting the biased gallery.
                counts[str(label)] = counts.get(str(label), 0.) + reach[j]
    if args.selection == "high-reach":
        rows.sort(key=lambda r: -r["reach"])
    elif args.selection == "uncertain":
        if rows[0]["right_probability"] is None:
            raise ValueError("Uncertain-route selection requires an internal node")
        rows.sort(key=lambda r: (abs(r["right_probability"] - .5), -r["reach"]))
    elif args.selection == "errors":
        rows = [r for r in rows if r["label"] != r["prediction"]]
        rows.sort(key=lambda r: -r["reach"])
    selected = rows[:args.count]
    if not selected:
        raise ValueError("No matching examples in scan; increase --scan-limit or change --selection")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figures, evidence = [], []
    for rank, row in enumerate(selected):
        pixels, label = data[row["index"]]
        x = pixels.unsqueeze(0).to(device).requires_grad_(True)
        class_index = label if args.class_index is None else args.class_index
        output = model(x, uniform=uniform)
        score = node_score(output, args.node, args.target, args.direction, class_index).sum()
        image_gradient, feature_gradient = torch.autograd.grad(score, (x, output.maps))
        weights = feature_gradient.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * output.maps).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()
        saliency = image_gradient.abs().amax(1)[0].cpu().numpy()
        original = ((pixels + 1) / 2).permute(1, 2, 0).numpy().clip(0, 1)
        positions = [(top, left) for top in range(0, x.shape[2], args.patch)
                     for left in range(0, x.shape[3], args.patch)]
        effects, final_effects, route_effects = [], [], []
        with torch.no_grad():
            base_score = float(score.item())
            base_final = float(output.log_probs[0, class_index].exp().item())
            base_route = (float(output.route_logits[args.node].sigmoid().item())
                          if args.node in output.route_logits else None)
            for begin in range(0, len(positions), 64):
                chunk = positions[begin:begin + 64]
                masked = x.detach().repeat(len(chunk), 1, 1, 1)
                for j, (top, left) in enumerate(chunk):
                    masked[j, :, top:top + args.patch, left:left + args.patch] = 0  # RGB 0.5
                changed = model(masked, uniform=uniform)
                effects.extend((base_score - node_score(changed, args.node, args.target, args.direction, class_index)).cpu().tolist())
                final_effects.extend((base_final - changed.log_probs[:, class_index].exp()).cpu().tolist())
                if base_route is not None:
                    route_effects.extend((base_route - changed.route_logits[args.node].sigmoid()).cpu().tolist())
        occlusion = np.zeros(x.shape[-2:])
        saliency_patch = []
        for (top, left), effect in zip(positions, effects):
            occlusion[top:top + args.patch, left:left + args.patch] = effect
            saliency_patch.append(float(cam[top:top + args.patch, left:left + args.patch].mean()))
        top_index = int(np.argmax(saliency_patch))
        random_indices = random.Random(42 + rank).sample(range(len(positions)), min(20, len(positions)))
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(original)
        axes[1].imshow(normalize(saliency), cmap="inferno")
        axes[2].imshow(original)
        axes[2].imshow(normalize(cam), cmap="jet", alpha=.45)
        limit = max(float(np.abs(occlusion).max()), 1e-8)
        heatmap = axes[3].imshow(occlusion, cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(heatmap, ax=axes[3], fraction=.046, pad=.04)
        for ax, title in zip(axes, ["Image", "Input sensitivity", "Node Grad-CAM", "Occlusion: score drop"]):
            ax.set_title(title)
            ax.axis("off")
        if np.ptp(cam) < 1e-12:
            axes[2].set_title("Grad-CAM (constant)")
        fig.suptitle(f"Node {args.node}, {args.target} {args.direction if args.target == 'router' else class_index}; "
                     f"label={label}, prediction={row['prediction']}, reach={row['reach']:.3f}")
        fig.tight_layout()
        path = out / f"example_{rank:02d}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        figures.append(path.name)
        evidence.append({**row, "id": data.records[row["index"]]["id"], "class_index": class_index,
                         "base_target_score": base_score, "positions": positions, "target_score_drops": effects,
                         "final_class_probability_drops": final_effects, "right_probability_drops": route_effects,
                         "highest_cam_patch_score_drop": effects[top_index],
                         "random_patch_mean_score_drop": float(np.mean([effects[i] for i in random_indices])),
                         "cam_is_constant": bool(np.ptp(cam) < 1e-12)})
    write_json(out / "evidence.json", {"checkpoint": args.checkpoint, "node": args.node,
               "target": args.target, "selection": args.selection, "scanned_count": scan_count,
               "reach_weighted_class_mass_in_scan": counts, "receptive_field": receptive_field(),
               "mask_baseline_rgb": .5, "patch_size": args.patch, "examples": evidence})
    (out / "gallery.md").write_text("# Node evidence\n\nThese maps measure sensitivity, not semantic proof. "
                                   "Occlusion uses constant grey patches; values can be negative.\n\n" +
                                   "\n\n".join(f"![Node evidence]({p})" for p in figures) + "\n")
    print(f"Wrote {len(figures)} explanations to {out}")


def compress_command(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if Path(args.out).exists() and any(Path(args.out).iterdir()):
        raise ValueError("Use a new output directory")
    device, model, checkpoint, config = checked_checkpoint(args)
    if not isinstance(model, NeuralTree):
        raise ValueError("Only trees can be pruned")
    data, cache = load_data(config)
    if not len(data["val"]):
        raise ValueError("Compression selection requires a held-out validation split")
    loaders = {s: DataLoader(ds, batch_size=config["train"]["batch_size"]) for s, ds in data.items()}
    mode = "uniform" if config["train"].get("uniform", False) else "soft"
    def score(candidate):
        return {s: evaluate(candidate, loader, device, from_features=cache is not None, mode=mode)
                for s, loader in loaders.items()}
    reference = score(model)
    candidates = model.internal if args.node is None else [args.node]
    if not candidates or any(n not in model.internal for n in candidates):
        raise ValueError("Select an existing internal node")
    ranking = []
    for node in candidates:
        candidate = copy.deepcopy(model)
        candidate.prune(node)
        metrics = evaluate(candidate, loaders["val"], device, from_features=cache is not None, mode=mode)
        ranking.append({"node": node, "parameters": parameter_count(candidate), "val": metrics})
        del candidate
    ranking.sort(key=lambda r: (r["val"]["errors"], r["val"]["cross_entropy"], r["parameters"]))
    chosen = ranking[0]["node"]
    model.prune(chosen)
    # Reconstruct the topology exactly, including repeated compression passes.
    original_internal = set(range(2 ** model.depth - 1))
    config["model"]["pruned"] = [int(k) for k, node in model.nodes.items()
                                  if int(k) in original_internal and node.router is None]
    config["train"].update({"epochs": args.epochs, "lr": args.lr, "device": args.device,
                            "auxiliary": 0., "balance": 0., "penalty_decay_epochs": 0})
    last = run(config, args.out, initial_model=model)
    compressed, _ = load_checkpoint(last, device)
    measured = score(compressed)
    accepted = (measured["train"]["errors"] / measured["train"]["count"] <= .001 and
                measured["val"]["accuracy"] >= reference["val"]["accuracy"] - .01)
    write_json(Path(args.out) / "compression.json", {
        "reference_checkpoint": args.checkpoint, "reference_sha256": file_hash(args.checkpoint),
        "reference": reference, "ranking_before_finetuning": ranking, "pruned_node": chosen,
        "compressed": measured, "parameters": parameter_count(compressed), "accepted": accepted,
        "criterion": "train accuracy >=99.9%, validation loss in accuracy <=1 percentage point relative to this reference"})
    print(f"Pruned node {chosen}; parameters={parameter_count(compressed):,}; accepted={accepted}")


def comparison_key(config, metadata):
    recipe = copy.deepcopy(config)
    recipe["model"].pop("width", None)
    recipe["model"].pop("node_width", None)
    recipe["train"].pop("seed", None)
    recipe["train"].pop("device", None)
    recipe["train"].pop("workers", None)
    recipe["data"].pop("root", None)
    recipe["manifest_sha256"] = metadata["manifest_sha256"]
    recipe["source_sha256"] = metadata["source_sha256"]
    recipe["cache_sha256"] = metadata.get("cache_sha256")
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:10]


def shrink_command(args):
    device, model, checkpoint, config = checked_checkpoint(args)
    if config["data"].get("cache"):
        raise ValueError("Trunk compression requires image training, not cached features")
    data, _ = load_data(config)
    if not len(data["val"]):
        raise ValueError("Compression requires validation data")
    loaders = {s: DataLoader(ds, batch_size=config["train"]["batch_size"]) for s, ds in data.items()}
    mode = "uniform" if config["train"].get("uniform", False) else "soft"
    def score(candidate):
        return {s: evaluate(candidate, loader, device, mode=mode) for s, loader in loaders.items()}
    reference = score(model)
    original_count = parameter_count(model)
    student, config["model"] = shrink_tree(model, config["model"], args.width, args.pool_size)
    before = score(student)
    config["train"].update(epochs=args.epochs, lr=args.lr, device=args.device,
                            auxiliary=0., balance=0., penalty_decay_epochs=0)
    last = run(config, args.out, initial_model=student)
    student, _ = load_checkpoint(last, device)
    measured = score(student)
    accepted = (measured["train"]["accuracy"] >= .999 and
                measured["val"]["accuracy"] >= reference["val"]["accuracy"] - .01)
    write_json(Path(args.out) / "compression.json", {
        "method": "Structured L1 channel selection and spatial pooling reduction, then fine-tuning",
        "model_config": config["model"],
        "reference_checkpoint": args.checkpoint, "reference_sha256": file_hash(args.checkpoint),
        "reference": reference, "before_finetuning": before, "compressed": measured,
        "reference_parameters": original_count, "parameters": parameter_count(student),
        "parameter_reduction": 1 - parameter_count(student) / original_count, "accepted": accepted,
        "criterion": "train accuracy >=99.9%, validation accuracy drop <=1 percentage point"})
    print(f"Trunk compressed: {original_count} -> {parameter_count(student)}; accepted={accepted}")


def report_command(args):
    from matplotlib.ticker import MaxNLocator
    plt = plotting()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    histories = sorted({p for root in args.runs for p in Path(root).rglob("history.jsonl")})
    if not histories:
        raise ValueError("No history.jsonl files found")
    rows, groups, epoch_figures = [], {}, []
    for index, path in enumerate(histories):
        history = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not history:
            continue
        config = json.loads((path.parent / "config.json").read_text())
        metadata = json.loads((path.parent / "metadata.json").read_text())
        last = history[-1]
        key = comparison_key(config, metadata)
        row = {"run": str(path.parent), "group": key, "kind": config["model"].get("kind", "tree"),
               "seed": config["train"]["seed"], "parameters": last["parameters"], "epoch": last["epoch"],
               "target_epochs": config["train"]["epochs"], "budget_complete": last["epoch"] >= config["train"]["epochs"],
               "train_count": last["train"]["count"], "train_errors": last["train"]["errors"],
               "train_cross_entropy": last["train"]["cross_entropy"],
               "val_error": None if last["val"] is None else 1 - last["val"]["accuracy"],
               "val_cross_entropy": None if last["val"] is None else last["val"]["cross_entropy"],
               "first_exact_epoch": next((r["epoch"] for r in history if r["train"]["errors"] == 0), None),
               "first_near_epoch": next((r["epoch"] for r in history if r["train"]["errors"] / r["train"]["count"] <= .001), None),
               "logged_seconds": sum(r["epoch_seconds"] for r in history),
               "cached_features": metadata.get("from_features", False)}
        rows.append(row)
        groups.setdefault(key, []).append(row)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        epochs = [r["epoch"] for r in history]
        for split in ("train", "val"):
            if all(r[split] is not None for r in history):
                axes[0].plot(epochs, [1 - r[split]["accuracy"] for r in history], label=split)
                axes[1].plot(epochs, [r[split]["cross_entropy"] for r in history], label=split)
        for ax in axes:
            if row["first_exact_epoch"] is not None:
                ax.axvline(row["first_exact_epoch"], color="green", linestyle=":", label="first exact interpolation")
            if row["first_near_epoch"] is not None and row["first_near_epoch"] != row["first_exact_epoch"]:
                ax.axvline(row["first_near_epoch"], color="orange", linestyle=":", label="first near interpolation")
            ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.legend(fontsize=7)
        axes[0].set_ylabel("Classification error")
        axes[1].set_ylabel("Cross-entropy")
        fig.suptitle(path.parent.name[:80], fontsize=10)
        fig.tight_layout()
        name = f"epochs_{index:03d}.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        epoch_figures.append(name)
    if not rows:
        raise ValueError("Histories are empty")
    with (out / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figures = []
    for key, members in groups.items():
        eligible = [r for r in members if r["budget_complete"] and r["val_error"] is not None]
        if not eligible:
            continue
        capacities = sorted({r["parameters"] for r in eligible})
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for metric, axis in [("val_error", axes[0]), ("val_cross_entropy", axes[1])]:
            means, spreads = [], []
            for n in capacities:
                values = [r[metric] for r in eligible if r["parameters"] == n]
                means.append(np.mean(values))
                spreads.append(np.std(values, ddof=1) if len(values) > 1 else 0.)
            axis.scatter([r["parameters"] for r in eligible], [r[metric] for r in eligible], alpha=.4)
            axis.errorbar(capacities, means, yerr=spreads, marker="o", label="mean ± seed standard deviation")
            axis.set_xlabel("Total parameters (including frozen trunk)")
            axis.set_ylabel(metric.replace("_", " "))
            axis.legend(fontsize=7)
        axes[0].scatter([r["parameters"] for r in eligible], [r["train_errors"] / r["train_count"] for r in eligible],
                        marker="x", color="green", label="training error")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Comparable recipe {key}; final-epoch results")
        fig.tight_layout()
        name = f"capacity_{key}.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        figures.append(name)
    text = ["# Measured experiment summary", "",
            "Generated from actual run logs. This is a results worksheet, not the final submission report. "
            "Curves do not automatically establish double descent. Incomplete runs remain in the table but are omitted "
            "from capacity curves. Error bars are seed standard deviations, not confidence intervals. "
            "Resumed runs in a new directory show only the epochs logged there.", "",
            "| Run | Parameters | Epoch | Training errors | Validation error | Exact interpolation epoch |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        val = "—" if r["val_error"] is None else f"{r['val_error']:.4f}"
        text.append(f"| {r['run']} | {r['parameters']} | {r['epoch']} | {r['train_errors']}/{r['train_count']} | {val} | {r['first_exact_epoch']} |")
    text.extend(["", "## Figures", ""])
    text.extend(f"![Measured curve]({name})\n" for name in figures + epoch_figures)
    text.extend(["", "## Interpretation to complete after experiments", "",
                 "Discuss convergence, seed variability, routing versus the matched baseline, node evidence and occlusion, "
                 "compression acceptance, and missing experiments. Add the real private repository URL and code commit "
                 "to the final PDF. No repository or scientific conclusion is inferred by this script."])
    (out / "SUMMARY.md").write_text("\n".join(text) + "\n")
    print(f"Wrote results.csv, SUMMARY.md and plots to {out}")


def build_parser():
    root = argparse.ArgumentParser(prog="python -m neural_tree", description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)

    def command(name, handler, description, *, checkpoint=False, output=True):
        parser = commands.add_parser(name, help=description, description=description)
        parser.set_defaults(handler=handler)
        if output:
            parser.add_argument("--out", required=True)
        if checkpoint:
            parser.add_argument("--checkpoint", required=True)
            parser.add_argument("--device", default="auto")
        return parser

    parser = command("data", data_command, "Prepare or verify a dataset manifest.", output=False)
    parser.add_argument("dataset", choices=["cifar10", "cats_dogs", "verify"])
    parser.add_argument("--root", default="data")
    parser.add_argument("--output", required=True, help="Manifest JSON path (input for verify)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default="Microsoft Research Cats and Dogs, PetImages archive")
    parser.add_argument("--archive", help="Optional downloaded source archive to hash")

    parser = command("train", train_command, "Train or resume a tree or baseline.")
    parser.add_argument("--config")
    parser.add_argument("--resume")
    for name, kind in [("epochs", int), ("batch-size", int), ("seed", int), ("workers", int), ("lr", float)]:
        parser.add_argument(f"--{name}", type=kind)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--width", type=float)
    parser.add_argument("--node-width", type=int)
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--pool-size", type=int, choices=[1, 4])
    parser.add_argument("--kind", choices=["tree", "baseline"])
    parser.add_argument("--subset", type=int)
    parser.add_argument("--validation-subset", type=int)
    parser.add_argument("--label-noise", type=float)
    parser.add_argument("--noise-seed", type=int)
    parser.add_argument("--cache")
    parser.add_argument("--all-train", action="store_true")
    parser.add_argument("--uniform", action="store_true")

    parser = command("evaluate", evaluate_command, "Evaluate a checkpoint on a selected split.", checkpoint=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--mode", choices=["soft", "hard", "uniform"], default="soft")

    parser = command("cache", cache_command, "Cache shared features for head sweeps.", checkpoint=True)
    parser.add_argument("--include-test", action="store_true")

    parser = command("sweeps", sweeps_command, "Plan or run a capacity and seed grid.", output=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--grid", help="Execute a previously saved grid.json")
    parser.add_argument("--out")
    parser.add_argument("--widths", nargs="+", type=float, default=[.25, .5, .75, 1., 1.5, 2.])
    parser.add_argument("--node-widths", nargs="+", type=int, help="For frozen-feature sweeps; keeps trunk width fixed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--execute", action="store_true")

    parser = command("benchmark", benchmark_command, "Measure training and inference cost.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--full-epoch", action="store_true", help="Also time a complete training and evaluation pass")

    parser = command("explain", explain_command, "Generate node explanations and occlusion checks.", checkpoint=True)
    parser.add_argument("--node", type=int, default=0)
    parser.add_argument("--target", choices=["router", "class", "reach"], default="router")
    parser.add_argument("--direction", choices=["left", "right"], default="right")
    parser.add_argument("--class-index", type=int, help="Defaults to each example's true class")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--selection", choices=["high-reach", "uncertain", "errors", "first"], default="high-reach")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--scan-limit", type=int, default=512)
    parser.add_argument("--patch", type=int, default=8)

    parser = command("compress", compress_command, "Prune and fine-tune a subtree.", checkpoint=True)
    parser.add_argument("--node", type=int, help="If omitted, rank all internal nodes")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser = command("shrink", shrink_command, "Shrink CNN channels and pooling, then fine-tune.", checkpoint=True)
    parser.add_argument("--width", type=float, required=True)
    parser.add_argument("--pool-size", type=int, choices=[1, 4], default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=.001)

    parser = command("report", report_command, "Plot and summarize measured run logs.")
    parser.add_argument("--runs", nargs="+", required=True, help="Run directories or sweep parent directories")
    return root


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
