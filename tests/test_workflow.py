"""Tiny synthetic fixtures exercise mechanics; these are not dataset experiments."""
import copy
import json
from pathlib import Path
from neural_tree.__main__ import main

import numpy as np
from PIL import Image
import pytest
import torch

from neural_tree.core import file_hash, write_json, load_data, run, load_checkpoint


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    root = tmp_path / "images"
    root.mkdir()
    rows = []
    rng = np.random.default_rng(1)
    for i in range(12):
        path = root / f"{i}.png"
        Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
        rows.append({"id": str(i), "path": path.name, "sha256": file_hash(path), "group": str(i),
                     "label": i % 2, "split": "train" if i < 8 else ("val" if i < 10 else "test")})
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"dataset": "cats_dogs", "root": str(root), "classes": ["cat", "dog"],
                          "image_size": 32, "records": rows})
    return {"data": {"manifest": str(manifest), "subset": None, "subset_seed": 42, "all_train": False},
            "model": {"kind": "tree", "classes": 2, "width": .125, "depth": 1, "pool_size": 1},
            "train": {"epochs": 2, "batch_size": 4, "lr": .001, "seed": 42, "workers": 0,
                      "auxiliary": .1, "balance": .01, "penalty_decay_epochs": 1, "device": "cpu", "uniform": False}}


def invoke(command, *args):
    main([command, *map(str, args)])


def test_resume_matches_uninterrupted_training_and_rejects_changed_inputs(config, tmp_path):
    full = run(config, tmp_path / "full")
    partial = copy.deepcopy(config)
    partial["train"]["epochs"] = 1
    first = run(partial, tmp_path / "resumed")
    resumed = run(config, tmp_path / "resumed", resume=first)
    a, _ = load_checkpoint(full)
    b, _ = load_checkpoint(resumed)
    for key, tensor in a.state_dict().items():
        torch.testing.assert_close(tensor, b.state_dict()[key], rtol=0, atol=0)
    changed = copy.deepcopy(config)
    changed["train"]["epochs"] = 3
    changed["train"]["lr"] = .1
    with pytest.raises(ValueError, match="Resume permits"):
        run(changed, tmp_path / "resumed", resume=resumed)
    with pytest.raises(ValueError, match="Run already exists"):
        run(config, tmp_path / "full")


def test_cli_workflow_and_frozen_cache(config, tmp_path):
    checkpoint = run(config, tmp_path / "tree")
    invoke("evaluate", "--checkpoint", checkpoint, "--out", tmp_path / "eval.json", "--device", "cpu")
    metrics = json.loads((tmp_path / "eval.json").read_text())["metrics"]
    assert metrics["count"] == 2
    cache_path = tmp_path / "features.pt"
    invoke("cache", "--checkpoint", checkpoint, "--out", cache_path, "--device", "cpu", "--include-test")
    frozen = copy.deepcopy(config)
    frozen["data"]["cache"] = str(cache_path)
    frozen["model"]["node_width"] = 5
    frozen_checkpoint = run(frozen, tmp_path / "frozen")
    frozen_model, frozen_state = load_checkpoint(frozen_checkpoint)
    source, _ = load_checkpoint(checkpoint)
    assert frozen_state["metadata"]["trainable_parameters"] < frozen_state["metadata"]["parameters"]
    for key, value in source.trunk.state_dict().items():
        torch.testing.assert_close(value, frozen_model.trunk.state_dict()[key])
    incompatible = copy.deepcopy(frozen)
    incompatible["data"]["subset"] = 4
    with pytest.raises(ValueError, match="Cache training selection"):
        load_data(incompatible)
    incompatible = copy.deepcopy(frozen)
    incompatible["model"]["width"] = .5
    with pytest.raises(ValueError, match="Cached trunk is fixed"):
        load_data(incompatible)
    invoke("benchmark", "--checkpoint", checkpoint, "--out", tmp_path / "benchmark.json",
           "--device", "cpu", "--batch-size", 2, "--steps", 1, "--full-epoch")
    assert json.loads((tmp_path / "benchmark.json").read_text())["inference"]["hard"]["linear_conv_macs_per_image"] > 0
    invoke("explain", "--checkpoint", checkpoint, "--out", tmp_path / "explain",
           "--device", "cpu", "--count", 1, "--patch", 16)
    assert (tmp_path / "explain" / "example_00.png").exists()
    invoke("compress", "--checkpoint", checkpoint, "--out", tmp_path / "compressed",
           "--device", "cpu", "--node", 0, "--epochs", 1)
    assert (tmp_path / "compressed" / "compression.json").exists()
    invoke("shrink", "--checkpoint", checkpoint, "--out", tmp_path / "shrunk",
           "--device", "cpu", "--width", .0625, "--epochs", 1)
    shrink = json.loads((tmp_path / "shrunk" / "compression.json").read_text())
    assert shrink["parameters"] < shrink["reference_parameters"]
    assert shrink["compressed"]["train"]["count"] == 8
    invoke("report", "--runs", tmp_path / "tree", tmp_path / "frozen", "--out", tmp_path / "report")
    assert (tmp_path / "report" / "results.csv").exists()
    cfg_path = tmp_path / "config.json"
    write_json(cfg_path, config)
    invoke("sweeps", "--config", cfg_path, "--out", tmp_path / "sweep",
           "--widths", .0625, "--seeds", 42, "--execute")
    status = next((tmp_path / "sweep").rglob("status.json"))
    assert json.loads(status.read_text())["status"] == "complete"


def test_all_train_removes_validation(config):
    config["data"]["all_train"] = True
    data, _ = load_data(config)
    assert len(data["train"]) == 10 and len(data["val"]) == 0


def test_unified_cli_verifies_data_trains_and_resumes(config, tmp_path):
    invoke("data", "verify", "--output", config["data"]["manifest"])
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    out = tmp_path / "cli"
    invoke("train", "--config", config_path, "--out", out,
           "--epochs", 1, "--node-width", 5, "--subset", 4)
    _, first = load_checkpoint(out / "last.pt")
    assert first["epoch"] == 0
    assert first["config"]["model"]["node_width"] == 5
    assert first["metadata"]["train_samples"] == 4
    invoke("train", "--resume", out / "last.pt", "--out", out, "--epochs", 2)
    _, resumed = load_checkpoint(out / "last.pt")
    assert resumed["epoch"] == 1
    assert resumed["config"]["model"] == first["config"]["model"]
    assert len((out / "history.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("command", ["evaluate", "cache", "explain", "compress"])
def test_checkpoint_commands_reject_changed_manifest(command, config, tmp_path, capsys):
    checkpoint = run(config, tmp_path / "tree")
    manifest = Path(config["data"]["manifest"])
    manifest.write_text(manifest.read_text() + "\n")
    with pytest.raises(SystemExit) as error:
        invoke(command, "--checkpoint", checkpoint, "--out", tmp_path / command,
               "--device", "cpu")
    assert error.value.code == 2
    assert "Manifest changed since checkpoint creation" in capsys.readouterr().err
