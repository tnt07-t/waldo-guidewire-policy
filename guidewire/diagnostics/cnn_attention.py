"""Eight-training-case CNN–attention debugging only. Real data stays on Perlmutter.

The parent process enforces a ten-minute wall limit, including worker imports,
checkpoint loading, extraction and head training. No feature cache/checkpoint
is saved. A new private output directory holds configuration and results.
"""

# Adapted from the team's bounded ViT-GRU runner at b9d1bdf.
# Separate CNN-attention entry point; Tran's implementation stays unchanged.
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]


class CNNAdapter:
    """Keep shared CNN unchanged; extract separate goal/history one image at a time."""

    def __init__(self, encoder):
        self.encoder = encoder
        self.backbone = encoder.backbone

    def encode(self, goal, history, image_chunk_size=1):
        import torch
        device = next(self.backbone.parameters()).device
        self.encoder.eval()
        with torch.no_grad():
            g = self.encoder(goal.unsqueeze(0).to(device))[0]
            h = torch.cat([self.encoder(frame.unsqueeze(0).to(device)) for frame in history])
        return {"goal_features": g, "history_features": h}


def training_inputs(dataset):
    """Read only selected training metadata; do not decode images here."""
    import torch

    if dataset.partition != "train" or dataset.split_position != 0:
        raise ValueError("This debugging check uses split position 0 training only")
    if len(set(dataset.indices)) != len(dataset.indices):
        raise ValueError("Duplicate training rows")
    rows, labels = [], []
    for row in dataset.indices:
        record = dataset.records[row]
        depth, offsets = record["depth_relative_mm"], record["history_offsets_s"]
        if len(depth) != 6 or len(offsets) != 6:
            raise ValueError("Expected six aligned depths and times")
        if any(type(v) not in (int, float) or not math.isfinite(v)
               for v in list(depth) + list(offsets)):
            raise ValueError("Metadata must contain finite numbers")
        if depth[-1] != 0 or offsets[-1] != 0:
            raise ValueError("Relative depth/time must end at zero")
        if any(a > b for a, b in zip(offsets, offsets[1:])):
            raise ValueError("History times must be chronological")
        if type(record["label"]) is not int or record["label"] not in (0, 1):
            raise ValueError("Expected integer action labels 0 or 1")
        rows.append(list(zip(depth, offsets)))
        labels.append(record["label"])
    if not rows:
        raise ValueError("Empty training partition")
    metadata = torch.tensor(rows, dtype=torch.float64)
    # One population standard deviation per channel over all training steps.
    scales = metadata.reshape(-1, 2).std(dim=0, correction=0)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    normalized = (metadata / scales).float()
    if not torch.isfinite(scales).all() or not torch.isfinite(normalized).all():
        raise ValueError("Non-finite metadata scales or scaled values")
    return normalized, torch.tensor(labels, dtype=torch.long), scales


def select_eight(labels, seed=0):
    """Return dataset-local indices: four examples per label, sampled once."""
    rng = random.Random(seed)
    indices = []
    for label in (0, 1):
        candidates = [i for i, value in enumerate(labels.tolist()) if value == label]
        if len(candidates) < 4:
            raise ValueError("Need at least four training cases per label")
        indices.extend(rng.sample(candidates, 4))
    rng.shuffle(indices)
    return indices


def freeze_and_snapshot(backbone):
    backbone.requires_grad_(False)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.grad = None
    return {name: value.detach().cpu().clone()
            for name, value in backbone.state_dict().items()}


def verify_frozen(backbone, snapshot):
    import torch

    if backbone.training or any(p.requires_grad or p.grad is not None
                                for p in backbone.parameters()):
        raise RuntimeError("Encoder is not frozen in evaluation mode")
    current = backbone.state_dict()
    if current.keys() != snapshot.keys() or any(
            not torch.equal(current[name].detach().cpu(), value)
            for name, value in snapshot.items()):
        raise RuntimeError("Encoder parameters or buffers changed")


def extract_selected(dataset, indices, prepare, encoder):
    """Encode only eight cases, once, one image per encoder call."""
    import torch

    if len(indices) != 8 or len(set(indices)) != 8:
        raise ValueError("Expected eight distinct selected cases")
    goals, histories = [], []
    for index in indices:
        prepared = prepare.prepare_case(dataset[index])
        features = encoder.encode(prepared["goal"], prepared["history"], image_chunk_size=1)
        # encode uses inference_mode. Clone OUTSIDE that mode so backward may
        # save these ordinary, detached tensors when learning the head weights.
        goals.append(features["goal_features"].detach().clone())
        histories.append(features["history_features"].detach().clone())
    return torch.stack(histories), torch.stack(goals)


def fit_head(head, history, goal, metadata, labels, *, max_updates=300,
             head_seconds=120, clock=time.monotonic):
    """Full-batch Adam; bounded training fit, not held-out evaluation."""
    import torch
    from torch.nn import functional as F

    if type(max_updates) is not int or not 1 <= max_updates <= 300:
        raise ValueError("max_updates must be between 1 and 300")
    if not math.isfinite(head_seconds) or not 0 < head_seconds <= 120:
        raise ValueError("head_seconds must be positive and at most 120")
    if (len(history) != 8 or labels.shape != (8,) or labels.dtype != torch.long
            or labels.tolist().count(0) != 4 or labels.tolist().count(1) != 4):
        raise ValueError("Expected eight cases with four integer labels per class")
    start = clock()
    head.train()
    parameters = list(head.parameters())
    before = [p.detach().clone() for p in parameters]
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    def scores():
        was_training = head.training
        head.eval()
        with torch.no_grad():
            logits = head(history, goal, metadata)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            result = {"loss": loss.item(),
                      "accuracy": (logits.argmax(-1) == labels).float().mean().item()}
        head.train(was_training)
        return result

    initial = scores()
    updates, stop_reason = 0, "max_updates"
    for _ in range(max_updates):
        if clock() - start >= head_seconds:
            stop_reason = "head_time_limit"
            break
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(history, goal, metadata), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
            raise RuntimeError("Missing or non-finite head gradients")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in parameters):
            raise RuntimeError("Non-finite head weights after optimizer step")
        updates += 1
    final = scores()
    changed = any(not torch.equal(old, p.detach()) for old, p in zip(before, parameters))
    if updates and not changed:
        raise RuntimeError("Optimizer steps did not change any head weights")
    return {"initial": initial, "final": final, "updates": updates,
            "stop_reason": stop_reason, "head_seconds": clock() - start,
            "finite_gradients": True if updates else None,
            "head_weights_changed": changed,
            "memorized_eight": final["accuracy"] == 1.0 and final["loss"] < initial["loss"]}


def run_worker(args):
    import hashlib
    import importlib.metadata
    import torch
    from guidewire.data.dataset import GuidewireDataset
    from guidewire.data.preprocessing import DINOv2Preprocessor
    from guidewire.encoders.cnn import FrameEncoder
    from guidewire.temporal.models import FeatureHead

    start = time.monotonic()
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; use an allocated GPU compute node")
        torch.cuda.manual_seed_all(0)
        torch.cuda.reset_peak_memory_stats(device)
    dataset = GuidewireDataset(args.data_dir, split_position=0, partition="train")
    metadata, labels, scales = training_inputs(dataset)
    indices = select_eight(labels)
    encoder = CNNAdapter(FrameEncoder().to(device))
    snapshot = freeze_and_snapshot(encoder.backbone)
    extraction_start = time.monotonic()
    history, goal = extract_selected(dataset, indices, DINOv2Preprocessor(), encoder)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    extraction_seconds = time.monotonic() - extraction_start
    verify_frozen(encoder.backbone, snapshot)
    # New head and optimizer on every invocation; never resume the tiny fit.
    head = FeatureHead("attention", feature_dim=512, width=64, dropout=0.1,
                       position_embeddings=True, classifier_width=64).to(device)
    result = fit_head(head, history, goal, metadata[indices].to(device), labels[indices].to(device),
                      max_updates=args.max_updates, head_seconds=args.head_seconds)
    verify_frozen(encoder.backbone, snapshot)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                           capture_output=True, text=True)
    report = {
        "status": "completed" if result["updates"] else "no_updates",
        "purpose": "Training memorization/debugging only; no generalization or temporal-use claim",
        "config": {"checkpoint": "ResNet18_Weights.DEFAULT", "architecture": "cnn_temporal_attention",
                   "feature_dim": 512, "attention_heads": 4, "dropout": 0.1,
                   "split_position": 0, "partition": "train",
                   "seed": 0, "cases": 8, "per_label": 4, "projection_width": 64, "classifier_width": 64, "position_embeddings": True,
                   "layers": 1, "image_chunk_size": 1, "optimizer": "Adam", "lr": 1e-3,
                   "max_updates": args.max_updates, "head_seconds": args.head_seconds,
                   "overall_seconds": 600, "device": str(device)},
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "training_partition_cases": len(dataset), "fold_field": dataset.fold,
        "metadata_scale_depth_time": scales.tolist(),
        "metadata_scaling": "Training-only population std over all steps, no centering; constant scale=1",
        # Reproducible selection is private, like all other derived artifacts.
        "selected_training_row_indices": [dataset.indices[i] for i in indices],
        "encoder_unchanged": True, "extraction_seconds": extraction_seconds,
        "worker_seconds": time.monotonic() - start,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
        "code_sha256": {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                        for name in ("guidewire/data/dataset.py", "guidewire/data/preprocessing.py", "guidewire/encoders/cnn.py",
                         "guidewire/temporal/models.py", "guidewire/diagnostics/cnn_attention.py")},
        "python": sys.version.split()[0],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "Pillow")},
        **result,
    }
    write_report(args.output_dir, report)
    print("Eight-case check finished: updates={}, loss {:.6f} -> {:.6f}, training accuracy={:.3f}".format(
        result["updates"], result["initial"]["loss"], result["final"]["loss"], result["final"]["accuracy"]), flush=True)
    return 0 if result["updates"] else 2


def write_report(directory, report):
    path = Path(directory) / "debug_summary.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def supervise(command, *, seconds=600, cwd=REPO_ROOT):
    """Bound even a worker stuck in native code/download; stop its process group."""
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    def stop():
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    try:
        return process.wait(timeout=seconds), False
    except subprocess.TimeoutExpired:
        stop()
        return 124, True
    except KeyboardInterrupt:
        stop()
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New private directory under PSCRATCH")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--max-updates", type=int, default=300, help="May reduce, never exceed 300")
    parser.add_argument("--head-seconds", type=float, default=120, help="May reduce, never exceed 120")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.max_updates <= 300 or not math.isfinite(args.head_seconds) or not 0 < args.head_seconds <= 120:
        parser.error("Use 1–300 updates and a positive head time limit of at most 120 seconds")
    if not os.environ.get("PSCRATCH") or args.data_dir is None:
        parser.error("Run on Perlmutter with PSCRATCH and DATA_DIR set")
    scratch = Path(os.environ["PSCRATCH"]).resolve()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if any(scratch not in p.parents for p in (args.data_dir, args.output_dir)):
        parser.error("Data and output must be beneath personal PSCRATCH")
    if args.output_dir == args.data_dir or args.data_dir in args.output_dir.parents:
        parser.error("Keep outputs outside the supplied dataset directory")
    if not args.data_dir.is_dir():
        parser.error("Data directory does not exist")
    if args.worker:
        return run_worker(args)
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new run directory")
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    command = [sys.executable, "-m", "guidewire.diagnostics.cnn_attention", "--worker",
               "--data-dir", str(args.data_dir), "--output-dir", str(args.output_dir),
               "--device", args.device, "--max-updates", str(args.max_updates),
               "--head-seconds", str(args.head_seconds)]
    start = time.monotonic()
    code, timed_out = supervise(command)
    path = args.output_dir / "debug_summary.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    if timed_out:
        report.update(status="overall_time_limit", stop_reason="overall_time_limit")
    elif code:
        report.setdefault("status", "worker_failed")
    report.update(process_seconds=time.monotonic() - start, process_exit_code=code,
                  overall_limit_reached=timed_out)
    write_report(args.output_dir, report)
    print("Result status: {}. Private report: debug_summary.json".format(report["status"]), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
