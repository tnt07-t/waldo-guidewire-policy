"""Development comparison on frozen features. Real data stays on Perlmutter.

No test predictions. Features live in RAM only; reports/checkpoints stay in a
new private scratch directory. Existing evaluation.py remains unchanged.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn

from evaluation import check_partitions, collect, metrics, episode_interval
from temporal_models import FeatureHead


def metadata_scale(metadata):
    """Population channel std over TRAINING steps; preserve zero and sign."""
    if metadata.ndim != 3 or metadata.shape[1:] != (6, 2) or not torch.isfinite(metadata).all():
        raise ValueError("Expected finite metadata [N,6,2]")
    scale = metadata.double().reshape(-1, 2).std(0, unbiased=False).float()
    return torch.where(scale == 0, torch.ones_like(scale), scale)


def tensors(features, device):
    # Existing collect stores [depth1..6, time1..6]. Restore aligned channels.
    meta = torch.tensor(features["metadata"], dtype=torch.float32)
    meta = torch.stack((meta[:, :6], meta[:, 6:]), -1)
    return (torch.tensor(features["history"], dtype=torch.float32, device=device),
            torch.tensor(features["goal"], dtype=torch.float32, device=device),
            meta.to(device), torch.tensor(features["y"], dtype=torch.long, device=device))


def collect_static_images(dataset, prepare, encoder, name, include_latest):
    """Decode/encode only goal and optionally latest; skip earlier image IO."""
    goals, latest = [], []
    shared = {k: [] for k in ('metadata','y','episode','case_id','row_index')}
    for row in dataset.indices:
        record = dataset.records[row]
        if len(record['paths']) != 7 or len(record['depth_relative_mm']) != 6 or len(record['history_offsets_s']) != 6:
            raise ValueError('Expected goal plus six history paths and aligned metadata')
        image = dataset._read_image(record['paths'][0], row)
        def encode(image):
            value = prepare.prepare_image(image)['tensor'].unsqueeze(0)
            with torch.inference_mode():
                encoded = (encoder._encode_images(value,1)[0] if name=='dinov2' else
                           encoder(value.to(next(encoder.parameters()).device))[0])
            if encoded.ndim != 1 or not torch.isfinite(encoded).all():
                raise ValueError('Expected finite image feature vector')
            return encoded.cpu().numpy()
        goals.append(encode(image))
        if include_latest:
            latest.append(encode(dataset._read_image(record['paths'][-1],row)))
        shared['metadata'].append(record['depth_relative_mm'] + record['history_offsets_s'])
        for key, source in (('y','label'),('episode','episode'),('case_id','case_id')):
            shared[key].append(record[source])
        shared['row_index'].append(row)
    shared = {k: np.asarray(v) for k,v in shared.items()}
    g = np.asarray(goals)
    history = np.zeros((len(g),6,g.shape[1]),dtype=g.dtype)
    if include_latest:
        history[:,-1] = np.asarray(latest)
    return shared, {name: dict(shared, goal=g, history=history)}


def probabilities(head, data):
    head.eval()
    with torch.no_grad():
        return head(*data[:3]).softmax(-1)[:, 1].cpu().numpy()


def assess(head, data):
    """Full-partition eval-mode metrics; no updates or dropout sampling."""
    head.eval()
    with torch.no_grad():
        logits = head(*data[:3])
        probability = logits.softmax(-1)[:, 1].cpu().numpy()
        result = metrics(data[3].cpu().numpy(), probability)
        result["loss"] = nn.functional.cross_entropy(logits, data[3]).item()
        predicted = probability >= .5
        result["prediction_counts"] = {"retract": int((~predicted).sum()),
                                       "advance": int(predicted.sum())}
        result["mean_advance_probability"] = float(probability.mean())
    return result


def fit(head, train, validation, *, epochs=100, patience=15, batch_size=32,
        lr=0.001, weight_decay=0.001, seed=0, epoch_callback=None,
        final_checkpoint_callback=None):
    """Train-only weighted loss; select checkpoint by validation BA, ties earliest."""
    if min(epochs, patience, batch_size) < 1 or lr <= 0 or weight_decay < 0:
        raise ValueError("Invalid training budget")
    counts = torch.bincount(train[3], minlength=2)
    if (counts == 0).any():
        raise ValueError("Both training labels required")
    weights = len(train[3]) / (2 * counts.float())
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    best_score, best_epoch, best_state = -1., 0, None
    history = []
    initial_metrics = {"train": assess(head, train), "validation": assess(head, validation)}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(train[3]), generator=generator)
        total = 0.
        for indices in order.split(batch_size):
            indices = indices.to(train[3].device)
            optimizer.zero_grad(set_to_none=True)
            logits = head(*(x[indices] for x in train[:3]))
            loss = loss_fn(logits, train[3][indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError("Missing/non-finite gradient")
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in head.parameters()):
                raise RuntimeError("Non-finite weights")
            total += loss.item() * len(indices)
        train_metrics, val_metrics = assess(head, train), assess(head, validation)
        score = val_metrics["balanced_accuracy"]
        history.append({"epoch": epoch, "batch_weighted_loss_mean": total / len(train[3]),
                        "validation_balanced_accuracy": score, "train": train_metrics,
                        "validation": val_metrics})
        if score > best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(head.state_dict())
        if epoch_callback is not None:
            epoch_callback(history)
        if epoch - best_epoch >= patience:
            break
    final_metrics = {"train": history[-1]["train"], "validation": history[-1]["validation"]}
    if final_checkpoint_callback is not None:
        final_checkpoint_callback(head.state_dict(), epoch)
    head.load_state_dict(best_state)
    head.eval()
    return {"best_epoch": best_epoch, "epochs_run": epoch, "history": history,
            "initial_metrics": initial_metrics,
            "final_epoch_metrics": final_metrics,
            "training_seconds": time.perf_counter() - start}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_learning_reports(directory, results):
    """Aggregate histories only: readable selected/final comparison and CSV."""
    lines = ['# Learning diagnostics', '',
             'Full-partition evaluation mode; train and validation loss are unweighted cross-entropy.', '',
             '| Model | Best epoch | Epochs | Selected train BA | Selected validation BA | Final train BA | Final validation BA |',
             '|---|---:|---:|---:|---:|---:|---:|']
    with (directory/'learning_curves.csv').open('w',newline='',encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['model','epoch','partition','loss','balanced_accuracy','retract_recall','advance_recall','predicted_retract','predicted_advance'])
        for name,r in results.items():
            if 'history' not in r:
                continue
            final = r['final_epoch_metrics']
            lines.append('| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(
                name,r['best_epoch'],r['epochs_run'],r['train']['balanced_accuracy'],r['validation']['balanced_accuracy'],
                final['train']['balanced_accuracy'],final['validation']['balanced_accuracy']))
            for row in r['history']:
                for partition in ('train','validation'):
                    m=row[partition]
                    writer.writerow([name,row['epoch'],partition,m['loss'],m['balanced_accuracy'],m['retract_recall'],
                                     m['advance_recall'],m['prediction_counts']['retract'],m['prediction_counts']['advance']])
    (directory/'learning_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def benchmark(case, prepare, encoder, head, scale, device, *, warmup=5, repeats=20):
    """Batch-one raw in-memory case -> logits; disk IO excluded, no feature reuse."""
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def once():
        sync()
        start = time.perf_counter()
        selected = ([] if head.kind == 'goal_metadata_mlp' else
                    [case["history"][-1]] if head.kind in ('latest_mlp','latest_goal_mlp') else case["history"])
        goal = prepare.prepare_image(case["goal"])["tensor"]
        frames = torch.stack([prepare.prepare_image(f)["tensor"] for f in selected]) if selected else None
        meta = torch.tensor(list(zip(case["depth_relative_mm"], case["history_offsets_s"])),
                            dtype=torch.float32, device=device).unsqueeze(0) / scale
        sync()
        boundary = time.perf_counter()
        with torch.no_grad():
            if hasattr(encoder, "encode"):
                g = encoder._encode_images(goal.unsqueeze(0), 1)
                h = encoder._encode_images(frames, 1).unsqueeze(0) if frames is not None else g.new_zeros(1,6,g.shape[-1])
            else:
                g = encoder(goal.unsqueeze(0).to(device))
                h = (torch.cat([encoder(f.unsqueeze(0).to(device)) for f in frames]).unsqueeze(0)
                     if frames is not None else g.new_zeros(1,6,g.shape[-1]))
            if head.kind in ("latest_mlp",'latest_goal_mlp'):
                # Only the final entry is consumed by this head.
                h = h.expand(-1, 6, -1)
            head(h, g, meta)
        sync()
        end = time.perf_counter()
        return ((boundary - start) * 1000, (end - boundary) * 1000, (end - start) * 1000)

    head.eval()
    for _ in range(warmup):
        once()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    values = np.asarray([once() for _ in range(repeats)])
    return {"scope": "same in-memory training case; batch one; preparation + transfer + encoder + head; excludes disk IO; no feature reuse",
            "images": 1 if head.kind == 'goal_metadata_mlp' else 2 if head.kind in ('latest_mlp','latest_goal_mlp') else 7,
            "warmup": warmup, "repeats": repeats,
            "milliseconds": {k: {"median": float(np.median(values[:, i])),
                                  "p95": float(np.percentile(values[:, i], 95))}
                             for i, k in enumerate(("preprocessing", "model", "total"))},
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}


def experiment_variants(args):
    """Predeclared sweep: baseline once each, six attention configurations."""
    if args.attention_sweep:
        variants = [dict(name=k, kind=k, width=64, classifier_width=64,
                         position_embeddings=False) for k in ("latest_mlp", "mean_mlp")]
        for width in (32, 64, 128):
            for position in (True, False):
                variants.append(dict(name="attention-w{}-pos{}".format(width, int(position)),
                                     kind="attention", width=width, classifier_width=64,
                                     position_embeddings=position))
        return variants
    return [dict(name=k, kind=k, width=args.width,
                 classifier_width=args.classifier_width or args.width,
                 position_embeddings=not args.no_position_embeddings) for k in args.models]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=os.environ.get("DATA_DIR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument("--encoder", choices=("resnet18", "dinov2"), default="resnet18")
    parser.add_argument("--models", nargs="+", choices=("latest_goal_mlp", "goal_metadata_mlp", "latest_mlp", "mean_mlp", "attention", "gru", "lstm"),
                        default=["latest_mlp", "attention"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--attention-sweep", action="store_true", help="Static + mean + attention widths 32/64/128 with/without positions; fixed 64-wide classifier")
    parser.add_argument("--no-position-embeddings", action="store_true")
    parser.add_argument("--classifier-width", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument('--classifier-activation',choices=('relu','leaky_relu'),default='relu',
                        help='Final MLP activation only; leaky_relu uses fixed negative slope 0.01')
    parser.add_argument("--benchmark", action="store_true", help="Also time raw-case preparation + encoder + head")
    args = parser.parse_args(argv)
    if not os.environ.get("PSCRATCH") or args.data_dir is None:
        parser.error("Run on Perlmutter with PSCRATCH and DATA_DIR set")
    scratch = Path(os.environ["PSCRATCH"]).resolve()
    args.data_dir, args.output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    if any(scratch not in p.parents for p in (args.data_dir, args.output_dir)):
        parser.error("Data/output must be beneath personal PSCRATCH")
    if args.output_dir == args.data_dir or args.data_dir in args.output_dir.parents:
        parser.error("Keep output outside dataset")
    if args.output_dir.exists():
        parser.error("Output exists; use a new run directory")
    if (min(args.epochs, args.patience, args.batch_size) < 1 or not np.isfinite(args.lr)
            or args.lr <= 0 or not np.isfinite(args.weight_decay) or args.weight_decay < 0
            or (args.classifier_width is not None and args.classifier_width <= 0)
            or not 0 <= args.dropout < 1 or args.width <= 0 or args.width % 4):
        parser.error("Invalid model/training settings")
    if len(set(args.models)) != len(args.models) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Models/seeds must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use an allocated compute node")
    from dataset import GuidewireDataset
    from preprocessing import DINOv2Preprocessor
    datasets = {p: GuidewireDataset(args.data_dir, split_position=args.split_position, partition=p)
                for p in ("train", "validation", "test")}
    check_partitions(datasets)
    if args.encoder == "resnet18":
        from cnn import FrameEncoder
        encoder = FrameEncoder().to(args.device).eval()
        backbone, feature_dim = encoder.backbone, 512
    else:
        from vit import DINOv2Encoder
        encoder = DINOv2Encoder(device=args.device)
        backbone, feature_dim = encoder.backbone, 384
    backbone.requires_grad_(False)
    snapshot = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    args.output_dir.mkdir(parents=True, mode=0o700)
    write_json(args.output_dir / "status.json", {"status": "extracting"})
    start = time.perf_counter()
    static_controls = not args.attention_sweep and set(args.models) <= {'latest_goal_mlp','goal_metadata_mlp','latest_mlp'}
    include_latest = set(args.models) != {'goal_metadata_mlp'}
    def extract(dataset):
        return (collect_static_images(dataset,DINOv2Preprocessor(),encoder,args.encoder,include_latest) if static_controls else
                collect(dataset,DINOv2Preprocessor(),{args.encoder:encoder}))
    _, train_features = extract(datasets['train'])
    _, val_features = extract(datasets['validation'])
    extraction_seconds = time.perf_counter() - start
    train = tensors(train_features[args.encoder], args.device)
    validation = tensors(val_features[args.encoder], args.device)
    scale = metadata_scale(train[2])
    train = (train[0], train[1], train[2] / scale, train[3])
    validation = (validation[0], validation[1], validation[2] / scale, validation[3])
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(feature_dim=feature_dim, extraction_visual_roles=(('goal + latest' if include_latest else 'goal only')
                                                                if static_controls else 'goal + six history images'),
                  metadata_scaling="train-only population std; no centering",
                  metadata_scale=scale.cpu().tolist(), preparation="RGB full-view 518 bicubic mean padding",
                  encoder_weights="ResNet18_Weights.DEFAULT" if args.encoder == "resnet18" else "vit_small_patch14_dinov2.lvd142m")
    source_dir = Path(__file__).resolve().parent
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_dir, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=source_dir, capture_output=True, text=True)
    import importlib.metadata
    versions = {}
    for package in ("numpy", "scikit-learn", "Pillow", "torchvision", "timm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {"format_version": 2, "purpose": "development validation; no test predictions", "config": config,
              "git_commit": git.stdout.strip(), "git_dirty": bool(dirty.stdout.strip()),
              "python": platform.python_version(), "torch": torch.__version__,
              "versions": versions,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "partition_counts": {p: len(d) for p, d in datasets.items()},
              "source_hashes": {p: hashlib.sha256((source_dir / p).read_bytes()).hexdigest()
                                for p in ("temporal_models.py", "temporal_evaluation.py", "evaluation.py", "cnn.py", "vit.py", "preprocessing.py", "dataset.py")},
              "extraction_seconds": extraction_seconds, "results": {}}
    predictions = {k: val_features[args.encoder][k].tolist() for k in ("case_id", "episode", "row_index", "y")}
    predictions["advance_probabilities"] = {}
    majority = int(train[3].float().mean() > .5)
    report["results"]["majority"] = {"validation": metrics(validation[3].cpu().numpy(),
                                                                   np.full(len(validation[3]), majority))}
    variants = experiment_variants(args)
    report["variants"] = variants
    for variant in variants:
        kind = variant["kind"]
        for seed in args.seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.device == "cuda":
                torch.cuda.manual_seed_all(seed)
            head = FeatureHead(kind, feature_dim, variant["width"], args.dropout,
                               position_embeddings=variant["position_embeddings"],
                               classifier_width=variant["classifier_width"],
                               classifier_activation=args.classifier_activation).to(args.device)
            checkpoint_config = dict(config, width=variant["width"],
                                     classifier_width=variant["classifier_width"],
                                     position_embeddings=variant["position_embeddings"])
            result = fit(head, train, validation, epochs=args.epochs, patience=args.patience,
                         batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, seed=seed,
                         epoch_callback=lambda history: write_json(
                             args.output_dir / (variant["name"] + "-seed" + str(seed) + "-history.json"), history),
                         final_checkpoint_callback=lambda state, epoch: torch.save(
                             {"head_state": state, "config": checkpoint_config, "kind": kind,
                              "seed": seed, "checkpoint_role": "final", "checkpoint_epoch": epoch},
                             args.output_dir / (variant["name"] + "-seed" + str(seed) + "-final.pt")))
            train_p, val_p = probabilities(head, train), probabilities(head, validation)
            key = variant["name"] + ":seed=" + str(seed)
            result.update(variant=variant, train=assess(head, train),
                          validation=assess(head, validation),
                          validation_balanced_accuracy_95pct=episode_interval(
                              validation[3].cpu().numpy(), val_p, val_features[args.encoder]["episode"]),
                          trainable_parameters=sum(p.numel() for p in head.parameters()))
            result['feature_roles'] = ('latest image + goal; no metadata' if kind == 'latest_goal_mlp' else
                                       'goal + six chronological depth/time pairs' if kind == 'goal_metadata_mlp' else
                                       'latest + goal + six depth/time pairs' if kind == 'latest_mlp' else
                                       'history + goal + six depth/time pairs')
            if args.benchmark:
                if kind in ('latest_goal_mlp','goal_metadata_mlp','latest_mlp'):
                    dataset = datasets['train']
                    row = dataset.indices[0]
                    record = dataset.records[row]
                    case = dict(goal=dataset._read_image(record['paths'][0],row),
                                history=[] if kind=='goal_metadata_mlp' else [dataset._read_image(record['paths'][-1],row)],
                                depth_relative_mm=record['depth_relative_mm'],history_offsets_s=record['history_offsets_s'])
                else:
                    case = datasets['train'][0]
                result["latency"] = benchmark(case, DINOv2Preprocessor(),
                                               encoder, head, scale, args.device)
            report["results"][key] = result
            predictions["advance_probabilities"][key] = val_p.tolist()
            torch.save({"head_state": head.state_dict(), "config": checkpoint_config, "kind": kind,
                        "seed": seed, "best_epoch": result["best_epoch"],
                        "checkpoint_role": "selected", "checkpoint_epoch": result["best_epoch"]},
                       args.output_dir / (variant["name"] + "-seed" + str(seed) + ".pt"))
            write_json(args.output_dir / "metrics.json", report)
            write_json(args.output_dir / "validation_predictions.private.json", predictions)
            write_learning_reports(args.output_dir,report['results'])
            print(key, "validation BA", result["validation"]["balanced_accuracy"], flush=True)
    if backbone.training or any(p.requires_grad or p.grad is not None for p in backbone.parameters()) or any(
            not torch.equal(v.detach().cpu(), snapshot[k]) for k, v in backbone.state_dict().items()):
        raise RuntimeError("Frozen encoder changed")
    report["encoder_unchanged"] = True
    write_json(args.output_dir / "metrics.json", report)
    lines = ["# Development comparison", "", "Validation-selected; not final test results.", "",
             "| Model/seed | Train BA | Validation BA | Retract recall | Advance recall | Parameters |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, r in report["results"].items():
        v = r["validation"]
        train_ba = "{:.3f}".format(r["train"]["balanced_accuracy"]) if "train" in r else "—"
        lines.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            key, train_ba, v["balanced_accuracy"], v["retract_recall"], v["advance_recall"], r.get("trainable_parameters", 0)))
    (args.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(args.output_dir / "status.json", {"status": "completed"})


if __name__ == "__main__":
    main()
