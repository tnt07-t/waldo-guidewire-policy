"""Train the CNN-GRU head on one split, printing metrics to the terminal.

The frozen ResNet encodes history and goal images once. The small head
trains with class-weighted cross-entropy, AdamW (weight decay 1.0 by
default), fixed MLP dropout 0.3, and optional Gaussian feature noise.
--config moderate, strict or extra-strict selects a 50-epoch budget and
feature-noise std 1.0, 2.0 or 3.0. All keep weight decay 1.0. The baseline
default remains 200 epochs with noise off. Explicit flags override presets.

Runs train for the requested epoch count and retain the final weights in
memory. Metrics are printed each epoch and averaged over the final window.
No run folders, logs, CSVs, plots, summaries, or checkpoints are saved.

--shuffle-train-labels provides a seeded null control; validation stays real.
--feature-cache optionally reuses frozen features outside the repository.
This explicit cache is the only disk output and must stay on Perlmutter.
The test partition is never loaded. Real images and features stay on Perlmutter.
"""

import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from dataset import GuidewireDataset
from model_cnngru import CNNGRUModel
from preprocessing import DINOv2Preprocessor

CLASS_NAMES = ("RETRACT", "ADVANCE")  # label 0, label 1
REPO_ROOT = Path(__file__).resolve().parent
REGULARIZATION_CONFIGS = {
    "baseline": {"weight_decay": 1.0, "feature_noise": 0.0, "epochs": 200},
    "moderate": {"weight_decay": 1.0, "feature_noise": 1.0, "epochs": 50},
    "strict": {"weight_decay": 1.0, "feature_noise": 2.0, "epochs": 50},
    "extra-strict": {"weight_decay": 1.0, "feature_noise": 3.0, "epochs": 50},
}


class PreparedCases(Dataset):
    """Raw cases -> model tensors only (previews dropped) for DataLoader workers."""

    def __init__(self, cases, prepare):
        self.cases = cases
        self.prepare = prepare

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, index):
        prepared = self.prepare.prepare_case(self.cases[index])
        return {
            "history": prepared["history"],                     # [6, 3, H, W]
            "goal": prepared["goal"],                           # [3, H, W]
            "depth": torch.tensor(prepared["depth_relative_mm"], dtype=torch.float32),
            "offsets": torch.tensor(prepared["history_offsets_s"], dtype=torch.float32),
            "label": torch.tensor(prepared["label"], dtype=torch.long),
        }


@torch.no_grad()
def extract_features(cnn, cases, device, batch_size, num_workers):
    """Encode all images once. Returns CPU tensors keyed like model inputs."""
    # A private generator keeps extraction from drawing global random numbers
    # (DataLoader takes a worker base seed), so cached and fresh runs match.
    loader = DataLoader(cases, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, generator=torch.Generator().manual_seed(0))
    parts = {key: [] for key in ("history", "goal", "depth", "offsets", "label")}
    cnn.eval()
    for batch in loader:
        parts["history"].append(cnn(batch["history"].to(device)).cpu())  # [b, 6, 512]
        parts["goal"].append(cnn(batch["goal"].to(device)).cpu())        # [b, 512]
        for key in ("depth", "offsets", "label"):
            parts[key].append(batch[key])
    features = {key: torch.cat(value) for key, value in parts.items()}
    for key in ("history", "goal"):
        if not torch.isfinite(features[key]).all():
            raise ValueError("Frozen encoder returned non-finite {} features".format(key))
    return features


FEATURE_KEYS = ("history", "goal", "depth", "offsets", "label")
# Bump when anything that changes cached features changes (encoder, prep).
CACHE_VERSION = 1


def cache_metadata(data_dir, split_position, partition, size):
    """What a cached feature file must match to be reused."""
    return {"version": CACHE_VERSION, "data_dir": str(Path(data_dir).expanduser().resolve()),
            "split_position": split_position,
            "partition": partition, "preprocessor": "dinov2", "size": size,
            "encoder": "resnet18-torchvision-default-frozen-avgpool"}


def cache_path(cache_dir, metadata):
    return Path(cache_dir) / "split{}-{}-{}{}-v{}.pt".format(
        metadata["split_position"], metadata["partition"], metadata["preprocessor"],
        metadata["size"], metadata["version"])


def check_cache_dir(cache_dir):
    """Refuse a cache inside the repository: it holds real-image features."""
    resolved = Path(cache_dir).expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError("--feature-cache must be outside the repository "
                         "(features from real images must not be committed)")
    return resolved


def save_cached_features(path, features, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unique per process, so parallel workers filling one cache never share a file.
    tmp = path.with_suffix(".{}.tmp".format(os.getpid()))
    torch.save({"metadata": metadata,
                "features": {key: features[key].cpu() for key in FEATURE_KEYS}}, tmp)
    tmp.replace(path)  # atomic, so a killed run never leaves a partial cache


def load_cached_features(path, metadata):
    """Cached CPU features, or None if absent; a mismatched file is an error."""
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu")
    if payload.get("metadata") != metadata:
        raise ValueError("Feature cache {} was made with {}, expected {}".format(
            path, payload.get("metadata"), metadata))
    return payload["features"]


def class_weights(labels):
    """Inverse-frequency weights n / (2 * count_c); equal classes give 1, 1."""
    counts = torch.bincount(labels, minlength=2).double()
    if (counts == 0).any():
        raise ValueError("Training partition must contain both classes")
    return (len(labels) / (2 * counts)).float()


def shuffled_labels(labels, seed):
    """Shuffled-label null: permute labels across cases with a seeded generator.

    Class counts (and so class weights) are unchanged; only the pairing of
    labels with features is broken.
    """
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    return labels.cpu()[order].to(labels.device)


WINDOW_METRICS = ("train_bal_acc", "val_bal_acc", "val_auc", "val_loss")


def window_summary(rows, window):
    """Mean and population std of WINDOW_METRICS over the last ``window`` epochs.

    ``rows`` are per-epoch dicts as in metrics.csv (values may be strings).
    A shorter run uses all its epochs; NaN values (e.g. AUC with one class)
    are skipped, and a metric with no finite value gets NaN mean and std.
    """
    if not rows or window <= 0:
        raise ValueError("Need at least one epoch row and a positive window")
    tail = rows[-window:]
    summary = {"window": len(tail), "first_epoch": int(tail[0]["epoch"]),
               "last_epoch": int(tail[-1]["epoch"])}
    for key in WINDOW_METRICS:
        values = [float(row[key]) for row in tail if not math.isnan(float(row[key]))]
        if values:
            mean = sum(values) / len(values)
            std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        else:
            mean = std = math.nan
        summary[key] = {"mean": mean, "std": std}
    return summary


def roc_auc(scores, labels):
    """Rank-based ROC AUC (Mann-Whitney U); ties count half. NaN if one class."""
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return math.nan
    greater = (positive[:, None] > negative[None, :]).double().mean()
    ties = (positive[:, None] == negative[None, :]).double().mean()
    return (greater + 0.5 * ties).item()


def binary_metrics(logits, labels, weight):
    """Loss (class-weighted, same as training), accuracy, balanced accuracy,
    per-class recall, ROC AUC and confusion matrix (rows true, cols pred)."""
    logits = logits.detach().float().cpu()
    labels = labels.cpu()
    loss = F.cross_entropy(logits, labels, weight=weight.cpu()).item()
    preds = logits.argmax(dim=-1)
    confusion = torch.zeros(2, 2, dtype=torch.long)
    for true, pred in zip(labels.tolist(), preds.tolist()):
        confusion[true, pred] += 1
    support = confusion.sum(dim=1)
    recall = [
        (confusion[c, c] / support[c]).item() if support[c] else math.nan
        for c in range(2)
    ]
    valid = [r for r in recall if not math.isnan(r)]
    return {
        "loss": loss,
        "acc": (preds == labels).double().mean().item(),
        "bal_acc": sum(valid) / len(valid) if valid else math.nan,
        "recall_retract": recall[0],
        "recall_advance": recall[1],
        "auc": roc_auc(torch.softmax(logits, dim=-1)[:, 1], labels),
        "confusion": confusion.tolist(),
    }


@torch.no_grad()
def evaluate(model, data, weight, batch_size=256):
    model.eval()
    logits = torch.cat([
        model.forward_features(data["history"][i:i + batch_size],
                               data["depth"][i:i + batch_size],
                               data["offsets"][i:i + batch_size],
                               data["goal"][i:i + batch_size], return_logits=True)
        for i in range(0, len(data["label"]), batch_size)
    ])
    return binary_metrics(logits, data["label"], weight)


def format_epoch(epoch, epochs, train, val, lr, seconds):
    return (
        "epoch {:>4}/{} | train loss {:.4f} acc {:.3f} bal {:.3f} "
        "| val loss {:.4f} acc {:.3f} bal {:.3f} auc {:.3f} rec R/A {:.2f}/{:.2f} "
        "| lr {:.1e} {:.1f}s".format(
            epoch, epochs, train["loss"], train["acc"], train["bal_acc"],
            val["loss"], val["acc"], val["bal_acc"], val["auc"],
            val["recall_retract"], val["recall_advance"],
            lr, seconds,
        )
    )


def train_model(model, train, val, *, epochs, batch_size, lr, weight_decay,
                grad_clip, seed, log=print):
    """Train all epochs; return in-memory rows and leave final weights in model."""
    weight = class_weights(train["label"].cpu()).to(train["label"].device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(train["label"])
    rows = []
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()  # Frozen ResNet stays in eval mode.
        order = torch.randperm(n, generator=generator).to(train["label"].device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            logits = model.forward_features(train["history"][idx], train["depth"][idx],
                                            train["offsets"][idx], train["goal"][idx],
                                            return_logits=True)
            loss = F.cross_entropy(logits, train["label"][idx], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

        train_m = evaluate(model, train, weight)
        val_m = evaluate(model, val, weight)
        seconds = time.perf_counter() - start
        rows.append({"epoch": epoch, "seconds": seconds,
                     **{"train_" + k: v for k, v in train_m.items() if k != "confusion"},
                     **{"val_" + k: v for k, v in val_m.items() if k != "confusion"}})
        log(format_epoch(epoch, epochs, train_m, val_m,
                         optimizer.param_groups[0]["lr"], seconds))
    return {"rows": rows}


def describe_partition(name, data):
    counts = torch.bincount(data["label"].cpu(), minlength=2).tolist()
    return "{}: {} cases ({} {}, {} {})".format(
        name, len(data["label"]), counts[0], CLASS_NAMES[0], counts[1], CLASS_NAMES[1])


def parse_args(argv=None):
    """Parse options for one terminal-only training run."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True,
                        help="Zero-based position in splits.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", choices=REGULARIZATION_CONFIGS, default="baseline",
                        help="Regularization preset; explicit flags override its defaults")
    parser.add_argument("--size", type=int, default=518,
                        help="Input size; shared preprocessor requires a multiple of 14")
    parser.add_argument("--epochs", type=int,
                        help="Epoch budget (baseline 200; regularized presets 50)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float,
                        help="AdamW weight decay (all presets: 1.0)")
    parser.add_argument("--feature-noise", type=float,
                        help="Training-only Gaussian std (baseline 0; moderate 1; strict 2; extra-strict 3)")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=8,
                        help="Cases per frozen-ResNet forward during feature extraction")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for image decoding/preparation")
    parser.add_argument("--feature-cache",
                        help="Directory outside the repo for cached frozen-ResNet features "
                             "(e.g. on $SCRATCH); reused when split/partition/size match")
    parser.add_argument("--shuffle-train-labels", action="store_true",
                        help="Shuffled-label null: permute training labels across cases "
                             "(seeded by --seed); validation labels stay real")
    parser.add_argument("--summary-window", type=int, default=20,
                        help="Report mean/std of validation metrics over the last N epochs")
    args = parser.parse_args(argv)
    for name, value in REGULARIZATION_CONFIGS[args.config].items():
        if getattr(args, name) is None:
            setattr(args, name, value)

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    for name in ("epochs", "batch_size", "feature_batch_size", "summary_window"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    for name in ("weight_decay", "feature_noise"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error("--{} must be finite and non-negative".format(name.replace("_", "-")))
    if args.feature_cache:
        try:
            args.feature_cache = str(check_cache_dir(args.feature_cache))
        except ValueError as exc:
            parser.error(str(exc))
    return args


PARTITIONS = ("train", "validation")


def load_features(args, cnn, device, log=print):
    """Frozen-ResNet features per partition as CPU tensors, from the cache if possible.

    ``cnn=None`` means cache only: a missing cache file is an error instead
    of an extraction.
    """
    prepare = DINOv2Preprocessor(size=args.size)
    features = {}
    for name in PARTITIONS:
        start = time.perf_counter()
        metadata = cache_metadata(args.data_dir, args.split_position, name, args.size)
        path = cache_path(args.feature_cache, metadata) if args.feature_cache else None
        part = load_cached_features(path, metadata) if path else None
        source = "Loaded cached"
        if part is None and cnn is None:
            raise ValueError("Feature cache file missing: {}".format(path))
        if part is None:
            raw = GuidewireDataset(args.data_dir, split_position=args.split_position,
                                   partition=name)
            part = extract_features(cnn, PreparedCases(raw, prepare), device,
                                    args.feature_batch_size, args.num_workers)
            source = "Encoded"
            if path:
                save_cached_features(path, part, metadata)
                source = "Encoded and cached"
        features[name] = part
        log("{} {} in {:.1f}s".format(source, describe_partition(name, part),
                                      time.perf_counter() - start))
    return features


def run(args, features=None, echo=True):
    """Train one configuration without saving results; return the model and metrics.

    Optional CPU features are copied, so repeated experiments do not mutate
    the caller's inputs. Each call re-seeds training. ``echo=False`` is silent.
    """
    def log(line=""):
        if echo:
            print(line, flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNGRUModel(feature_noise=args.feature_noise).to(device)
    if features is None:
        features = load_features(args, model.encoder.cnn, device, log)
    else:
        for name in PARTITIONS:
            log("Reused in-memory {}".format(describe_partition(name, features[name])))
    # Fresh device copies: the caller's features stay untouched.
    data = {name: {key: value.to(device, copy=True) for key, value in features[name].items()}
            for name in PARTITIONS}
    if args.shuffle_train_labels:
        real = data["train"]["label"]
        data["train"]["label"] = shuffled_labels(real, args.seed)
        log("NULL RUN: training labels shuffled across cases with seed {} "
            "({} of {} keep their real label); validation labels are real.".format(
                args.seed, (data["train"]["label"] == real).sum().item(), len(real)))

    model.encoder.set_metadata_stats(data["train"]["depth"], data["train"]["offsets"])
    weight = class_weights(data["train"]["label"].cpu())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log("Device: {} | split {} | seed {}".format(device, args.split_position, args.seed))
    log("Trainable parameters: {:,}".format(trainable))
    log("Config: {} | effective epochs {} | lr {:g} | batch size {}".format(
        args.config, args.epochs, args.lr, args.batch_size))
    log("Regularization: weight decay {:g} | fixed MLP dropout 0.3 | feature noise {:g}"
        .format(args.weight_decay, args.feature_noise))
    log("Metadata mean [depth_mm, offset_s]: {} std: {}".format(
        [round(v, 3) for v in model.encoder.meta_mean.tolist()],
        [round(v, 3) for v in model.encoder.meta_std.tolist()]))
    log("Class weights [{}, {}]: {}".format(
        *CLASS_NAMES, [round(v, 3) for v in weight.tolist()]))
    log("Metrics: loss is class-weighted CE; bal = balanced accuracy; "
        "rec R/A = validation recall for RETRACT/ADVANCE.")

    result = train_model(
        model, data["train"], data["validation"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        seed=args.seed, log=log)

    final = result["rows"][-1]
    log("Final epoch {}: train bal {:.3f} | val bal {:.3f} auc {:.3f} loss {:.4f}".format(
        final["epoch"], final["train_bal_acc"], final["val_bal_acc"],
        final["val_auc"], final["val_loss"]))
    window = window_summary(result["rows"], args.summary_window)
    log("Last {} epochs ({}-{}) mean +- std: train bal {:.3f} +- {:.3f} | "
        "val bal {:.3f} +- {:.3f} auc {:.3f} +- {:.3f} loss {:.4f} +- {:.4f}".format(
            window["window"], window["first_epoch"], window["last_epoch"],
            *(window[key][stat] for key in ("train_bal_acc", "val_bal_acc",
                                            "val_auc", "val_loss")
              for stat in ("mean", "std"))))
    gap = window["train_bal_acc"]["mean"] - window["val_bal_acc"]["mean"]
    log("Last-window train - validation balanced-accuracy gap: {:+.3f}".format(gap))
    if args.shuffle_train_labels:
        log("Null training balanced accuracy (noise fitting): {:.3f}".format(
            window["train_bal_acc"]["mean"]))

    return {"model": model, "rows": result["rows"],
            "final_epoch": final, "last_window": window, "generalization_gap": gap}


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
