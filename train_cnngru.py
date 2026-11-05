"""Train the CNN-GRU policy head on one split and log metrics every epoch.

Pipeline:
1. Prepare train and validation cases with the shared preprocessor and
   encode every goal/history image ONCE with the frozen ResNet-18. The
   features are held in memory only; nothing is written to disk.
2. Fit depth/offset standardization on the training partition.
3. Train the GRU + MLP head with class-weighted cross-entropy, AdamW,
   gradient clipping and early stopping on validation balanced accuracy.
4. After every epoch, print one line of train and validation metrics
   (both computed in eval mode, so dropout does not distort the train
   numbers). With --output-dir, also write metrics.csv, summary.json and
   the best checkpoint.

The test partition is never loaded. Real images, features and outputs
must remain on Perlmutter.
"""

import argparse
import copy
import csv
import json
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
    loader = DataLoader(cases, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)
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


def class_weights(labels):
    """Inverse-frequency weights n / (2 * count_c); equal classes give 1, 1."""
    counts = torch.bincount(labels, minlength=2).double()
    if (counts == 0).any():
        raise ValueError("Training partition must contain both classes")
    return (len(labels) / (2 * counts)).float()


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
                               data["goal"][i:i + batch_size])
        for i in range(0, len(data["label"]), batch_size)
    ])
    return binary_metrics(logits, data["label"], weight)


def format_epoch(epoch, epochs, train, val, lr, seconds, best):
    return (
        "epoch {:>4}/{} | train loss {:.4f} acc {:.3f} bal {:.3f} "
        "| val loss {:.4f} acc {:.3f} bal {:.3f} auc {:.3f} rec R/A {:.2f}/{:.2f} "
        "| lr {:.1e} {:.1f}s{}".format(
            epoch, epochs, train["loss"], train["acc"], train["bal_acc"],
            val["loss"], val["acc"], val["bal_acc"], val["auc"],
            val["recall_retract"], val["recall_advance"],
            lr, seconds, "  *best" if best else "",
        )
    )


def train_model(model, train, val, *, epochs, batch_size, lr, weight_decay,
                grad_clip, patience, seed, log=print):
    """Train the head on cached features with early stopping.

    ``train`` / ``val``: dicts of tensors on the model's device with keys
    history [N, T, 512], goal [N, 512], depth [N, T], offsets [N, T], label [N].
    Selection: highest validation balanced accuracy, ties -> lower val loss.
    Returns the per-epoch rows, best epoch/metrics and the best state_dict
    (the model is left holding the best weights).
    """
    weight = class_weights(train["label"].cpu()).to(train["label"].device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = len(train["label"])

    rows, best, best_state, stale = [], None, None, 0
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()  # FrameEncoder keeps the frozen ResNet in eval mode
        order = torch.randperm(n, generator=generator).to(train["label"].device)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            logits = model.forward_features(train["history"][idx], train["depth"][idx],
                                            train["offsets"][idx], train["goal"][idx])
            loss = F.cross_entropy(logits, train["label"][idx], weight=weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

        train_m = evaluate(model, train, weight)
        val_m = evaluate(model, val, weight)
        improved = best is None or val_m["bal_acc"] > best["val"]["bal_acc"] or (
            val_m["bal_acc"] == best["val"]["bal_acc"]
            and val_m["loss"] < best["val"]["loss"])
        if improved:
            best = {"epoch": epoch, "train": train_m, "val": val_m}
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        seconds = time.perf_counter() - start
        rows.append({"epoch": epoch, "seconds": seconds, "best": improved,
                     **{"train_" + k: v for k, v in train_m.items() if k != "confusion"},
                     **{"val_" + k: v for k, v in val_m.items() if k != "confusion"}})
        log(format_epoch(epoch, epochs, train_m, val_m,
                         optimizer.param_groups[0]["lr"], seconds, improved))
        if stale >= patience:
            log("Early stop: no validation balanced-accuracy gain for {} epochs."
                .format(patience))
            break

    model.load_state_dict(best_state)
    return {"rows": rows, "best": best, "best_state": best_state}


def describe_partition(name, data):
    counts = torch.bincount(data["label"].cpu(), minlength=2).tolist()
    return "{}: {} cases ({} {}, {} {})".format(
        name, len(data["label"]), counts[0], CLASS_NAMES[0], counts[1], CLASS_NAMES[1])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True,
                        help="Zero-based position in splits.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=518,
                        help="Input size; shared preprocessor requires a multiple of 14")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20,
                        help="Epochs without validation balanced-accuracy gain before stopping")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=8,
                        help="Cases per frozen-ResNet forward during feature extraction")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for image decoding/preparation")
    parser.add_argument("--output-dir",
                        help="Optional new directory for metrics.csv, summary.json, best_model.pt")
    args = parser.parse_args()

    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    for name in ("epochs", "patience", "batch_size", "feature_batch_size"):
        if getattr(args, name) <= 0:
            parser.error("--{} must be positive".format(name.replace("_", "-")))
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if args.output_dir and Path(args.output_dir).exists():
        parser.error("--output-dir already exists; choose a fresh directory")
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNGRUModel(dropout=args.dropout).to(device)
    prepare = DINOv2Preprocessor(size=args.size)
    data = {}
    for name in ("train", "validation"):
        start = time.perf_counter()
        raw = GuidewireDataset(args.data_dir, split_position=args.split_position,
                               partition=name)
        data[name] = {key: value.to(device) for key, value in extract_features(
            model.encoder.cnn, PreparedCases(raw, prepare), device,
            args.feature_batch_size, args.num_workers).items()}
        print("Encoded {} in {:.1f}s".format(
            describe_partition(name, data[name]), time.perf_counter() - start), flush=True)

    model.encoder.set_metadata_stats(data["train"]["depth"], data["train"]["offsets"])
    weight = class_weights(data["train"]["label"].cpu())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Device: {} | split {} | seed {}".format(device, args.split_position, args.seed))
    print("Trainable parameters: {:,}".format(trainable))
    print("Metadata mean [depth_mm, offset_s]: {} std: {}".format(
        [round(v, 3) for v in model.encoder.meta_mean.tolist()],
        [round(v, 3) for v in model.encoder.meta_std.tolist()]))
    print("Class weights [{}, {}]: {}".format(
        *CLASS_NAMES, [round(v, 3) for v in weight.tolist()]))
    print("Metrics: loss is class-weighted CE; bal = balanced accuracy; "
          "rec R/A = validation recall for RETRACT/ADVANCE.", flush=True)

    result = train_model(
        model, data["train"], data["validation"],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        patience=args.patience, seed=args.seed,
        log=lambda line: print(line, flush=True))

    best = result["best"]
    print("\nBest epoch {}: train bal {:.3f} | val bal {:.3f} acc {:.3f} auc {:.3f} loss {:.4f}"
          .format(best["epoch"], best["train"]["bal_acc"], best["val"]["bal_acc"],
                  best["val"]["acc"], best["val"]["auc"], best["val"]["loss"]))
    print("Validation confusion (rows true, cols pred; RETRACT, ADVANCE):")
    for name, row in zip(CLASS_NAMES, best["val"]["confusion"]):
        print("  {:<8} {}".format(name, row))

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True)
        with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        summary = {"args": vars(args), "device": str(device),
                   "trainable_parameters": trainable,
                   "class_weights": weight.tolist(),
                   "meta_mean": model.encoder.meta_mean.tolist(),
                   "meta_std": model.encoder.meta_std.tolist(),
                   "epochs_run": len(result["rows"]), "best": best}
        (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        torch.save(result["best_state"], out / "best_model.pt")
        print("Wrote metrics.csv, summary.json, best_model.pt to", out)


if __name__ == "__main__":
    main()
