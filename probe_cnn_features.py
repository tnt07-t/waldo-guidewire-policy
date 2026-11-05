"""Frozen ResNet linear probe (no GRU) on one split.

Encodes train/validation once with the shared frozen FrameEncoder, builds
simple feature designs, and trains a single Linear layer with class-weighted
cross-entropy. Compares designs to ask whether 512-d CNN features carry
ADVANCE/RETRACT signal without a sequence model.

Designs:
  latest_goal  concat(history[:, -1], goal)   -> 1024-d
  mean_goal    concat(history.mean(1), goal)  -> 1024-d
  goal_only    goal                           -> 512-d

Preset ``best`` (from split-0 / seeds 0–1 study): ``--remove-maxpool`` and
design ``goal_only`` only. Also writes ``best_model.pt`` loadable by
``GoalOnlyLinearModel`` / ``infer_goal_linear.py``. See docs/grp1/PROBE_LINEAR.md.

The test partition is never loaded. Real images/features stay on Perlmutter.
Writes aggregate metrics under docs/grp1/results/<run-name>/.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

# Metrics mirrored from train_cnngru so unit tests do not import torchvision via cnn.
CLASS_NAMES = ("RETRACT", "ADVANCE")
REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "docs" / "grp1" / "results"
DESIGNS = ("latest_goal", "mean_goal", "goal_only")
# Best mean config across seeds 0–1 on split 0 (docs/grp1/PROBE_LINEAR.md).
BEST_PRESET = {"remove_maxpool": True, "designs": ["goal_only"]}


def class_weights(labels):
    counts = torch.bincount(labels, minlength=2).double()
    if (counts == 0).any():
        raise ValueError("Training partition must contain both classes")
    return (len(labels) / (2 * counts)).float()


def roc_auc(scores, labels):
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if len(positive) == 0 or len(negative) == 0:
        return math.nan
    greater = (positive[:, None] > negative[None, :]).double().mean()
    ties = (positive[:, None] == negative[None, :]).double().mean()
    return (greater + 0.5 * ties).item()


def binary_metrics(logits, labels, weight):
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


def build_features(design, history, goal):
    """history [N,6,512], goal [N,512] -> design matrix [N, D]."""
    if design == "latest_goal":
        return torch.cat([history[:, -1], goal], dim=-1)
    if design == "mean_goal":
        return torch.cat([history.mean(dim=1), goal], dim=-1)
    if design == "goal_only":
        return goal
    raise ValueError("Unknown design: {}".format(design))


def train_linear_probe(train_x, train_y, val_x, val_y, *, epochs, lr, weight_decay):
    """Fit nn.Linear on train; select epoch by validation balanced accuracy."""
    device = train_x.device
    weight = class_weights(train_y.cpu()).to(device)
    model = nn.Linear(train_x.shape[1], 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = None
    best_state = None
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        # Full-batch: N is ~267 on split 0; SGD noise is not the point of this probe.
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y, weight=weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_m = binary_metrics(model(train_x), train_y, weight)
            val_m = binary_metrics(model(val_x), val_y, weight)
        improved = best is None or val_m["bal_acc"] > best["val"]["bal_acc"] or (
            val_m["bal_acc"] == best["val"]["bal_acc"] and val_m["loss"] < best["val"]["loss"]
        )
        if improved:
            best = {"epoch": epoch, "train": train_m, "val": val_m}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rows.append({
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_bal_acc": train_m["bal_acc"],
            "val_loss": val_m["loss"],
            "val_bal_acc": val_m["bal_acc"],
            "val_auc": val_m["auc"],
            "best": improved,
        })

    model.load_state_dict(best_state)
    return {"model": model, "best": best, "rows": rows, "weight": weight.cpu()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--split-position", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--designs", nargs="+", default=None, choices=DESIGNS,
                        help="Feature designs to probe (default: all three)")
    parser.add_argument("--preset", choices=("best",),
                        help="best = remove-maxpool + goal_only (winning probe config)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--remove-maxpool", action="store_true",
                        help="Optional FrameEncoder stem change for matched ablations")
    parser.add_argument("--run-name",
                        help="Folder name under docs/grp1/results/")
    args = parser.parse_args()
    if not args.data_dir:
        parser.error("Set DATA_DIR or provide --data-dir")
    if args.epochs <= 0 or args.feature_batch_size <= 0:
        parser.error("--epochs and --feature-batch-size must be positive")
    if args.preset == "best":
        if args.designs is not None and list(args.designs) != BEST_PRESET["designs"]:
            parser.error("--preset best fixes designs to goal_only; omit --designs")
        args.remove_maxpool = True
        args.designs = list(BEST_PRESET["designs"])
    elif args.designs is None:
        args.designs = list(DESIGNS)
    if args.run_name is None:
        tags = ["probe-cnn", "split{}".format(args.split_position),
                "seed{}".format(args.seed)]
        if args.preset == "best":
            tags.append("best")
        if args.remove_maxpool:
            tags.append("nomaxpool")
        if args.designs == ["goal_only"]:
            tags.append("goalonly")
        if args.size != 518:
            tags.append("size{}".format(args.size))
        tags.append(time.strftime("%Y%m%d-%H%M%S"))
        args.run_name = "-".join(tags)
    if "/" in args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a plain folder name")
    return args


def main():
    # Heavy imports stay in main so local unit tests need only torch.
    from cnn import FrameEncoder
    from dataset import GuidewireDataset
    from preprocessing import DINOv2Preprocessor
    from train_cnngru import PreparedCases, extract_features


    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = RESULTS_DIR / args.run_name
    if out.exists():
        sys.exit("Run folder already exists: {}".format(out))
    out.mkdir(parents=True)
    log_path = out / "probe.log"

    def log(line=""):
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("Run folder: {}".format(out.relative_to(out.parents[2])))
    prepare = DINOv2Preprocessor(size=args.size)
    cnn = FrameEncoder(pretrained=True, remove_maxpool=args.remove_maxpool).to(device)
    cnn.eval()

    data = {}
    for name in ("train", "validation"):
        start = time.perf_counter()
        raw = GuidewireDataset(args.data_dir, split_position=args.split_position,
                               partition=name)
        feats = extract_features(
            cnn, PreparedCases(raw, prepare), device,
            args.feature_batch_size, args.num_workers)
        data[name] = {k: v.to(device) for k, v in feats.items()}
        counts = torch.bincount(data[name]["label"].cpu(), minlength=2).tolist()
        log("Encoded {}: {} cases ({} {}, {} {}) in {:.1f}s".format(
            name, len(data[name]["label"]), counts[0], CLASS_NAMES[0],
            counts[1], CLASS_NAMES[1], time.perf_counter() - start))

    log("Device: {} | split {} | seed {} | remove_maxpool={}".format(
        device, args.split_position, args.seed, args.remove_maxpool))
    log("Designs: {}".format(", ".join(args.designs)))

    summaries = []
    for design in args.designs:
        train_x = build_features(design, data["train"]["history"], data["train"]["goal"])
        val_x = build_features(design, data["validation"]["history"], data["validation"]["goal"])
        log("\n=== design {} | dim {} ===".format(design, train_x.shape[1]))
        result = train_linear_probe(
            train_x, data["train"]["label"], val_x, data["validation"]["label"],
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay)
        best = result["best"]
        log("Best epoch {}: train bal {:.3f} | val bal {:.3f} acc {:.3f} auc {:.3f} loss {:.4f}"
            .format(best["epoch"], best["train"]["bal_acc"], best["val"]["bal_acc"],
                    best["val"]["acc"], best["val"]["auc"], best["val"]["loss"]))
        log("Validation confusion (rows true, cols pred; RETRACT, ADVANCE):")
        for name, row in zip(CLASS_NAMES, best["val"]["confusion"]):
            log("  {:<8} {}".format(name, row))

        design_dir = out / design
        design_dir.mkdir()
        with (design_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result["rows"][0]))
            writer.writeheader()
            writer.writerows(result["rows"])
        linear_state = result["model"].state_dict()
        torch.save(linear_state, design_dir / "best_linear.pt")
        if design == "goal_only":
            # Full baseline checkpoint for GoalOnlyLinearModel / infer_goal_linear.py.
            bundle = {
                "design": "goal_only",
                "remove_maxpool": args.remove_maxpool,
                "size": args.size,
                "split_position": args.split_position,
                "seed": args.seed,
                "best_epoch": best["epoch"],
                "val_metrics": best["val"],
                "linear": {k: v.cpu() for k, v in linear_state.items()},
            }
            torch.save(bundle, out / "best_model.pt")
            log("Wrote best_model.pt (GoalOnlyLinearModel linear head)")
        summaries.append({
            "design": design,
            "feature_dim": train_x.shape[1],
            "best": best,
        })

    summary = {
        "args": {k: v for k, v in vars(args).items() if k != "data_dir"},
        "device": str(device),
        "designs": summaries,
        "note": (
            "Linear probe on frozen ResNet features; no GRU. "
            "Compare val bal-acc/AUC to the CNN-GRU policy (~0.61 / ~0.5 on split 0). "
            "Best mean config: goal_only + remove_maxpool (docs/grp1/PROBE_LINEAR.md)."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log("\nWrote summary.json and per-design metrics under {}".format(out))


if __name__ == "__main__":
    main()
