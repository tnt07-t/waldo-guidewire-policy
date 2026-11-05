"""Run the H4 regularization sweep in a few long-lived processes on one GPU.

Each run is one train_cnngru.py configuration and writes the same run
folder docs/grp1/results/h4-<setting>[-null]-split<S>-seed<N>/ that the
command-line script would, with the same results: every run re-seeds from
its own --seed. What changes is the overhead. Instead of 270 Python
processes that each import torch and reload features, the frozen-ResNet
features are encoded once per split into the --feature-cache, and
--workers processes share the GPU, each loading every split's features
once and training its share of the runs back to back. The head is tiny, so
one GPU easily holds 8 such workers.

  python sweep_h4.py --feature-cache "$PSCRATCH/waldo-feature-cache" --workers 8

Steps:
1. Fill the feature cache for every split (sequential, one process).
2. Start --workers worker processes; worker i takes runs i, i+N, i+2N, ...
3. When all workers finish, run summarize_runs.py over every setting and
   its shuffled-label null and write h4-summary.txt in the repo root.

The sweep resumes: a run whose folder has summary.json (written last) is
skipped, and a folder without it, left by a killed job, is deleted and
rerun. Rerun the same command after a time-out. Per-epoch lines go to each
run's train.log; the terminal shows one line per finished run. Worker
output is also kept in h4-sweep-logs/worker-<i>.log.

Real images and features stay on Perlmutter; the test partition is never
loaded.
"""

import argparse
import os
import selectors
import shutil
import subprocess
import sys
import time

import torch

import train_cnngru
from train_cnngru import REPO_ROOT, RESULTS_DIR

# (tag, train_cnngru.py flags). r0 is the baseline; the rest are the
# H4 sweep in docs/grp1/NEXT_EXPERIMENTS.md, one family at a time.
SETTINGS = (
    ("r0", ()),
    ("wd0.1", ("--weight-decay", "0.1")),
    ("wd0.3", ("--weight-decay", "0.3")),
    ("wd1", ("--weight-decay", "1.0")),
    ("do0.5", ("--dropout", "0.5")),
    ("do0.7", ("--dropout", "0.7")),
    ("fn0.25", ("--feature-noise", "0.25")),
    ("fn0.5", ("--feature-noise", "0.5")),
    ("fn1", ("--feature-noise", "1.0")),
    ("fd0.2", ("--feature-dropout", "0.2")),
    ("fd0.4", ("--feature-dropout", "0.4")),
    ("frd0.17", ("--frame-dropout", "0.17")),
    ("frd0.33", ("--frame-dropout", "0.33")),
    ("ls0.1", ("--label-smoothing", "0.1")),
    ("ls0.2", ("--label-smoothing", "0.2")),
)
PREFIX = "h4-"
LOG_DIR = REPO_ROOT / "h4-sweep-logs"
SUMMARY_FILE = REPO_ROOT / "h4-summary.txt"
FIXED_FLAGS = ("--epochs", "200", "--patience", "0")


def run_name(tag, split, seed, null):
    return "{}{}{}-split{}-seed{}".format(PREFIX, tag, "-null" if null else "", split, seed)


def build_runs(splits, seeds, settings=SETTINGS):
    """Every (name, split, train_cnngru.py flags) in a fixed order.

    Real and null runs of one (setting, split, seed) are adjacent, and splits
    vary fastest after seeds, so any prefix of the list is a balanced sample.
    """
    runs = []
    for tag, flags in settings:
        for split in splits:
            for seed in seeds:
                for null in (False, True):
                    argv = ["--split-position", str(split), "--seed", str(seed),
                            *FIXED_FLAGS, *flags,
                            "--run-name", run_name(tag, split, seed, null)]
                    if null:
                        argv.append("--shuffle-train-labels")
                    runs.append((run_name(tag, split, seed, null), split, argv))
    return runs


def shard(runs, index, count):
    """Runs for worker ``index`` of ``count`` (round robin)."""
    if not 0 <= index < count:
        raise ValueError("shard index must be in [0, count)")
    return runs[index::count]


def run_state(name):
    """'done', 'partial' (folder without summary.json) or 'todo'."""
    run_dir = RESULTS_DIR / name
    if (run_dir / "summary.json").is_file():
        return "done"
    return "partial" if run_dir.exists() else "todo"


def clear_partial(name):
    """Delete an unfinished h4 run folder so it can be rerun."""
    if not name.startswith(PREFIX):
        raise ValueError("Refusing to delete a non-H4 folder: {}".format(name))
    shutil.rmtree(RESULTS_DIR / name)


def common_flags(args):
    return ["--data-dir", args.data_dir, "--feature-cache", args.feature_cache]


def fill_cache(args):
    """Encode every split once into the feature cache (cheap if already there)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = None
    for split in args.splits:
        probe = train_cnngru.parse_args(
            common_flags(args) + ["--split-position", str(split),
                                  "--run-name", PREFIX + "cache-probe"])
        if cnn is None:
            from cnn import FrameEncoder
            cnn = FrameEncoder(pretrained=True).to(device)
        train_cnngru.load_features(probe, cnn, device,
                                   log=lambda line: print("cache split {}: {}".format(split, line),
                                                          flush=True))


def worker(args):
    """Train this worker's share of the runs, reusing features per split."""
    runs = shard(build_runs(args.splits, args.seeds), args.shard_index, args.shard_count)
    features = {}
    failed = 0
    for position, (name, split, argv) in enumerate(runs, 1):
        state = run_state(name)
        if state == "done":
            continue
        if state == "partial":
            clear_partial(name)
        start = time.perf_counter()
        try:
            run_args = train_cnngru.parse_args(common_flags(args) + argv)
            if split not in features:
                features[split] = train_cnngru.load_features(
                    run_args, None, None, log=lambda line: None)
            summary = train_cnngru.run(run_args, features[split], echo=False)
        except Exception as exc:  # keep going; the run is retried on the next sweep
            failed += 1
            print("[worker {} {}/{}] FAILED {}: {!r}".format(
                args.shard_index, position, len(runs), name, exc), flush=True)
            continue
        window = summary["last_window"]
        print("[worker {} {}/{}] {} in {:.0f}s | last-{} train bal {:.3f} val bal {:.3f} "
              "val auc {:.3f}".format(
                  args.shard_index, position, len(runs), name, time.perf_counter() - start,
                  window["window"], window["train_bal_acc"]["mean"],
                  window["val_bal_acc"]["mean"], window["val_auc"]["mean"]), flush=True)
    return failed


def summary_groups():
    groups = []
    for tag, _ in SETTINGS:
        groups.append("{}=docs/grp1/results/{}{}-split*".format(tag, PREFIX, tag))
        groups.append("{}:null=docs/grp1/results/{}{}-null-*".format(tag, PREFIX, tag))
    return groups


def summarize():
    command = [sys.executable, str(REPO_ROOT / "summarize_runs.py"), "--window", "20",
               "--min-effect", "0.03", "--at-epochs", "10,25,50,100,200", *summary_groups()]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    SUMMARY_FILE.write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout + result.stderr, flush=True)
    return result.returncode


def coordinate(args):
    """Fill the cache, run the workers, then summarize."""
    runs = build_runs(args.splits, args.seeds)
    states = [run_state(name) for name, _, _ in runs]
    print("{} runs: {} done, {} partial, {} to do".format(
        len(runs), states.count("done"), states.count("partial"), states.count("todo")),
        flush=True)
    fill_cache(args)

    LOG_DIR.mkdir(exist_ok=True)
    threads = max(1, (os.cpu_count() or 1) // args.workers)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    procs = []
    for index in range(args.workers):
        command = [sys.executable, "-u", str(REPO_ROOT / "sweep_h4.py"),
                   "--data-dir", args.data_dir, "--feature-cache", args.feature_cache,
                   "--splits", *map(str, args.splits), "--seeds", *map(str, args.seeds),
                   "--shard", "{}/{}".format(index, args.workers)]
        log = (LOG_DIR / "worker-{}.log".format(index)).open("w", encoding="utf-8")
        procs.append((subprocess.Popen(command, cwd=REPO_ROOT, env=env, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT), log))
    # Relay worker lines to the terminal and to each worker log as they arrive.
    selector = selectors.DefaultSelector()
    for proc, log in procs:
        selector.register(proc.stdout, selectors.EVENT_READ, log)
    open_streams = len(procs)
    while open_streams:
        for key, _ in selector.select():
            line = key.fileobj.readline()
            if not line:
                selector.unregister(key.fileobj)
                open_streams -= 1
                continue
            print(line, end="", flush=True)
            key.data.write(line)
            key.data.flush()
    codes = [proc.wait() for proc, _ in procs]
    for _, log in procs:
        log.close()

    states = [run_state(name) for name, _, _ in runs]
    print("Sweep finished: {} of {} runs done; worker exit codes {}".format(
        states.count("done"), len(runs), codes), flush=True)
    summarize()
    return 0 if states.count("done") == len(runs) else 1


def parse_shard(text):
    index, sep, count = text.partition("/")
    try:
        index, count = int(index), int(count)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected I/N, got {!r}".format(text))
    if not sep or count <= 0 or not 0 <= index < count:
        raise argparse.ArgumentTypeError("Expected I/N with 0 <= I < N, got {!r}".format(text))
    return index, count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR"))
    parser.add_argument("--feature-cache",
                        help="Directory outside the repo for cached features (e.g. on $PSCRATCH); "
                             "required unless --dry-run or --summarize-only")
    parser.add_argument("--workers", type=int, default=8,
                        help="Worker processes sharing the GPU (default 8)")
    parser.add_argument("--splits", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dry-run", action="store_true",
                        help="List the runs and their state, then exit")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only rewrite h4-summary.txt from existing run folders")
    parser.add_argument("--shard", type=parse_shard, help=argparse.SUPPRESS)  # internal
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not (args.dry_run or args.summarize_only):
        if not args.data_dir:
            parser.error("Set DATA_DIR or provide --data-dir")
        if not args.feature_cache:
            parser.error("--feature-cache is required to run the sweep")
        try:
            args.feature_cache = str(train_cnngru.check_cache_dir(args.feature_cache))
        except ValueError as exc:
            parser.error(str(exc))
    args.shard_index, args.shard_count = args.shard or (None, None)
    return args


def main():
    args = parse_args()
    if args.dry_run:
        for name, _, argv in build_runs(args.splits, args.seeds):
            print("{:<8} {:<34} {}".format(run_state(name), name, " ".join(argv)))
        return 0
    if args.summarize_only:
        return summarize()
    if args.shard is not None:
        return 1 if worker(args) else 0
    return coordinate(args)


if __name__ == "__main__":
    sys.exit(main())
