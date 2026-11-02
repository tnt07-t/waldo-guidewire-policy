# CNN 512-d feature diagnostics

`cnn_diagnostics.py` probes whether the **pooled ResNet-18 512-d vectors** a later GRU would receive still carry coarse wire, goal-marker, and motion signal. It is separate from `cnn.py`: no localization heads, no direction head, no training, and no feature/image writes.

These checks do **not** measure ADVANCE/RETRACT accuracy. They only ask whether frozen encoder outputs look informative enough to feed a temporal model. Owner-run one-case GPU diagnostics on Perlmutter passed (2026-11-02); laptop invented-input tests cover bookkeeping only.

## What each probe scores

All scores use the same 512-d interface the GRU would see (`history` → `(6, 512)`, `goal` → `(512)`).

1. **Wire-ish sensitivity (history frames):** grid-occlude squares on each prepared history image; record L2 change in the 512-d vector. Per frame: max/median delta and normalized argmax location. Across frames: range/std of per-frame max scores (`wire_scores`). Larger, structured occlusion response suggests the vector is not ignoring local structure.
2. **Goal marker:** detect red-ish pixels on the prepared goal preview; occlude that mask versus an equal-area control patch; compare 512-d L2 changes. Also report goal-versus-history feature distances. A marker/control ratio above ~1 suggests the goal vector reacts more to the painted target than to a matched control edit.
3. **Motion (not part of the CNN):** calculations on the history feature sequence and the wire scores from (1), optionally correlated with `depth_relative_mm` step sizes. Reports mean step L2, first-vs-last L2/cosine distance, wire-score path, and feature–depth step correlation. No learned direction output.

Occlusion fill is `0.0` in normalized tensor space (approximately ImageNet mean after `DINOv2Preprocessor`).

## Synthetic check — laptop or Perlmutter

Needs PyTorch and Pillow only (no torchvision import at module import time):

```sh
python3 -m unittest discover -s tests -p 'test_cnn_diagnostics.py' -v
```

Local result (2026-11-02, tool-verified): five invented-input tests passed with system Python 3.9 / torch 2.2.2. They do not load ResNet weights or real cases.

## One-case run — Perlmutter terminal

After the laptop file is available in your Perlmutter clone. Uses the existing `pytorch/2.13.0` module (torch + torchvision). Grid occlusion encodes many patched images per frame; the successful owner run used a short shared-interactive GPU allocation (`salloc` / `srun`). Confirm `cases.jsonl` exists under `DATA_DIR` before encoding (a missing scratch copy previously failed; the shared studentdata tree is a valid `DATA_DIR` when the personal copy is absent).

```sh
# Run on Perlmutter (GPU session recommended for the real-case probe).
module load pytorch/2.13.0
export TORCH_HOME="${PSCRATCH:?}/model-cache/torch"
# Prefer a personal scratch copy when present; otherwise the shared studentdata path.
export DATA_DIR="${PSCRATCH}/CS5470_FinalProject/shorthorizon2"
if [ ! -f "$DATA_DIR/cases.jsonl" ]; then
  export DATA_DIR=/global/cfs/cdirs/m4999/studentdata/shorthorizon2
fi
test -f "$DATA_DIR/cases.jsonl"

cd "${REPO_DIR:?Set REPO_DIR to your waldo-guidewire-policy clone}"
python3 -m unittest discover -s tests -p 'test_cnn_diagnostics.py' -v
python3 cnn_diagnostics.py --data-dir "$DATA_DIR" --split-position 0 --partition train --case-index 0
```

Optional: `--patch` / `--stride` (default 74), `--size`, `--remove-maxpool`, `--case-index`.

Share only permitted aggregates printed by the script (device, shapes, occlusion summaries, marker/control ratio, direction scalars). Do not transfer images, masks, feature vectors, or sensitive errors off Perlmutter.

## Owner-run one-case aggregates (2026-11-02)

Device `cuda`; history `(6, 512)`; goal `(512,)`. Unit tests OK on login and GPU node. Exact remote git HEAD not supplied.

| Probe | Result (permitted aggregates) |
| --- | --- |
| Wire occlusion | score_range `0.246`; max/median ~`1.62`–`1.76`; max L2 ~`3.02`–`3.27`; argmax often `(0.357, 0.643)` |
| Goal marker | marker/control `2.95` (L2 `3.965` vs `1.344`); goal–history L2 mean `2.897` |
| Motion math | mean step L2 `2.474`; first–last L2 `2.931`; cosine distance `0.012`; depth-step corr `-0.226` |

Goal-marker response and non-collapsed history trajectory support trying a GRU on frozen 512-d features. Wire argmax still needs preview confirmation; feature–depth step correlation is weak/negative on this case. Full evidence and limits: [PROGRESS.md](PROGRESS.md).

## How to read outcomes

| Signal | Encouraging pattern | Weak pattern |
| --- | --- | --- |
| Wire occlusion | max L2 ≫ median; argmax not stuck on padding | flat deltas across the grid |
| Goal marker | marker L2 > control L2; goal far from history | marker ≈ control; goal ≈ a history frame |
| Motion | non-trivial step L2 / first–last change; optional depth correlation | near-zero feature trajectory over the six frames |

Weak results mean a GRU on frozen 512-d may lack wire/goal/motion cues—not that the diagnostics script failed. Strong results still do not prove policy accuracy. The completed one-case GPU run need not be repeated without a relevant change; a second case or preview overlay is the useful follow-up.

Encoder architecture and smoke-check commands remain in [cnn.md](cnn.md). Outcomes belong in [PROGRESS.md](PROGRESS.md).
