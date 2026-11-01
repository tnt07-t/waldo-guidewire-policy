# Group 1 — Andrew and Suhani

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 1 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- CNN–GRU: a shared CNN encodes frame-map pairs, combines features with depth, and feeds a GRU action predictor.
- CNN–LSTM: replace GRU with LSTM while keeping visual inputs and depth encoding consistent.
- Current-input baseline: test whether history helps; exact inputs and implementation remain to be agreed.

These are team proposals, not confirmed course requirements. Individual ownership is open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Suhani and Andrew work on CNNs and understanding the data. By Sunday (2026-11-01) — aim to have the CNN work done or finalizing. This is a target, not a completion report.

Upstream commit `023d125` implements the frozen ResNet-18 encoder in `cnn.py`: 512 features/image, separate goal/history calls, retained default maxpool, frozen backbone and evaluation-mode BatchNorm. The first shared GPU evaluation has completed; aggregate scores and timing are recorded once in EVALUATION.md. Basic dataset validation has already passed; do not repeat it without a concrete reason.

**Evaluation clarification:** the first shared frozen-feature evaluation has since completed; aggregate results are recorded in [EVALUATION.md](../EVALUATION.md). It trains a separate scikit-learn logistic classifier on detached ResNet features. `cnn.py` is unchanged, no trainable classifier was added inside ResNet, and the CNN weights/BatchNorm statistics stayed fixed. Sabrina confirmed that the CNN should remain frozen for the coming experiments. High probe training accuracy does not mean the CNN was fine-tuned.

Sabrina authorized the shared before-GRU evaluation milestone on 2026-11-01. [EVALUATION.md](../EVALUATION.md) describes the controlled ResNet/DINOv2 comparisons and diagnostic ablations. The new probes do not implement Group 1 GRU/LSTM models or change their ownership; remaining team contracts still need agreement.

## Progress log

- 2026-10-30 — **Friday/Sunday plan (owner-reported):** Suhani and Andrew work on CNNs and understanding the data, targeting CNN work done or finalizing by Sunday. Exact architecture ownership and completion are unconfirmed. Next: coordinate model inputs with the shared raw reader before integration.
