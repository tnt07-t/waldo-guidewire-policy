# Group 2 — Tran and Sabrina

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Tran sets up data loading; Sabrina works on models (specific model and deliverable unconfirmed). Group 2 is also planning visual data analysis; its exact scope and Sunday deliverable remain open. These are assignments and plans, not completion reports. The current task is original-resolution data loading; model implementation is deferred.

The first raw-case reader has been reviewed against the accepted one-case scope and passes laptop syntax checks. Synthetic image-loading tests and real-data execution are pending; the laptop lacks Pillow. Batching, preprocessing, and model code are deferred. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason. The Perlmutter Python/Pillow environment check passed; see the shared progress log for evidence and limits.

Before model implementation, agree on preprocessing and model input contracts, coordinate shared components with Group 1, and clarify architecture ownership.

## Progress log

- 2026-10-30 — **Friday assignments (owner-reported):** Tran handles data loading; Sabrina works on models, with specific architecture ownership open. Visual analysis is planned; its Sunday deliverable is not yet defined.
- 2026-10-30 — **Raw reader implemented locally:** original-resolution case reading is in `dataset.py`, with usage in [LOADER.md](../LOADER.md) and synthetic checks in `tests/test_dataset.py`. These checks create invented data; they are not a new evaluation dataset. Syntax and documentation checks pass. Image-loading tests have not run because laptop Pillow is unavailable; real-data execution is pending. Next: test on Perlmutter after the approved code handoff. No preprocessing or model code added.
