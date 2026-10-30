# Group 2 — Tran and Sabrina

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 2 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- ViT–GRU: replace CNN with a pretrained vision transformer, retaining GRU and consistent inputs.
- CNN–temporal attention: replace GRU with temporal self-attention, retaining CNN and consistent inputs.

These are team proposals, not confirmed course requirements. Friday task ownership is recorded below; exact architecture ownership remains open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Tran sets up data loading; Sabrina works on models (specific model and deliverable unconfirmed). Group 2 is also planning visual data analysis; its exact scope and Sunday deliverable remain open. These are assignments and plans, not completion reports. The current task is original-resolution data loading; model implementation is deferred.

The first raw-case reader passed six synthetic tests and decoded one real training case on Perlmutter at `9221135` (owner-reported). The owner confirmed that both the goal and six history frames display in the Perlmutter notebook, with history depth/time annotations. This confirms display execution, not visual/semantic correctness. Batching, preprocessing, and model code remain pending. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

Before model implementation, agree on preprocessing and model input contracts, coordinate shared components with Group 1, and clarify architecture ownership.

## Progress log

- 2026-10-30 — **Friday assignments (owner-reported):** Tran handles data loading; Sabrina works on models, with specific architecture ownership open. Visual analysis is planned; its Sunday deliverable is not yet defined.
- 2026-10-30 — **Notebook display working (owner-reported):** selected the existing module Python with its available notebook support and successfully displayed the goal and six history images with depth/time annotations on Perlmutter. No new environment or package installation needed. This resolves the notebook setup obstacle and confirms display execution, not target correctness, wire visibility, or semantic alignment. Reusable loading and history-display cells based on those snippets are now included in the shared notebook, without outputs; the other inspection sections remain placeholders. Laptop checks validate cell syntax and the absence of saved outputs; the packaged notebook itself has not been rerun remotely. Next: assess the images before agreeing on preprocessing; keep sensitive outputs on Perlmutter.
- 2026-10-30 — **First loader milestone passed (owner-run on Perlmutter, `9221135`):** all six synthetic tests passed. The first case in split position 0's training partition decoded as seven RGB images at original `(1065, 1050)` size, with six depths and six time offsets; the selected partition contains 267 cases. No preprocessing applied. This verifies one-case decoding and synthetic loader behavior, not full real-data decoding or visual/semantic correctness. Notebook display also succeeded as recorded above. Batching and models remain deferred. Loader implementation is `922f200`; shared handoff incorporates teammate commit `77947cd`.
