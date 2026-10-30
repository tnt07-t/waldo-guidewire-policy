# Group 1 — Andrew and Suhani

Read the [shared approach notes](../APPROACHES.md), [dataset notes](../DATASET.md), and [project README](../../README.md) before continuing. This file holds Group 1 plans, progress, and permitted aggregate results. Shared environment, preprocessing, interfaces, and evaluation decisions belong in the approach notes.

## Proposed scope — supplied by project owner

- CNN–GRU: a shared CNN encodes frame-map pairs, combines features with depth, and feeds a GRU action predictor.
- CNN–LSTM: replace GRU with LSTM while keeping visual inputs and depth encoding consistent.
- Current-input baseline: test whether history helps; exact inputs and implementation remain to be agreed.

These are team proposals, not confirmed course requirements. Individual ownership is open. The shared optional JEPA extension remains in the approach notes.

## Current status and next steps

Owner-reported schedule (2026-10-30): Friday — Suhani and Andrew work on CNNs and understanding the data. By Sunday (2026-11-01) — aim to have the CNN work done or finalizing. This is a target, not a completion report.

No Group 1 implementation or experiment results have been reported in this workspace. Basic dataset validation has already passed as recorded in the dataset notes; do not repeat it without a concrete reason.

Before implementation, agree on the shared environment and contracts, clarify the frame-map pairing and current-input baseline, and assign individual owners. No model implementation is currently authorized.

## Progress log

- 2026-10-30 — **Friday/Sunday plan (owner-reported):** Suhani and Andrew work on CNNs and understanding the data, targeting CNN work done or finalizing by Sunday. Exact architecture ownership and completion are unconfirmed. Next: coordinate model inputs with the shared raw reader before integration.
