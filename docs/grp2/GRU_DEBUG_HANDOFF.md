# ViT experiment handoff

## Current state

Five frozen experiments are complete; last-block adaptation has a time-limited eight-epoch result (best saved validation BA 57.03%). Do not repeat completed runs, tiny fitting, masking or pretrained readiness just to resume. Read [PROGRESS](PROGRESS.md) for evidence and [GRU_TRIAL](GRU_TRIAL.md) for the implemented recipe and design rationale.

The next task is interpretation and presentation, or a separately agreed experiment—not an automatic retry. Review weak balanced training discrimination and missing final reload/frozen checks before drawing conclusions about fine-tuning. Preserve the existing models and independent experiment-review tools.

## Reusable environment and artifacts

Tran handles SSH and Perlmutter execution. In a new remote shell, `source "$HOME/.waldo-env.sh"` restores the PyTorch module, existing scratch venv, data/cache settings and `$HOME/CS5470_vit_gru_debug` worktree. Prior missing-timm errors came from omitted environment activation; tests on main do not verify debug-branch code. Preserve remote notebooks and private artifacts before updates.

Frozen histories are under the existing waldo-gru-training, waldo-static-training and waldo-grid-training scratch roots. Adaptation attempts are under `$PSCRATCH/waldo-static-finetune/lastblock-summary-adam-split0-seed0-01` and `-02`. Accuracy/loss SVGs and the combined aggregate export were generated under `$PSCRATCH/waldo-poster/accuracy-loss/`; generation was user-confirmed, not independent visual verification. Local aggregate data and Sheet links are indexed in [MIDPOINT](../../tran/MIDPOINT.md). The generic experiment-review default roots omit the grid directory; use the existing standalone plotter or explicitly configure discovery.

## Future execution boundaries

Finish local implementation and synthetic verification before considering GPU time. Current adaptation defaults cap training/evaluation and the overall worker at 1200 seconds each; loading/scoring/reload consume the same overall budget. Historical 15-minute allocations cannot accommodate this full envelope. Agree on allocation/runtime before any future run, use a fresh output directory and release compute promptly. Do not increase limits or retry silently.

Saved-artifact analysis requires no GPU. Request only allowlisted aggregate fields: actual settings/revision, status, stop reason, epochs/updates, best/final metrics, runtime and verification results. A parent `worker_failed` can mask a budget stop; absence of verification fields is not a pass or failure. No case-level data, raw images, features, private predictions, complete reports or weights belong in Git. Test remains reserved.

## Presentation handoff

The first Google Doc tab contains Tran’s six-step narrative, five architecture diagrams and comparison chart, checked within three pages for Tran’s section alone. Preserve other contributors. The Sheet has six detail tabs, 305 epochs and 12 accuracy/loss charts; further Sheet edits require a proposal and explicit approval. Presentation claims must distinguish accuracy from BA and the budget-limited saved result from a converged run.
