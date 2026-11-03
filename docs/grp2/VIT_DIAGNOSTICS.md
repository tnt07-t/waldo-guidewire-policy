# ViT feature check

## What we tested

Tran ran the notebook diagnostics on Perlmutter to check whether the ViT's 384-value image summaries respond to the goal marker and wire-tip region. The notebook contains the runnable steps; real images and outputs stay on Perlmutter.

- **Goal:** visually compare the supplied marked and unmarked reference images. The image comparison worked, but feature-distance results were not reported, so marker sensitivity is not established.
- **Wire:** cover a 28×28 region around the tip, then separately cover three equally sized non-wire regions. Compare each edited image's features with the original.

## Result — one history frame

Relative L2 measures feature change relative to the original vector's length. Cosine distance measures the change in its direction. Neither is accuracy.

- Wire tip: relative L2 **0.0725**; cosine distance **0.00263**.
- Comparison 1: **0.1005**; **0.00506**.
- Comparison 2: **0.0789**; **0.00312**.
- Comparison 3: **0.0800**; **0.00320**.

Covering the tip changed the summary less than covering any comparison region. This did not provide evidence of stronger tip sensitivity in this example.

## What we learned and what comes next

The check confirmed that image edits affect the features, but was limited for our task: it examined one frame, while ADVANCE/RETRACT prediction uses a sequence. Region selection and background content also affect the distances. The result does not establish whether the features retain useful wire information, and does not justify rejecting or fine-tuning the ViT.

Tran chose to stop further manual masking. The [frozen ViT–GRU trial](GRU_TRIAL.md) now has a locally tested implementation: next check that it can learn eight training examples on Perlmutter, then separately scope full training and validation. Once it works, compare with a latest-frame model to test whether history helps. These tests address learning and temporal usefulness more directly.

The workflow was shared in `0c4cbee`; execution and environment evidence are summarized in [Group 2 progress](PROGRESS.md). GRU debugging code now exists in the separate worktree; real-data GRU training results remain pending.
