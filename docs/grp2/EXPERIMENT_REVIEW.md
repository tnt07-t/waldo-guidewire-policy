# Experiment review — no code edits needed

The notebook is already populated with the experiment history, rather than asking
you to assemble a dashboard yourself. After the code is published and pulled, run
this on a **Perlmutter login node**:

```sh
source "$HOME/.waldo-env.sh"
python -B experiment_review.py --prepare-notebook
```

It prints an exact absolute notebook path. In JupyterLab, **click File → Open from
Path**, paste that path, then **click Run → Run All Cells**. Select a working Python
kernel if prompted. No GPU, model downloads, extra plotting packages, manually
generated graphs, run-directory moves, or notebook code edits are required.

## What you see immediately

- A concise history of the four completed development trials: original late-goal
  GRU/Adam, AdamW regularization, early-goal GRU, and static latest+goal+metadata.
  Each records its question, change, confirmed result, interpretation limits and
  next decision. The score progression uses their documented order.
- Actual saved loss/BA overlays, with up to four main/recent runs selected
  automatically. All discovered reports remain available in expandable details.
- Per-trial training/validation loss, balanced accuracy and both class recalls;
  selected-epoch markers; confusion counts; selected versus final metrics; settings,
  revision and encoder/reload checks when the saved reports recorded them.
- Earlier setup and diagnostics, the successful eight-case learning check and
  Sabrina's probe comparison, including its unfavorable/inconclusive results.
- Discussed next steps kept separate from completed experiments. Known spatial-grid
  code/report presence is reflected automatically; file presence alone is not
  treated as verified implementation or successful training.

The recorded journal is `docs/grp2/EXPERIMENT_HISTORY.json`. It contains only
permitted aggregate outcomes and narrative already recorded in the shared project
notes; it never contains private rows, metadata scales, predictions or weights.
The Git-tracked notebook template itself has no execution outputs or personal
paths. Its helper reads the current journal when Run All is clicked, so future Git
updates can improve the history without replacing your private notebook copy.

## Saved reports and historical summaries stay distinct

The helper scans only the established scratch output trees:
`waldo-gru-training`, `waldo-static-training`, and `waldo-evaluation`. It supports
the full PyTorch trainer's `summary.json`, `run_context.json`, `history.json` and
Sabrina's `metrics.json`. It never scans the dataset or all scratch and never opens
images, model checkpoints or private prediction files.

Known trials are linked automatically using documented paths and saved settings,
epochs and score fingerprints. An unknown baseline directory can be found by its
saved fingerprint. Conflicting evidence or multiple matching reports is shown
explicitly; no arbitrary report is selected. Missing saved provenance stays unknown,
even if a folder name appears to imply a recipe.

If an old report is missing or not accessible from this account, its documented
result remains in the history with a clear label. Epoch curves are shown only from
actual saved histories; there are no invented curves or automatic training reruns.
Linked reports supply selected-epoch training scores and final metrics that were
not previously extracted into the shared notes. Probe fits have no fabricated
epoch history and retain their different scaling/selection protocol.

## After another experiment

**Click Run → Run All Cells** to refresh. New compatible reports in the established
output trees appear automatically and the overlays include recent training runs.
No list of run IDs or filter code needs maintaining for normal use. The raw index,
older checks and proposal details are collapsed to keep the first view concise.
New report schemas or nonstandard output roots need an implementation change;
current known runs require no manual adjustments.

Meaningful hypotheses, rationale and decisions are curated in the existing shared
progress/reference notes and the journal when work changes; they are not guessed
from scores or filenames. No extra overlapping personal log is needed. The journal
distinguishes proposed changes, local code and completed runs.

After pulling helper/journal updates, **click Kernel → Restart Kernel and Run All
Cells**. The preparation command preserves an existing scratch notebook and does
not overwrite saved outputs or notes. The private copy embeds the existing checkout
and scratch-root paths so the Jupyter kernel need not inherit terminal exports.

## Privacy and interpretation

The executed copy lives at `$PSCRATCH/waldo-review/experiment_review.ipynb`. Keep
it, its outputs, and every real report/history/checkpoint on Perlmutter. Never
commit or transfer that executed private copy. The source checkout stays in place
to supply the helper. The setup command copies the output-free template only;
report reads are read-only, and the review never changes saved evidence.

Historical dates come from the documented milestones. Extra discovered reports
use report-file modification times, which can change after copying. Selected and
final scores remain separate; snapshots from shared notes do not silently override
private report values. Compare split, seed, optimizer, scaler, feature roles and
capacity before interpreting differences. A single seed and repeated validation
selection do not establish superiority. No scores are averaged across incompatible
trials, no final-test data is read, and extraction/head duration is not online
decision latency.

## Verification

```sh
python -B -m unittest discover -s tests -v
```

Tests use invented reports to check discovery, private-field exclusion, no report
mutations, legacy/static/probe identities, report errors, selected/final alignment,
SVG XML/HTML escaping, conservative journal matching and conflicts, automatic
recent-run selection, useful missing-report fallback, notebook code execution with
an injected display stub, and private-copy preservation. Actual Perlmutter Jupyter
Run All and visual rendering remain to be verified after publication.
