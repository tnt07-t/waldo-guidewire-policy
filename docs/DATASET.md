# Dataset

## Sources and restrictions

Sources: project-owner instructions and the dataset-supplied `README.md` text pasted from Perlmutter on 2026-10-30. The supplied README confirms the aggregate counts, package/schema descriptions, variable history spacing, and proxy-label limitations below. Assistant tools have not directly accessed the dataset.

Instructor source on Perlmutter: `/global/cfs/cdirs/m4999/studentdata/shorthorizon2`.

The instructor recommends a separate copy in each student's `$PSCRATCH`. Use `cp -a` to preserve internal hard-linked duplicate images; see the README for portable commands.

The dataset is sensitive and MUST NOT leave Perlmutter. Do not download images, case records, or derived sensitive artifacts to laptops, GitHub, external APIs, or external tracking services. Shared documentation may contain schema descriptions, aggregate counts, and portable setup paths without private records. Local inspection and any logs containing sensitive details must remain on Perlmutter.

Preserve the supplied package. The instructor permits corrections to questionable labels or goal markings, but corrections must be traceable and recorded separately from supplied data. A correction-record format has not been agreed.

## Task and aggregate counts

- ADVANCE = 1: 259 cases.
- RETRACT = 0: 157 cases.
- Total: 416 decision points from 120 episodes.

These are inferred operator-action labels, not independently verified optimal actions. Owner-reported JSONL checks confirm these case, episode, and label counts. Assistant tools have not independently accessed the dataset.

## Package structure

- `README.md`: dataset-supplied documentation; general schema and limitations text supplied by owner on 2026-10-30.
- `manifest.json`: all case records and five episode-based splits.
- `cases.jsonl`: the same records in the same order, one JSON object per line.
- `splits.json`: the same five splits.
- `validation.json`: checks performed during dataset preparation; it does not establish that a student's copy has passed integrity checks.
- `cases/<case>/`: `goal.png` (target-marked reference), `goal_raw.png` (unannotated reference), `f1.png` through `f6.png` (chronological history), and `metadata.json`.
- `frames/<episode>/<frame>.png`: unique unannotated frames.

Duplicate images are hard links within the package. It does not depend on original videos or another dataset. All recorded image paths are relative to the dataset root.

The supplied README states that original video/dataset paths, external build provenance, source row references, and auxiliary clarity scores were omitted. No model-generated annotations or predictions are included. Original frame indices and timestamps do not imply a dependency on the videos.

## Case schema

- `case_id`, `episode`: case identifier and episode grouping.
- `label`, `label_name`: action label.
- `paths`: seven task images, target-marked goal first followed by six chronological history frames ending at now.
- `clean_paths`: corresponding unannotated images.
- `render_dir`, `image_roles`: rendering directory and image roles; exact value structure remains uninspected.
- `image_size_wh`: width 1065, height 1050 pixels.
- `goal_xy`: target pixel coordinates measured from the top-left.
- `source_frames`, `source_times_s`: aligned with the seven images.
- `history_offsets_s`: six times relative to now; spacing may vary.
- `depth_relative_mm`: six depths relative to current depth; last value is zero.

The goal image is a separate reference, NOT a seventh chronological history step. The proposal's “artery map” may refer to this image; that correspondence is unconfirmed.

Owner-reported first-record type inspection matches the listed fields: `case_id`, `episode`, `label_name`, and `render_dir` are strings; `label` is an integer; the remaining listed fields are lists. This checks first-record types only, not all records, list lengths, or image contents. Owner-reported summaries confirm 416 records and 416 unique `case_id` values.

## Preprocessing and splits

No model preprocessing or evaluation protocol has been approved or implemented. Owner-reported structural inspection confirms `splits.json` is a list of five objects, each with `fold` (integer), `train`, `validation`, and `test` (lists).

Partition sizes by object position (not the uninspected `fold` value):

- Object 0: train 267, validation 80, test 69.
- Object 1: train 271, validation 73, test 72.
- Object 2: train 270, validation 75, test 71.
- Object 3: train 263, validation 72, test 81.
- Object 4: train 241, validation 52, test 123.

Owner-reported validation using zero-based positions in `cases.jsonl` passes for all five objects: indices are valid, every record appears exactly once across train/validation/test, and all three partition pairs have zero shared episodes. This supports row-position indexing and verifies coverage and episode separation under that interpretation. Memberships do not all match `case_id` values. The supplied README does not explicitly define indexing, and the `fold` field name alone does not establish standard five-fold cross-validation or the intended reporting protocol.

Pending verification on Perlmutter:

- Agree on the evaluation/reporting protocol and check cross-fold relationships if needed; within-split coverage and episode separation have passed under row-position indexing.
- Compare manifest, JSONL, and split definitions; check full image decoding and visual/content correctness if needed. Metadata lengths, path safety, referenced image availability, and PNG header dimensions have passed the checks below; full metadata alignment remains unverified.

## Image and metadata checks

Owner-run read-only checks on Perlmutter passed across 416 cases and 5,543 unique referenced image paths:

- Expected list lengths: seven paths, clean paths, image roles, source frames, and source times; six history offsets and depths; two goal coordinates and image dimensions.
- Last history offset and relative depth equal zero.
- Referenced paths are relative, resolve inside the dataset root, and are readable.
- PNG signatures and header dimensions match 1065 × 1050.
- 5,127 referenced paths have filesystem link counts greater than one, confirming hard links are present. This does not establish that every intended hard link was preserved.

Reported errors: none. These checks do not decode full images, verify visual content, compare all records with the manifest, or establish semantic alignment of metadata and frames. The path count is not a count of unique image contents or filesystem inodes.

Proposed preprocessing and loader shapes are recorded in `APPROACHES.md`; they are suggestions, not dataset requirements.

## Verification log

- 2026-10-30 — Dataset validation [completed for stated checks; owner-run on Perlmutter]. Confirmed 416 unique cases, 120 episodes, labels 0:157 and 1:259. All five supplied split objects pass index bounds, exact once-only coverage, and zero shared episodes between partitions under zero-based record indexing. Across 5,543 referenced image paths, metadata-length, zero-ending depth/time, path safety/readability, and PNG header/dimension checks reported no errors; 5,127 paths have multiple hard links. Supplied dataset README reviewed. No data changed. Remaining limits: full decoding, visual/content correctness, semantic metadata alignment, manifest consistency, and cross-fold evaluation protocol are not verified. Subsequent raw-reader tests and one-case decoding/display are recorded in [Group 2 progress](grp2/PROGRESS.md); these do not establish full-dataset decoding or semantic correctness. Do not repeat completed checks without a changed copy, code change, or concrete concern.

## Open questions

- What do the five splits mean, and which evaluation protocol should the team use?
- Does “artery map” mean the supplied goal reference?
- What correction format and review process should the team adopt?
