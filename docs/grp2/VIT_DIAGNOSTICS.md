# ViT feature diagnostics — Group 2

## Purpose and current status

The pretrained DINOv2 encoder already returns a 384-value whole-image summary for the separate goal and each history frame. CPU execution and grouping checks passed previously. It has not learned this project's ADVANCE/RETRACT task. It receives images only, not depth, time, labels or an instruction identifying the goal. The proposed downstream GRU/classifier would learn task relationships; useful visual information must survive the encoder for that to work.

The next bounded step requested on 2026-11-02 is to inspect whether the summaries respond to the marker and wire-tip region. Cells V1–V6 in `data_inspect.ipynb` implement this workflow. Local checks cover notebook structure and Python syntax only; these new cells have not run with pretrained weights or real data. There are no new diagnostic results yet.

Sensitivity is weaker evidence than understanding: a response to marker removal does not establish target location encoding or destination semantics. Wire occlusion does not establish a wire detector. There is no validated numerical pass/fail threshold.

## Perlmutter walkthrough

1. **Preserve remote work.** Save the existing remote notebook and make a separate backup on Perlmutter before any code transfer or pull. The laptop notebook cannot contain the remote edits/outputs. No commit, push or remote update has been performed by this change. Transfer/reconcile the new diagnostic cells through an authorized handoff; do not overwrite the sole remote notebook.
2. **Use the existing environment and compute access.** Start a notebook in an allocated compute session, with the existing scratch venv containing timm. Kernel selection does not allocate a GPU. The base module alone lacks timm. The coordinate plot also needs matplotlib; verify its import in the chosen kernel. Exact account/venv/allocation details are private and not established here. Do not install or recreate environments automatically. GPU is preferred; CPU execution is supported but this multi-image workflow should run on compute resources.
3. **Run Load data.** The existing cell expects the scratch dataset layout from README. If using another confirmed Perlmutter dataset root, set `data_dir` accordingly in your remote working copy. Use `partition="train"`; split position 0 is an inspection example, not a final evaluation decision.
4. **Run V1.** Choose training-case index 0 initially. Compare original/prepared goal and latest-history images. Check wire-tip and circle visibility after resizing to 518×518; no cropping is applied. If the tip is not identifiable, do not guess its location.
5. **Run V2 once.** Load/reuse the encoder and define single-image extraction plus distance measurements. Public weights may download into the existing cache. Images are never sent to an external inference service. The calls match `vit.py`'s backbone feature/head path and run without training.
6. **Run V3's first cell.** Inspect marked, clean and amplified difference previews. They must represent the same scene apart from the annotation. If confirmed, change `diag_pair_confirmed` to `True` in the following cell and run it. This measures repeat-image noise and marker-removal change. Do not interpret unmatched scenes as marker sensitivity.
7. **Run V4.** View the latest history frame with coordinate axes. Identify the tip and three non-wire controls, ideally with comparable brightness/texture. Coordinates are `(x, y)` in the prepared image, with origin at its top-left; they are not original-resolution coordinates.
8. **Edit and run V5.** Replace the four `None` entries with observed centers. Each 28×28 patch must lie within the resized image, not padding. The code clones the tensor, fills one patch with the normalization mean, displays the edit and compares its features to the original. Inspect the displayed patches to verify their placement.
9. **Read V6 and repeat cautiously.** Change the case index in V1 and rerun V1, V3–V5 for 3–5 varied training cases; reuse V2's encoder. Choose fresh coordinates for each case. Keep this exploratory; do not inspect test cases to guide model choices.

## Choosing the wire and comparison regions

**What control 1, 2 and 3 mean:** these are three comparison regions you choose that do not contain the wire, not extra models. Cover each separately with the same size square used for the wire tip. If covering any region changes the features substantially, a wire-tip change alone tells us little. A consistently larger change at the tip supports sensitivity to that region. Three comparisons reduce reliance on one unusual region; three is not a model requirement.

**How to choose coordinates:** in V4, read x horizontally (increasing right) and y vertically (increasing down), starting at (0, 0) in the top-left. Use the displayed 518×518 prepared image, not the original. Center the wire square on its visible tip. Choose three distinct non-wire regions with reasonably similar brightness/texture; avoid text, borders, padding and choosing only blank background. Keep centers at least 14 pixels inside the actual image so the entire 28×28 square fits. Replace each `None` in V5 with a numeric `(x, y)` pair. Inspect the covered-image previews: the tip square must cover the tip and comparison squares must avoid the wire. Adjust coordinates if necessary; skip a case if you cannot confidently identify the tip.

## Reading the measurements

- **Relative L2:** Euclidean distance between original and changed feature vectors, divided by the original vector's length. Measures relative feature change, not action probability.
- **Cosine distance:** one minus cosine similarity. Measures a difference in vector direction; zero means the same direction.
- **Repeat unchanged image:** should be approximately zero; unexpected differences warrant investigating execution/reproducibility first.
- **Marker removal:** larger than repeat noise supports marker sensitivity. It does not show whether the representation preserves marker position, or treats the interior as a destination.
- **Wire versus controls:** consistently larger effects for the tip region support sensitivity to that region. Background structure, artificial mask edges and distributed wire information confound interpretation; weak effects are inconclusive.

Do not compare absolute distances across different encoders as a quality ranking. A hypothetical later marker-relocation or patch-token probe may address spatial information more directly, but is not implemented or a prerequisite added here. No red-color threshold detector is used; the goal check uses the supplied clean counterpart.

## Recording and next decision

Keep real images, features, coordinates, identifiers and detailed outputs on Perlmutter. Share only permitted aggregate distances, case count, device/software versions, qualitative observations and limitations. Preserve your remote notebook before creating an output-free shareable copy.

After this bounded pass, discuss the frozen ViT–GRU design: depth/time scaling and per-frame inputs, separate goal fusion, ownership, development split, validation metrics and model selection. Freezing is a proposed baseline, not a demonstrated best choice. These probes alone cannot decide fine-tuning. GPU inference performance measurement, training and action evaluation remain separate outstanding work; dataset-wide extraction is not part of this workflow.
