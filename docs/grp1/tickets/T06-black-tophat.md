# T06 - Fast black top-hat filter

**Status:** Todo
**Code written:** laptop (no execution)
**Tested:** Perlmutter, by the owner
**Depends on:** T01
**Files:** `preprocessing.py` (or a small new module if it keeps `preprocessing.py` readable), `tests/test_preprocessing.py`

## Why

The wire is dark and about 3-4 px wide.
A black top-hat with a square window wider than the wire keeps thin dark structures and removes smooth anatomy:
`black_tophat(x) = closing(x) - x`, where `closing = erode(dilate(x))`.
A one-off prototype on invented images found Pillow's `MaxFilter`/`MinFilter` at about 260 ms per case for a 256 crop, too slow, while a separable max version took about 16 ms.
Those numbers are only a rough guide; T12 measures the real cost on Perlmutter.

## Scope

- Implement `black_tophat(image, size)` on a float tensor of shape `[H, W]` (or `[N, 1, H, W]`), returning the same shape with non-negative values.
- Use `torch.nn.functional.max_pool2d` with stride 1 for dilation and `-max_pool2d(-x)` for erosion, so it also runs on GPU.
- Pad with edge replication (`F.pad(..., mode="replicate")`), not zeros, so image borders do not create fake dark edges.
- `size` must be an odd integer of at least 3; raise `ValueError` otherwise.

## Out of scope

- Choosing the default `size` (T13 picks it visually; start with 11).
- Non-square or disk-shaped windows.

## Acceptance criteria

Tests on invented images:

- A 4 px wide dark line on a smooth gradient background: the top-hat is high on the line and near zero on the background.
- A dark blob wider than `size` is mostly removed.
- A constant image returns all zeros.
- Output shape equals input shape; the input tensor is not modified.
- Results match a slow reference (nested-loop or Pillow-based closing) on a small random image.

## Run on Perlmutter (owner)

```bash
module load pytorch/2.13.0
cd "${REPO_DIR:?}" && git pull
python3 -m unittest tests.test_preprocessing -v
```
