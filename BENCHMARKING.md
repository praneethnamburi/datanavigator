# datanavigator -- responsiveness benchmarks

Per-frame `browser.update()` wall-time stats, tracked across releases.
Two harnesses:

- [`tests/qt_learning/08_benchmark.py`](tests/qt_learning/08_benchmark.py)
  -- **synthetic** DUSTrack-shaped widget load on top of one `imshow`.
  Isolates the widget-rendering cost we touched in 1.4.0.
- [`tests/qt_learning/09_benchmark_dustrack.py`](tests/qt_learning/09_benchmark_dustrack.py)
  -- **real DUSTrack UI** via DLCProject on the interosseous_pn24-x
  video; the whole stack (video decode + annotation traces + signal
  subplots + widgets). Closer to user-felt responsiveness.

## Summary

### Synthetic (b4 env, PyQt5 5.15.2, matplotlib 3.7.3, 3 runs averaged)

| Release / Branch       | Median ms | Mean ms | p95 ms | fps (median) | Speedup |
|------------------------|-----------|---------|--------|--------------|---------|
| 1.3.0 (master)         | 28.5      | 29.1    | 34.0   | 35.1         | 1.00x   |
| 1.4.0-qt               | 15.5      | 16.1    | 19.4   | 64.5         | **1.84x** |

### Real DUSTrack UI (dlc env, PySide6 6.4.2, matplotlib 3.8.4, interosseous_pn24-x video)

| Release / Branches              | Median ms | Mean ms | p95 ms | fps (median) | Speedup |
|---------------------------------|-----------|---------|--------|--------------|---------|
| 1.3.0 (datanavigator+dustrack)  | 141.6     | 141.7   | 144.1  | 7.1          | 1.00x   |
| 1.4.0-qt (both)                 | 128.6     | 128.8   | 130.8  | 7.8          | **1.10x** |

### How to read the gap between the two

The synthetic harness's ~13 ms gap (28.5 -> 15.5) and the real DUSTrack
harness's ~13 ms gap (141.6 -> 128.6) are essentially the same absolute
saving. Both come from the same Phase 2 + Phase 3 change: every
persistent non-plot widget (memoryslot display, state variables,
buttons) moved off the matplotlib canvas and into native Qt widgets
that don't participate in the per-frame raster.

In real DUSTrack the ~13 ms is a smaller fraction of the total update
cost because each frame also costs: video decode, ~7 annotation
Line2D updates, 2-3 signal subplots, title relayout. None of those
were touched in 1.4.0. So real-world speedup is ~10% / +0.7 fps,
not the 1.84x the synthetic shows in isolation. Held-down arrow
scrubbing will feel marginally smoother, not transformatively so.

Further wins on real DUSTrack require touching the actual plot update
path. Two candidates, in expected-impact order, tracked in
[`TODO.md`](TODO.md) for follow-up in 1.4.0 (which is themed around
performance — the items below stay in scope):

1. **Blit-mode rendering** for the imshow + ~7 annotation Line2D
   traces. matplotlib's `canvas.copy_from_bbox` / `restore_region` /
   `ax.draw_artist` re-rasters only the artists that changed, skipping
   axes / titles / signal subplots when they didn't. Estimated saving:
   tens of ms per frame on real DUSTrack.
2. **Video-frame pre-decoding / lookahead** in `video_reader.py`. For
   forward scrubbing, decode frames N+1..N+k in a worker thread while
   user is on frame N. PyAV decode disappears from the timing path
   on cache hit. Estimated saving: another ~30-50 ms.

## Methodology

- Each iteration: set `_current_idx`, call `browser.update()`, call
  `QApplication.processEvents()` to drain Qt's paint queue. Time the
  pair. `processEvents()` is essential because matplotlib's
  `canvas.draw_idle()` only *schedules* a repaint; timing `update()`
  alone misses the actual rasterization that happens on Qt's next idle
  tick.
- Headless via `QT_QPA_PLATFORM=offscreen` so the script is shell-
  runnable. Qt still rasterizes; only the OS compositor / window
  swap is skipped. The dominant cost is the Agg rasterization (CPU
  side), which runs unchanged under offscreen.
- First 10 iterations discarded as warmup (font cache, lazy Qt widget
  realization, first-touch canvas allocations).
- Known-cosmetic Qt warnings (`QFontDatabase: Cannot find font
  directory`, offscreen plugin `does not support raise()` /
  `propagateSizeHints()`) are filtered via a Qt message handler so
  benchmark output is clean.

## How to run

```powershell
# Quick local benchmark, printed to stdout:
C:\Users\praneeth\anaconda3\envs\b4\python.exe `
    C:\dev\datanavigator\tests\qt_learning\08_benchmark.py

# Same, with a results block appended to this file under a label:
C:\Users\praneeth\anaconda3\envs\b4\python.exe `
    C:\dev\datanavigator\tests\qt_learning\08_benchmark.py `
    --record "1.4.0-qt rerun"
```

To compare a branch against the documented baseline: run with
`--record <label>` on each branch, the script appends a timestamped
block to *Raw results* below; eyeball the numbers against the
Summary table above.

## Hardware / environment

- Machine: Windows 11, the development workstation
- Conda env: `b4` (Python 3.10.13)
- Qt binding: PyQt5 5.15.2 via qtpy
- matplotlib: 3.7.3 with QtAgg backend

## Raw results

<!-- The benchmark script appends timestamped blocks below when run with
     --record LABEL. Newest results land at the bottom. -->

### 1.3.0 baseline -- 2026-05-17

- git: `master @ b71761d`
- N=190 (10 warmup discarded)
- 3 runs

| metric  | run 1 | run 2 | run 3 |
|---------|-------|-------|-------|
| median  | 28.24 | 28.53 | 28.83 |
| mean    | 28.98 | 29.18 | 29.14 |
| p95     | 34.11 | 34.45 | 33.32 |
| min     | 26.29 | 26.35 | 26.33 |
| max     | 42.43 | 36.93 | 38.97 |

### 1.4.0-qt -- 2026-05-17

- git: `1.4.0-qt @ 01e8cdb`
- N=190 (10 warmup discarded)
- 3 runs

| metric  | run 1 | run 2 | run 3 |
|---------|-------|-------|-------|
| median  | 15.45 | 15.57 | 15.49 |
| mean    | 15.98 | 16.22 | 16.15 |
| p95     | 18.68 | 19.81 | 19.76 |
| min     | 14.37 | 14.40 | 14.49 |
| max     | 21.42 | 24.89 | 25.37 |

### 1.4.0-qt smoke after warnings filter -- 2026-05-17 10:46:04

- git: `1.4.0-qt @ 01e8cdb-dirty`
- source: `C:\dev\datanavigator\datanavigator\__init__.py`
- version: `1.3.0`
- backend: QtAgg, qt_api=pyqt5, qt_plat=offscreen
- N=190 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 14.88 | 15.57 | 15.79 | 17.40 | 21.95 | 22.14 | 64.2 |

### DUSTrack UI -- 1.4.0-qt -- 2026-05-17 10:51:35

- datanavigator: `1.4.0-qt@fc1b6f3-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.4.0-qt@9ba5e7f` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 126.82 | 128.59 | 128.80 | 130.77 | 134.75 | 134.75 | 7.8 |

### DUSTrack UI -- 1.3.0 baseline (master) -- 2026-05-17 10:52:37

- datanavigator: `master@b71761d-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@dc0cff3` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 139.34 | 141.28 | 141.51 | 144.03 | 147.08 | 147.08 | 7.1 |

### DUSTrack UI -- 1.3.0 baseline (master) run2 -- 2026-05-17 10:53:23

- datanavigator: `master@b71761d-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@dc0cff3` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=90 (10 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 140.15 | 141.83 | 141.98 | 144.10 | 146.46 | 146.46 | 7.1 |
