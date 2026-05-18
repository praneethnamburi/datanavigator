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

| Release / Branches                   | Median ms | Mean ms | p95 ms | fps (median) | Speedup |
|--------------------------------------|-----------|---------|--------|--------------|---------|
| 1.3.0 (datanavigator+dustrack)       | 141.6     | 141.7   | 144.1  | 7.1          | 1.00x   |
| 1.4.0-qt (both)                      | 128.6     | 128.8   | 130.8  | 7.8          | 1.10x   |
| 1.4.0-qt + cache_quick_wins          | 93.8      | 93.8    | 95.9   | 10.7         | **1.51x** |

### How the cache_quick_wins layer fits in

Probe 11 (`tests/qt_learning/11_profile_dustrack_update.py`) broke
the 128.6 ms 1.4.0-qt baseline into per-segment cost and surfaced two
items inside the update() body — independent of canvas raster — that
were doing work proportional to the dataset, not the frame:

- `VideoPointAnnotator.update_frame_marker` was rebuilding
  `np.hstack([ann.to_trace(label).T for ann in ...])` and recomputing
  `nanmin` / `nanmax` per frame to set trace ylim. The output only
  changes when annotations / label / frames_of_interest change, never
  when only `_current_idx` moves. Cost: ~18.7 ms / frame on
  interosseous_pn24-x (36,715 frames × N labels).
- `VideoAnnotation.update_display_trace` was calling `to_trace(label)`
  and `set_ydata(...)` for every label per frame. Same observation:
  the trace contents are a pure function of the annotation data, not
  the frame index. Cost: ~15.8 ms / frame.

Fix: bump a `_revision` counter on every `VideoAnnotation.data`
mutation (`add` / `remove` / `add_at_frame` / `add_label` /
`sort_labels` / `sort_data` / `clip_*` / etc.) and cache both code
paths on it. Per-frame work for these two segments dropped to ~0.3 ms
combined. No API change; both methods accept the same arguments and
produce the same visual output.

Profile (probe 11) deltas, 105 frames on the same interosseous video:

| segment                  | 1.4.0-qt | + cache | delta   |
|--------------------------|----------|---------|---------|
| frame_marker             |  18.71   |   0.02  | -18.69  |
| annotation_visibility    |  15.84   |   0.30  | -15.54  |
| process_events (raster)  |  83.02   |  82.39  |  ~0     |
| total                    | 129.17   |  94.28  | -34.89  |

Process raster is now **87% of total** — every remaining big-win item
lives behind blit-mode.

### Remaining headroom

1. ~~**Blit-mode rendering.**~~ **Probe 12
   (`tests/qt_learning/12_benchmark_blit_feasibility.py`) showed
   blit does NOT help on QtAgg with this figure size.** The
   matplotlib-blit pattern assumes raster cost lives in Agg
   rasterization. On QtAgg with a ~1100x700 figure containing a
   706x558 imshow + trace axes, the dominant cost is the Qt widget
   buffer upload (`canvas.blit(bbox)` ~ 83 ms), not the Agg raster
   (`draw_artist(imshow)` ~ 13 ms; the smaller dynamic artists, ~0.5
   ms combined). A single full-figure blit cost the same as
   per-axis blits because the per-axis bboxes covered roughly the
   whole figure area. The 1.4.0-qt widget swap (Phase 2/3) already
   captured the part Qt-side widget movement could capture; the
   remaining cost is the canvas pixmap upload itself, which neither
   blit nor pre-decode can address.

   The actual operations that *could* reduce raster cost on QtAgg
   are architectural -- either an OpenGL-backed canvas (matplotlib
   has experimental backends; not robust on Windows) or rendering
   the imshow outside matplotlib (`QGraphicsView` /
   `QOpenGLWidget` / raw `QLabel(QPixmap)`, keeping matplotlib only
   for the traces). Both are sized like the spec's 2.0.0
   from-scratch Qt rewrite, not under-the-hood 1.4.0.

2. ~~**Video-frame pre-decoding / lookahead.**~~ Investigated
   2026-05-17 and **declined**. The accuracy story is fine
   (`PyAV.to_ndarray('rgb24')` allocates a fresh ndarray and is
   deterministic, so a single-worker-thread design with serialized
   decode is provably correct), but the win is only ~7.7 ms / frame
   on this video -- ~8% of the post-cache_quick_wins budget -- and
   not worth the concurrency surface (lock contention on backward
   jumps, worker lifecycle on figure close, byte-equivalence
   regression tests). Reopen only if rendering pipeline changes
   shift decode into a meaningful fraction.

The under-the-hood perf arc on `1.4.0-qt` is **complete** as of
2026-05-17. Cumulative win: **141.6 -> 93.8 ms median, 7.1 -> 10.7
fps, 1.51x speedup over 1.3.0.** Further gains require architectural
change (OpenGL-backed canvas, render imshow outside matplotlib via
QPixmap/QGraphicsView, or a 2.0.0 from-scratch Qt rewrite) -- the
remaining 82 ms is the Qt widget pixmap upload, which neither blit
nor pre-decode can address.

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

### DUSTrack UI -- 1.4.0-qt + cache_quick_wins -- 2026-05-17 20:10:47

- datanavigator: `1.4.0-qt@6c052ef-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.4.0-qt@9ba5e7f` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- N=85 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 91.72 | 93.77 | 93.80 | 95.85 | 99.24 | 99.24 | 10.7 |
