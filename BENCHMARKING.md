# datanavigator -- responsiveness benchmarks

Per-frame `browser.update()` wall-time stats, tracked across releases.
The harness is [`tests/qt_learning/08_benchmark.py`](tests/qt_learning/08_benchmark.py).

## Summary

The synthetic-frame harness mimics DUSTrack-shaped visual complexity:
one `imshow` + memoryslot display + 2 state variables + 8 buttons.
200 frames, 10 warmup discarded. Run on b4 conda env (PyQt5 5.15.2,
matplotlib 3.7.3, Windows 11), 3 runs averaged.

| Release / Branch       | Median ms | Mean ms | p95 ms | fps (median) | Speedup vs baseline |
|------------------------|-----------|---------|--------|--------------|---------------------|
| 1.3.0 (master)         | 28.5      | 29.1    | 34.0   | 35.1         | 1.00x (baseline)    |
| 1.4.0-qt               | 15.5      | 16.1    | 19.4   | 64.5         | **1.84x**           |

Source of the 1.4.0 win: Phase 2 (TextView -> QLabel overlay) and
Phase 3 (Buttons -> QPushButton in QToolBar) moved every persistent
non-plot widget *off* the matplotlib canvas. The per-frame canvas
raster only includes the actual plot (the imshow) now, not the ~11
widget axes that previously lived inside the figure.

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
