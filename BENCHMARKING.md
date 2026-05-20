# datanavigator -- VideoBrowser per-frame benchmarks

Per-frame `browser.update()` wall-time stats on the synthetic
DUSTrack-shaped harness, plus the dnav-side render-pipeline lessons
(blit feasibility on QtAgg, fast_render Tier 2 architecture,
pre-decode rationale) that informed `VideoBrowser` design choices.

Production workflow numbers — real DUSTrack UI on the
`interosseous_pn24-x` pia02 video (probes 09 / 11 / 14, rc1↔rc2
deltas, cold-open) — live in
[`DUSTrack/BENCHMARKING.md`](../DUSTrack/BENCHMARKING.md). The split
mirrors the 1.5.0a1 package split: VideoBrowser stays in dnav and is
benchmarked here synthetically; the DUSTrack workflow that layers on
top is benchmarked there.

Harness:
[`tests/qt_learning/08_benchmark.py`](tests/qt_learning/08_benchmark.py)
— a synthetic DUSTrack-shaped widget load on top of one `imshow`.
Isolates the widget-rendering cost we touched in 1.4.0.

## Summary -- synthetic (b4 env, PyQt5 5.15.2, matplotlib 3.7.3, 3 runs averaged)

| Release / Branch       | Median ms | Mean ms | p95 ms | fps (median) | Speedup |
|------------------------|-----------|---------|--------|--------------|---------|
| 1.3.0 (master)         | 28.5      | 29.1    | 34.0   | 35.1         | 1.00x   |
| 1.4.0-qt               | 15.5      | 16.1    | 19.4   | 64.5         | **1.84x** |

The synthetic harness exists to isolate the dnav widget + canvas
side of the budget. For the user-felt total (image + scatter +
trace lines + signal subplots + Qt sidebar widgets) on the real
DUSTrack UI, see the dustrack doc.

## Render-pipeline lessons (dnav-side)

These shaped VideoBrowser's design even though the headline numbers
were measured through the DUSTrack UI (= dustrack-side). Captured
here because the *feature surface* lives in dnav.

### Blit-mode rendering doesn't help on QtAgg with large figures

Probe 12 (`DUSTrack/tests/qt_learning/12_benchmark_blit_feasibility.py`)
showed blit does NOT help on QtAgg with the VideoBrowser figure
size. The matplotlib-blit pattern assumes raster cost lives in Agg
rasterization, which is true on Agg / TkAgg / inline backends. On
QtAgg with a ~1100x700 figure containing a 706x558 imshow + two
trace axes, the dominant cost is the **Qt widget pixmap upload**,
not the Agg raster. A single full-figure blit cost the same as
per-axis blits because the per-axis bboxes covered roughly the
whole figure area.

Probe 12 measured top-line: 100.95 ms baseline (full draw) median
→ 110.71 ms median with blit, i.e. blit was **slower** on this
stack. Internal breakdown of `blit_render()`:

| op                          |   ms | notes |
|-----------------------------|-----:|-------|
| `canvas.restore_region`     | 0.29 | per-axis, cheap |
| `draw_artist` of light artists | 0.49 | frame_marker / FOI / scatter combined |
| `draw_artist` of imshow (Agg) | 13.63 | Agg rasterization of the 706×558 RGB pixmap |
| **`canvas.blit(bbox)`**     | **83.39** | **Qt widget pixmap upload — the dominant cost** |
| `flush_events`              | 0.50 | cheap |
| reference: `canvas.draw()` (synchronous, no blit) | ~80 | for comparison |

`canvas.blit()` and `canvas.draw()` cost the same because both
push the same total pixel area to the Qt widget. The 1.4.0-qt
widget swap (Phase 2/3) already captured the part Qt-side widget
movement could capture; the remaining cost is the canvas pixmap
upload itself.

The actual operations that *could* reduce raster cost on QtAgg
are architectural — either an OpenGL-backed canvas (matplotlib
has experimental backends; not robust on Windows) or rendering
the imshow outside matplotlib (`QGraphicsView` / `QOpenGLWidget`
/ raw `QLabel(QPixmap)`, keeping matplotlib only for the traces).
Both are sized like a 2.0.0 from-scratch Qt rewrite, not
under-the-hood 1.4.0 — except for the targeted fast_render Tier 2
described below.

### Video-frame pre-decoding / lookahead -- declined

Investigated 2026-05-17 and **declined**. The accuracy story is
fine (`PyAV.to_ndarray('rgb24')` allocates a fresh ndarray and is
deterministic, so a single-worker-thread design with serialized
decode is provably correct), but the win is only ~7.7 ms / frame
on the production video — ~8% of the post-cache_quick_wins budget
— and not worth the concurrency surface (lock contention on
backward jumps, worker lifecycle on figure close, byte-equivalence
regression tests). Reopen only if rendering pipeline changes
shift decode into a meaningful fraction.

## 1.5.0 fast_render Tier 2 -- architecture

The architectural change probe 13 surveyed was implemented in 1.5.0
as an opt-in second tier of `VideoBrowser`
(`VideoBrowser(..., fast_render=True)`, threaded through
`VideoPointAnnotator` and `DUSTrack` downstream). Tier 1
(`fast_render=False`, default) is unchanged. Production timing on
the DUSTrack UI lives in
[`DUSTrack/BENCHMARKING.md`](../DUSTrack/BENCHMARKING.md) -- this
section captures the dnav-side architecture.

In Tier 2:
- The video frame renders to a `QGraphicsView` + `QGraphicsPixmapItem`
  parented to the QMainWindow's central widget, above the matplotlib
  canvas. `QImage(rgb)->QPixmap.fromImage->setPixmap` per frame.
- The annotation scatter renders to a `QGraphicsItemGroup` of
  per-marker `QGraphicsEllipseItem`s, wrapped by `_QtScatterArtist`
  (a duck-typed match for the matplotlib `PathCollection` subset
  `VideoAnnotation` calls).
- Mouse pick / button-press over the image region routes through
  `_QtPickAdapter` (an `eventFilter` on the view's viewport) which
  fires synthetic mpl-shaped events; the existing `pick_event` /
  `button_press_event` callbacks in pointtracking.py work unchanged.
- The matplotlib canvas covers only the trace region (figure size
  reduced to `(12, 3)` in Tier 2), so its per-frame raster cost is
  much smaller than the full `(12, 8)` Tier 1 figure.

### Tier 2 ergonomic cost

The image region is no longer an mpl `Axes`, so users who relied on
the Tier-1 ergonomic of adding ad-hoc mpl overlays (e.g.
`self._ax_image.plot(...)` in a subclass override) get neither the
overlay nor an error: `_ax_image` is the `_QtImagePane` instance.
The audit before the 1.5.0 plan confirmed no portfolio code does
this today; DUSTrack's `_apply_dark_theme` was the only external
touch on the image axis (`_ax_image.set_facecolor(ax_color)`), and
it now routes through `VideoPointAnnotator.set_image_background_color`
which dispatches per tier. Forward-looking constraint: subclassing
with `fast_render=True` requires Qt-side overlays for image-region
augmentations.

## Methodology (synthetic harness)

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

## Raw results -- synthetic

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
