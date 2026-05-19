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
| 1.5.0-fast-render Tier 2             | 36.0      | 36.0    | 37.2   | 27.8         | **3.94x** |
| 1.4.0rc1 / 1.1.0rc1 (rerun 2026-05-19, pia02 vid) | 38.1 | 38.7 | 43.2 | 26.2 | 3.71x |
| **1.4.0rc2 / 1.1.0rc2** (2026-05-19, pia02 vid)   | **46.7** | **47.1** | **50.5** | **21.4** | **3.03x** |

The bottom two rows were measured on `pia02_s001_007_LFA2.mp4` —
`g.video_list[0]` is now a pia02 video on this DLC project (which
grew to 120 videos). The rc1 rerun (38.1 ms) reproduces the published
1.5.0-fast-render Tier 2 baseline (36.0 ms) within ~5%, so the
videos are perf-equivalent and the rc1↔rc2 delta is directly
comparable.

**rc2 regression vs rc1, accepted as the cost of new features.**
The +8.6 ms / +22% jump from rc1 to rc2 is the price of the rc2
UX surface (workflow-grouped sidebar, EnhanceWidget with two sliders
+ None/Auto, ConfirmOverlay, save-on-close guard, layer-lifecycle
buttons, statevars promoted from QLabel overlay to full Qt widget
with dropdown/toggle controls). Probe 11 ran on both branches with
`fast_render=False` to isolate the dnav-side segments: cache-keyed
segments (`annotation_visibility` 0.28 → 0.33 ms, `frame_marker`
0.02 → 0.02 ms) confirm the `_TrackedFrameDict` mutation guard is
NOT a contributor. The cost lives in **`update_assets`** (1.38 → 3.01
ms; sidebar grew from a flat list to five workflow groups + the new
EnhanceWidget), **`statevars_display`** (0.20 → 0.51 ms; Qt
widget replaces QLabel overlay), and the **Qt raster drain on the
fast_render path** (the bigger sidebar + statevars widget cost extra
paint events drained in `process_events`). Tradeoff judged worth
it 2026-05-19 — rc2 ships at ~21 fps which is still well above the
interactive threshold and the UX value bought is high.

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

Full probe-11 segment breakdown, 105 frames on the interosseous
video (medians, ms). Segments in call order inside
`VideoPointAnnotator.update()` (+ `VideoBrowser.update()` body
inlined for finer attribution); `process_events` is timed
separately outside the update body and is the actual canvas
rasterization (what blit would target).

| segment                  | 1.4.0-qt | + cache | delta   | what it does |
|--------------------------|---------:|--------:|--------:|--------------|
| annotation_visibility    |    15.84 |    0.30 |  -15.54 | scatter `set_offsets` + per-label trace `set_ydata` (cached: only on annotation revision change) |
| statevars_display        |     0.32 |    0.23 |   -0.09 | Qt overlay text push |
| frame_marker             |    18.71 |    0.02 |  -18.69 | `_frame_marker_{x,y}` `set_data` + FOI `set_data` + ylim recompute (cached: only on annotation revision change) |
| decode                   |     7.70 |    7.45 |   -0.25 | `self.data[idx].asnumpy()` — PyAV+TOC decode |
| image_process            |     0.95 |    1.00 |   +0.05 | DUSTrack `enhance_ultrasound_image` (CLAHE + gamma + brightness; see probe 10) |
| imshow_set_data          |     0.80 |    0.77 |   -0.03 | `self._im.set_data(processed)` |
| title                    |     0.17 |    0.16 |   -0.01 | `self._ax.set_title(self.titlefunc(self))` |
| update_assets            |     1.47 |    1.49 |   +0.02 | buttons / memslots / events display push |
| plt_draw                 |     0.00 |    0.00 |    0.00 | `plt.draw()` × 2 (scheduling only, no synchronous raster) |
| **subtotal (update body)** | **45.97** | **11.43** | **-34.54** | sum of the above |
| process_events (raster)  |    83.02 |   82.39 |   -0.63 | `app.processEvents()` drains Qt's paint queue (the actual rasterization) |
| **total (update + raster)** | **129.17** | **94.28** | **-34.89** | per-frame budget |

Process raster is now **87% of total** — every remaining big-win
item lives behind it (and probe 12 below shows it's not behind
blit either).

### Remaining headroom

1. ~~**Blit-mode rendering.**~~ **Probe 12
   (`tests/qt_learning/12_benchmark_blit_feasibility.py`) showed
   blit does NOT help on QtAgg with this figure size.** The
   matplotlib-blit pattern assumes raster cost lives in Agg
   rasterization, which is true on Agg / TkAgg / inline backends. On
   QtAgg with a ~1100x700 figure containing a 706x558 imshow + two
   trace axes, the dominant cost is the **Qt widget pixmap upload**,
   not the Agg raster. A single full-figure blit cost the same as
   per-axis blits because the per-axis bboxes covered roughly the
   whole figure area.

   Probe 12 measured top-line: 100.95 ms baseline (full draw)
   median -> 110.71 ms median with blit, i.e. blit was **slower**
   on this stack. Internal breakdown of `blit_render()`:

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

## 1.5.0 -- `fast_render` (Tier 2 Qt-native video pane)

The architectural change probe 13 surveyed was implemented in 1.5.0
as an opt-in second tier (`VideoBrowser(..., fast_render=True)`,
threaded through `VideoPointAnnotator` and `DUSTrack`). Tier 1
(`fast_render=False`, default) is unchanged.

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

### Probe 14 result

`tests/qt_learning/14_benchmark_fast_render.py` measured the same
real DUSTrack UI on the same `interosseous_pn24-x` video used by
probes 09/11/12/13:

| segment / total        | 1.4.0-qt + cache | 1.5.0 fast_render | delta |
|------------------------|-----------------:|------------------:|------:|
| update body            |            11.43 |             11.42 |  ~0    |
| process_events (raster + upload) |  82.39 |             24.42 | -57.97 |
| **total (median)**     |        **93.80** |         **35.97** | -57.83 |
| **fps (median)**       |         **10.7** |          **27.8** | +17.1  |

**Speedup: 2.6x over 1.4.0-qt + cache_quick_wins, 3.94x over
1.3.0 baseline.**

The plan's aspirational threshold was 25 ms / 40 fps (probe 13's
~17-22 ms prediction); we landed at 36 ms / 28 fps. The gap is in
process_events (24 ms vs predicted 6-12 ms). Diagnostic probes
during 14 development surfaced two contributors:

- **`constrained_layout=True` re-runs on every draw**: cost ~25 ms
  on a trace-only canvas where the layout solver runs over many
  Line2D objects. Tier 2 explicitly uses `constrained_layout=False`
  with a manual `subplots_adjust` to bypass this; without that
  change, fast_render measured 60-70 ms (1.4x speedup -- not the
  ship-worthy claim).
- **`canvas.draw()` is ~15 ms** even on a 1200x100 px trace canvas
  with mostly invisible Line2Ds. Hidden artists still pay some
  Agg-render overhead; the trace canvas keeps ~30 lines per axis
  (one per (label, annotation) pair), of which only ~3 are visible
  in any given frame. Reducing the line count would help; this is
  scoped for a future iteration.

The Qt-side pane was microbenched independently at ~0.6 ms / frame
(versus QLabel's 0.25 ms): the `QGraphicsView` path is well below
the matplotlib trace canvas's residual cost. There's no point
swapping to QLabel until the trace canvas drops below ~5 ms.

### Tier 2 ergonomic cost (1.5.0)

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

### DUSTrack UI -- 1.5.0-fast-render Tier 2 -- 2026-05-17 22:39:30

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 66.23 | 68.32 | 68.63 | 70.88 | 74.03 | 75.15 | 14.6 |

### DUSTrack UI -- 1.5.0-fast-render Tier 2 (trace-only canvas) -- 2026-05-17 22:43:19

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 64.81 | 71.05 | 71.66 | 81.18 | 84.92 | 87.60 | 14.1 |

### DUSTrack UI -- 1.5.0-fast-render Tier 2 (final) -- 2026-05-17 22:51:32

- datanavigator: `1.5.0-fast-render@52f5da1-dirty` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `1.5.0-fast-render@4f271b4-dirty` source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True
- N=185 (15 warmup discarded)

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 34.43 | 35.97 | 36.03 | 37.21 | 37.97 | 39.82 | 27.8 |

### DUSTrack UI -- 1.4.0rc1 / 1.1.0rc1 rerun -- 2026-05-19

- datanavigator: `HEAD@1e8fa51` (tag `v1.4.0rc1`) source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `HEAD@29775c0` (tag `v1.1.0rc1`) source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True (default)
- video: `pia02_s001_007_LFA2.mp4` (g.video_list[0], 36,715 frames)
- N=185 (15 warmup discarded)
- purpose: rerun of the rc1 baseline on the same env / video / harness
  as the rc2 measurement below, so the rc1↔rc2 delta is apples-to-apples.

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 35.85 | 38.13 | 38.68 | 43.15 | 48.12 | 49.16 | 26.2 |

### DUSTrack UI -- 1.4.0rc2 / 1.1.0rc2 -- 2026-05-19

- datanavigator: `1.4.0rc2@c5bcbd0` source `C:\dev\datanavigator\datanavigator\__init__.py`
- dustrack: `main@e8e9653-dirty` (a tiny dlcinterface.py edit committed
  shortly after as `a03877c`; not material to the per-frame budget)
  source `C:\dev\DUSTrack\dustrack\__init__.py`
- backend: QtAgg, qt_api=pyside6, qt_plat=default
- fast_render: True (default)
- video: `pia02_s001_007_LFA2.mp4` (g.video_list[0], 36,715 frames)
- N=185 (15 warmup discarded)
- delta vs rc1 rerun above: **+8.58 ms median (+22.5%), -4.8 fps median (-18.3%)**.
  Root-causing (see summary table commentary above + probe 11
  segment diff below) attributes the regression to the rc2 UX
  additions (workflow-grouped sidebar, statevars Qt widget,
  EnhanceWidget, layer-lifecycle buttons) and **not** to the
  `_TrackedFrameDict` mutation guard. Accepted 2026-05-19 as the
  cost of the new UX surface; gates the 1.4.0 / 1.1.0 final cut
  on no *further* regression beyond this point.

| min | median | mean | p95 | p99 | max | fps (median) |
|---|---|---|---|---|---|---|
| 44.69 | 46.71 | 47.09 | 50.50 | 53.53 | 61.70 | 21.4 |

### Probe 11 segment diff -- rc1 vs rc2 -- 2026-05-19

Ran `tests/qt_learning/11_profile_dustrack_update.py`'s logic with
`fast_render=False` (the canonical probe was written before
fast_render became default-on and crashes on `_im = None` when it's
on; the dnav-side segments measured are identical with or without
fast_render — fast_render only swaps the image pane). Same env,
same video, 185 measured frames each.

| Segment                  | rc1 ms | rc2 ms | Δ ms   | Notes |
|--------------------------|-------:|-------:|-------:|-------|
| annotation_visibility    |  0.28  |  0.33  |  +0.05 | flat — mutation guard NOT a contributor |
| statevars_display        |  0.20  |  0.51  |  +0.31 | new Qt statevars widget replaces QLabel overlay |
| frame_marker             |  0.02  |  0.02  |   0.00 | flat — cache invariant holds |
| decode                   |  7.69  |  7.74  |  +0.05 | flat — PyAV decode unchanged |
| image_process            |  0.99  |  0.00  |  -0.99 | rc2 default opens raw (clahe=1.0/gamma=1.0); bypass branch fires |
| imshow_set_data          |  0.77  |  0.90  |  +0.13 | small |
| title                    |  0.13  |  0.17  |  +0.04 | small |
| update_assets            |  1.38  |  3.01  |  +1.63 | sidebar grew: workflow groups + EnhanceWidget + new buttons |
| plt_draw                 |  0.00  |  0.00  |   0.00 | flat |
| process_events (raster)  | 81.69  | 83.07  |  +1.38 | small on mpl path; the bulk of fast_render-path regression lives here on the production path |
| **TOTAL (mpl path)**     | **93.40** | **95.90** | **+2.50** | mpl path: ~2.5 ms regression |
| (TOTAL fast_render path) | (38.13) | (46.71) | (+8.58) | for reference — production hot path |

The ~6 ms gap between the mpl-path regression (+2.5 ms) and the
fast_render-path regression (+8.6 ms) is most likely extra Qt paint
events drained in `process_events` on the fast_render path, caused
by the same expanded sidebar + statevars widget that show up in
`update_assets` and `statevars_display` on the mpl path. Could be
recovered by caching the per-frame asset push or auditing
paint-event frequency for the new widgets — deferred to 1.5.0's
`fast_traces` work or a `1.x.x` patch if the workflow ever runs
into the headroom.
