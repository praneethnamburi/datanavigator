# TODO

Working list of deferred work for the next datanavigator release. The
authoritative roadmap lives in
[`pn-specs/specs/datanavigator.md`](https://github.com/praneethnamburi/datanavigator)
(internal); this file is for items concrete enough to act on.

## 1.4.0 (deferred from the 1.2.0 audit; decoupled from the 1.3.0 decord swap)

The 1.4.0 theme is **under-the-hood performance**. The Qt swap (done)
brought ~1.84x speedup on the synthetic harness and ~1.10x on real
DUSTrack. The cache_quick_wins layer (done) added another 35 ms
saving on real DUSTrack, taking the cumulative speedup to **1.51x**
(141.6 -> 93.8 ms median, 7.1 -> 10.7 fps). See
[`BENCHMARKING.md`](BENCHMARKING.md). The next big win is blit-mode
rendering against the remaining 82 ms canvas raster.

### Done in this release (on `1.4.0-qt` branch)

- ~~**Qt under-the-hood swap.**~~ Phases 1-3 landed: `_qt_window`
  discovery, TextView -> QLabel overlay (Phase 2), Buttons ->
  QPushButton in QToolBar (Phase 3). Soft Qt mode -- mpl fallback
  unchanged. `copy_to_clipboard` migrated PySide2 -> qtpy.
- ~~**add_separator API + TextView _pos propagation** (Phase 4a).~~
  Promotes DUSTrack's mpl-internals-reaching `_add_dummy_button`
  hack to a real cross-backend Buttons API.
- ~~**Skip mpl spacer Axes on TextView Qt path** (Phase 4b).~~ The
  Axes was created but never used on the Qt path; rendered as an
  empty rectangle in the figure. DUSTrack interactive smoke surfaced
  it.
- ~~**Per-frame trace recompute cache (`cache_quick_wins`).**~~
  Profile probe 11 found `update_frame_marker` (~18.7 ms) and
  `update_display_trace` (~15.8 ms) were rebuilding label-trace
  arrays + ylim per frame, even though their output depends only on
  annotation contents, not `_current_idx`. Added a `_revision`
  counter on `VideoAnnotation` that bumps on every `data` mutation;
  both code paths now cache on it. ~35 ms / frame saved, no API
  change. Regression test:
  `test_video_annotation_revision_bumps_on_mutation`.

### Further perf wins -- to chase in a future session

After the cache_quick_wins layer, ~87% of the remaining 94 ms / frame
is canvas raster (`processEvents()`). Ordered by expected impact:

- ~~**Blit-mode rendering for the imshow + annotation Line2D traces.**~~
  **Probe 12 (`tests/qt_learning/12_benchmark_blit_feasibility.py`)
  measured this and found blit does NOT help on QtAgg.** With a
  hand-coded blit of a live DUSTrack instance (animated dynamic
  artists, cached backgrounds, restore_region + draw_artist +
  canvas.blit), the per-frame budget went 100.95 ms -> 110.71 ms
  median (worse). Internal breakdown:

  ```
  restore_region       0.29 ms    (cheap, as expected)
  draw_artist_light    0.49 ms    (cheap dynamic artists)
  draw_artist_imshow   13.63 ms   (Agg raster of imshow)
  canvas.blit(bbox)    83.39 ms   <-- Qt widget pixmap upload
  flush_events         0.50 ms    (cheap)
  ```

  The dominant raster cost is the Qt buffer upload, not Agg
  rasterization. `canvas.blit()` pays the same upload cost that
  `canvas.draw()` pays (~80 ms reference) because both push the same
  pixel area. Per-axis blits did not help (bbox sum ~= figure bbox);
  a single full-figure blit cost the same.

  Eliminating raster cost on QtAgg requires architectural change
  (OpenGL canvas, or rendering the imshow outside matplotlib via a
  QGraphicsView / QOpenGLWidget / raw QLabel(QPixmap)). That work is
  sized like the spec's 2.0.0 from-scratch Qt rewrite, not
  under-the-hood 1.4.0. Tracked in
  [`BENCHMARKING.md`](BENCHMARKING.md).

- ~~**Video-frame pre-decoding / lookahead in `video_reader.py`.**~~
  Investigated 2026-05-17 and **declined**. The accuracy invariant
  ("pre-decoded frame == fresh-decoded frame") is achievable --
  PyAV's `to_ndarray('rgb24')` allocates a fresh ndarray each call
  and is deterministic, so a single-worker-thread design with
  serialized decode is provably correct. But the win is only ~7.7
  ms / frame (~8% of the post-cache_quick_wins budget), and the
  concurrency surface (lock contention on backward jumps, worker
  lifecycle on figure close, byte-equivalence regression tests)
  isn't worth that saving. Reopen only if rendering pipeline
  changes shift the per-frame budget so decode becomes a meaningful
  fraction.

After cache_quick_wins, the under-the-hood perf arc on the 1.4.0-qt
branch is **complete**. Cumulative win: 141.6 -> 93.8 ms median,
7.1 -> 10.7 fps, **1.51x speedup** over 1.3.0. Going further requires
architectural change (OpenGL canvas, QPixmap-backed imshow, or
2.0.0 from-scratch rewrite) -- separate, dedicated session.

### Unresolved -- low priority

- **PyQt6 vs PySide6 perceived hover+key inversion.** During Phase 4f-i
  investigation, user reported "hover + t works in dustrack1a1 (PyQt6
  binding) but not in dlc (PySide6 binding)" with Caps Lock confirmed
  off in both. But probe 20 (tests/qt_learning/20_raw_qt_vs_mpl_key.py)
  showed byte-identical raw Qt event sequences in both envs. Phase 4i's
  case-insensitive single-letter fallback in
  GenericBrowser._resolve_keypress masks any user-facing symptom, but
  the underlying perceived inversion has no data-supported explanation.
  Filed for completeness; reopen if it resurfaces with new evidence.

### Smaller wins (worth knowing about, defer if time-bound)

- **Cache last QLabel size in `QtTextOverlay._reposition`.** Skip the
  `move()` call when text dimensions haven't changed. ~<0.5 ms/update.
- **Skip `update_display()` when neither state nor text changed.**
  `super().update()` in concrete browsers re-push memoryslot /
  state-variable text on every frame even when unchanged. ~<0.5 ms.
- **Audit the 33 `plt.draw()` sites** for any not already coalescing
  (modern matplotlib makes `plt.draw == canvas.draw_idle`; if any
  call sites call `canvas.draw()` synchronously, those could be
  switched to `draw_idle`). Likely zero left; worth confirming.

### Carry-over items not directly perf-related

- **Migrate to `src/datanavigator/` layout.** Convention compliance.
  Rides the perf work naturally since both touch lots of files.
- **Split `pointtracking.py` (1853 LOC)** into annotator-UI /
  annotation-IO / DLC-interop pieces. Rides the blit work above,
  which will rearrange `pointtracking.py`'s update loop anyway.
- **Migrate `DUSTrack-shaped` paths to `dustrack.VideoAnnotation`.**
  Grep anchor: `DUSTrack-shaped` (5 occurrences in
  `pointtracking.py`, all on `VideoAnnotation`). Leaves datanavigator
  modality-agnostic and gives DUSTrack the DLC story end-to-end.
- **Full type-hint sweep** on `assets.py`, `videos.py`, `utils.py`,
  `signals.py`, `core.py`, `__init__.py`, `examples.py`,
  `opticalflow.py` -- the 1.2.0 audit only modernized files it was
  already touching.

### Axis-limit polish (from the 1.3.1 audit)

A focused audit of axis-limit changes (`set_xlim/ylim`, `autoscale`,
`relim`, `fill_between`, `Grouper`) accompanied the 1.3.1 fix for
`Event._get_ylim` (events.py:609, xdata vs ydata). Three further
items surfaced -- none are bugs at the user-visible level today,
all are real correctness / forward-compat improvements that
belong on the Qt-branch surface (which is already in motion). None
were folded into 1.3.1 because they are either behavior changes
(not patch-release material) or live on files already moved on
this branch.

- **`videos.py:295-298` `VideoPlotBrowser` playhead tick is y-clipped.**
  The vertical playhead is drawn as a `Line2D` from `y=ylim[0]` to
  `y=ylim[1]` captured at setup time. Two consequences: (1) if the
  user zooms in y or the signal y-range grows, the playhead is
  clipped to its original ydata range; (2) the tick's own ydata
  caps `dataLim`, so subsequent `relim()` can't shrink past it.
  Fix: `this_ax.axvline(0, color='k')` instead of
  `this_ax.plot([0,0], ylim, "k")`. Behavior change (playhead now
  spans the current axis bounds, not the setup-time bounds).
  Tier 2's separate (12,3) trace canvas will rework this region
  anyway -- worth fixing in the same pass.
- **`components.py:152-156` `ComponentBrowser` signal separators are
  static at setup ylim.** Same pattern as the playhead tick -- the
  per-signal separator lines are drawn at `[x_pos]*2, this_ylim`
  with `this_ylim` captured once. Fix: `axvline(x_pos, color='k',
  linewidth=0.2)`. `components.py` is unchanged on this branch,
  so it's a clean edit. Behavior change (separators follow zoom).
- **`core.py:158-170` `reset_axes` uses deprecated `SubplotBase`
  filter, and `relim()` misses collections on older matplotlib.**
  `isinstance(ax, maxes.SubplotBase)` was deprecated in mpl 3.7
  and is on track for removal; the over-restrictive filter also
  drops inset and colorbar axes that callers might reasonably
  want included. `relim()` ignored `PolyCollection` (i.e.
  `fill_between` artists -- e.g. fill-display events) until
  recent mpl. Fix: drop the `SubplotBase` filter (use
  `ax.get_subplotspec() is not None` only if exclusion is
  intentional); decide the matplotlib floor for this branch.
  Pair with the floor question that's already pending for the
  Qt rework.
- **`reset_axes` hard-codes autoscale -- no hook for browser-supplied
  limits (tackle before the next release).** `reset_axes` (the `r`
  key + the init call) always `relim()` + `autoscale()`. A browser
  that wants *explicit / robust* per-view limits has no clean way in:
  it must **override `reset_axes` wholesale** (which then loses the
  collection-walk + SubplotBase handling above). Two real cases hit
  this in `pn-projects/projects/walking/review.py`: (a) annotation
  spans drawn with `axvspan` (axes-fraction y) pin autoscale to the
  current ylim, so it sticks/accumulates across items; and (b) a
  spiky per-beat RMSSD trace whose `max()` blows the axis up, where
  the meaningful frame is a robust statistic, not the data extent.
  The walking reviewers worked around it by (1) a non-polluting
  `Polygon`-via-`add_artist` span helper and (2) overriding
  `reset_axes` to re-apply explicit limits. Proposal: give
  `reset_axes` a delegation hook -- e.g. an optional
  `fit_view()` / `_reset_limits` callback (or a per-axes
  "autoscale vs explicit" policy) -- so `r` / init can call the
  browser's limit function instead of forcing autoscale, and
  annotations can be marked non-autoscaling. Would let the walking
  reviewers drop both workarounds.

## Nice-to-have

- `components.py` — simplify the verbose palette idiom
  `plt.get_cmap("tab20")([np.r_[0:1.5:0.05]])[0][:, :3]` to
  `plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]`.
- `pointtracking.py:861-887` — `rstc_paths` dict in
  `check_labels_with_lk` is populated by slice-assignment but never
  returned. Dead-but-populated state; safe removal requires deleting
  both the init at line 861 and the fill at line 885.
- Suppress the ~80 `FigureCanvasAgg is non-interactive` UserWarnings
  in the test suite via a `pytest` `filterwarnings` entry (the
  warnings come from `plt.show(block=False)` calls in production
  code that are intentional in interactive mode and inert under
  Agg). Cleaner alternative: have the production code skip
  `plt.show` when the backend is non-interactive.

## Stretch

- Retire `pn-utilities/pntools/gui.py` re-export shim once all callers
  migrate to direct `import datanavigator`.
- Coverage report wired into CI as an informational artifact (don't
  gate releases on it — per `pn-specs/CONVENTIONS.md`).
- Documentation: cookbook of common UI extension patterns for
  `PlotBrowser` (wobble's tremor editors are the canonical worked
  example; right now those patterns live only in the spec).
