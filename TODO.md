# TODO

Working list of deferred work for the next datanavigator release. The
authoritative roadmap lives in
[`pn-specs/specs/datanavigator.md`](https://github.com/praneethnamburi/datanavigator)
(internal); this file is for items concrete enough to act on.

## 1.4.0 (deferred from the 1.2.0 audit; decoupled from the 1.3.0 decord swap)

The 1.4.0 theme is **under-the-hood performance**. The Qt swap (done)
brought ~1.84x speedup on the synthetic harness and ~1.10x on real
DUSTrack (see [`BENCHMARKING.md`](BENCHMARKING.md)). The real-DUSTrack
saving is dominated by costs we haven't touched yet -- the items below
chase those.

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

### Further perf wins -- to chase in a future session

The Qt swap captured the widget-side cost (~13 ms / frame), but real
DUSTrack's per-frame budget is dominated by other things that weren't
touched. Ordered by expected impact:

- **Blit-mode rendering for the imshow + annotation Line2D traces.**
  matplotlib's blitting API (`canvas.copy_from_bbox` /
  `canvas.restore_region` / `ax.draw_artist`) re-rasters only the
  artists that changed instead of the full canvas. For DUSTrack, the
  imshow + ~7 annotation lines update per frame; the axes / titles /
  signal subplots don't. Estimated saving: tens of ms per frame on
  real DUSTrack. Substantial code change in `videos.py`,
  `pointtracking.py`, and `signals.py` -- needs a fixture-style
  setup_func / update_func split.
- **Video-frame pre-decoding / lookahead in `video_reader.py`.** For
  forward scrubbing, decode frames N+1..N+k in a worker thread while
  user is on frame N. Cache hit on the next arrow press, no PyAV
  call in the timing path. Estimated saving: another ~30-50 ms on
  real DUSTrack (depending on codec). Concurrency to think about
  carefully; share with the PyAV TOC cache shape from 1.3.0.

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
