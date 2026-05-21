# Change Log
All notable changes to this project will be documented in this file.

## [1.5.0a1] - unreleased

Structural refactor: `pointtracking.py` (VideoPointAnnotator,
VideoAnnotation, VideoAnnotations) and `opticalflow.py` (lucas_kanade,
lucas_kanade_rstc) relocated to the `dustrack` package. `datanavigator`
narrows to **data-navigation primitives** -- browsers, asset managers,
events; `dustrack 1.2.0a1` owns its DeepLabCut workflow + the VPA
labeling UI end-to-end. All portfolio consumers of
`dnav.VideoAnnotation` / `VideoPointAnnotator` / `lucas_kanade*` flip
to `dustrack.*` in lockstep; no transitional re-exports.

### Changed
- **`VideoReader` now exposes `fname` and `name` attributes**
  (`datanavigator/video_reader.py`). Previously only the `utils.Video`
  subclass carried these (its `__init__` set them by hand); lifting
  them to the base lets callers that previously required a
  `utils.Video` instance — notably `dustrack.VideoAnnotation`, which
  reads `self.video.fname` / `self.video.name` — accept a plain open
  `VideoReader`. Backs the dustrack 1.2.0a2 cold-open win
  (`dustrack._DUSTrackBase.add_annotation_layers` shares the
  browser's single open reader across every annotation layer instead
  of opening the file once per layer — see the dustrack changelog for
  the av.open numbers).
- **`VideoReader(..., pix_fmt=None)` auto-detects monochrome sources**
  and decodes them directly as `(H, W)` gray ndarrays. `pix_fmt=None`
  (default) probes the source's encoded pixel format: `gray` /
  `yuvj400p` / etc. → gray decode path; anything else → the historical
  `rgb24` path. Pass `pix_fmt='gray'` to force gray decode on a color
  source (extracts Y via swscale, skipping chroma noise); pass
  `pix_fmt='rgb24'` to force the historical contract on a monochrome
  source. The vendored `PyAVReaderIndexed` gained the same kwarg.
  Threaded the parameter through `_open_with_cache` /
  `_build_and_cache_all` so the TOC sidecar still amortises across
  pix_fmt choices. On a 706x558 h265 monochrome clip, sequential
  decode goes from 2.28 ms/frame (`rgb24 + cvtColor(RGB2GRAY)` —
  current path) to 0.37 ms/frame (`to_ndarray('gray')` — new path),
  a **~6.16x speedup**; on dustrack's full `decode + gray()` loop
  including the postprocess short-circuit, end-to-end was **164 fps
  → 1886 fps** (11.5x) on the S-corpus
  `pia02_s001_011_RFA2_min1_15s_mono.mp4` test clip. Bench:
  `S:/_corpus/dustrack/mono_encode_bench_2026-05-21/`. Six new tests
  in `tests/test_video_reader.py` cover auto-detect on both source
  shapes, explicit overrides in both directions, and the invalid-
  value guard.

The relocation **preserves full git history** on both sides. On the
dustrack side, `git log --follow dustrack/pointtracking.py` traces
every dnav-era commit (rc1-rc2 perf work, label-aware y-refit,
`_TrackedFrameDict` mutation guard, etc.). On the dnav side, the
deletion shows up as a single coherent commit -- the files' history
is reachable through their new home in dustrack.

### Removed
- `datanavigator.pointtracking` (relocated to `dustrack.pointtracking`):
  `VideoPointAnnotator`, `VideoAnnotation`, `VideoAnnotations`,
  `_TrackedFrameDict`. Cross-package: VPA remains a subclass of
  `datanavigator.VideoBrowser`; asset managers, events, utils, `_qt`
  scaffolding stay in dnav.
- `datanavigator.opticalflow` (relocated to `dustrack.opticalflow`):
  `lucas_kanade`, `lucas_kanade_rstc`. The frame-list siblings
  `lucas_kanade_2` / `lucas_kanade_rstc_2` continue to live in
  `dustrack.postprocess`.

### Changed
- README + `__init__.py` module docstring restated: datanavigator is
  data-navigation primitives. The "Point tracking" + "Optical flow"
  sections of the module docstring are gone; `__all__` drops the
  relocated names. A new "See also" section in README points at
  DUSTrack as the canonical downstream consumer.
- `tests/test_fast_render_parity.py` narrowed to Tier-2 (Qt-pane +
  pick-adapter) tests only. The Tier-1 VideoAnnotation positional
  parity test relocated to `dustrack/tests/test_fast_render_parity.py`;
  both files preserve full pre-split history via `git log --follow`.

### Migration

Direct `import datanavigator as dnav` users of `VideoAnnotation` /
`VideoPointAnnotator` / `lucas_kanade*` need:

```python
import dustrack
# was: dnav.VideoAnnotation(json_path, video_path).to_signals()
ann = dustrack.VideoAnnotation(json_path, video_path)
signals = ann.to_signals()

# was: dnav.VideoPointAnnotator(video, ["pn", "buffer"])
# now use DUSTrack instead -- direct VPA instantiation is being
# phased out:
v = dustrack.DUSTrack(video, ["pn", "buffer"])

# was: dnav.lucas_kanade(video, start, end, init_points)
from dustrack import lucas_kanade
```

The `datanavigator` floor in `dustrack`'s `pyproject.toml` moves to
`>=1.5.0a1` so the two releases ship in lockstep.

### Pickle compatibility

A scan of `C:\data\_cache\` (352 files, 2026-05-19) returned zero hits
for `datanavigator.pointtracking` or `datanavigator.opticalflow` in
pickle headers, so no proactive `sys.modules` shim ships in this
release. If you hit `ModuleNotFoundError: No module named
'datanavigator.pointtracking'` when loading an older pickle (e.g. an
archive on a different drive that wasn't part of the cache scan),
register the import alias yourself before loading:

```python
import sys
import dustrack.pointtracking
import dustrack.opticalflow
sys.modules["datanavigator.pointtracking"] = dustrack.pointtracking
sys.modules["datanavigator.opticalflow"] = dustrack.opticalflow

import pickle
with open(path, "rb") as f:
    obj = pickle.load(f)
```

If this surfaces for multiple users, a follow-up 1.5.0a2 will fold the
alias into `datanavigator/__init__.py` directly.

## [1.4.0] - 2026-05-19

Major release shipping in two arcs from 1.3.1.

**Arc 1: Qt-native rendering and event plumbing** (pre-released as
1.4.0rc1, 2026-05-18). The video frame routes through `QGraphicsView`
+ `QPixmapItem` (`fast_render=True`; default off, DUSTrack 1.1.0
defaults on) and the annotation scatter through
`QGraphicsItemGroup`, leaving matplotlib responsible only for the
trace canvas. Wheel-zoom + middle-button pan/reset on the image pane;
`r` resets both image zoom and trace axes. Qt-native buttons
(`QPushButton` in a `QToolBar`) and statevariables overlay (`QLabel`)
replace matplotlib widgets; `qtpy` shim makes the Qt binding
pluggable. Per-frame trace-recompute cache (`_revision` counter on
`VideoAnnotation`) gates the trace recomputation. Result: 3.94×
real-DUSTrack speedup (36 ms median, 27.8 fps) over 1.3.x. Full rc1
notes:
<https://github.com/praneethnamburi/datanavigator/releases/tag/v1.4.0rc1>.

**Arc 2: Sidebar consolidation + data-layer cache invariant** (this
release).
(a) The button host swaps from `QToolBar` to a `QDockWidget` +
`QVBoxLayout` left column, (b) state-variables are promoted from a
read-only text overlay to interactive Qt controls (dropdowns / toggles
/ labels) mounted in that same column beneath the buttons, and (c) a
focused robustness subset folded from the originally-planned rc3 band:
inner per-label annotation dicts become a `dict` subclass that bumps
`_revision` on every mutator, making the rc1-era cache invariant a
data-structure invariant rather than reviewer-side discipline. (a) +
(b) share a single goal -- one column of controls for an interactive
UI, no scattered widgets across the QMainWindow's dock areas. (c)
closes the bug class that bit `check_labels_with_lk` and DUSTrack's
`copy_existing_annotations_from_overlay` in rc1 at the data layer.

The Added / Changed / Removed / Fixed sections below describe the
Arc 2 work; refer to the v1.4.0rc1 GitHub release for the granular
Arc 1 changelog.

### Added
- `_TrackedFrameDict` -- internal `dict` subclass that wraps every
  per-label inner annotation dict at `VideoAnnotation.data[label]`.
  Bumps `parent._revision` on `__setitem__` / `__delitem__` / `pop` /
  `popitem` / `clear` / `update` / `setdefault`. Turns the rc1-era
  "route writes through `add()` / `remove()` / `add_at_frame()`"
  discipline into a data-structure invariant: any future direct
  mutation through `ann.data[label][frame] = ...` correctly bumps
  revision without reviewer-side enforcement. Parent reference is a
  weakref to avoid lifetime extension. `__reduce__` drops the
  weakref for pickle round-tripping. The 1.4.0rc1 bypass sites
  (`check_labels_with_lk` and DUSTrack's
  `copy_existing_annotations_from_overlay`) were already fixed in
  rc2 via the public API; this guard makes the *next* bypass
  unrepresentable. Perf measured on 100k-frame synthetic data:
  bulk-load 14 ms / 1M entries (negligible), per-frame write
  sub-microsecond, 1M-entry read path 13% over bare dict -- well
  under the 36 ms / frame budget on `interosseous_pn24-x`.
- `VideoAnnotation.data` is now a property; setter calls
  `_wrap_label_dicts` which is idempotent (re-wraps only
  foreign-bound or bare inner dicts). Wholesale-reassignment sites
  (`sort_data`, `clip_labels`, `keep_overlapping_frames`,
  `keep_overlapping_continuous_frames`, `from_multiple_files`) were
  refactored to assemble in one shot and route through the setter --
  the previous per-label `self.data[label] = {...}` pattern would
  have written bare dicts into the wrapped outer container.
  `add_label` updated to use `self.data = {**self._data, label: {}}`
  for the same reason. `_revision = 0` hoisted before the first
  data assignment in `__init__` so the guard finds it during
  initial wrapping.

### Added
- `Buttons.register_style(name, styler)` + `Buttons.add(..., style_tag=)`
  / `Buttons.add_multi(specs with per-spec style_tag=)` + a new
  `Buttons.reapply_styles()` method. A `style_tag=` declares which
  registered styler should run on the freshly built button at the
  tail of `_finalize_button`. Resolution is two-tier: consumer
  registry first, then the dnav-shipped built-ins in the new
  `datanavigator.styles` module
  (`primary` / `secondary` / `neutral` / `warn`); unknown tags raise
  `KeyError` at add-time so typos / "forgot to register first" fail
  loud. Consumers can shadow a built-in by re-registering the same
  name -- DUSTrack rc2 does exactly this with its per-group
  `workflow / display / niche / utilities / swap` palette. Replaces
  the pre-rc2 pattern of consumer-side "collect buttons into lists,
  run a batch styling pass at end-of-setup"; each button now lives
  in one place (the `add` call) with its styling tag declared
  inline.
- `Buttons.add_multi(*specs)` -- N buttons side-by-side in a single
  row. Each `spec` is a dict of kwargs accepted by `Buttons.add()`
  (`text=`, `action_func=`, `type_=`, ...); the call returns the list
  of created buttons in spec order. On the Qt path a child QWidget
  with a QHBoxLayout hosts the row inside the buttons column (new
  `_qt.make_qt_button_row(figure, specs)` helper), so the row
  consumes exactly one vertical slot regardless of N. On the mpl
  fallback the row's width is divided evenly across N buttons at a
  shared y. A new lazy `Buttons._mpl_row_cursor` separates "row
  index used for y placement" from "button count" so `add_multi`
  (one row, N buttons) and `add_separator(style="double")` (one call,
  two slots) keep the mpl-path vertical rhythm correct; for
  single-button-only flows the cursor equals `len(self)` throughout,
  preserving pre-rc2 placement. First consumers: DUSTrack's
  "Trace: line / Trace: dot" and "Freeze plot axes / Unfreeze plot
  axes" pairs (reclaims one sidebar row each).
- `Buttons.add_separator(name=None, style="single")` takes a new
  `style` kwarg. `"single"` (default) is the pre-existing single
  sunken `QFrame.HLine`; `"double"` builds two stacked HLines (via
  the new module-level `_qt._make_qt_separator_widget` helper) for
  a stronger group break. `add_qt_separator(figure, style=...)`
  in `_qt.py` carries the corresponding parameter. mpl fallback in
  `assets.py` doubles the invisible-button slot count so the
  vertical rhythm matches.
- `_QtStatevarsWidget` appends a trailing double separator after
  the last state-variable row, marking the visual end of the
  statevars section in the left column. Motivated by DUSTrack
  asking for an "after state variables" group boundary 2026-05-18.
- `VideoPointAnnotator._add_default_buttons()` -- overridable hook
  for the post-`__init__` button installation (currently just the
  `Refresh UI` action). Subclasses with a hand-curated sidebar
  order (DUSTrack) override to a no-op and add the same buttons at
  the desired position. Pre-fix, `Refresh UI` was added inline at
  the end of `VideoPointAnnotator.__init__`, locking it to slot 0
  of the buttons column for every subclass.
- `StateVariables.add(name, states, widget="label")` takes a new
  `widget` kwarg. Allowed values: `"label"` (default; read-only text
  line, matching pre-rc2 behavior), `"dropdown"` (`QComboBox`),
  `"toggle"` (mutually-exclusive `QButtonGroup` of checkable
  `QToolButton`). The hint is read by the rc2 Qt sidebar; on non-Qt
  backends (Agg) it's ignored and the value renders as plain text via
  the legacy `TextView` path.
- `_qt._QtStatevarsWidget` -- the new layout-managed sidebar widget.
  Builds one row per state variable, picks control class from
  `StateVariable.widget`, and on user interaction calls
  `state.set_state(value)` followed by `parent.update()` -- the same
  generic redraw every keybind handler already invokes after
  `state.cycle()`. No new callback API is exposed; consumers stay on
  the `add(...)` surface.
- `make_qt_statevars_widget(figure, statevars_container)` -- mount
  function for the new widget. Inserts it into the QDockWidget left
  column (above an `addStretch`) so any buttons added later still
  stack above it.
- `tests/qt_learning/08_rc2_statevars_widget.py` -- Qt-headless smoke
  exercising all three `widget` values plus the dropdown-pick /
  toggle-click round-trip through `parent.update()`.
- `TestStateVariableWidgetHint` in `tests/test_core.py` -- pure-model
  pytest coverage for the new kwarg (default, each allowed value,
  rejection of unknown values, propagation through
  `StateVariables.add`, and the Agg `TextView` fallback).
- `VideoAnnotation.keep_overlapping_frames()` -- sibling of
  `keep_overlapping_continuous_frames()` (the `alt+q` action) without
  the consecutive-runs constraint: drops frames where any label is
  missing, but preserves isolated fully-labeled frames. Motivated by
  DUSTrack's "Train DLC model" pre-flight, which needs a way to drop
  partial-label frames before training -- DLC tolerates per-bodypart
  NaN in its CSV but partial frames degrade the trained model in
  practice. Not wired to a keybinding; DUSTrack is the only caller.
- `VideoPointAnnotator.keep_overlapping_frames()` -- wrapper that
  calls the annotation method and triggers `self.update()`.
- `test_video_annotation_keep_overlapping_frames`,
  `test_video_annotation_keep_overlapping_frames_keeps_non_continuous`,
  `test_video_annotation_keep_overlapping_frames_no_overlap_aborts`,
  and `test_video_point_annotator_keep_overlapping_frames` in
  `tests/test_pointtracking.py`. The "keeps_non_continuous" case is
  the load-bearing one -- distinguishes the new method from the
  continuous variant.
- `AssetContainer.remove(name)` -- counterpart to `add()`. Pops and
  returns the asset whose `.name` matches; raises `KeyError` if no
  asset carries that name. The container only manages membership;
  the caller is responsible for tearing down any plot handles /
  Qt widgets the popped asset owns. Inherited unchanged by every
  AssetContainer subclass (`Buttons`, `Selectors`, `MemorySlots`,
  `StateVariables`, `VideoAnnotations`). Motivated by DUSTrack
  1.1.0rc2's new "Remove layer" UI affordance which needs a way
  to drop a `VideoAnnotation` from a live session without
  restarting the annotator.
- `VideoAnnotation.reload()` -- inverse of `save()`. Wholesale-
  replaces `self.data` with the result of `load()` (so the property
  setter rewraps each per-label inner dict as `_TrackedFrameDict`),
  then bumps `_revision` explicitly so per-frame caches keyed on
  `(label_list, _revision)` invalidate. If `fname` is `None` or
  doesn't exist, `load()`'s empty-fallback branch returns
  `{str(i): {} for i in range(n)}` -- so "reload from disk if a
  file exists, otherwise reset to empty" is one method call. Mirrors
  the explicit-bump pattern already used by `sort_data`,
  `sort_labels`, `clip_trailing_empty_labels`, `remove_empty_labels`.
  Drives the DUSTrack 1.1.0rc2 "Discard unsaved annotations" button.
- `VideoPointAnnotator.remove_annotation_layer(name)` -- removes an
  annotation layer from a live session. Tears down the layer's
  scatter + trace artists via `VideoAnnotation.clear_display()`,
  drops it from `self.annotations` via `AssetContainer.remove`, then
  resyncs the `annotation_layer` / `annotation_overlay` state-
  variables (rotation lists + current selections) through the new
  `_refresh_annotation_state_lists` helper. Active-layer handoff:
  if the removed layer was the primary, the previous-in-rotation
  layer is auto-selected (falling through to the first surviving
  layer if the removed one was at index 0); if it was the overlay,
  the overlay clears to `None`. Raises `ValueError` if it would
  leave the container empty -- consumers wanting a "reset contents"
  semantic should use `VideoAnnotation.reload()` on the surviving
  layer instead. Treats every named entry the same at the dnav
  layer; the `"buffer"` exclusion is a DUSTrack-side UI concern.
  Drives the DUSTrack 1.1.0rc2 "Remove layer" button.
- `VideoPointAnnotator._refresh_annotation_state_lists()` -- helper
  extracted from the inline statevar-rotation refresh that lived at
  the tail of `add_annotation_layers`. Single source of truth for
  the `annotation_layer` / `annotation_overlay` rotation resync,
  shared by `add_annotation_layers` (extending the rotation) and
  the new `remove_annotation_layer` (shrinking it). Also clamps
  each statevariable's `_current_state_idx` so the position is
  never out-of-bounds after a shrink, providing a last-resort
  safety net for the caller's "pick a new selection" decision.
- `test_video_annotation_reload_with_file_on_disk`,
  `test_video_annotation_reload_without_file`,
  `test_asset_container_remove_happy_path_and_keyerror`,
  `test_remove_annotation_layer_swaps_active_and_clears_overlay`,
  `test_remove_annotation_layer_refuses_only_layer`, and
  `test_remove_annotation_layer_preserves_overlay_when_unrelated`
  in `tests/test_pointtracking.py`. Cover the new `reload` /
  `remove_annotation_layer` / `AssetContainer.remove` surfaces.
- `VideoAnnotations.reorder(names)` -- permute the underlying
  `AssetContainer._list` so layer names follow `names` (which must be
  a permutation of the current `self.names`). Idempotent when `names`
  already matches current order; raises `ValueError` otherwise. Sits
  on the plural `VideoAnnotations` rather than the base
  `AssetContainer` to limit blast radius to the annotation-layer use
  case it was built for. Membership-only -- callers that own the
  rotation of an `annotation_layer` / `annotation_overlay`
  state-variable are responsible for resyncing via
  `_refresh_annotation_state_lists` after a reorder. Drives the
  DUSTrack 1.1.0rc2 layer-regrouping pass.
- `VideoPointAnnotator.refresh()` + `VideoAnnotation.invalidate_caches()`
  + `F5` keybinding (and inherited "Refresh UI" button) -- escape
  hatch for the rare case of command-line direct mutations of `.data`
  that bypass the public `add()` / `remove()` / `add_at_frame()` API
  and therefore skip the `_revision` bump the trace-display and
  frame-marker caches key on. `invalidate_caches()` nulls
  `_trace_display_cache_key`; `refresh()` calls it on every annotation
  layer, nulls `_frame_marker_cache`, then `update()`s. The
  `_TrackedFrameDict` guard above makes a direct
  `ann.data[label][frame] = ...` correctly bump revision, but
  `refresh()` remains the insurance for any future bump miss in code
  paths not yet thought through.
- **Grouped Qt keybindings cheatsheet.** `show_key_bindings` now opens
  a modeless `QDialog` with sections (Annotation / Frame navigation /
  Layer / label / LK / interpolate / View / File / Other) instead of
  the pre-rc2 matplotlib `TextView` popup. `add_key_binding` gains
  `group=` and `on_button=` kwargs; `on_button=True` appends
  `"  (key)"` to the matching button's label via identity-matched
  lookup against `Buttons._action_funcs`, so buttons advertise their
  own shortcut. `pointtracking.set_key_bindings` declares groups for
  every binding mapping to the 5-step DUSTrack workflow (Select
  annotation layer → Select annotation number → Navigate to frame →
  Edit annotation → LK augmentation / refine).
  `GenericBrowser.set_default_keybindings` follows suit. Stdout
  fallback when no Qt window is available. Coverage:
  `tests/test_key_bindings.py` (new module),
  `tests/qt_learning/21_dustrack_keybindings_smoke.py`.

### Changed
- `_qt._get_buttons_widget` replaces the pre-rc2 `_get_buttons_toolbar`.
  Buttons now stack in a `QVBoxLayout` inside a `QDockWidget` on
  `LeftDockWidgetArea` (pre-rc2: `QToolBar` on `LeftToolBarArea`). The
  `QDockWidget` is borderless and non-floatable. Cached attribute on
  the `QMainWindow` is `_dnav_left_column` (the rc2 left-column
  struct); the legacy `_dnav_buttons_widget` attribute survives as an
  alias pointing at the buttons sub-widget.
- `Buttons.add_separator()` on the Qt path now inserts a sunken
  `QFrame.HLine` into the buttons `QVBoxLayout` (pre-rc2:
  `QToolBar.addSeparator()` `QAction`). Visual is equivalent;
  separator detection in tests now walks layout items.
- `VideoAnnotation.set_plot_type(type_)` now records the choice on
  `self._plot_type` in addition to applying the visual style; the
  `plot_type` property setter becomes a thin delegate so the two
  APIs are symmetric. Pre-fix, `set_plot_type("line")` only
  updated trace handle linestyle/marker, leaving `_plot_type` at
  its `__init__` default `"dot"`; the next `re_setup_display`
  (which fires on every existing layer inside
  `add_annotation_layers`) called `setup_display` → `set_plot_type
  (self.plot_type)`, read the stale `"dot"`, and reverted the
  visual. Manifested as the DUSTrack `dlccorr` "renders as dots
  after Reduce jitter" regression: `apply_manual_corrections` set
  dlccorr to line via the method (visual only); adding the LK
  output layer triggered `re_setup_display` on every layer,
  including dlccorr, which then reverted to dot. Two new tests in
  `tests/test_pointtracking.py` guard the sync and the
  re-setup-survival.
- `VideoPointAnnotator` non-fast_render path: the matplotlib gridspec
  drops its dedicated state-variables column. Layout shrinks from
  `3x2 width_ratios=[1, 4]` to `3x1`; the image axis is now
  full-width. State-variables move from the in-figure mpl axis into
  the QDockWidget left column. The `_ax_statevar` attribute is still
  set (to `None`) for backwards compatibility but is unused.
- `_QtStatevarsWidget._build_row` switched from `QHBoxLayout`
  (name | control) to `QVBoxLayout` (name above control). Side-by-side
  starved the combo of width so long state values
  (e.g. `dlc_iteration-3_250000` from a DUSTrack DLC project) elided
  to `dlc_...250000`; stacking gives the combo the full column width
  and the dropdown is set to `AdjustToContents` so it grows to fit
  the widest entry. Rows are now visually grouped with a sunken
  `QFrame.HLine` separator between adjacent state variables.
  Reported during DUSTrack 1.1.0rc2 testing.
- `_QtStatevarsWidget` no longer renders the bold "State variables:"
  title row; the trailing double separator + group rule already
  delimit the section. The widget now also paints itself with a
  slightly darker background (palette `base.darker(120)`, theme-
  adaptive) so the statevars area reads as a visually distinct
  group from the buttons column above it. Requested during DUSTrack
  1.1.0rc2 sidebar polish.
- Left-column dock host gets `setMinimumWidth(_LEFT_COLUMN_MIN_WIDTH=300)`
  (a touch above the pre-rc2 `sidebar_width=280` default; tuned
  empirically during DUSTrack 1.1.0rc2 testing). Combined with
  `_QtStatevarsWidget` switching its horizontal size policy from
  `Fixed` → `Preferred`, the statevars widget now fills the column
  instead of sticking to its own sizeHint; each combo (already
  `Expanding` horizontal) therefore reaches the column edge even when
  its current state value is short (e.g. `select` / `place`). Reported
  during the same DUSTrack 1.1.0rc2 testing pass that motivated the
  stacked-layout fix above.
- `StateVariables.show()` tries the new Qt widget path first and falls
  back to `TextView` on non-Qt backends. The `pos` / `fax` arguments
  are only consulted on the fallback (`pos` is ignored on the Qt
  path; the widget's position is dock-managed).
- `make_image_pane` no longer creates the fast_render `pane.sidebar`
  QLabel. That QLabel was the fast_render-Tier-2 statevariables text
  sink; rc2's left column subsumes that role for both Tier 1 and Tier
  2. The `sidebar_width` kwarg is kept on the signature for API
  stability but is now ignored. Fast_render `make_image_pane` now
  returns a pane with just `[image_pane, trace_canvas]`.
- **`r` keybinding is now cursor-aware in fast_render (Tier 2).**
  `VideoPointAnnotator._reset_view_all` dispatches on `event.inaxes`
  (set by `_patch_event_for_image_pane` in `__call__`): cursor over
  the Tier 2 image pane resets the image zoom/pan only; cursor over
  `_ax_trace_x` / `_ax_trace_y` resets the trace pair with x pinned
  to `(0, ann.n_frames)` (the full video range) and y autoscaled to
  data; cursor elsewhere or `event is None` falls back to the same
  trace treatment plus the image-pane reset, preserving muscle memory
  for `r` hit while hovering a button or off-figure. Pre-fix, a user
  who panned the trace x-axis to inspect a feature then hit `r` to
  refit the trace y also lost their image-pane zoom (and vice versa).
  Pinning x to the full video range — rather than autoscaling to the
  current annotation extent — keeps frames outside the annotation
  envelope visible, which is the usual case when extending
  annotations to a new region. Dissolves the Tier 2 image-zoom /
  trace-axes coupling the 2026-05-19 trace-scaling audit flagged as a
  follow-up. Tier 1 is unaffected (no image pane to scope). Binding
  description is now `"Reset view under cursor (traces use full-video
  x)"`. Regression test: `test_r_keybinding_cursor_aware_dispatch` in
  `tests/test_pointtracking.py` covers all three dispatch branches.
- **`alt+r` is the autoscale-to-data-extent sibling of `r`** (Tier 2).
  `VideoPointAnnotator._reset_view_to_data_extent` mirrors
  `_reset_view_all`'s dispatch structure, but the trace branch and
  fallback autoscale x and y to the data extent (`reset_axes(axis=
  "both", axes=[trace_x, trace_y])`) instead of pinning x to the full
  video. The image-pane branch is identical to `r`. Use when the
  annotated region is much narrower than the full video and you want
  to zoom in on it. Regression test:
  `test_alt_r_keybinding_cursor_aware_data_extent_dispatch`.
- `GenericBrowser.reset_axes` takes a new optional keyword `axes=`
  that restricts the walk to a given subset (default `None` walks
  `self.figure.axes` as before). Added to support the cursor-aware
  `r` / `alt+r` dispatch above — the trace branches pass
  `[_ax_trace_x, _ax_trace_y]` to scope the refit to the trace pair.
  Regression test: `test_reset_axes_axes_scope_kwarg` in
  `tests/test_core.py`.
- **Label-aware y-refit for multi-label tracking.** New
  `VideoPointAnnotator._fit_y_to_active_label` helper computes the
  y-extent from the active layer's active-label trace (and the overlay
  layer's same-named label, if an overlay is set and contains the
  label) with a 5% margin, then `set_ylim` on both trace axes. Wired
  to `annotation_label` and `label_range` statevariable changes via a
  new `StateVariable.add_on_change(callback)` callback list, so both
  keyboard (digit keys, `'` / `;`, `q` / `w`,
  `select_label_with_mouse`) and Qt-dropdown driven label switches
  refit; `VideoPointAnnotator._on_active_label_change` de-dupes via
  `_last_active_label` so the increment_label_range double-fire
  (`cycle()` then `set_state()`) is a single fit. Layer flips
  (primary or overlay) intentionally do NOT trigger a refit -- the
  layer-flip comparison workflow keeps its current y window. The `r`
  trace branch now calls the helper directly (was
  `reset_axes(axis="y", ...)`), so pressing `r` over a trace gives a
  comfortable view of the active label rather than compressing it into
  the union of every label's data extent; `alt+r` retains the union
  autoscale (sibling, explicitly documented). Single-label sessions
  see no behavior change -- the helper walks exactly one label.
  Binding descriptions tightened: `r` -> `"Reset view under cursor
  (traces: full-video x, active label y)"`, `alt+r` -> `"Reset view
  under cursor (traces: data-extent x, all-labels y)"`. Regression
  tests in `tests/test_pointtracking.py`:
  `test_fit_y_to_active_label_active_only`,
  `test_fit_y_to_active_label_with_overlay`,
  `test_fit_y_to_active_label_empty_is_noop`,
  `test_label_switch_triggers_yfit`,
  `test_layer_switch_does_not_trigger_yfit`,
  `test_label_range_switch_triggers_yfit`,
  `test_r_keybinding_trace_branch_fits_active_label_only`,
  `test_alt_r_keybinding_trace_branch_keeps_union`.

### Removed
- `_qt.make_sidebar_text_sink` / `_qt._QtSidebarTextSink` (fast_render
  Tier 2 statevariables text sink). Use the rc2 widget path via
  `StateVariables.show()` -- it covers Tier 1 and Tier 2 in one mount
  function.

### Fixed
- **Trace-pane scaling audit (Bug B: x-axis preserved across mid-session
  layer adds).** `VideoAnnotation.setup_display_trace` no longer
  unconditionally calls `ax_x.set_xlim(0, self.n_frames)`. The call is
  now guarded with `ax_x.get_autoscalex_on()`, so the *first*
  annotation constructed on the axes claims xlim (flipping autoscale
  off as a side effect) and every subsequent layer-add is a no-op.
  Pre-fix, each in-session `_adopt_layer` call in DUSTrack (Reduce
  jitter / post-train DLC refresh / first-time Apply manual
  corrections) blew away any user pan/zoom on the trace time-axis,
  forcing reliance on DUSTrack's `Freeze plot axes` workaround. Press
  `r` to refit. Regression tests: `test_setup_display_trace_xlim_guard`
  + `test_video_point_annotator_xlim_preserved_across_layer_add`.
- **Trace-pane scaling audit (Bug A: Manual y-policy).**
  `VideoPointAnnotator.update_frame_marker`'s cache-miss branch no
  longer unconditionally re-applies `set_ylim(nanlim(trace_data))` on
  every annotation mutation, label switch, or frame-of-interest
  toggle. The two `set_ylim` calls are guarded with
  `get_autoscaley_on()`, so the first cache miss with real data fits
  y (claiming autoscale) and subsequent mutations leave the user's
  view alone. Press `r` to refit. Pre-fix, every `add` / `remove` /
  LK interpolate / copy-from-overlay / label change re-fitted y,
  forcing reliance on DUSTrack's `Freeze plot axes` workaround. The
  all-NaN case is now also a true no-op (pre-fix it silently
  consumed the autoscale claim by setting ylim to itself). Regression
  test: `test_video_point_annotator_ylim_manual_policy` (covers
  first-fit / mutation-preserves / label-switch-preserves /
  FOI-preserves / r-restores-refit).
- **`reset_axes` polish (1.3.1 audit punchlist).**
  `GenericBrowser.reset_axes` swaps the deprecated
  `isinstance(ax, maxes.SubplotBase)` filter for
  `ax.get_subplotspec() is not None` (mpl 3.7+ deprecation, slated for
  removal). The method now also folds each axis's `Collection`
  datalims into `ax.dataLim` before calling `autoscale()`, so
  `fill_between` artists -- e.g. DUSTrack `display_type="fill"`
  events -- contribute to the autoscale extent. mpl's `Axes.relim()`
  walks Lines + Patches + Images but not Collections, so pressing
  `r` previously left fill artists outside the refit ylim.
  Regression tests: `test_reset_axes_includes_fill_artists` +
  `test_reset_axes_skips_non_subplot_axes`.
- Phase 2 smoke (`tests/qt_learning/04_phase2_smoke.py`) now pins the
  local source on `sys.path[0]` matching the pattern in 07. Pre-rc2 a
  script-mode run would silently import the env's installed
  datanavigator (which on older envs lacks `TextView._overlay`),
  producing an `AttributeError` unrelated to the test's intent.
- **`fast_render` window resize re-fits the image.**
  `_QtImagePane.resizeEvent` pre-rc2 gated the auto-re-fit on
  `transform().isIdentity()`, but the *initial* `_fit_view()` leaves
  a non-identity scale transform, so the check was False on every
  subsequent resize and the image never re-fit. New
  `_PanZoomGraphicsView.user_adjusted` flag is set True on wheel-zoom
  and on real pan drags (cleared by `reset_view()`); `resizeEvent`
  now re-fits when the flag is False, preserving the original intent
  ("don't clobber a manual zoom") without the false negative on
  auto-fit. Initial QMainWindow aspect ratio also tuned:
  `make_image_pane` resizes the host window after `setCentralWidget`
  so total height = `canvas_h * 3`, matching the layout's 2:1 stretch
  hint and giving the image pane ~2x the canvas height (~600 px for
  DUSTrack). Regression test:
  `test_tier2_resize_refits_until_user_adjusts`.
- **`z` selector listens on both trace axes.** The interval-picker
  `Selector` had `ax_list=[self._ax_trace_x]`, so pressing `z` while
  hovering the y-trace was a no-op. Now
  `[self._ax_trace_x, self._ax_trace_y]` -- both trace panes accept
  interval bracketing for LK-RSTC.
- `GenericBrowser.copy_to_clipboard` (Ctrl+C) now grabs the entire
  `QMainWindow` via `QWidget.grab()` instead of `figure.savefig`-ing
  the matplotlib canvas alone. On a Qt backend the clipboard image
  now includes every Qt-side widget parented to the window --
  left-column dock (buttons + state-variables), fast_render image
  pane, and any downstream sidebars (DUSTrack). Pre-fix, a DUSTrack
  window pasted into a doc as a thin trace strip with the entire
  sidebar / image pane missing. mpl/Agg fallback retained via
  `find_qt_window(...) is None`.
- **Labels promoted to first-class schema.** Three coordinated
  changes:
  (1) `VideoAnnotation.save` no longer calls `remove_empty_labels()`
  on the way out. Empty-but-declared labels round-trip through JSON
  as `"label": {}` instead of being silently pruned. The method
  stays available for callers that explicitly want a lean export
  (DUSTrack pre-flight before DLC training calls it directly).
  (2) `VideoAnnotation.to_trace(label)` is now schema-tolerant: a
  label missing from `self.data` returns the full NaN array
  (matching the docstring's "unannotated frames are NaN" promise,
  treated as every-frame-unannotated). Lets `update_frame_marker`
  iterate every layer with one shared active label even when a
  layer doesn't carry that label.
  (3) The default-label bootstrap shrank from 10 placeholders to
  1, since the no-prune save would otherwise write a slate of
  empties on every fresh save. `VideoAnnotation(n_labels=N)` and
  `VideoPointAnnotator(..., n_labels=N)` still let callers declare a
  larger schema up front.
  Net fix: DUSTrack's `apply_manual_corrections` no longer crashes
  with `AssertionError: label in self.labels` when the user has
  manually corrected only one of two labels (the patch overlay's
  empty label was getting save-pruned, the corrections layer was
  then missing that label, and `update_frame_marker` blew up
  hstacking traces across layers). Tests:
  `test_video_annotation_save_preserves_empty_labels`,
  `test_to_trace_returns_nan_for_missing_label`,
  `test_corrections_layer_shaped_update_frame_marker`,
  `test_add_annotation_layers_unions_declared_labels`,
  `test_video_annotation_default_n_labels_is_one`.
- `VideoPointAnnotator._ensure_target_has_labels(target, labels)`
  static helper. Used by the three LK interpolate paths
  (`interpolate_with_lk`, `interpolate_with_lk_norstc`,
  `check_labels_with_lk`) to declare missing labels on the target
  layer before bulk `add` calls. Pre-rc2 the 10-default-empty
  bootstrap made cross-layer label-set parity implicit; with
  first-class labels the LK callers declare what they need.
- `add_annotation_layers` synthesis takes the union of every
  layer's *declared* labels (was: labels-with-data only), so
  empty-but-declared labels flow back into the layer set on
  reload and a layer missing a peer's declared label gets it
  added.

## [1.3.1] - 2026-05-18

Patch release. One bug fix on the event-overlay path.

### Fixed
- `Event._get_ylim` (`datanavigator/events.py:599`) was reading
  `line.get_xdata()` while claiming to return a y-axis limit. For
  fill-display events (`display_type="fill"`) this made
  `ax.fill_between` draw a polygon spanning the *x*-data extent in
  the *y* direction; autoscale then grew the y-axis to fit. Visible
  reproducer: `EventPickerDemo`, add a 2-event by pressing `2` at
  two x positions — pre-fix the y-axis rescaled from ~`(0, 1)` to
  ~`(0, 10)` for the demo's `np.random.rand(100)` / `sr=10` signal.
  Line-display events were affected too: the vertical ticks marking
  the event were drawn at the wrong y-extent, contributing oversized
  ydata to `dataLim` on the next autoscale. Regression tests:
  `tests/test_events.py::test_get_ylim_reads_ydata_not_xdata` (unit)
  and
  `tests/test_examples.py::test_event_picker_fill_event_preserves_ylim`
  (end-to-end repro from the bug report).

## [1.3.0] - 2026-05-17

Video-reader backend swap: drop the `decord` runtime dependency in
favour of an in-tree PyAV+TOC reader. Public API is preserved — all
existing call sites (`vr[i].asnumpy()`, slicing, `vr.get_batch([...]).asnumpy()`,
`len(vr)`, iteration, `vr.get_avg_fps()`, `Video(VideoReader)`
subclass) continue to work unchanged.

The swap is motivated by the 2026-05-16 frame-parity test (canonical
record at `pn-portfolio/plans/20260516_decord_pyav_parity.md`):
decord disagrees with the ffmpeg-CLI oracle on 8/28 frames of a
production VFR telemed clip (PSNR 32–37 dB on the divergent frames);
PyAV+TOC matches the oracle pixel-exact on all 28 frames across
11 test clips and every codec exercised (libx264 / libx264+B /
h264_nvenc ±B / hevc_nvenc / paired CPU↔GPU). The swap is therefore
not just a dependency cleanup — it is a **correctness improvement**
on real lab videos.

### Added
- `VideoReader` and `cpu` re-exported at the `datanavigator` package
  root (`from datanavigator import VideoReader, cpu`). These provide
  a decord-compatible API surface — `VideoReader(uri, ctx, width,
  height, num_threads, fault_tol)` constructor signature, `vr[i]`
  / slice / `get_batch([...])` returning objects with `.asnumpy()`,
  `len()`, iteration, `get_avg_fps()`. `cpu(0)` is a no-op sentinel
  for call-site parity (PyAV is CPU-only here).
- `datanavigator/_vendor/pims_pyav_reader.py` — `PyAVReaderIndexed`
  class vendored from the PIMS project (BSD-3-Clause, upstream
  commit `d599596`, 2023-01-24). The full upstream license is
  reproduced in `datanavigator/_vendor/LICENSE-PIMS`. PIMS itself
  is **not** a runtime dependency.
- `tests/test_video_reader.py` — unit-test coverage for the shim
  (indexing, slicing, `get_batch`, `len` / iteration, `get_avg_fps`,
  `cpu` sentinel, `Video.gray` RGB-fix regression gate, TOC sidecar
  cache flow, `get_frame_timestamp` shape/monotonicity, v1/v2 sidecar
  paths).
- A `dnav_pyav_toc` reader entry in the parity harness
  (`tests/decord_pyav_parity/readers.py`) — confirms the vendored+
  wrapped reader matches the upstream PIMS reader row-for-row.
- **TOC sidecar cache (schema v2).** `PyAVReaderIndexed.__init__`
  walks every packet in the file at open time to build the
  frame-index TOC — that's the cost decord skipped (and was sometimes
  wrong because of). To avoid paying it on every open, the wrapped
  reader now caches the TOC next to the video as `<video>.dnav-toc`
  (JSON content; the `.json` extension is intentionally omitted so
  `*.json` walkers in downstream tooling — e.g. DUSTrack annotation
  discovery in `dlcinterface.py`, ad-hoc notebooks — don't pick the
  sidecar up). Keyed on path + size + mtime + SHA-256 of the
  first/last 64 KiB. Cache miss prints `datanavigator: building TOC
  for <name>...` and saves a sidecar on success; cache hit is silent
  and sub-second. The v2 sidecar also records per-frame `pts` and
  `duration` plus the stream `time_base`, so
  `vr.get_frame_timestamp(...)` is a cache hit on second open. v1
  sidecars (TOC only) remain readable for normal video access; the
  first `get_frame_timestamp` call on a v1-cached file lazily builds
  the per-frame table and upgrades the sidecar to v2 in place.
  Read-only data directories degrade gracefully (warning to stderr,
  data built in memory anyway).
  `datanavigator.precompute_toc(paths, force=False)` batch-warms the
  cache for a sequence of videos before an interactive session
  (pass `force=True` to upgrade v1 sidecars to v2).
- **`VideoReader.get_frame_timestamp(indices)`** — decord-compatible
  per-frame timestamp lookup. Returns an `(N, 2)` `float64` ndarray
  of `[start, end]` times in seconds. Used by
  `immersionlab/tobii.py:243` to derive per-frame clocks from Tobii
  eye-tracker recordings (`vr.get_frame_timestamp(range(len(vr))).mean(-1)`).
  On a v2 cache hit this is an array slice; on first call against a
  never-cached or v1-cached file, it triggers a full demux+decode
  pass announced with
  `datanavigator: indexing frame timestamps for <name>...`.

### Changed
- **Video-reading backend swapped from `decord` to PyAV+TOC**, with
  the public API preserved (see Added). Internal imports in
  `utils.py`, `videos.py`, and `opticalflow.py` now route through
  the in-tree shim.
- CI matrix re-adds `macos-latest`. `continue-on-error` removed from
  Windows runners — test failures now block on all three OSes. A
  brew-based `ffmpeg` install step is added for the macOS runner.

### Fixed
- **`Video.gray` and `opticalflow.lucas_kanade` grayscale conversion
  channel order.** Both previously called `cv.COLOR_BGR2GRAY` on
  RGB-decoded frames, silently swapping R and B before the
  grayscale weighting. Now use `cv.COLOR_RGB2GRAY`. This is an
  observable behavior change for any caller depending on the buggy
  luminance — call sites that compared grayscale frames across the
  upgrade boundary will see different values. Affected files:
  `datanavigator/utils.py` (`Video.gray`) and
  `datanavigator/opticalflow.py` (`lucas_kanade.gray` inner helper).

### Removed
- `decord` runtime dependency (replaced by in-tree PyAV+TOC reader).
- `numpy<2` pin in `pyproject.toml` and `requirements.yml` — the pin
  was held by `decord` + `tables<3.10`; both gates are now gone.
- Implicit `tables<3.10` upper bound: the dep is now `tables>=3.10`
  to make NumPy-2 compatibility an explicit hard requirement.

### Infrastructure
- `pyproject.toml`: `decord` → `av`; `numpy<2` → `numpy`;
  `tables` → `tables>=3.10`.
- `requirements.yml`: same set of changes (`decord` → `av` in the
  pip section; `numpy<2` → `numpy`; `pytables` → `pytables>=3.10`).
- `.github/workflows/ci.yml`: `macos-latest` re-added to matrix;
  `continue-on-error` dropped from Windows; brew step added for
  macOS ffmpeg install.

## [1.2.0] - 2026-05-16
Audit-and-polish release. No public API changes.

### Added
- `utils.find_nearest` and siblings (`find_nearest_idx`,
  `find_nearest_val`, `find_nearest_idx_val`), surfaced via the
  package's `utils` module. Cash-out from the `pntools` retirement
  pass; previously unreleased on `master` since `4cdbe64`.
- GitHub Actions CI: pytest matrix across Python 3.10–3.12 and
  `{ubuntu-22.04, macos-latest, windows-latest}`. Ubuntu is the
  authoritative runner; macOS / Windows are `continue-on-error` for
  this release while the `numpy<2` pin keeps `decord` / `tables`
  wheel availability flaky on those platforms. ffmpeg is installed
  via OS-conditional steps (`apt`, `brew`, `choco`) so the workflow
  works on Apple Silicon `macos-latest`.
- `__all__` on `datanavigator/__init__.py`. `from datanavigator
  import *` now exposes 39 documented names (down from 53 — dropped
  the stdlib leaks `os` / `sys` / `shutil` and the 11 submodule
  attributes). Submodule access via `datanavigator.utils.X` or
  `from datanavigator import utils` is unchanged.
- `TODO.md` seeded with deferred-to-1.3.0 items, nice-to-have polish,
  and stretch goals.
- `[project.optional-dependencies] dev = ["pytest"]` in
  `pyproject.toml`, plus `pytest` in `requirements.yml` (previously
  the test suite was unrunnable as-shipped — no env in the repo
  could run it).
- New regression tests in `tests/test_pointtracking.py`:
  `test_video_annotation_to_dlc_populates_values` (synthetic) and
  `test_video_annotation_to_dlc_real_data_roundtrip` (skipif-gated
  against the OPR02 DLC fixture on the S: drive). Pin the
  chained-assignment fix above.

### Changed
- **Cross-platform default paths.** On macOS / Linux,
  `datanavigator._config.CLIP_FOLDER` and `CACHE_FOLDER` now default
  to `~/datanavigator/_clipcollection` and `~/datanavigator/_cache`.
  Windows defaults are unchanged (`C:\\data\\_clipcollection`,
  `C:\\data\\_cache`). `CLIP_FOLDER` / `CACHE_FOLDER` environment
  variable overrides are unchanged.
- **Python floor raised to 3.10** (3.7–3.9 are EOL since 2023-06).
- DeepLabCut HDF5 paths in `VideoAnnotation` (`load`, `_load_dlc`,
  `_dlc_df_to_annotation_dict`, `_dlc_trace_to_annotation_dict`,
  `to_dlc`) are now docstring-tagged `DUSTrack-shaped` so the
  upcoming 1.3.0 migration to `dustrack.VideoAnnotation` has a
  greppable anchor.
- Module-level docstring added to `pointtracking.py`.
- Opportunistic PEP 604 / PEP 585 type-hint modernization in the
  files this audit was already touching: `events.py`,
  `pointtracking.py`, `components.py`, `plots.py` (`Optional[X]` →
  `X | None`, `Union[A, B]` → `A | B`, `List` / `Tuple` / `Dict` →
  `list` / `tuple` / `dict`). The remaining modules retain legacy
  typing imports; a portfolio-wide sweep is deferred to 1.3.0.
- `pyproject.toml`: added `[tool.black]`, `[tool.isort]`, and
  `[tool.pytest.ini_options]` blocks so contributors don't need to
  reconstruct the conventions from CONVENTIONS.md.

### Fixed
- `VideoBrowser.extract_clip` and `VideoPlotBrowser.extract_clip` now
  fall back to `libx264` when `h264_nvenc` (NVIDIA hardware encoder)
  is unavailable. Previously hardcoded; failed with
  `Unknown encoder 'h264_nvenc'` or `Cannot load nvcuda.dll` on
  CPU-only hosts, CI runners, and macOS / Apple Silicon.
- `is_pathname_valid` now catches `ValueError` (which Python 3.5+
  raises for "embedded null character" in `os.lstat`) in addition
  to the stale `TypeError` catch. The function now returns `False`
  for NUL-bearing paths on modern Python instead of leaking the
  exception.
- `tests/test_utils.py` path-validity assertions are now
  platform-aware: Windows tests with `<` (a reserved char on
  Windows), POSIX tests with an embedded NUL byte. The old test
  failed on Linux because POSIX accepts `<` in path components.
- `VideoPlotBrowser` setup no longer crashes on matplotlib ≥ 3.8:
  `Grouper.join` was removed in 3.8, replaced by `Axes.sharex()`.
  This was a latent bug surfaced by the 1.2.0 audit's fresh-env
  test run (matplotlib 3.10.9). `test_video_plot_browser_init`
  passes again.
- `VideoPlotBrowser.onclick` right-click-to-seek now works on
  Python ≥ 3.11. The check compared `str(event.button)` to the
  literal `"MouseButton.RIGHT"`, which silently broke when
  Python 3.11 changed `IntEnum.__str__` to return the integer
  value — `str(MouseButton.RIGHT)` is now `"3"`. Replaced with an
  enum comparison (`event.button == MouseButton.RIGHT`). The
  failure had been mis-triaged in the 1.2.0 audit as a headless-
  Agg dispatch quirk; it was a runtime defect for every user on
  Python ≥ 3.11, not test-only. Confirmed by running
  `test_video_plot_browser_init` on Python 3.12 + matplotlib 3.10.7.
- `VideoAnnotation.to_dlc` no longer uses pandas chained assignment
  (`df.loc[row][col] = val`), which pandas emits a `FutureWarning`
  on every call and which will silently no-op once Copy-on-Write
  becomes the default in pandas 3.0. Validated byte-equivalent to
  the pre-fix code on real OPR02 4-point DLC labeled-data (438
  frames × 4 labels = 1752 annotations).
- `tests/conftest.py` now sets `matplotlib.use("Agg", force=True)`
  before any pyplot import, so the test suite runs cleanly headless
  without depending on `MPLBACKEND=Agg` being set in the
  environment.
- `tests/test_opticalflow.py::test_lucas_kanade_rstc` no longer
  returns a tuple (pytest `PytestReturnNotNoneWarning`); the
  removed return statement was replaced with shape assertions on
  the reverse / RSTC paths.
- Typo in the ffmpeg-not-found warning ("Cound" → "Could") at
  `datanavigator/__init__.py`.
- Dead code after the early return in `Event.to_portions`.
- Stale debug `print(label_orig_to_internal)` in
  `_dlc_df_to_annotation_dict`.
- Three commented-out blocks in `pointtracking.py` (a duplicated
  `ctrl+alt+c` key binding, the `self.ann.update_display` /
  `show_one_trace` alternative paths in
  `VideoPointAnnotator.update`, and the `set_xlim` conditional in
  `update_annotation_visibility`) plus an unclosed-quote comment.
- `plots.py:add_selectors` "Unable to add selectors" diagnostic now
  goes through `logging.getLogger(__name__).debug` instead of stdout
  `print`.

### Notes
- The `numpy<2` pin is retained: it tracks transitive constraints in
  `tables` (PyTables) and `decord`, not anything in datanavigator
  itself. It will be revisited in 1.3.0 once CI confirms
  `tables >= 3.10` (numpy-2-compatible) works across the matrix.
- `src/datanavigator/` layout migration is deferred to 1.3.0, which
  is already touching the package layout via the `pointtracking.py`
  split.

## [1.1.4] - 2025-12-08
Adding support to export video annotations as pysampled.Data with signal_names and signal_coords.

## [1.1.3] - 2025-12-02
Minor bugfixes. 
1. Restrict numpy version to less than 2 in pyproject.toml.
2. Exception handling when checking video files

Minor feature add. Export annotation overlaid on video from the VideoAnnotation class.

## [1.1.2] - 2025-09-10
Minor bugfix. Fixed error in the palette code when seaborn is not installed
Minor feature add. Export h5 files as json.

## [1.1.1] - 2025-05-14
Minor bugfix related to adding annotation layers after creating the VideoPointAnnotator interface.

## [1.1.0] - 2025-05-12

### Changed
- **Headline:** removed the 10-label limit on the `VideoPointAnnotator` interface — the new limit is 1000 labels. Implemented as a two-tier state machine: a `label_range` state variable (which decade: 0–9, 10–19, ...) and an `annotation_label` state variable (which one within the decade), so number keys can still drive the keyboard UI.

### Fixed
- Number-key event handling: `label_range` and `annotation_label` now stay in sync when the user types a digit or clicks a picker — picking a label now also updates the active decade. (Previously, switching decades from the UI left the keyboard mapping stale.)
- Existing-label vs new-label key paths reordered so typing a digit that matches an existing label activates it instead of re-creating it.
- Annotation initialization: starting from a fresh annotation now adds labels through `Annotations.add_label(...)` and re-sets up the display, instead of poking `ann.data[label] = {}` directly.
- `plot_type` is now assigned directly (`self.annotations["buffer"].plot_type = "line"`) rather than via a setter that didn't exist.
- Removed a recursive `_update_statevariable_annotation_label` helper that called itself unconditionally.
- Final key-press dispatch refactor: digit detection via `str(event.key).isdigit()`, cleaner decade math.

### Changed (layout)
- `VideoPointAnnotator` figure switched from `plt.subplots(3, 1, ...)` + `tight_layout()` to an explicit `gridspec` layout with a dedicated state-variable panel in the top-left cell. The image axes now share the right column with the trace axes spanning both columns underneath. Resolves stacking / overlap issues that appeared as the state-variable panel grew with the new label_range UI.

## [1.0.0] - 2025-05-08

First major release after type hints, 91% test coverage, and `black` formatting.
