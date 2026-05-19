# Change Log
All notable changes to this project will be documented in this file.

## [1.4.0rc2] - 2026-05-18

Release candidate for 1.4.0. Two themes layered on top of 1.4.0rc1:
(a) the button host swaps from `QToolBar` to a `QDockWidget` +
`QVBoxLayout` left column, and (b) state-variables are promoted from a
read-only text overlay to interactive Qt controls (dropdowns / toggles
/ labels) mounted in that same column beneath the buttons. Both
changes share a single goal -- one column of controls for an
interactive UI, no scattered widgets across the QMainWindow's dock
areas.

### Added
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

### Removed
- `_qt.make_sidebar_text_sink` / `_qt._QtSidebarTextSink` (fast_render
  Tier 2 statevariables text sink). Use the rc2 widget path via
  `StateVariables.show()` -- it covers Tier 1 and Tier 2 in one mount
  function.

### Fixed
- Phase 2 smoke (`tests/qt_learning/04_phase2_smoke.py`) now pins the
  local source on `sys.path[0]` matching the pattern in 07. Pre-rc2 a
  script-mode run would silently import the env's installed
  datanavigator (which on older envs lacks `TextView._overlay`),
  producing an `AttributeError` unrelated to the test's intent.
- `GenericBrowser.copy_to_clipboard` (Ctrl+C) now grabs the entire
  `QMainWindow` via `QWidget.grab()` instead of `figure.savefig`-ing
  the matplotlib canvas alone. On a Qt backend the clipboard image
  now includes every Qt-side widget parented to the window --
  left-column dock (buttons + state-variables), fast_render image
  pane, and any downstream sidebars (DUSTrack). Pre-fix, a DUSTrack
  window pasted into a doc as a thin trace strip with the entire
  sidebar / image pane missing. mpl/Agg fallback retained via
  `find_qt_window(...) is None`.

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
