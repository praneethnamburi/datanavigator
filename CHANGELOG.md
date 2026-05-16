# Change Log
All notable changes to this project will be documented in this file.

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
  wheel availability flaky on those platforms.
- `TODO.md` seeded with deferred-to-1.3.0 items, nice-to-have polish,
  and stretch goals.
- `[project.optional-dependencies] dev = ["pytest"]` in
  `pyproject.toml`, plus `pytest` in `requirements.yml` (previously
  the test suite was unrunnable as-shipped — no env in the repo
  could run it).

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
- `VideoPlotBrowser` setup no longer crashes on matplotlib ≥ 3.8:
  `Grouper.join` was removed in 3.8, replaced by `Axes.sharex()`.
  This was a latent bug surfaced by the 1.2.0 audit's fresh-env
  test run (matplotlib 3.10.9). `test_video_plot_browser_init`
  passes again.
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
