# Change Log
All notable changes to this project will be documented in this file.

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
