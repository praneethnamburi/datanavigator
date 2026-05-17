# TODO

Working list of deferred work for the next datanavigator release. The
authoritative roadmap lives in
[`pn-specs/specs/datanavigator.md`](https://github.com/praneethnamburi/datanavigator)
(internal); this file is for items concrete enough to act on.

## 1.3.0 (deferred from the 1.2.0 audit)

- **Migrate to `src/datanavigator/` layout.** Convention compliance
  (`pn-specs/CONVENTIONS.md`). Deferred from 1.2.0 because the 1.3.0
  `pointtracking.py` split already touches the layout.
- **Split `pointtracking.py` (1853 LOC)** into annotator-UI / annotation-IO
  / DLC-interop pieces. Rides the Qt rework.
- **Qt under-the-hood swap.** Host `matplotlib.FigureCanvas` inside Qt;
  preserve public API. Absorb `core.py`'s optional PySide2 `copy_to_clipboard`
  path. The natural moment for the `pointtracking.py` split, the type-hint
  sweep on remaining files, and the `assets.py` / `videos.py` cleanups
  below.
- **Migrate `DUSTrack-shaped` paths to `dustrack.VideoAnnotation`.**
  Grep anchor: `DUSTrack-shaped` (5 occurrences in
  `pointtracking.py`, all on `VideoAnnotation`). Leaves datanavigator
  modality-agnostic and gives DUSTrack the DLC story end-to-end.
- **Swap `decord` → PyAV+TOC** (greenlit by the 2026-05-16 parity
  test, see
  [`pn-specs/plans/20260516_decord_pyav_parity.md`](https://github.com/praneethnamburi/pn-specs/blob/master/plans/20260516_decord_pyav_parity.md)
  — internal). Lifts the `numpy<2` pin as a side effect. Also a
  correctness improvement: decord disagrees with the ffmpeg-CLI
  oracle on 8/28 frames of the production VFR ultrasound clip;
  PyAV+TOC matches all 28. Implementation: vendor PIMS's
  `PyAVReaderIndexed` (BSD-3, single class, ~300 LOC, attribution
  header) OR add `pims` as a dep — decide during the Phase 1 planning
  session. Re-export `VideoReader` / `cpu` / `Video` from
  `datanavigator/__init__.py` so downstream repos (DUSTrack,
  immersionToolbox, pn-projects, blender-ScriptViz) do a one-line
  `from decord import ...` → `from datanavigator import ...` swap.
  Re-run `tests/decord_pyav_parity/run_parity.py --full` against the
  new in-repo `datanavigator.VideoReader` as a regression gate.
- **Fix RGB/BGR latent bug at `datanavigator/utils.py:322` and
  `datanavigator/opticalflow.py:60`.** Decord (and PyAV with `rgb24`)
  return RGB but the code calls `cv.COLOR_BGR2GRAY` — swaps red and
  blue before greyscale. Behavior is observable but small (LK
  gradients are mostly robust to the swap, which is why this never
  surfaced). Bundle with the decord swap above (both touch the same
  files). Audit `immersionlab/cep.py` for the same `BGR2GRAY` pattern
  during the immersionToolbox migration phase.
- **Re-add `macos-latest` to the CI matrix + drop `continue-on-error`
  from macOS / Windows runners.** Unblocked by the decord swap above
  — decord's missing Apple Silicon wheel was the original reason for
  exclusion, and the `numpy<2` pin was the original reason for the
  flag on Windows.
- **Full type-hint sweep** on `assets.py`, `videos.py`, `utils.py`,
  `signals.py`, `core.py`, `__init__.py`, `examples.py`,
  `opticalflow.py` — the 1.2.0 audit only modernized files it was
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
