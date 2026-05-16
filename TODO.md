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
- **Lift the `numpy<2` pin.** Currently pinned transitively via
  `tables` (PyTables 3.10 adds numpy-2 support) and `decord` (no
  numpy-2 wheels). Re-evaluate once CI confirms both work on numpy 2.x
  across the matrix.
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
