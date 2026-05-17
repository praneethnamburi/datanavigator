# decord vs PyAV+TOC frame-parity test

**Original purpose** (2026-05-16): decide whether `datanavigator`
should swap `decord` for `PyAV + frame-index TOC`. Verdict:
`SHIP_PYAV_TOC` — shipped in `datanavigator 1.3.0` (`fe5899c`).

**Current purpose** (post-migration): **ongoing regression check** for
the in-tree shim against an ffmpeg-CLI oracle. The shim's behavior is
now load-bearing for DUSTrack, immersionToolbox, pn-projects, and
blender-ScriptViz, and `tests/test_video_reader.py` only pins
internal self-consistency — this harness is the only place
shim output is compared against an external ground truth. Re-run
after any change to the vendored `PyAVReaderIndexed`, a PyAV (`av`)
upgrade, or an NVENC driver update that might shift encoder output.

The `dnav_pyav_toc` reader entry exercises the shipped shim
end-to-end; the `decord` reader entry stays as the historical
reference (decord is the only external dep this harness still pulls
in — install it into the `parity-test` env only).

Files do not start with `test_` so they are not collected by
`pytest`. Emits `_results/PARITY_REPORT.md` with a `## VERDICT: ...`
line.

## Run

```
conda activate parity-test
cd C:\dev\datanavigator\tests\decord_pyav_parity
python run_parity.py --quick    # ~30 s, sanity check
python run_parity.py --full     # ~5 min, the real run
```

The first run also generates transcoded + synthetic clips into
`S:\_corpus\_parity_test\` (idempotent — skipped on subsequent runs).

## Plan reference

Canonical record (committed; includes the 2026-05-17 final
regression-run appendix):
`C:/dev/pn-portfolio/plans/20260516_decord_pyav_parity.md`

Original (user-local) planning thread:
`C:\Users\praneeth\.claude\plans\ok-do-the-tests-mighty-salamander.md`

## Corpus note

`S:/_corpus/_parity_test/_raw.y4m` (7.4 GB Y4M source for the
`pair_cpu` / `pair_gpu` encodes) was deleted in Phase 6 cleanup. The
already-transcoded `.mp4` clips remain and are sufficient for normal
regression runs; if you need to regenerate the transcodes from
scratch, recreate the Y4M from any libx264-decoded source first.

