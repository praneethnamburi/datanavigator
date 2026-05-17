# decord vs PyAV+TOC frame-parity test (scratch)

**Purpose**: decide whether `datanavigator` should swap `decord` for
`PyAV + frame-index TOC`. Decision-producing — emits
`_results/PARITY_REPORT.md` with a `## VERDICT: ...` line.

**Scope**: scratch. **Delete this directory after the migration completes
(Phase 6 of the plan in
`pn-specs/plans/20260516_decord_pyav_parity.md`).** Retained meanwhile
as a regression check: when datanavigator 1.3.0 ships an in-repo
`VideoReader` shim, swap it in as the `pyav_toc` reader and re-run
the harness to confirm it still matches the ffmpeg-CLI oracle on the
corpus. Files do not start with `test_` so they are not collected by
`pytest`.

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

Canonical record (committed):
[`pn-specs/plans/20260516_decord_pyav_parity.md`](../../../pn-specs/plans/20260516_decord_pyav_parity.md)
— relative path won't resolve cross-repo; see `C:/dev/pn-specs/plans/20260516_decord_pyav_parity.md`.

Original (user-local) planning thread:
`C:\Users\praneeth\.claude\plans\ok-do-the-tests-mighty-salamander.md`

