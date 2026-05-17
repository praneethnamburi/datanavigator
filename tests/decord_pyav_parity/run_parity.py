"""decord vs PyAV+TOC frame-parity test — single entrypoint.

Usage:
    python run_parity.py --quick   # ~5 indices/clip, sanity check
    python run_parity.py --full    # ~30 indices/clip, the real run

Writes _results/PARITY_REPORT.md with a top-line VERDICT.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import clip_gen
import oracle
import readers
import report


# ---------- helpers ----------

def keyframe_indices(path: str) -> list[int]:
    """Return frame indices of keyframes via PyAV demuxer pass."""
    import av
    out: list[int] = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        i = -1
        for packet in container.demux(stream):
            if packet.stream.type != "video":
                continue
            decoded = packet.decode()
            for frame in decoded:
                i += 1
                if frame.key_frame:
                    out.append(i)
    return out


def pick_indices(n_frames: int, keyframes: list[int],
                 mode: str, seed: int = 20260516) -> list[int]:
    """Boundary + keyframes + mid-GOP + random; per the plan."""
    rng = np.random.default_rng(seed)
    if n_frames <= 0:
        return []
    boundary = {0, 1, max(0, n_frames - 2), n_frames - 1}
    if mode == "quick":
        kf = set(keyframes[:1])
        mid = {(k + 5) % n_frames for k in kf}
        idx = sorted((boundary | kf | mid) - {-1})
        # cap at 5
        return idx[:5]
    # full
    kf = set(keyframes[: 5])
    mid = {(k + 5) % n_frames for k in kf}
    rand = set(int(x) for x in rng.integers(0, n_frames, size=16))
    idx = sorted((boundary | kf | mid | rand) & set(range(n_frames)))
    return idx


def measure_one(reader_out: np.ndarray, oracle_out: np.ndarray) -> dict:
    if reader_out.shape != oracle_out.shape:
        return dict(
            pixel_exact=False,
            l1_mean=float("nan"),
            psnr_db=float("nan"),
            matches_oracle=False,
        )
    exact = report.pixel_exact(reader_out, oracle_out)
    p = report.psnr(reader_out, oracle_out)
    return dict(
        pixel_exact=exact,
        l1_mean=report.l1_mean(reader_out, oracle_out),
        psnr_db=p,
        matches_oracle=report.matches_oracle(exact, p),
    )


# ---------- RGB/BGR adjudication ----------

def adjudicate_rgb_bgr(rgb_probe_path: Path) -> dict[str, str]:
    """Each reader reads frame 0; we inspect top-half dominant channel."""
    out: dict[str, str] = {}
    for name, fn in readers.READERS.items():
        try:
            f = fn(rgb_probe_path, 0)
            top = f[5, 5, :]  # row 5 of 64-px-tall stripe -> firmly in red half
            r, g, b = int(top[0]), int(top[1]), int(top[2])
            verdict = "RGB" if r > b else "BGR" if b > r else "AMBIGUOUS"
            out[name] = f"{verdict} (top px = R{r:3d} G{g:3d} B{b:3d})"
        except Exception as e:
            out[name] = f"ERROR: {type(e).__name__}: {e}"
    return out


# ---------- Seek-correctness via the counter clips ----------

def seek_correctness(clip_path: Path, indices: list[int]) -> dict[tuple[str, int], bool]:
    """For each (reader, requested_index), OCR the burned-in counter.

    Strategy: we know the counter clips are 30 fps, black bg, white "%{n}"
    text at top-left starting at frame 0. For seek correctness we don't
    need to OCR — instead, pull a reference frame for the requested index
    via the FFmpeg oracle, then check the reader's output is pixel-exact
    against it. If not pixel-exact, *something* about the seek is off.

    But for stronger evidence: the only thing changing frame-to-frame
    is the digits in the corner. If reader returns the *wrong* frame
    (e.g. shows "5" instead of "12"), pixel comparison vs the oracle for
    index 12 will fail catastrophically (MSE huge, PSNR low). We treat
    "matches_oracle" as a proxy for "seek correct" on the counter clips.
    """
    out: dict[tuple[str, int], bool] = {}
    for idx in indices:
        try:
            ref = oracle.read_frame(clip_path, idx)
        except Exception:
            for reader_name in readers.READERS:
                out[(reader_name, idx)] = False
            continue
        for reader_name, fn in readers.READERS.items():
            try:
                f = fn(clip_path, idx)
                p = report.psnr(f, ref)
                # On counter clips, exact-match for non-mid-GOP is normal;
                # PSNR ≥ 40 dB indicates we got the right frame's pixels
                # (small chroma differences from intra-only re-encode are OK).
                out[(reader_name, idx)] = p >= 40.0
            except Exception:
                out[(reader_name, idx)] = False
    return out


# ---------- Workflow-level test (DUSTrack-style frame extraction) ----------

def _extract_frames_via_reader(reader_fn, video_path: Path,
                                indices: Sequence[int], out_dir: Path) -> list[Path]:
    """Mimic DUSTrack._extract_frames_decord: write img%05d.png to out_dir.

    Skips coords (no crop) for the parity test — the crop is a local op
    that doesn't depend on which reader we use.
    """
    import imageio.v3 as iio
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_frames = max(int(i) for i in indices) + 1
    indexlength = max(1, int(np.ceil(np.log10(max(1, n_frames)))))
    for idx in indices:
        f = reader_fn(video_path, idx)
        p = out_dir / f"img{str(idx).zfill(indexlength)}.png"
        iio.imwrite(str(p), f)
        written.append(p)
    return written


def workflow_parity(video_path: Path, indices: list[int], results_dir: Path) -> dict:
    decord_dir = results_dir / "workflow_decord"
    pyav_dir = results_dir / "workflow_pyav_toc"
    if decord_dir.exists():
        shutil.rmtree(decord_dir)
    if pyav_dir.exists():
        shutil.rmtree(pyav_dir)
    a = _extract_frames_via_reader(readers.decord_read_frame, video_path, indices, decord_dir)
    b = _extract_frames_via_reader(readers.pyav_toc_read_frame, video_path, indices, pyav_dir)
    assert len(a) == len(b)

    sha_match = 0
    arr_match = 0
    import imageio.v3 as iio
    for pa, pb in zip(a, b):
        h1 = hashlib.sha256(pa.read_bytes()).hexdigest()
        h2 = hashlib.sha256(pb.read_bytes()).hexdigest()
        if h1 == h2:
            sha_match += 1
            arr_match += 1
            continue
        # PNG metadata might differ; compare pixel arrays
        ia = iio.imread(str(pa))
        ib = iio.imread(str(pb))
        if ia.shape == ib.shape and np.array_equal(ia, ib):
            arr_match += 1
    n = len(a)
    return {
        "n_frames": n,
        "sha256_identical_pct": 100.0 * sha_match / n if n else 0.0,
        "array_match_pct": 100.0 * arr_match / n if n else 0.0,
        "clip_used": str(video_path),
        "frame_indices": str(indices),
    }


# ---------- env fingerprint ----------

def env_fingerprint() -> dict[str, str]:
    import decord, av, cv2, pims
    try:
        ff = subprocess.run([clip_gen.FFMPEG, "-version"],
                            capture_output=True, text=True).stdout.splitlines()[0]
    except Exception:
        ff = "ffmpeg ???"
    try:
        nv = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        nv = "?"
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "decord": decord.__version__,
        "av (PyAV)": av.__version__,
        "pims": pims.__version__,
        "cv2": cv2.__version__,
        "ffmpeg": ff,
        "gpu": nv,
    }


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="5 indices/clip")
    ap.add_argument("--full", action="store_true", help="up to 30 indices/clip")
    args = ap.parse_args()
    mode = "quick" if (args.quick or not args.full) else "full"
    print(f"[run_parity] mode={mode}")

    results_dir = HERE / "_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Phase 0: ensure corpus exists
    print("[1/5] ensuring corpus...")
    gen_paths = clip_gen.ensure_corpus()
    asis_paths = clip_gen.corpus_asis()
    clips: dict[str, Path] = {**asis_paths, **gen_paths}
    # Skip rgb_probe from per-clip parity (it's 1-frame, used separately)
    rgb_path = clips.pop("rgb_probe", None)

    # Phase 1: RGB/BGR adjudication
    print("[2/5] RGB/BGR adjudication...")
    rgb_results = {}
    if rgb_path is not None:
        rgb_results = adjudicate_rgb_bgr(rgb_path)
        for k, v in rgb_results.items():
            print(f"   {k:12s} -> {v}")
    else:
        print("   (no rgb_probe clip generated)")

    # Phase 2: per-clip parity vs ffmpeg-CLI oracle
    print(f"[3/5] per-clip parity across {len(clips)} clips × {len(readers.READERS)} readers...")
    rows: list[dict] = []
    for clip_name, clip_path in clips.items():
        try:
            n = readers.decord_len(clip_path)
        except Exception as e:
            print(f"   [{clip_name}] decord_len failed: {e}; skipping")
            continue
        try:
            kfs = keyframe_indices(str(clip_path))
        except Exception as e:
            print(f"   [{clip_name}] keyframe scan failed: {e}; assuming kf=[0]")
            kfs = [0]
        kf_set = set(kfs)
        indices = pick_indices(n, kfs, mode)
        print(f"   [{clip_name}] n={n} kf={len(kfs)} -> indices={indices}")
        for idx in indices:
            try:
                ref = oracle.read_frame(clip_path, idx)
            except Exception as e:
                print(f"     oracle FAILED on {idx}: {e}")
                continue
            for reader_name, fn in readers.READERS.items():
                try:
                    f = fn(clip_path, idx)
                except Exception as e:
                    rows.append(dict(
                        clip=clip_name, reader=reader_name, index=idx,
                        is_keyframe=idx in kf_set,
                        pixel_exact=False, l1_mean=float("nan"),
                        psnr_db=float("nan"), matches_oracle=False,
                        error=f"{type(e).__name__}: {e}",
                    ))
                    continue
                m = measure_one(f, ref)
                rows.append(dict(
                    clip=clip_name, reader=reader_name, index=idx,
                    is_keyframe=idx in kf_set, **m,
                ))

    # Phase 3: seek correctness on counter clips (already covered by per-clip
    # parity, but flag explicitly)
    print("[4/5] seek-correctness check on counter clips (via oracle PSNR)...")
    for clip_name in ("counter_x264", "counter_nvenc"):
        sub = [r for r in rows if r["clip"] == clip_name and r["reader"] == "pyav_toc"]
        n = len(sub)
        ok = sum(r["matches_oracle"] for r in sub)
        print(f"   {clip_name}: pyav_toc matches_oracle {ok}/{n}")
        for r in sub:
            r["seek_correct"] = r["matches_oracle"]

    # Phase 4: workflow-level parity (DUSTrack-style)
    print("[5/5] workflow-level (DUSTrack-style PNG extraction)...")
    workflow_clip = clips.get("atem_continuous") or next(iter(clips.values()))
    try:
        n = readers.decord_len(workflow_clip)
        kfs = keyframe_indices(str(workflow_clip))
        wf_indices = pick_indices(n, kfs, mode)
        workflow_results = workflow_parity(workflow_clip, wf_indices, results_dir)
        for k, v in workflow_results.items():
            print(f"   {k}: {v}")
    except Exception as e:
        traceback.print_exc()
        workflow_results = {"error": f"{type(e).__name__}: {e}",
                            "array_match_pct": None}

    # Write report. derive_verdict needs the short channel-order verdict
    # ("RGB" / "BGR"), so pass that. The table rendering in write_report
    # still gets the long-form mapping via the same rgb_results dict.
    fp = env_fingerprint()
    rgb_short = {k: v.split()[0] for k, v in rgb_results.items()}
    out = results_dir / "PARITY_REPORT.md"
    verdict = report.write_report(
        out, rows, rgb_short, workflow_results, fp,
    )
    # Decorate the channel-order section with the full pixel-value notes
    # so the report keeps the diagnostic data without confusing derive_verdict.
    extra = ["", "### Pixel diagnostics (per-reader top-row sample)", ""]
    for reader, full in rgb_results.items():
        extra.append(f"- `{reader}`: {full}")
    extra.append("")
    out.write_text(out.read_text(encoding="utf-8") + "\n" + "\n".join(extra),
                   encoding="utf-8")
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
