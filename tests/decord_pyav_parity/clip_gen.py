"""Materialize transcoded + synthetic clips into S:/_corpus/_parity_test/.

Idempotent: skips files that already exist. Run via ``python -m clip_gen``
or via ``run_parity.py`` which imports ``ensure_corpus()``.

Generated clips:
    x264_bf3.mp4          libx264 with -bf 3 (mid-GOP seek stress, B-frame)
    nvenc_h264.mp4        h264_nvenc plain (immersionToolbox-flagged case)
    nvenc_h264_bf3.mp4    h264_nvenc with -bf 3 (B-frame on GPU encoder)
    nvenc_hevc.mp4        hevc_nvenc
    pair_cpu.mp4          libx264 from a raw y4m (CPU side of A/B pair)
    pair_gpu.mp4          h264_nvenc from the same y4m (GPU side)
    counter_x264.mp4      libx264 with drawtext frame counter (ground truth)
    counter_nvenc.mp4     h264_nvenc with drawtext frame counter
    rgb_probe.mp4         red-top blue-bottom stripe for RGB/BGR adjudication
"""
from __future__ import annotations

import subprocess
from pathlib import Path

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"
SRC_ATEM = Path(r"S:\_corpus\atem\061_continuous\061_continuous_atem.mp4")
SRC_TELEMED = next(
    Path(r"S:\_corpus\telemed").glob("*.mp4"), None
) if Path(r"S:\_corpus\telemed").exists() else None

OUT = Path(r"S:\_corpus\_parity_test")

# Font for drawtext on Windows. Path-escaped per ffmpeg's filter requirements.
# Use a forward-slash + escaped colon form which ffmpeg's filter parser accepts.
DRAWTEXT_FONT = r"C\:/Windows/Fonts/consola.ttf"


def _run(cmd: list[str]) -> None:
    print(">", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def _skip_if_exists(p: Path) -> bool:
    if p.exists() and p.stat().st_size > 0:
        print(f"  [skip] {p.name} ({p.stat().st_size:,} bytes)")
        return True
    return False


def ensure_corpus(duration_s: int = 10) -> dict[str, Path]:
    """Materialize every test clip. Returns mapping name -> path."""
    OUT.mkdir(parents=True, exist_ok=True)
    if not SRC_ATEM.exists():
        raise FileNotFoundError(f"missing source: {SRC_ATEM}")

    paths: dict[str, Path] = {}

    # 1. x264 with B-frames
    p = OUT / "x264_bf3.mp4"
    paths["x264_bf3"] = p
    if not _skip_if_exists(p):
        _run([FFMPEG, "-y", "-i", str(SRC_ATEM),
              "-c:v", "libx264", "-preset", "medium",
              "-bf", "3", "-g", "60", "-keyint_min", "60",
              "-pix_fmt", "yuv420p", "-an", "-t", str(duration_s), str(p)])

    # 2. NVENC plain
    p = OUT / "nvenc_h264.mp4"
    paths["nvenc_h264"] = p
    if not _skip_if_exists(p):
        _run([FFMPEG, "-y", "-i", str(SRC_ATEM),
              "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23",
              "-g", "60", "-pix_fmt", "yuv420p", "-an", "-t", str(duration_s), str(p)])

    # 3. NVENC with B-frames
    p = OUT / "nvenc_h264_bf3.mp4"
    paths["nvenc_h264_bf3"] = p
    if not _skip_if_exists(p):
        _run([FFMPEG, "-y", "-i", str(SRC_ATEM),
              "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "23",
              "-g", "60", "-bf", "3", "-pix_fmt", "yuv420p",
              "-an", "-t", str(duration_s), str(p)])

    # 4. HEVC NVENC (use telemed as source if available, else ATEM)
    src_hevc = SRC_TELEMED if SRC_TELEMED and SRC_TELEMED.exists() else SRC_ATEM
    p = OUT / "nvenc_hevc.mp4"
    paths["nvenc_hevc"] = p
    if not _skip_if_exists(p):
        _run([FFMPEG, "-y", "-i", str(src_hevc),
              "-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "25",
              "-g", "60", "-pix_fmt", "yuv420p", "-an", "-t", str(duration_s), str(p)])

    # 5. Identical-source CPU/GPU pair via y4m
    y4m = OUT / "_raw.y4m"
    pair_cpu = OUT / "pair_cpu.mp4"
    pair_gpu = OUT / "pair_gpu.mp4"
    paths["pair_cpu"] = pair_cpu
    paths["pair_gpu"] = pair_gpu
    if not (_skip_if_exists(pair_cpu) and _skip_if_exists(pair_gpu)):
        if not y4m.exists():
            _run([FFMPEG, "-y", "-i", str(SRC_ATEM), "-t", str(duration_s),
                  "-pix_fmt", "yuv420p", str(y4m)])
        if not pair_cpu.exists():
            _run([FFMPEG, "-y", "-i", str(y4m),
                  "-c:v", "libx264", "-crf", "20", "-g", "60", "-bf", "3",
                  "-pix_fmt", "yuv420p", "-an", str(pair_cpu)])
        if not pair_gpu.exists():
            _run([FFMPEG, "-y", "-i", str(y4m),
                  "-c:v", "h264_nvenc", "-cq", "23", "-g", "60", "-bf", "3",
                  "-pix_fmt", "yuv420p", "-an", str(pair_gpu)])
        # keep _raw.y4m around; might be 1-2 GB but lets the pair regen cheaply

    # 6. Frame counter clips (ground-truth oracle for seek correctness)
    for codec_name, codec_args in [
        ("counter_x264", ["-c:v", "libx264", "-preset", "medium"]),
        ("counter_nvenc", ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "20"]),
    ]:
        p = OUT / f"{codec_name}.mp4"
        paths[codec_name] = p
        if _skip_if_exists(p):
            continue
        drawtext = (
            f"drawtext=fontfile='{DRAWTEXT_FONT}':text='%{{n}}':"
            "fontsize=72:fontcolor=white:x=20:y=20:start_number=0"
        )
        _run([FFMPEG, "-y",
              "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=10",
              "-vf", drawtext,
              *codec_args,
              "-bf", "3", "-g", "30", "-pix_fmt", "yuv420p", str(p)])

    # 7. RGB/BGR probe (red top, blue bottom)
    p = OUT / "rgb_probe.mp4"
    paths["rgb_probe"] = p
    if not _skip_if_exists(p):
        filt = (
            "color=c=red:s=64x32:d=1[a];"
            "color=c=blue:s=64x32:d=1[b];"
            "[a][b]vstack=inputs=2"
        )
        _run([FFMPEG, "-y", "-f", "lavfi", "-i", filt,
              "-frames:v", "1", "-pix_fmt", "yuv420p",
              "-c:v", "libx264", str(p)])

    return paths


# Pre-existing corpus clips (as-is, no generation needed)
def corpus_asis() -> dict[str, Path]:
    out: dict[str, Path] = {}
    candidates = [
        ("atem_continuous", Path(r"S:\_corpus\atem\061_continuous\061_continuous_atem.mp4")),
        ("atem_with_gap",   Path(r"S:\_corpus\atem\061_with_gap\061_with_gap_atem.mp4")),
    ]
    if SRC_TELEMED is not None:
        candidates.append(("telemed_vfr", SRC_TELEMED))
    for name, p in candidates:
        if p.exists():
            out[name] = p
        else:
            print(f"  [missing] {name}: {p}")
    return out


if __name__ == "__main__":
    paths = ensure_corpus()
    for k, v in paths.items():
        print(f"  {k:18s} -> {v}")
