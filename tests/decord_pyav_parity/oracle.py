"""ffmpeg-CLI reference oracle.

Uses `select=eq(n,i)` to extract a single frame by container frame index
(frame-exact, including on VFR). Pipes raw PPM and parses to
np.ndarray (H, W, 3) uint8 RGB.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

FFMPEG = r"C:\ffmpeg\bin\ffmpeg.exe"


def read_frame(path: str | Path, index: int) -> np.ndarray:
    """Return frame `index` from `path` via ffmpeg CLI as (H, W, 3) uint8 RGB."""
    path = str(path)
    cmd = [
        FFMPEG, "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-vf", f"select=eq(n\\,{index})",
        "-vsync", "0", "-frames:v", "1",
        "-f", "image2pipe", "-c:v", "ppm", "-",
    ]
    out = subprocess.run(cmd, check=True, capture_output=True).stdout
    return _parse_ppm(out)


def _parse_ppm(data: bytes) -> np.ndarray:
    """Parse a binary PPM (P6) into (H, W, 3) uint8 RGB."""
    # Header: P6\n[# comments\n]*W H\nMAX\n<binary RGB>
    i = 0
    def nextline():
        nonlocal i
        j = data.index(b"\n", i)
        line = data[i:j]
        i = j + 1
        return line

    magic = nextline()
    if magic != b"P6":
        raise ValueError(f"not a P6 PPM: {magic!r}")
    # skip comments
    while True:
        # Peek
        if data[i:i+1] == b"#":
            nextline()
            continue
        break
    w, h = map(int, nextline().split())
    maxv = int(nextline())
    if maxv != 255:
        raise ValueError(f"PPM maxval={maxv}, expected 255")
    raw = data[i:i + w * h * 3]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
    return arr
