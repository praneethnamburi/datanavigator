"""Reader implementations under test.

All readers expose: read_frame(path, index) -> np.ndarray (H, W, 3) uint8 RGB
and read_batch(path, indices) -> np.ndarray (N, H, W, 3) uint8 RGB.

The contract is "give me frame `i` as RGB." Each reader normalizes its
native return format to RGB so the parity comparison is apples-to-apples.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------- decord ----------

import decord

decord.bridge.set_bridge("native")


@lru_cache(maxsize=8)
def _decord_vr(path: str):
    return decord.VideoReader(path, ctx=decord.cpu(0), num_threads=1)


def decord_read_frame(path: str | Path, index: int) -> np.ndarray:
    vr = _decord_vr(str(path))
    return vr.get_batch([int(index)]).asnumpy()[0]


def decord_read_batch(path: str | Path, indices: Sequence[int]) -> np.ndarray:
    vr = _decord_vr(str(path))
    return vr.get_batch([int(i) for i in indices]).asnumpy()


def decord_len(path: str | Path) -> int:
    return len(_decord_vr(str(path)))


# ---------- PyAV naive (what people write first) ----------

import av


def pyav_naive_read_frame(path: str | Path, index: int) -> np.ndarray:
    """Open, time-seek to the keyframe before the target, decode forward.

    Expected to be inaccurate on mid-GOP for B-frame-heavy streams —
    that's the point of including it as the "naive baseline".
    """
    path = str(path)
    with av.open(path) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        if fps <= 0:
            fps = 30.0
        target_time = index / fps
        target_pts = int(target_time / float(stream.time_base))
        container.seek(target_pts, stream=stream, any_frame=False, backward=True)
        for frame in container.decode(stream):
            # frame index from PTS
            fi = int(round((float(frame.pts) - 0.0) * float(stream.time_base) * fps))
            if fi >= index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(f"frame {index} not found in {path}")


# ---------- PyAV + frame-index TOC (via pims, BSD-3) ----------

import pims


@lru_cache(maxsize=8)
def _pims_reader(path: str):
    # PyAVReaderIndexed walks the file once to build a packet TOC,
    # then random-access reads are seek-to-packet + decode-up-to-frame.
    return pims.PyAVReaderIndexed(path)


def pyav_toc_read_frame(path: str | Path, index: int) -> np.ndarray:
    r = _pims_reader(str(path))
    return np.asarray(r[int(index)])


def pyav_toc_read_batch(path: str | Path, indices: Sequence[int]) -> np.ndarray:
    r = _pims_reader(str(path))
    out = [np.asarray(r[int(i)]) for i in indices]
    return np.stack(out, axis=0)


def pyav_toc_len(path: str | Path) -> int:
    return len(_pims_reader(str(path)))


# ---------- OpenCV (positive control — known to be wrong on NVENC) ----------

import cv2


def opencv_read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if not ok:
            raise IndexError(f"opencv could not read frame {index} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


# ---------- Registry ----------

READERS = {
    "decord":     decord_read_frame,
    "pyav_naive": pyav_naive_read_frame,
    "pyav_toc":   pyav_toc_read_frame,
    "opencv":     opencv_read_frame,
}
