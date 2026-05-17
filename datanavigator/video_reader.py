"""decord-compatible ``VideoReader`` shim backed by PyAV + a frame-index TOC.

Wraps :class:`datanavigator._vendor.pims_pyav_reader.PyAVReaderIndexed` so
existing call sites that were written against decord
(``vr[i].asnumpy()``, ``vr.get_batch([...]).asnumpy()``, ``len(vr)``,
``vr.get_avg_fps()``, ``cpu(0)``) continue to work unchanged.

The 2026-05-16 parity test (``tests/decord_pyav_parity/``) confirmed
PyAV+TOC matches the ffmpeg-CLI oracle pixel-exact on every clip and
codec exercised, where decord diverged on 8/28 frames of a production
VFR clip — so this swap is a correctness improvement, not just a
dependency cleanup.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Union

import numpy as np

from ._vendor.pims_pyav_reader import PyAVReaderIndexed


class _CpuCtx:
    """Sentinel returned by :func:`cpu`. Exists only for decord call-site parity."""

    def __init__(self, device_id: int = 0) -> None:
        self.device_id = int(device_id)

    def __repr__(self) -> str:
        return f"cpu({self.device_id})"


def cpu(device_id: int = 0) -> _CpuCtx:
    """decord.cpu() compatibility shim. Returns a sentinel; PyAV is CPU-only."""
    return _CpuCtx(device_id)


class _Frame(np.ndarray):
    """ndarray subclass exposing ``.asnumpy()`` for decord parity."""

    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    def __array_finalize__(self, obj):
        if obj is None:
            return

    def asnumpy(self) -> np.ndarray:
        return np.asarray(self)


class _FrameBatch:
    """Holds a list of frames and stacks them on ``.asnumpy()``."""

    def __init__(self, frames: Sequence[np.ndarray]) -> None:
        self._frames = list(frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        for f in self._frames:
            yield _Frame(f)

    def __getitem__(self, i):
        return _Frame(self._frames[i])

    def asnumpy(self) -> np.ndarray:
        if not self._frames:
            return np.empty((0,), dtype=np.uint8)
        return np.stack([np.asarray(f) for f in self._frames], axis=0)


class VideoReader:
    """decord-compatible random-access video reader backed by PyAV+TOC.

    The constructor signature mirrors ``decord.VideoReader`` so existing
    call sites and subclasses continue to work; ``ctx``, ``width``,
    ``height``, ``num_threads``, ``fault_tol`` are accepted but ignored
    (PyAV honors the source resolution and is CPU-only here).
    """

    def __init__(
        self,
        uri,
        ctx=None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        # decord accepted both string paths and file-like objects; preserve
        # that for downstream call sites (e.g. videos.VideoBrowser opens
        # the file as a BufferedReader and hands the handle in). Resolve
        # file-likes to their underlying path so PyAV's libavformat-side
        # open() — which works on filenames, not Python streams — keeps
        # working regardless of caller style.
        if hasattr(uri, "read") and hasattr(uri, "name"):
            self._uri = str(uri.name)
        else:
            self._uri = str(uri)
        self._reader = PyAVReaderIndexed(self._uri)
        self._avg_fps = self._probe_avg_fps(self._uri)

    @staticmethod
    def _probe_avg_fps(path: str) -> float:
        import av

        with av.open(path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            return float(rate) if rate is not None else 0.0

    def __len__(self) -> int:
        return int(len(self._reader))

    def __iter__(self):
        for i in range(len(self)):
            yield _Frame(self._reader.get_frame(i))

    def __getitem__(self, key) -> Union[_Frame, _FrameBatch]:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return _FrameBatch([self._reader.get_frame(i) for i in range(start, stop, step)])
        if isinstance(key, (list, tuple, np.ndarray)):
            return _FrameBatch([self._reader.get_frame(int(i)) for i in key])
        idx = int(key)
        if idx < 0:
            idx += len(self)
        return _Frame(self._reader.get_frame(idx))

    def get_batch(self, indices: Iterable[int]) -> _FrameBatch:
        idxs = [int(i) for i in indices]
        n = len(self)
        norm = [i + n if i < 0 else i for i in idxs]
        return _FrameBatch([self._reader.get_frame(i) for i in norm])

    def get_avg_fps(self) -> float:
        return float(self._avg_fps)
