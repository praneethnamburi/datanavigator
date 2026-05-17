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

TOC build cost is paid once per video and cached as a sidecar
(``<video>.dnav-toc``, JSON content) keyed on path + size + mtime +
SHA-256 of the first/last 64 KiB. Subsequent opens are O(stat) + small
read. The composite suffix is deliberately *not* ``.json`` to avoid
collisions with downstream tooling that walks ``*.json`` (DUSTrack
annotation discovery, ad-hoc notebooks, etc.). Use :func:`precompute_toc`
to batch-warm the cache before an annotation session.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence, Union

import numpy as np

from ._vendor.pims_pyav_reader import PyAVReaderIndexed


_SIDECAR_SUFFIX = ".dnav-toc"
_SIDECAR_SCHEMA_VERSION = 1
_SHA_PROBE_BYTES = 64 * 1024  # 64 KiB head + 64 KiB tail


# ---------- TOC sidecar cache ----------

def _sidecar_path(video_path: str) -> Path:
    """Return the sidecar path: ``<video>.dnav-toc`` (JSON content, no .json
    suffix — keeps the file invisible to ``*.json`` walkers in downstream
    tooling like DUSTrack annotation discovery)."""
    return Path(str(video_path) + _SIDECAR_SUFFIX)


def _cache_key(video_path: str) -> dict:
    """Build the cache-validation key: path + size + mtime + head/tail SHA-256.

    Cheap (one stat + at most 128 KiB read). Catches the common ways a
    video file changes between sessions (re-encode, replace, truncation)
    without needing to hash the whole file.
    """
    p = Path(video_path)
    st = p.stat()
    size = int(st.st_size)
    mtime = float(st.st_mtime)
    with open(video_path, "rb") as f:
        head = f.read(min(_SHA_PROBE_BYTES, size))
        if size > _SHA_PROBE_BYTES:
            f.seek(max(0, size - _SHA_PROBE_BYTES))
            tail = f.read(_SHA_PROBE_BYTES)
        else:
            tail = head
    return {
        "size": size,
        "mtime": mtime,
        "head_sha256": hashlib.sha256(head).hexdigest(),
        "tail_sha256": hashlib.sha256(tail).hexdigest(),
    }


def _load_sidecar(video_path: str, current_key: dict) -> dict | None:
    """Load the cached TOC if the sidecar matches ``current_key``, else None.

    Returns None on any failure mode (missing, corrupted, wrong schema,
    key mismatch) so the caller can fall back to building from scratch.
    """
    sidecar = _sidecar_path(video_path)
    if not sidecar.exists():
        return None
    try:
        with open(sidecar, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("schema_version") != _SIDECAR_SCHEMA_VERSION:
        return None
    if data.get("key") != current_key:
        return None
    toc = data.get("toc")
    if not isinstance(toc, dict) or "lengths" not in toc or "ts" not in toc:
        return None
    return toc


def _serialize_payload(payload: dict) -> str:
    """Pretty-print the outer structure but keep the long ``lengths`` /
    ``ts`` arrays on a single line each — so the ``key`` / ``schema_version``
    header stays human-readable in any editor while the file size is close
    to the fully-compact form (≈3× smaller than fully pretty-printed at
    1800 frames). Implementation: serialize the arrays compactly, slot
    them into the indented outer dict via unique placeholders that can't
    appear in the rest of the payload (SHA hex / ints / version int).
    """
    lengths_compact = json.dumps([int(x) for x in payload["toc"]["lengths"]])
    ts_compact = json.dumps([int(x) for x in payload["toc"]["ts"]])
    outer = {
        "key": payload["key"],
        "schema_version": payload["schema_version"],
        "toc": {
            "lengths": "__DNAV_TOC_LENGTHS__",
            "ts": "__DNAV_TOC_TS__",
        },
    }
    text = json.dumps(outer, indent=2, sort_keys=True)
    text = text.replace('"__DNAV_TOC_LENGTHS__"', lengths_compact)
    text = text.replace('"__DNAV_TOC_TS__"', ts_compact)
    return text + "\n"


def _save_sidecar(video_path: str, toc: dict, key: dict) -> bool:
    """Atomically write the sidecar. Returns False if the write failed
    (e.g. read-only directory) — not fatal; the caller continues with the
    in-memory TOC.
    """
    sidecar = _sidecar_path(video_path)
    payload = {
        "schema_version": _SIDECAR_SCHEMA_VERSION,
        "key": key,
        "toc": {
            "lengths": [int(x) for x in toc["lengths"]],
            "ts": [int(x) for x in toc["ts"]],
        },
    }
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=str(sidecar.parent),
            prefix=sidecar.name + ".",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(_serialize_payload(payload))
            os.replace(tmp_path, sidecar)
            return True
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except OSError as e:
        print(
            f"datanavigator: could not write TOC cache to {sidecar}: {e}",
            file=sys.stderr,
        )
        return False


def _build_and_cache_toc(video_path: str, *, announce: bool) -> PyAVReaderIndexed:
    """Build the TOC by demuxing the whole file, then save the sidecar.

    ``announce`` controls the user-facing "Building TOC..." print.
    """
    if announce:
        print(f"datanavigator: building TOC for {Path(video_path).name}...")
    reader = PyAVReaderIndexed(video_path)
    try:
        key = _cache_key(video_path)
        _save_sidecar(video_path, reader._toc, key)
    except OSError as e:
        # stat or read failed somehow; reader is still usable, just uncached
        print(
            f"datanavigator: TOC built but cache key could not be computed for "
            f"{video_path}: {e}",
            file=sys.stderr,
        )
    return reader


def _open_with_cache(video_path: str) -> PyAVReaderIndexed:
    """Open ``video_path``, using the sidecar TOC cache when available."""
    if not Path(video_path).is_file():
        # Network paths, file-likes resolved to non-file URIs, etc.: skip the
        # cache and let PyAVReaderIndexed handle whatever was passed in.
        return PyAVReaderIndexed(video_path)
    try:
        key = _cache_key(video_path)
    except OSError:
        return PyAVReaderIndexed(video_path)

    cached_toc = _load_sidecar(video_path, key)
    if cached_toc is not None:
        return PyAVReaderIndexed(video_path, toc=cached_toc)
    return _build_and_cache_toc(video_path, announce=True)


def precompute_toc(
    paths: Iterable[Union[str, os.PathLike]],
    *,
    force: bool = False,
    show_progress: bool = True,
) -> dict:
    """Batch-build and cache TOCs for a sequence of video files.

    Useful before an annotation session, so the per-video "Building
    TOC..." pause doesn't happen interactively::

        import glob, datanavigator
        datanavigator.precompute_toc(glob.glob("data/*.mp4"))

    Args:
        paths: Iterable of video file paths.
        force: If True, rebuild + overwrite even when a valid cache exists.
        show_progress: If True (default), wrap iteration in a tqdm bar.

    Returns:
        ``{path: status}`` where status is ``"hit"`` (cache already valid),
        ``"built"`` (rebuilt and cached), or ``f"error: {msg}"`` (skipped).
    """
    paths_list = [str(p) for p in paths]
    if show_progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(paths_list, desc="Building TOCs", unit="video")
        except ImportError:
            iterator = paths_list
    else:
        iterator = paths_list

    results: dict[str, str] = {}
    for path_str in iterator:
        try:
            if not Path(path_str).is_file():
                results[path_str] = "error: not a file"
                continue
            key = _cache_key(path_str)
            if not force and _load_sidecar(path_str, key) is not None:
                results[path_str] = "hit"
                continue
            reader = PyAVReaderIndexed(path_str)
            ok = _save_sidecar(path_str, reader._toc, key)
            results[path_str] = "built" if ok else "built (uncached)"
        except Exception as e:  # noqa: BLE001 — surface per-file errors as status
            results[path_str] = f"error: {e}"
    return results


# ---------- Shim API ----------

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

    A frame-index TOC is built from the video at open time and cached
    next to it as ``<video>.dnav-toc`` (JSON content; the ``.json``
    extension is intentionally omitted so ``*.json`` walkers in
    downstream tooling don't pick the sidecar up). The key is path +
    size + mtime + head/tail SHA-256. On a cache hit the second open is
    sub-second; on a miss the first open prints a one-line "building
    TOC..." notice. Use :func:`precompute_toc` to warm the cache for a
    set of videos before an interactive session.
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
        self._reader = _open_with_cache(self._uri)
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
