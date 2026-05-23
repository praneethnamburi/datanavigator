"""decord-compatible ``VideoReader`` shim backed by PyAV + a frame-index TOC.

Wraps :class:`datanavigator._vendor.pims_pyav_reader.PyAVReaderIndexed` so
existing call sites that were written against decord
(``vr[i].asnumpy()``, ``vr.get_batch([...]).asnumpy()``, ``len(vr)``,
``vr.get_avg_fps()``, ``vr.get_frame_timestamp([...])``, ``cpu(0)``)
continue to work unchanged.

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
annotation discovery, ad-hoc notebooks, etc.). Schema v2 also records
per-frame pts/duration so ``get_frame_timestamp`` is a cache hit;
v1 sidecars stay readable and lazy-upgrade on first call. Use
:func:`precompute_toc` to batch-warm the cache before an annotation
session.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np

from ._vendor.pims_pyav_reader import PyAVReaderIndexed

_SIDECAR_SUFFIX = ".dnav-toc"
_SIDECAR_SCHEMA_VERSION = 2
_SUPPORTED_SCHEMA_VERSIONS = (1, 2)
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
    """Load the cached payload if the sidecar matches ``current_key``, else None.

    Returns the full payload dict (``schema_version`` + ``toc`` + optional
    v2 ``timestamps`` / ``time_base``). Callers branch on
    ``payload["schema_version"]``. Returns None on any failure mode
    (missing, corrupted, unsupported schema, key mismatch).
    """
    sidecar = _sidecar_path(video_path)
    if not sidecar.exists():
        return None
    try:
        with open(sidecar, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if data.get("schema_version") not in _SUPPORTED_SCHEMA_VERSIONS:
        return None
    if data.get("key") != current_key:
        return None
    toc = data.get("toc")
    if not isinstance(toc, dict) or "lengths" not in toc or "ts" not in toc:
        return None
    return data


def _serialize_payload(payload: dict) -> str:
    """Pretty-print the outer structure but keep the long ``lengths`` /
    ``ts`` / ``frame_pts`` / ``frame_durations`` arrays on a single line
    each — so the ``key`` / ``schema_version`` / ``time_base`` header
    stays human-readable in any editor while the file size is close to
    the fully-compact form. Implementation: serialize the arrays
    compactly, slot them into the indented outer dict via unique
    placeholders that can't appear in the rest of the payload
    (SHA hex / ints / version int).
    """
    lengths_compact = json.dumps([int(x) for x in payload["toc"]["lengths"]])
    ts_compact = json.dumps([int(x) for x in payload["toc"]["ts"]])
    outer: dict = {
        "key": payload["key"],
        "schema_version": payload["schema_version"],
        "toc": {
            "lengths": "__DNAV_TOC_LENGTHS__",
            "ts": "__DNAV_TOC_TS__",
        },
    }
    replacements: list[tuple[str, str]] = [
        ('"__DNAV_TOC_LENGTHS__"', lengths_compact),
        ('"__DNAV_TOC_TS__"', ts_compact),
    ]
    if payload["schema_version"] >= 2:
        outer["time_base"] = payload["time_base"]
        outer["timestamps"] = {
            "frame_pts": "__DNAV_FRAME_PTS__",
            "frame_durations": "__DNAV_FRAME_DURATIONS__",
        }
        replacements.append(
            (
                '"__DNAV_FRAME_PTS__"',
                json.dumps([int(x) for x in payload["timestamps"]["frame_pts"]]),
            )
        )
        replacements.append(
            (
                '"__DNAV_FRAME_DURATIONS__"',
                json.dumps([int(x) for x in payload["timestamps"]["frame_durations"]]),
            )
        )
    text = json.dumps(outer, indent=2, sort_keys=True)
    for placeholder, compact in replacements:
        text = text.replace(placeholder, compact)
    return text + "\n"


def _save_sidecar(
    video_path: str,
    toc: dict,
    key: dict,
    *,
    timestamps: dict | None = None,
    time_base: dict | None = None,
) -> bool:
    """Atomically write the sidecar. Returns False if the write failed
    (e.g. read-only directory) — not fatal; the caller continues with the
    in-memory data. When ``timestamps`` + ``time_base`` are supplied, the
    sidecar is written as schema v2; otherwise as v1.
    """
    sidecar = _sidecar_path(video_path)
    if timestamps is not None and time_base is not None:
        payload = {
            "schema_version": 2,
            "key": key,
            "toc": {
                "lengths": [int(x) for x in toc["lengths"]],
                "ts": [int(x) for x in toc["ts"]],
            },
            "time_base": {
                "num": int(time_base["num"]),
                "den": int(time_base["den"]),
            },
            "timestamps": {
                "frame_pts": [int(x) for x in timestamps["frame_pts"]],
                "frame_durations": [int(x) for x in timestamps["frame_durations"]],
            },
        }
    else:
        payload = {
            "schema_version": 1,
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


def _build_toc_and_timestamps(video_path: str) -> tuple[dict, dict, dict]:
    """Single demux + per-frame decode pass producing the TOC, per-frame
    pts/durations (in stream time_base units), and the stream time_base.

    Returns ``(toc, timestamps, time_base)`` where:
    - ``toc = {"lengths": [...], "ts": [...]}`` (per-packet)
    - ``timestamps = {"frame_pts": [...], "frame_durations": [...]}`` (per-frame)
    - ``time_base = {"num": int, "den": int}``
    """
    import av

    packet_lengths: list[int] = []
    packet_ts: list[int] = []
    frame_pts: list[int] = []
    frame_durations: list[int] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        time_base = stream.time_base
        # Fallback frame duration in time_base units, derived from average_rate.
        if stream.average_rate is not None and float(stream.average_rate) > 0:
            avg_step = int(
                round(time_base.denominator / (float(stream.average_rate) * time_base.numerator))
            )
        else:
            avg_step = 0

        for packet in container.demux(stream):
            if packet.stream.type != "video":
                continue
            decoded = packet.decode()
            if len(decoded) == 0:
                continue
            packet_lengths.append(len(decoded))
            packet_ts.append(int(decoded[0].pts))
            for frame in decoded:
                pts = frame.pts
                if pts is None:
                    pts = (frame_pts[-1] + (frame_durations[-1] or avg_step)) if frame_pts else 0
                frame_pts.append(int(pts))
                duration = frame.duration
                frame_durations.append(int(duration) if duration is not None else avg_step)

    # Fix up trailing zero-duration frames using inter-frame deltas, falling
    # back to avg_step. Decord exposes a sensible duration here on VFR clips
    # too, and tobii's .mean(-1) on the (start, end) pair only makes sense
    # when end > start.
    for i in range(len(frame_durations) - 1):
        if frame_durations[i] == 0:
            frame_durations[i] = max(0, frame_pts[i + 1] - frame_pts[i]) or avg_step
    if frame_durations and frame_durations[-1] == 0:
        frame_durations[-1] = avg_step

    toc = {"lengths": packet_lengths, "ts": packet_ts}
    timestamps = {"frame_pts": frame_pts, "frame_durations": frame_durations}
    time_base_dict = {"num": time_base.numerator, "den": time_base.denominator}
    return toc, timestamps, time_base_dict


class _CacheLoadResult:
    """What _open_with_cache returns to VideoReader.__init__."""

    __slots__ = ("reader", "frame_pts", "frame_durations", "time_base_num", "time_base_den")

    def __init__(
        self,
        reader: PyAVReaderIndexed,
        frame_pts: np.ndarray | None,
        frame_durations: np.ndarray | None,
        time_base_num: int | None,
        time_base_den: int | None,
    ) -> None:
        self.reader = reader
        self.frame_pts = frame_pts
        self.frame_durations = frame_durations
        self.time_base_num = time_base_num
        self.time_base_den = time_base_den


# Source pix_fmts that are monochrome. When ``VideoReader(pix_fmt=None)``
# detects one of these in the source, it auto-selects the gray decode
# path. Listed exhaustively rather than via substring match because PyAV
# also exposes packed-luma formats whose names contain "y" but encode
# multi-channel data (e.g. ``yuyv422``).
_MONOCHROME_SOURCE_PIX_FMTS = frozenset(
    {
        "gray",
        "gray8",
        "gray9",
        "gray9le",
        "gray9be",
        "gray10",
        "gray10le",
        "gray10be",
        "gray12",
        "gray12le",
        "gray12be",
        "gray14",
        "gray14le",
        "gray14be",
        "gray16",
        "gray16le",
        "gray16be",
        "grayf32",
        "grayf32le",
        "grayf32be",
        "yuvj400p",
        "yuv400p",
    }
)


def _detect_source_pix_fmt(path: str) -> str:
    """Probe the source's encoded pix_fmt via PyAV stream metadata.

    Returns the format name string. The auto-detect logic in
    :class:`VideoReader` maps anything in :data:`_MONOCHROME_SOURCE_PIX_FMTS`
    to the gray decode path.
    """
    import av

    with av.open(path) as container:
        stream = container.streams.video[0]
        fmt = stream.codec_context.pix_fmt
    return fmt or ""


def _build_and_cache_all(video_path: str, *, announce: bool, pix_fmt: str) -> _CacheLoadResult:
    """Build TOC + per-frame timestamps in one pass, save v2 sidecar."""
    if announce:
        print(f"datanavigator: building TOC for {Path(video_path).name}...")
    toc, timestamps, time_base = _build_toc_and_timestamps(video_path)
    reader = PyAVReaderIndexed(video_path, toc=toc, pix_fmt=pix_fmt)
    try:
        key = _cache_key(video_path)
        _save_sidecar(video_path, toc, key, timestamps=timestamps, time_base=time_base)
    except OSError as e:
        print(
            f"datanavigator: TOC built but cache key could not be computed for "
            f"{video_path}: {e}",
            file=sys.stderr,
        )
    return _CacheLoadResult(
        reader=reader,
        frame_pts=np.asarray(timestamps["frame_pts"], dtype=np.int64),
        frame_durations=np.asarray(timestamps["frame_durations"], dtype=np.int64),
        time_base_num=time_base["num"],
        time_base_den=time_base["den"],
    )


def _open_with_cache(video_path: str, *, pix_fmt: str) -> _CacheLoadResult:
    """Open ``video_path``, using the sidecar cache when available."""
    if not Path(video_path).is_file():
        # Network paths, file-likes resolved to non-file URIs, etc.: skip the
        # cache and let PyAVReaderIndexed handle whatever was passed in.
        return _CacheLoadResult(
            PyAVReaderIndexed(video_path, pix_fmt=pix_fmt), None, None, None, None
        )
    try:
        key = _cache_key(video_path)
    except OSError:
        return _CacheLoadResult(
            PyAVReaderIndexed(video_path, pix_fmt=pix_fmt), None, None, None, None
        )

    payload = _load_sidecar(video_path, key)
    if payload is not None:
        toc = payload["toc"]
        reader = PyAVReaderIndexed(video_path, toc=toc, pix_fmt=pix_fmt)
        if payload["schema_version"] >= 2:
            ts = payload["timestamps"]
            tb = payload["time_base"]
            return _CacheLoadResult(
                reader=reader,
                frame_pts=np.asarray(ts["frame_pts"], dtype=np.int64),
                frame_durations=np.asarray(ts["frame_durations"], dtype=np.int64),
                time_base_num=int(tb["num"]),
                time_base_den=int(tb["den"]),
            )
        # v1 sidecar: TOC only. Timestamps lazy-built (and sidecar
        # lazy-upgraded) on first get_frame_timestamp() call.
        return _CacheLoadResult(reader, None, None, None, None)

    return _build_and_cache_all(video_path, announce=True, pix_fmt=pix_fmt)


DEFAULT_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
)


def _iter_video_files(
    sources: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
    *,
    extensions: Iterable[str],
    recursive: bool,
) -> list[Path]:
    """Resolve ``sources`` to a sorted, deduplicated list of video file paths.

    Each item may be a file (kept as-is) or a directory (walked for
    ``extensions``). Extension matching is case-insensitive. Order:
    each input is processed in turn, and within each directory the walk
    is sorted alphabetically; duplicates across inputs are dropped on
    second occurrence.
    """
    norm_exts = {ext.lower() for ext in extensions}
    if isinstance(sources, (str, os.PathLike)):
        items: list[Path] = [Path(sources)]
    else:
        items = [Path(s) for s in sources]

    out: list[Path] = []
    seen: set[Path] = set()

    def _add(fp: Path) -> None:
        try:
            key = fp.resolve()
        except OSError:
            key = fp
        if key in seen:
            return
        seen.add(key)
        out.append(fp)

    for item in items:
        if item.is_dir():
            walker = item.rglob("*") if recursive else item.glob("*")
            for fp in sorted(walker):
                if fp.is_file() and fp.suffix.lower() in norm_exts:
                    _add(fp)
        elif item.is_file():
            _add(item)
        # Silently skip non-existent entries; precompute_toc_folder reports
        # them as errors below if they were named explicitly.
    return out


def precompute_toc_folder(
    folder: Union[str, os.PathLike, Iterable[Union[str, os.PathLike]]],
    *,
    extensions: Iterable[str] = DEFAULT_VIDEO_EXTENSIONS,
    recursive: bool = True,
    force: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Walk a folder (or list of folders / files) and build TOCs for each video.

    Thin sibling of :func:`precompute_toc` that handles directory walking
    so callers don't have to glob themselves. Each ``folder`` entry may
    be a directory (walked for ``extensions``) or a file (kept as-is).
    Non-existent entries that were named explicitly are surfaced as
    ``"error: missing"`` in the returned dict; directories that turn up
    no matching videos are silently empty.

    Example::

        import datanavigator
        results = datanavigator.precompute_toc_folder(
            "M:/us_videos_for_tracking2",
        )
        # {'M:/.../foo.mp4': 'hit', 'M:/.../bar.mp4': 'built', ...}

    Args:
        folder: A directory path, a file path, or an iterable mixing both.
        extensions: File extensions to include (case-insensitive). Default
            covers the common video container formats.
        recursive: If True (default), recurse into subdirectories.
        force: Forwarded to :func:`precompute_toc` — rebuild even on hit.
        show_progress: Forwarded to :func:`precompute_toc` — tqdm bar.
        progress_callback: Forwarded to :func:`precompute_toc`.
        cancel_check: Forwarded to :func:`precompute_toc`.

    Returns:
        ``{path: status}`` per :func:`precompute_toc`, with an extra
        ``"error: missing"`` entry for any explicitly-named path that
        doesn't exist.
    """
    # Capture explicit-but-missing entries so the caller can audit them.
    if isinstance(folder, (str, os.PathLike)):
        explicit = [Path(folder)]
    else:
        explicit = [Path(s) for s in folder]
    missing = {str(p): "error: missing" for p in explicit if not p.exists()}

    files = _iter_video_files(folder, extensions=extensions, recursive=recursive)
    results = precompute_toc(
        files,
        force=force,
        show_progress=show_progress,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    # Merge missing-entries dict, but don't overwrite real statuses on key collisions.
    for path_str, status in missing.items():
        results.setdefault(path_str, status)
    return results


def precompute_toc(
    paths: Iterable[Union[str, os.PathLike]],
    *,
    force: bool = False,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> dict:
    """Batch-build and cache TOCs + per-frame timestamps for a sequence of
    video files.

    Useful before an annotation session, so the per-video "Building
    TOC..." pause doesn't happen interactively::

        import glob, datanavigator
        datanavigator.precompute_toc(glob.glob("data/*.mp4"))

    For folder-walking semantics, see :func:`precompute_toc_folder`.

    Args:
        paths: Iterable of video file paths.
        force: If True, rebuild + overwrite even when a valid cache exists.
            (A valid v1 sidecar counts as a hit; pass ``force=True`` to
            upgrade pre-1.3 sidecars to schema v2 with per-frame
            timestamps.)
        show_progress: If True (default), wrap iteration in a tqdm bar.
            Suppressed when ``progress_callback`` is set, so UI callers
            don't get both a bar and their own per-file updates.
        progress_callback: Optional ``fn(idx, total, path, status)``
            invoked after each video is resolved (hit / built / error).
            ``idx`` is the zero-based position in the input list,
            ``total`` is the full length, ``path`` is the string path
            passed in, ``status`` is the same per-file string written
            into the return dict. Lets UI consumers drive their own
            progress widget without parsing tqdm output.
        cancel_check: Optional zero-arg callable polled at the top of
            each video. If it returns truthy, the loop exits early and
            the partial ``{path: status}`` dict is returned (already-
            processed entries kept). Used by UI consumers' Cancel
            button.

    Returns:
        ``{path: status}`` where status is ``"hit"`` (cache already valid),
        ``"built"`` (rebuilt and cached), ``"built (uncached)"`` (built
        but sidecar save failed), or ``f"error: {msg}"`` (skipped).
    """
    paths_list = [str(p) for p in paths]
    # tqdm is suppressed when a UI callback is wired -- the caller is
    # rendering its own progress and a stdout bar would compete.
    bar = None
    if show_progress and progress_callback is None:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(paths_list), desc="Building TOCs", unit="video")
        except ImportError:
            bar = None

    results: dict[str, str] = {}
    total = len(paths_list)
    try:
        for idx, path_str in enumerate(paths_list):
            if cancel_check is not None and cancel_check():
                break
            try:
                if not Path(path_str).is_file():
                    status = "error: not a file"
                else:
                    key = _cache_key(path_str)
                    if not force and _load_sidecar(path_str, key) is not None:
                        status = "hit"
                    else:
                        toc, timestamps, time_base = _build_toc_and_timestamps(path_str)
                        ok = _save_sidecar(
                            path_str,
                            toc,
                            key,
                            timestamps=timestamps,
                            time_base=time_base,
                        )
                        status = "built" if ok else "built (uncached)"
            except Exception as e:  # noqa: BLE001 — surface per-file errors as status
                status = f"error: {e}"
            results[path_str] = status
            if progress_callback is not None:
                progress_callback(idx, total, path_str, status)
            if bar is not None:
                bar.update(1)
    finally:
        if bar is not None:
            bar.close()
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

    A frame-index TOC and per-frame pts/duration table are built from
    the video at open time and cached next to it as
    ``<video>.dnav-toc`` (JSON content; the ``.json`` extension is
    intentionally omitted so ``*.json`` walkers in downstream tooling
    don't pick the sidecar up). The key is path + size + mtime +
    head/tail SHA-256. On a cache hit the second open is sub-second;
    on a miss the first open prints a one-line "building TOC..." notice.
    Use :func:`precompute_toc` to warm the cache for a set of videos
    before an interactive session.
    """

    def __init__(
        self,
        uri,
        ctx=None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
        pix_fmt: str | None = None,
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
        # 1.5.0a2: surface ``fname`` + ``name`` on the base class so call
        # sites that previously required the ``utils.Video`` subclass
        # (notably ``dustrack.VideoAnnotation``, which reads
        # ``self.video.fname`` / ``.name``) can be handed a plain
        # ``VideoReader``. Lets DUSTrack share a single open reader
        # across every annotation layer of a session instead of opening
        # the file once per layer.
        self.fname = self._uri
        self.name = Path(self._uri).stem
        # Resolve pix_fmt. None (default) auto-detects from the source's
        # encoded pixel format: monochrome sources (h265 mono, yuvj400p,
        # etc.) go through PyAV's gray decode path which skips the
        # YUV->RGB color matrix entirely (~6x faster on h265 mono per
        # 2026-05-21 bench). Color sources keep the historical rgb24
        # default. Pass an explicit string to override.
        if pix_fmt is None:
            try:
                source_fmt = _detect_source_pix_fmt(self._uri)
            except Exception:  # noqa: BLE001 -- probe is advisory; fall back to rgb24
                source_fmt = ""
            pix_fmt = "gray" if source_fmt in _MONOCHROME_SOURCE_PIX_FMTS else "rgb24"
        self.pix_fmt = pix_fmt
        loaded = _open_with_cache(self._uri, pix_fmt=pix_fmt)
        self._reader = loaded.reader
        self._frame_pts: np.ndarray | None = loaded.frame_pts
        self._frame_durations: np.ndarray | None = loaded.frame_durations
        self._time_base_num: int | None = loaded.time_base_num
        self._time_base_den: int | None = loaded.time_base_den
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

    def get_frame_timestamp(self, indices) -> np.ndarray:
        """Return ``(N, 2)`` array of ``[start, end]`` times in seconds
        for each requested frame — decord-compatible.

        On a v2 cache hit this is a slice; on a v1 cache hit (legacy
        sidecar) or never-cached file, the per-frame pts/duration table
        is built from a full demux+decode pass on first call (announced
        with a one-line notice) and the sidecar is upgraded in place
        if writable.
        """
        if self._frame_pts is None or self._frame_durations is None:
            self._ensure_timestamps_loaded()
        # Normalize indices to a 1-D int64 array
        if isinstance(indices, range):
            idxs = np.arange(indices.start, indices.stop, indices.step or 1, dtype=np.int64)
        else:
            idxs = np.atleast_1d(
                np.asarray(
                    list(indices) if hasattr(indices, "__iter__") else [indices], dtype=np.int64
                )
            )
        starts_units = self._frame_pts[idxs]
        durations_units = self._frame_durations[idxs]
        time_base_seconds = self._time_base_num / self._time_base_den
        starts = starts_units.astype(np.float64) * time_base_seconds
        ends = (starts_units + durations_units).astype(np.float64) * time_base_seconds
        return np.column_stack([starts, ends])

    def _ensure_timestamps_loaded(self) -> None:
        """Lazy-build per-frame timestamps for v1-sidecar or never-cached
        opens. Upgrades the on-disk sidecar to v2 if the directory is
        writable."""
        print(
            f"datanavigator: indexing frame timestamps for " f"{Path(self._uri).name}...",
        )
        toc, timestamps, time_base = _build_toc_and_timestamps(self._uri)
        # Sanity-check the rebuilt TOC against the one we opened with —
        # mismatch means the file changed under us between cache load
        # and now, which would silently corrupt frame indexing.
        existing_lengths = list(self._reader._toc["lengths"])
        if list(toc["lengths"]) != existing_lengths:
            raise RuntimeError(
                f"datanavigator: TOC mismatch when computing frame timestamps "
                f"for {self._uri} — file may have changed between cache load "
                f"and now."
            )
        self._frame_pts = np.asarray(timestamps["frame_pts"], dtype=np.int64)
        self._frame_durations = np.asarray(timestamps["frame_durations"], dtype=np.int64)
        self._time_base_num = time_base["num"]
        self._time_base_den = time_base["den"]
        try:
            key = _cache_key(self._uri)
            _save_sidecar(
                self._uri,
                toc,
                key,
                timestamps=timestamps,
                time_base=time_base,
            )
        except OSError:
            pass
