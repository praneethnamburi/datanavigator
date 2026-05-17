"""Vendored ``PyAVReaderIndexed`` from PIMS (BSD-3-Clause).

Source: https://github.com/soft-matter/pims/blob/master/pims/pyav_reader.py
Upstream commit: d5995962b79004f9bb8ac58d2ff14735d4a0c610 (2023-01-24)

The full PIMS license text is reproduced in
``datanavigator/_vendor/LICENSE-PIMS``.

Only the ``PyAVReaderIndexed`` class is vendored — the minimum needed to
power datanavigator's frame-indexed video reader. The upstream class
inherits from ``pims.base_frames.FramesSequence`` and returns
``pims.frame.Frame`` (an ``ndarray`` subclass). To avoid pulling in
``pims`` + ``slicerator`` at runtime, this file provides minimal local
stand-ins (``Frame``, ``FramesSequence``) that preserve the public
behavior PyAVReaderIndexed relies on: ``__getitem__`` delegates to
``get_frame``, frames are ndarray-compatible. The PyAVReaderIndexed
class itself is reproduced verbatim so future re-syncs against upstream
are trivial diffs.
"""
from __future__ import annotations

import numpy as np
from numpy import asarray, ndarray

# Deviation from upstream: the original PIMS file wraps this import in
# ``try / except ImportError`` (av = None) because PIMS supports other
# readers and av is optional there. In datanavigator av is the only
# backend, so silently degrading to ``av is None`` produces a cryptic
# ``'NoneType' object has no attribute 'open'`` at ``av.open(...)``
# inside ``__init__``. Import unconditionally so a missing dep fails
# loud at ``import datanavigator`` time instead.
import av


def available():
    return av is not None


# ---------- Minimal local stand-ins for pims.frame.Frame and pims.base_frames.FramesSequence ----------
# These exist only to satisfy what PyAVReaderIndexed needs at runtime.
# They are NOT vendored from PIMS — they are minimal datanavigator
# substitutes so we don't need to depend on `pims` or `slicerator`.


class Frame(ndarray):
    """ndarray subclass carrying frame_no/metadata, minimal subset of pims.frame.Frame."""

    def __new__(cls, input_array, frame_no=None, metadata=None):
        obj = asarray(input_array).view(cls)
        if frame_no is None and hasattr(input_array, "frame_no"):
            frame_no = getattr(input_array, "frame_no")
        obj.frame_no = frame_no
        if hasattr(input_array, "metadata"):
            arr_metadata = dict(getattr(input_array, "metadata"))
        else:
            arr_metadata = dict()
        if metadata is None:
            metadata = {}
        arr_metadata.update(metadata)
        obj.metadata = arr_metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.frame_no = getattr(obj, "frame_no", None)
        self.metadata = getattr(obj, "metadata", None)


class FramesSequence:
    """Minimal stand-in for pims.base_frames.FramesSequence.

    Provides __getitem__ → get_frame delegation and basic iteration.
    Subclasses must implement __len__ and get_frame.
    """

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return [self.get_frame(i) for i in range(start, stop, step)]
        if isinstance(key, (list, tuple, np.ndarray)):
            return [self.get_frame(int(i)) for i in key]
        return self.get_frame(int(key))

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_frame(i)

    @classmethod
    def class_exts(cls):
        return set()


# ---------- BEGIN verbatim from upstream pims/pyav_reader.py ----------
# The following is the unmodified PyAVReaderIndexed class.
# Re-sync from upstream by replacing the block below.


def _next_video_packet(container_iter):
    for packet in container_iter:
        if packet.stream.type == 'video':
            decoded = packet.decode()
            if len(decoded) > 0:
                return decoded

    raise ValueError("Could not find any video packets.")


class PyAVReaderIndexed(FramesSequence):
    """Read images from the frames of a standard video file into an
    iterable object that returns images as numpy arrays.

    Parameters
    ----------
    filename : string

    Examples
    --------
    >>> video = Video('video.avi')  # or .mov, etc.
    >>> imshow(video[0]) # Show the first frame.
    >>> imshow(video[-1]) # Show the last frame.
    >>> imshow(video[1][0:10, 0:10]) # Show one corner of the second frame.

    >>> for frame in video[:]:
    ...    # Do something with every frame.

    >>> for frame in video[10:20]:
    ...    # Do something with frames 10-20.

    >>> for frame in video[[5, 7, 13]]:
    ...    # Do something with frames 5, 7, and 13.

    >>> frame_count = len(video) # Number of frames in video
    >>> frame_shape = video.frame_shape # Pixel dimensions of video
    """
    class_priority = 8

    @classmethod
    def class_exts(cls):
        return {'mov', 'avi',
                'mp4'} | super(PyAVReaderIndexed, cls).class_exts()

    def __init__(self, file, toc=None, format=None):
        if not hasattr(file, 'read'):
            file = str(file)
        self.file = file
        self.format = format
        self._container = None

        with av.open(self.file, format=self.format) as container:
            stream = [s for s in container.streams if s.type == 'video'][0]

            # Build a toc
            if toc is None:
                packet_lengths = []
                packet_ts = []
                for packet in container.demux(stream):
                    if packet.stream.type == 'video':
                        decoded = packet.decode()
                        if len(decoded) > 0:
                            packet_lengths.append(len(decoded))
                            packet_ts.append(decoded[0].pts)
                self._toc = {
                    'lengths': packet_lengths,
                    'ts': packet_ts,
                }
            else:
                self._toc = toc

            self._toc_cumsum = np.cumsum(self.toc['lengths'])
            self._len = self._toc_cumsum[-1]

            # PyAV always returns frames in color, and we make that
            # assumption in get_frame() later below, so 3 is hardcoded here:
            self._im_sz = stream.height, stream.width, 3
            self._time_base = stream.time_base

        self._load_fresh_file()

    def _load_fresh_file(self):
        if self._container is not None:
            self._container.close()

        if hasattr(self.file, 'seek'):
            self.file.seek(0)

        self._container = av.open(self.file, format=self.format)
        demux = self._container.demux(self._video_stream)
        self._current_packet = _next_video_packet(demux)
        self._current_packet_no = 0

    @property
    def _video_stream(self):
        return [s for s in self._container.streams if s.type == 'video'][0]

    def __len__(self):
        return self._len

    def __del__(self):
        if self._container is not None:
            self._container.close()

    @property
    def frame_shape(self):
        return self._im_sz

    @property
    def toc(self):
        return self._toc

    def get_frame(self, j):
        # Find the packet this frame is in.
        packet_no = self._toc_cumsum.searchsorted(j, side='right')
        self._seek_packet(packet_no)
        # Find the location of the frame within the packet.
        if packet_no == 0:
            loc = j
        else:
            loc = j - self._toc_cumsum[packet_no - 1]
        frame = self._current_packet[loc]  # av.VideoFrame

        return Frame(frame.to_ndarray(format='rgb24'), frame_no=j)

    def _seek_packet(self, packet_no):
        """Advance through the container generator until we get the packet
        we want. Store that packet in selfpp._current_packet."""
        packet_ts = self.toc['ts'][packet_no]
        # Only seek when needed.
        if packet_no == self._current_packet_no:
            return
        elif (packet_no < self._current_packet_no
                or packet_no > self._current_packet_no + 1):
            self._container.seek(packet_ts, stream=self._video_stream)

        demux = self._container.demux(self._video_stream)
        self._current_packet = _next_video_packet(demux)
        while self._current_packet[0].pts < packet_ts:
            self._current_packet = _next_video_packet(demux)

        self._current_packet_no = packet_no

    @property
    def pixel_type(self):
        # No need to detect dtype: PyAV always returns uint8.
        return np.uint8

    def __repr__(self):
        # May be overwritten by subclasses
        return """<Frames>
Source: {filename}
Length: {count} frames
Frame Shape: {frame_shape!r}
""".format(frame_shape=self.frame_shape,
           count=len(self),
           filename=self.file)

# ---------- END verbatim from upstream pims/pyav_reader.py ----------
