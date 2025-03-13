
"""
TextView - Show text array line by line.
    _parse_fax - Helper function to parse figure or axes.
    _parse_pos - Helper function to parse position strings.
get_palette - Get a color palette, with fallback if seaborn is not available.
"""
from __future__ import annotations

import os
import sys
import errno
import subprocess
from pathlib import Path

import cv2 as cv
import numpy as np
import portion as P
from decord import VideoReader
from matplotlib import axes as maxes
from matplotlib import pyplot as plt

def ticks_from_times(times, tick_lim):
    """Generate x, y arrays to supply to plt.plot function to plot a set of x-values (times) as ticks."""
    def nan_pad_x(inp):
            return [item for x in inp for item in (x, x, np.nan)]
    def nan_pad_y(ylim, n):
        return [item for y1, y2 in [ylim]*n for item in (y1, y2, np.nan)]
    return nan_pad_x(times), nan_pad_y(tick_lim, len(times))

def find_nearest(x, y):
    """
    Find the nearest x-values for every value in y.
    x and y are expected to be lists of floats.
    Returns:
        List with the same number of values as y, but each value is the closest value in x.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    return [x[np.argmin(np.abs(x - yi))] for yi in y]

class List(list):
    def next(self, val):
        """Next element in the list closest to val."""
        return min([x for x in self if x > val], default=max(self))
    
    def previous(self, val):
        """Previous element in the list closest to val."""
        return max([x for x in self if x < val], default=min(self))

# extend portion functionality in the class below
class _PNInterval(P.Interval):
    @property
    def atomic_durations(self):
        return [xi.upper - xi.lower for xi in self]
    
    @property
    def duration(self):
        return sum(self.atomic_durations)
    
    @property
    def fraction(self):
        # fractional duration relative to the enclosure
        return self.duration/self.enclosure.duration
    
portion = P.create_api(_PNInterval)


# TextView and supporting functions
def _parse_fax(fax, ax_pos=(0.01, 0.01, 0.98, 0.98)):
    assert isinstance(fax, (type(None), plt.Figure, maxes.Axes))
    if fax is None:
        f = plt.figure()
        ax = f.add_axes(ax_pos)
    elif isinstance(fax, plt.Figure):
        f = fax
        ax = f.add_axes(ax_pos)
    else:
        f = fax.figure
        ax = fax
    return f, ax

def _parse_pos(pos):
    if isinstance(pos, str):
        updown, leftright = pos.replace('middle', 'center').split(' ')
        assert updown in ('top', 'center', 'bottom')
        assert leftright in ('left', 'center', 'right')
        y = {'top': 1, 'center': 0.5, 'bottom': 0}[updown]
        x = {'left': 0, 'center': 0.5, 'right': 1}[leftright]
        pos = (x, y, updown, leftright)
    assert len(pos) == 4
    return pos

class TextView:
    """Show text array line by line"""
    def __init__(self, text, fax=None, pos='bottom left'):
        """
        text is an array of strings
        fax is either a figure or an axis handle
        """
        def rescale(xy, margin=0.03):
            return (1-2*margin)*xy + margin

        self.text = self.parse_text(text)
        self._text = None # matplotlib text object
        self._pos = _parse_pos(pos)
        self.figure, self._ax = _parse_fax(fax, ax_pos=(rescale(self._pos[0]), rescale(self._pos[1]), 0.02, 0.02))
        self.setup()
        self.update()
    
    def parse_text(self, text):
        if isinstance(text, dict):
            text = [f'{key} - {val}' for key, val in text.items()] 
        return text

    def setup(self):
        self._ax.axis('off')
        plt.show(block=False)

    def update(self, new_text=None):
        if new_text is not None:
            self.text = self.parse_text(new_text)
        if self._text is not None:
            self._text.remove()
        x, y, va, ha = self._pos
        self._text = self._ax.text(x, y, '\n'.join(self.text), va=va, ha=ha, family='monospace')
        plt.draw()


# attached to VideoAnnotation
def get_palette(palette_name='Set2', n_colors=10):
    try: # remove seaborn dependency by manually specifying a color palette
        import seaborn as sns
        return sns.color_palette(palette_name, n_colors=n_colors)
    except ModuleNotFoundError:
        palettes = {
            'Set2': [ # seaborn set 2
                (0.40, 0.76, 0.65),
                (0.99, 0.55, 0.38),
                (0.55, 0.63, 0.79),
                (0.91, 0.54, 0.76),
                (0.65, 0.85, 0.33),
                (1.00, 0.85, 0.18),
                (0.90, 0.77, 0.58),
                (0.70, 0.70, 0.70),
                (0.40, 0.76, 0.65),
                (0.99, 0.55, 0.38)
            ]}
        return palettes[palette_name][:n_colors]



def is_video(vid_file:str, verbose=False) -> bool: # Refactored from pntools.video.is_video
    """Check if a file is a video file."""
    codec_types = subprocess.getoutput(f'ffprobe -loglevel error -show_entries stream=codec_type -of default=nw=1 "{vid_file}"')
    if verbose:
        my_print = print
    else:
        my_print = lambda x:None

    ret = False
    if 'Invalid data' in codec_types:
        my_print('Not an audio or video file')
    if 'codec_type=video' in codec_types:
        ret = True
        my_print('Video stream found')
    if 'codec_type=audio' in codec_types:
        my_print('Audio stream found')
    return ret


class Video(VideoReader):
    def __init__(self, uri, ctx=cpu(0), width=-1, height=-1, num_threads=0, fault_tol=-1):
        assert os.path.exists(uri) and is_video(uri)
        self.fname = uri
        self.name = Path(uri).stem
        super().__init__(uri, ctx, width, height, num_threads, fault_tol)

    def gray(self, frame_num: int):
        return cv.cvtColor(self[frame_num].asnumpy(), cv.COLOR_BGR2GRAY)

### ------- FROM STACK OVERFLOW
# Sadly, Python fails to provide the following magic number for us.
ERROR_INVALID_NAME = 123
'''
Windows-specific error code indicating an invalid pathname.

See Also
----------
https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--0-499-
    Official listing of all such codes.
'''

def is_pathname_valid(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS;
    `False` otherwise.
    '''
    # If this pathname is either not a string or is but is empty, this pathname
    # is invalid.
    try:
        if not isinstance(pathname, str) or not pathname:
            return False

        # Strip this pathname's Windows-specific drive specifier (e.g., `C:\`)
        # if any. Since Windows prohibits path components from containing `:`
        # characters, failing to strip this `:`-suffixed prefix would
        # erroneously invalidate all valid absolute Windows pathnames.
        _, pathname = os.path.splitdrive(pathname)

        # Directory guaranteed to exist. If the current OS is Windows, this is
        # the drive to which Windows was installed (e.g., the "%HOMEDRIVE%"
        # environment variable); else, the typical root directory.
        root_dirname = os.environ.get('HOMEDRIVE', 'C:') \
            if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)   # ...Murphy and her ironclad Law

        # Append a path separator to this directory if needed.
        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        # Test whether each path component split from this pathname is valid or
        # not, ignoring non-existent and non-readable path components.
        for pathname_part in pathname.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            # If an OS-specific exception is raised, its error code
            # indicates whether this pathname is valid or not. Unless this
            # is the case, this exception implies an ignorable kernel or
            # filesystem complaint (e.g., path not found or inaccessible).
            #
            # Only the following exceptions indicate invalid pathnames:
            #
            # * Instances of the Windows-specific "WindowsError" class
            #   defining the "winerror" attribute whose value is
            #   "ERROR_INVALID_NAME". Under Windows, "winerror" is more
            #   fine-grained and hence useful than the generic "errno"
            #   attribute. When a too-long pathname is passed, for example,
            #   "errno" is "ENOENT" (i.e., no such file or directory) rather
            #   than "ENAMETOOLONG" (i.e., file name too long).
            # * Instances of the cross-platform "OSError" class defining the
            #   generic "errno" attribute whose value is either:
            #   * Under most POSIX-compatible OSes, "ENAMETOOLONG".
            #   * Under some edge-case OSes (e.g., SunOS, *BSD), "ERANGE".
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False
    # If a "TypeError" exception was raised, it almost certainly has the
    # error message "embedded NUL character" indicating an invalid pathname.
    except TypeError as exc:
        return False
    # If no exception was raised, all path components and hence this
    # pathname itself are valid. (Praise be to the curmudgeonly python.)
    else:
        return True
    # If any other exception was raised, this is an unrelated fatal issue
    # (e.g., a bug). Permit this exception to unwind the call stack.
    #
    # Did we mention this should be shipped with Python already?

def is_path_creatable(pathname: str) -> bool:
    '''
    `True` if the current user has sufficient permissions to create the passed
    pathname; `False` otherwise.
    '''
    # Parent directory of the passed path. If empty, we substitute the current
    # working directory (CWD) instead.
    dirname = os.path.dirname(pathname) or os.getcwd()
    return os.access(dirname, os.W_OK)

def is_path_exists_or_creatable(pathname: str) -> bool:
    '''
    `True` if the passed pathname is a valid pathname for the current OS _and_
    either currently exists or is hypothetically creatable; `False` otherwise.

    This function is guaranteed to _never_ raise exceptions.
    '''
    try:
        # To prevent "os" module calls from raising undesirable exceptions on
        # invalid pathnames, is_pathname_valid() is explicitly called first.
        return is_pathname_valid(pathname) and (
            os.path.exists(pathname) or is_path_creatable(pathname))
    # Report failure on non-fatal filesystem complaints (e.g., connection
    # timeouts, permissions issues) implying this path to be inaccessible. All
    # other exceptions are unrelated fatal issues and should not be caught here.
    except OSError:
        return False
    
### ------- END FROM STACK OVERFLOW
