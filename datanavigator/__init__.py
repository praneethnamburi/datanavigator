r"""
Interactive data visualization for signals, videos, and complex data objects.

datanavigator provides the modality-agnostic data-navigation primitives
-- browsers, asset managers, events. The point-tracking UI, annotation
containers (:class:`VideoAnnotation`, :class:`VideoAnnotations`), and
Lucas-Kanade helpers (``lucas_kanade``, ``lucas_kanade_rstc``) used to
live here too; in 1.5.0 they were relocated to the :mod:`dustrack`
package alongside its DeepLabCut workflow. See ``dustrack.DUSTrack`` /
``dustrack.VideoAnnotation`` for the new home. ``git log --follow
dustrack/annotations.py`` traces the full pre-relocation history.

Browsers

- :py:class:`GenericBrowser`: Generic class to browse data. Meant to be extended.
- :py:class:`SignalBrowser`: Browse an array of pysampled.Data elements, or 2D arrays.
- :py:class:`PlotBrowser`: Scroll through an array of complex data where a plotting function is defined for each element.
- :py:class:`VideoBrowser`: Scroll through the frames of a video.
- :py:class:`VideoPlotBrowser`: Browse through video and 1D signals synced to the video side by side.
- :py:class:`ComponentBrowser`: Browse signals (e.g., from periodic motion) as scatterplots of components (e.g., from UMAP, PCA).

Video I/O

- :py:class:`Video`: Extended VideoReader class with additional functionalities.
- :py:class:`VideoReader`: PyAV-backed video reader with a TOC cache.

Assets

- :py:class:`Button`: Custom button widget with a 'name' state.
- :py:class:`StateButton`: Button widget that stores a number/coordinate state.
- :py:class:`ToggleButton`: Button widget with a toggle state.
- :py:class:`Selector`: Select points in a plot using the lasso selection widget.
- :py:class:`StateVariable`: Manage state variables with multiple states.
- :py:class:`EventData`: Manage the data from one event type in one trial.
- :py:class:`Event`: Manage selection of a sequence of events.

Assetcontainers

- :py:class:`AssetContainer`: Container for managing assets such as buttons, memory slots, etc.
- :py:class:`Buttons`: Manager for buttons in a matplotlib figure or GUI.
- :py:class:`Selectors`: Manager for selector objects for picking points on line2D objects.
- :py:class:`MemorySlots`: Manager for memory slots to store and navigate positions.
- :py:class:`StateVariables`: Manager for state variables.
- :py:class:`Events`: Manager for event objects.
"""

import os
import shutil
import sys
import warnings

# Suppress PySide6 6.4.x shibokensupport signature parser warnings.
# These fire when something (matplotlib's NavigationToolbar2QT save path,
# DLC's GUI imports, etc.) causes shiboken to lazy-parse QFileDialog's
# API signatures. PySide6 6.4.x's parser doesn't recognize certain
# forward-referenced types (typing.List[PySide6.QtCore.QUrl], nested
# QFileDialog.Option enums); each unparseable line emits a RuntimeWarning.
# Fixed upstream in PySide6 6.5+; DLC envs commonly pin 6.4.2 so users
# see dozens of these on first DUSTrack launch. Cosmetic only -- the Qt
# API still works fine, just signature introspection is incomplete.
# Narrow module filter so any non-shiboken RuntimeWarning still surfaces.
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    module=r"shibokensupport\.signature\.parser",
)

from .__version__ import __version__
from ._config import (
    get_cache_folder,
    get_clip_folder,
    set_cache_folder,
    set_clip_folder,
)
from .assets import (
    AssetContainer,
    Button,
    Buttons,
    MemorySlots,
    Selector,
    Selectors,
    StateButton,
    StateVariable,
    StateVariables,
    ToggleButton,
)
from .components import ComponentBrowser
from .core import GenericBrowser
from .events import Event, EventData, Events, portion
from .examples import (
    ButtonDemo,
    EventPickerDemo,
    SelectorDemo,
    get_example_video,
)
from .plots import PlotBrowser
from .signals import SignalBrowser
from ._modals import confirm, prompt_text
from .utils import (
    TextView,
    Video,
    get_palette,
    is_path_exists_or_creatable,
    is_video,
    ticks_from_times,
)
from .video_reader import (
    DEFAULT_VIDEO_EXTENSIONS,
    VideoReader,
    cpu,
    precompute_toc,
    precompute_toc_folder,
)
from .videos import VideoBrowser, VideoPlotBrowser


def _check_ffmpeg():
    def check_command(command):
        """Check if a command is available in the system's PATH."""
        return shutil.which(command) is not None

    def print_install_instructions():
        """Print installation instructions for ffmpeg and ffprobe."""
        if sys.platform.startswith("win"):
            print("\nFFmpeg is not installed or not in PATH.")
            print("Download it from: https://ffmpeg.org/download.html")
            print("After installation, add FFmpeg's 'bin' folder to the system PATH.")
        else:
            print("\nFFmpeg is not installed or not in PATH.")
            print("On Debian/Ubuntu, install it with: sudo apt install ffmpeg")
            print("On macOS, install it with: brew install ffmpeg")
            print("On Fedora, install it with: sudo dnf install ffmpeg")
            print("On Arch Linux, install it with: sudo pacman -S ffmpeg")

    # Check if ffmpeg and ffprobe are available
    ffmpeg_found = check_command("ffmpeg")

    if not ffmpeg_found:
        print("Could not find ffmpeg.")
        print_install_instructions()


def _check_clip_folder():
    if not os.path.exists(get_clip_folder()):
        folder = os.getcwd()
        print(f"Using the current working directory-{folder}-for storing video clips.")
        set_clip_folder(folder)
        print("To change, use datanavigator.set_clip_folder(<folder_name>)")


_check_ffmpeg()
_check_clip_folder()


__all__ = [
    # Browsers
    "GenericBrowser",
    "SignalBrowser",
    "PlotBrowser",
    "VideoBrowser",
    "VideoPlotBrowser",
    "ComponentBrowser",
    # Video I/O
    "Video",
    "VideoReader",
    "cpu",
    "precompute_toc",
    "precompute_toc_folder",
    "DEFAULT_VIDEO_EXTENSIONS",
    # Asset widgets
    "Button",
    "StateButton",
    "ToggleButton",
    "Selector",
    "StateVariable",
    # Modal overlays
    "confirm",
    "prompt_text",
    # Asset containers
    "AssetContainer",
    "Buttons",
    "Selectors",
    "MemorySlots",
    "StateVariables",
    "Events",
    # Events / intervals
    "Event",
    "EventData",
    "portion",
    # Utilities
    "TextView",
    "get_palette",
    "is_path_exists_or_creatable",
    "is_video",
    "ticks_from_times",
    # Examples
    "get_example_video",
    "EventPickerDemo",
    "ButtonDemo",
    "SelectorDemo",
    # Configuration
    "get_cache_folder",
    "get_clip_folder",
    "set_cache_folder",
    "set_clip_folder",
    # Version
    "__version__",
]
