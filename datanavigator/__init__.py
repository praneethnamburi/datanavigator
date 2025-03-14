"""
Interactive data visualization for signals, videos, and complex data objects.

Classes:
    GenericBrowser - Generic class to browse data. Meant to be extended.
    SignalBrowser - Browse an array of pysampled.Data elements, or 2D arrays.
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element.
    VideoBrowser - Scroll through the frames of a video.
    VideoPlotBrowser - Browse through video and 1D signals synced to the video side by side.
    ComponentBrowser - Browse signals (e.g. from periodic motion) as scatterplots of components (e.g. from UMAP, PCA).
    
    Button - Custom button widget with a 'name' state.
    StateButton - Button widget that stores a number/coordinate state.
    ToggleButton - Button widget with a toggle state.
    Selector - Select points in a plot using the lasso selection widget.
    StateVariable - Manage state variables with multiple states.
    EventData - Manage the data from one event type in one trial.
    Event - Manage selection of a sequence of events.
    
    AssetContainer - Container for managing assets such as buttons, memory slots, etc.
    Buttons - Manager for buttons in a matplotlib figure or GUI.
    Selectors - Manager for selector objects for picking points on line2D objects.
    MemorySlots - Manager for memory slots to store and navigate positions.
    StateVariables - Manager for state variables.
    Events - Manager for event objects.
    
    Video - Extended VideoReader class with additional functionalities (helper for VideoPointAnnotator).
    VideoAnnotation - Manage one point annotation layer in a video.
    VideoAnnotations - Manager for multiple video annotation layers.
    VideoPointAnnotator - Annotate points in a video.

Functions:
    lucas_kanade - Track points in a video using the Lucas-Kanade algorithm.
    lucas_kanade_rstc - Track points in a video using Lucas-Kanade with reverse sigmoid tracking correction.
    test_lucas_kanade_rstc - Test function for Lucas-Kanade with reverse sigmoid tracking correction.

External requirements:
    `ffprobe` - Required for Video class to get video information.
"""
import os
import sys
import shutil

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
from .events import Event, EventData, Events

from .core import GenericBrowser
from .plots import PlotBrowser
from .signals import SignalBrowser
from .videos import VideoBrowser, VideoPlotBrowser
from .components import ComponentBrowser

from .opticalflow import lucas_kanade, lucas_kanade_rstc
from .pointtracking import VideoAnnotation, VideoAnnotations, VideoPointAnnotator

from .utils import (
    TextView,
    Video,
    find_nearest,
    get_palette,
    is_path_exists_or_creatable,
    is_video,
    portion,
    ticks_from_times,
)

from .examples import (
    get_example_video, 
    EventPickerDemo, 
    ButtonDemo, 
    SelectorDemo, 
)

def check_ffmpeg():
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
    ffprobe_found = check_command("ffprobe")

    if not (ffmpeg_found and ffprobe_found):
        print("One or both of FFmpeg and FFprobe are missing.")
        print_install_instructions()

        
if not os.path.exists(get_clip_folder()):
    folder = os.getcwd()
    print(f"Using the current working directory-{folder}-for storing video clips.")
    set_clip_folder(folder)
    print("To change, use datanavigator.set_clip_folder(<folder_name>)")
