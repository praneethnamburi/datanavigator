"""
Simple Graphical User Interface elements for browsing data.

This module provides classes and functions to create and manage graphical user interfaces for browsing and interacting with various types of data, including signals, plots, and videos. It includes support for event handling, and data visualization.

Classes:
    GenericBrowser - Generic class to browse data. Meant to be extended.
    SignalBrowser - Browse an array of pysampled.Data elements, or 2D arrays.
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element.
    VideoBrowser - Scroll through the frames of a video.
    VideoPlotBrowser - Browse through video and 1D signals synced to the video side by side.
    
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
from .core import (
    GenericBrowser,
    PlotBrowser,
    SignalBrowser,
    VideoBrowser,
    VideoPlotBrowser,
)
from .events import Event, EventData, Events
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

if not os.path.exists(get_clip_folder()):
    folder = os.getcwd()
    print(f"Using the current working directory-{folder}-for storing video clips.")
    set_clip_folder(folder)
    print("To change, use datanavigator.set_clip_folder(<folder_name>)")
