"""
This module contains various demo classes for signal browsing, event picking, button interactions,
and selection using matplotlib widgets. It also includes a utility function to download
a sample video file.

Functions:
    get_example_video: Downloads a sample video file for demonstration purposes.

Classes:
    EventPickerDemo: Demonstrates browsing signals and picking events of varying sizes using SignalBrowser.
    ButtonDemo: Shows how to create and interact with custom buttons on a Matplotlib figure.
    SelectorDemo: Illustrates selecting data points on a plot using the LassoSelector widget.
"""

import os
import urllib.request
from typing import Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pysampled
from matplotlib.widgets import LassoSelector as LassoSelectorWidget
from matplotlib.path import Path

from . import _config
from .core import Buttons
from .signals import SignalBrowser


def get_example_video(dest_folder: str=None, source_url: str=None) -> str:
    """
    Download a sample video file if it doesn't exist locally and return its path.

    Uses a default destination folder and source URL if none are provided.
    The default destination is retrieved from `_config.get_clip_folder()`.
    The default source URL points to a small jellyfish video.

    Args:
        dest_folder (str, optional): The folder where the video should be saved.
            Defaults to the folder specified in the configuration.
        source_url (str, optional): The URL from which to download the video.
            Defaults to a known test video URL.

    Returns:
        str: The local file path to the downloaded (or existing) video.

    Raises:
        AssertionError: If a `dest_folder` is provided but does not exist.
        urllib.error.URLError: If the video download fails (e.g., network issue, invalid URL).
    """
    if dest_folder is None:
        dest_folder = _config.get_clip_folder()
    else:
        assert os.path.exists(dest_folder), f"Folder {dest_folder} does not exist."
    
    example_video_path = os.path.join(dest_folder, "example_video.mp4")
    if os.path.exists(example_video_path):
        return example_video_path

    if source_url is None:
        source_url = "https://test-videos.co.uk/vids/jellyfish/mp4/h264/720/Jellyfish_720_10s_2MB.mp4"

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
    req = urllib.request.Request(source_url, headers=headers)

    with urllib.request.urlopen(req) as response:
        with open(example_video_path, "wb") as temp_file:
            temp_file.write(response.read())

    print(f"Downloaded video to: {example_video_path}")
    return example_video_path


class EventPickerDemo(SignalBrowser):
    """
    Demonstrates browsing signals and picking events using the SignalBrowser.

    This class extends SignalBrowser to showcase how to:
    - Load multiple signals (random noise in this case).
    - Define and manage different types of events ('pick1', 'pick2', 'pick3')
      with varying sizes and picking behaviors ('append', 'overwrite').
    - Assign keyboard shortcuts for adding, removing, and saving events.
    - Customize event display (line vs. fill).
    - Override the `update` method to ensure event displays are refreshed.

    Key Bindings:
        '1': Add a 'pick1' event (size 1, append).
        'alt+1': Remove the nearest 'pick1' event.
        'ctrl+1': Save 'pick1' events to file.
        '2': Add a 'pick2' event (size 2, append, fill display).
        'alt+2': Remove the nearest 'pick2' event.
        'ctrl+2': Save 'pick2' events to file.
        '3': Add a 'pick3' event (size 3, overwrite).
        'alt+3': Remove the nearest 'pick3' event.
        'ctrl+3': Save 'pick3' events to file.
    """

    def __init__(self) -> None:
        """
        Initializes the EventPickerDemo.

        Sets up 10 random signals, configures three event types with different
        properties and key bindings, and performs an initial plot update.
        """
        plot_data = [
            pysampled.Data(
                np.random.rand(100), sr=10, meta={"id": f"sig{sig_count:02d}"}
            )
            for sig_count in range(10)
        ]
        super().__init__(plot_data)
        self.memoryslots.disable()
        data_id_func = (lambda s: s.data[s._current_idx].meta["id"]).__get__(self)
        self.events.add(
            name="pick1",
            size=1,
            fname=os.path.join(_config.get_cache_folder(), "_pick1.json"),
            data_id_func=data_id_func,
            color="tab:red",
            pick_action="append",
            ax_list=[self._ax],
            add_key="1",
            remove_key="alt+1",
            save_key="ctrl+1",
            linewidth=1.5,
        )
        self.events.add(
            name="pick2",
            size=2,
            fname=os.path.join(_config.get_cache_folder(), "_pick2.json"),
            data_id_func=data_id_func,
            color="tab:green",
            pick_action="append",
            ax_list=[self._ax],
            add_key="2",
            remove_key="alt+2",
            save_key="ctrl+2",
            linewidth=1.5,
            display_type="fill",
        )
        self.events.add(
            name="pick3",
            size=3,
            fname=os.path.join(_config.get_cache_folder(), "_pick3.json"),
            data_id_func=data_id_func,
            color="tab:blue",
            pick_action="overwrite",
            ax_list=[self._ax],
            add_key="3",
            remove_key="alt+3",
            save_key="ctrl+3",
            linewidth=1.5,
        )
        self.update()

    def update(self, event: Optional[Any] = None) -> None:
        """The update method is often one that needs to be specified when extending a class."""
        self.events.update_display()
        return super().update(event)


class ButtonDemo(plt.Figure):
    """
    Demonstrates creating and using custom buttons within a Matplotlib figure.

    This class creates a figure and adds two buttons using the `datanavigator.core.Buttons` manager:
    - A 'Toggle' button.
    - A 'Push' button that triggers a callback function (`test_callback`) when clicked.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initializes the ButtonDemo figure.

        Creates the figure, adds a toggle and a push button, and displays the figure.

        Args:
            *args: Variable length argument list passed to the parent plt.Figure.
            **kwargs: Arbitrary keyword arguments passed to the parent plt.Figure.
        """
        super().__init__(*args, **kwargs)
        self.buttons = Buttons(parent=self)
        self.buttons.add(text="test", type_="Toggle")
        self.buttons.add(
            text="push button", type_="Push", action_func=self.test_callback
        )
        plt.show(block=False)

    def test_callback(self, event: Optional[Any] = None) -> None:
        """
        Callback function executed when the 'push button' is clicked.

        Prints the triggering mouse event to the console.

        Args:
            event (Optional[Any], optional): The matplotlib event object associated
                with the button click. Defaults to None.
        """
        print(event)


class SelectorDemo:
    """
    Demonstrates using the Matplotlib LassoSelector widget to select data points.

    Creates a scatter plot and adds buttons to start and stop the lasso selection mode.
    Selected points are highlighted, and selections can be toggled (selecting an already
    selected point unselects it).
    """

    def __init__(self) -> None:
        """
        Initializes the SelectorDemo.

        Creates a figure with axes, adds control buttons, plots random data,
        and activates the lasso selector. Sets a random seed for reproducible data.
        """
        np.random.seed(42) # Set random seed for reproducibility
        f, ax = plt.subplots(1, 1)
        self.buttons = Buttons(parent=f)
        self.buttons.add(text="Start selection", type_="Push", action_func=self.start)
        self.buttons.add(text="Stop selection", type_="Push", action_func=self.stop)
        self.ax = ax
        self.x = np.random.rand(10)
        self.t = np.r_[:1:0.1]
        self.plot_handles = {}
        (self.plot_handles["data"],) = ax.plot(self.t, self.x)
        (self.plot_handles["selected"],) = ax.plot([], [], ".")
        plt.show(block=False)
        self.start()
        self.ind = set()

    def get_points(self) -> np.ndarray:
        """
        Returns the data points currently plotted on the axes.

        Used by the LassoSelector to determine which points are inside the lasso path.

        Returns:
            np.ndarray: A 2D array where each row is (t, x) coordinate of a point.
        """
        return np.vstack((self.t, self.x)).T

    def onselect(self, verts: list) -> None:
        """
        Callback function executed when a lasso selection is completed.

        Determines which points fall within the lasso path defined by `verts`.
        Updates the set of selected indices (`self.ind`), toggling the selection
        state for points within the lasso. Refreshes the plot to show the
        currently selected points.

        Args:
            verts (list): A list of (x, y) tuples defining the vertices of the
                lasso path in display coordinates.
        """
        path = Path(verts)
        selected_ind = set(np.nonzero(path.contains_points(self.get_points()))[0])
        existing_ind = self.ind.intersection(selected_ind)
        new_ind = selected_ind - existing_ind
        self.ind = (self.ind - existing_ind).union(new_ind)
        idx = list(self.ind)
        if idx:
            self.plot_handles["selected"].set_data(self.t[idx], self.x[idx])
        else:
            self.plot_handles["selected"].set_data([], [])
        plt.draw()

    def start(self, event: Optional[Any] = None) -> None:
        """
        Activates the LassoSelector widget.

        Connected as the callback for the 'Start selection' button.

        Args:
            event (Optional[Any], optional): The event triggering the start. Defaults to None.
        """
        self.lasso = LassoSelectorWidget(self.ax, onselect=self.onselect)

    def stop(self, event: Optional[Any] = None) -> None:
        """
        Deactivates the LassoSelector widget.

        Connected as the callback for the 'Stop selection' button.

        Args:
            event (Optional[Any], optional): The event triggering the stop. Defaults to None.
        """
        self.lasso.disconnect_events()
