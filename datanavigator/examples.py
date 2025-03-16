"""
This module contains various demo classes for signal browsing, event picking, button interactions,
and selection using matplotlib widgets.

Classes:
    EventPicker: A class for picking events from signal data.
    ButtonDemo: A class demonstrating button interactions using matplotlib.
    SelectorDemo: A class demonstrating selection of points using LassoSelector.
    SignalBrowserKeyPress: A class for signal browsing with key press features.
"""

import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pysampled
from matplotlib.widgets import LassoSelector as LassoSelectorWidget

from . import _config
from .core import Buttons
from .signals import SignalBrowser


def get_example_video(dest_folder: str=None, source_url: str=None) -> str:
    """Download a sample video and return the path to that video."""
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
    A class demonstrating how to browse 10 (white noise) signals and pick events of different sizes.
    It demonstrates how to extend :py:class:`datanavigator.SignalBrowser`.
    """

    def __init__(self) -> None:
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
    """A class demonstrating button interactions using matplotlib."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.buttons = Buttons(parent=self)
        self.buttons.add(text="test", type_="Toggle")
        self.buttons.add(
            text="push button", type_="Push", action_func=self.test_callback
        )
        plt.show(block=False)

    def test_callback(self, event: Optional[Any] = None) -> None:
        """Demo callback function for button press events. Simply prints the event at the console."""
        print(event)


class SelectorDemo:
    """A class demonstrating selection of points using LassoSelector."""

    def __init__(self) -> None:
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
        """Get the points to be selected."""
        return np.vstack((self.t, self.x)).T

    def onselect(self, verts: list) -> None:
        """Select if not previously selected; Unselect if previously selected."""
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
        """Start the LassoSelector."""
        self.lasso = LassoSelectorWidget(self.ax, onselect=self.onselect)

    def stop(self, event: Optional[Any] = None) -> None:
        """Stop the LassoSelector."""
        self.lasso.disconnect_events()


# ComponentBrowser
    # import projects.gaitmusic as gm
    # mr = gm.MusicRunning01()
    # lf = mr(10).ot
    # sig_pieces = gm.gait_phase_analysis(lf, muscle_line_name='RSBL_Upper', target_samples=500)
    # gui.ComponentBrowser(sig_pieces)

    # Single-click on scatter plots to select a gait cycle.
    # Press r to refresh 'recent history' plots.
    # Double click on the time course plot to select a gait cycle from the time series plot.

class SignalBrowserKeyPress(SignalBrowser):
    """This demonstrates the old way of handling key press events in SignalBrowser. See :py:class:`EventPicker` for the new way."""

    def __init__(
        self,
        plot_data: List[pysampled.Data],  # Use List from typing module
        titlefunc: Optional[Callable] = None,
        figure_handle: Optional[Any] = None,
        reset_on_change: bool = False,
    ) -> None:
        super().__init__(plot_data, titlefunc, figure_handle, reset_on_change)
        self.event_keys = {"1": [], "2": [], "3": [], "t": [], "d": []}

    def __call__(self, event: Any) -> None:
        """Handle key press events."""
        from pprint import pprint

        super().__call__(event)
        if event.name == "key_press_event":
            sr = self.data[self._current_idx].sr
            if event.key in self.event_keys:
                if event.key == "1":
                    self.first = int(float(event.xdata) * sr)
                    self.event_keys[event.key].append(self.first)
                    print(f"first: {self.first}")
                elif event.key == "2":
                    self.second = int(float(event.xdata) * sr)
                    self.event_keys[event.key].append(self.second)
                    print(f"second: {self.second}")
                elif event.key == "3":
                    self.third = int(float(event.xdata) * sr)
                    self.event_keys[event.key].append(self.third)
                    print(f"third: {self.third}")
                elif event.key == "t":
                    pprint(self.event_keys, width=1)
                    self.export = self.event_keys
                elif event.key == "d":
                    for key in self.event_keys:
                        self.event_keys[key].clear()
                    pprint(self.event_keys, width=1)
