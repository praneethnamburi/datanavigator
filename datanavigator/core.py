"""
This module defines the core class called :py:class:`GenericBrowser`. It
defines basic functionalities for browsing data, such as navigating
using arrow keys, storing positions in memory (e.g. video frame
numbers), adding buttons, and assigning hotkeys to custom functions.
This class can be extended to create interactive browsers for various
types of data, including plots, signals, and videos.
"""

from __future__ import annotations

import inspect
import io
import os
from datetime import datetime, timedelta

import matplotlib as mpl
import pysampled
from decord import VideoReader
from matplotlib import axes as maxes
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from . import _config, utils
from .assets import Buttons, MemorySlots, Selectors, StateVariables
from .events import Events


class GenericBrowser:
    """
    Generic class that defines base functionality. Meant to be extended before use.

    Features:
        - Navigate using arrow keys.
        - Store positions in memory using number keys (e.g. for flipping between positions when browsing a video).
        - Quickly add toggle and push buttons.
        - Design custom functions and assign hotkeys to them (add_key_binding).

    Default Navigation (arrow keys):
        - ctrl+k      - show all keybindings
        - right       - forward one frame
        - left        - back one frame
        - up          - forward 10 frames
        - down        - back 10 frames
        - shift+left  - first frame
        - shift+right - last frame
        - shift+up    - forward nframes/20 frames
        - shift+down  - back nframes/20 frames
    """

    def __init__(self, figure_handle: plt.Figure = None):
        """
        Initialize the GenericBrowser.

        Args:
            figure_handle (plt.Figure, optional): Matplotlib figure handle. Defaults to None.
        """
        if figure_handle is None:
            figure_handle = plt.figure()
        assert isinstance(figure_handle, plt.Figure)
        self.figure = figure_handle
        self._keypressdict = {}  # managed by add_key_binding
        self._bindings_removed = {}

        # tracking variable
        self._current_idx = 0

        self._keybindingtext = None
        self.buttons = Buttons(parent=self)
        self.selectors = Selectors(parent=self)
        self.memoryslots = MemorySlots(parent=self)
        self.statevariables = StateVariables(parent=self)
        self.events = Events(parent=self)

        # for cleanup
        self.cid = []
        self.cid.append(self.figure.canvas.mpl_connect("key_press_event", self))
        self.cid.append(self.figure.canvas.mpl_connect("close_event", self))

    def update_assets(self):
        """Update the display of various assets."""
        if self.has("memoryslots"):
            self.memoryslots.update_display()
        if self.has("events"):
            self.events.update_display()
        if self.has("statevariables"):
            self.statevariables.update_display()

    def update(self, event=None):
        """
        Update the browser. Extended classes are expected to implement their update function.

        Args:
            event (optional): Event that triggered the update. Defaults to None.
        """
        self.update_assets()

    def update_without_clear(self, event=None):
        """
        Update the browser without clearing the axis.

        Args:
            event (optional): Event that triggered the update. Defaults to None.
        """
        self.update_assets()

    def mpl_remove_bindings(self, key_list: list[str]):
        """
        Remove existing key bindings in matplotlib.

        Args:
            key_list (list[str]): List of keys to remove bindings for.
        """
        for key in key_list:
            this_param_name = [
                k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v
            ]
            if this_param_name:  # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key

    def __call__(self, event):
        """
        Handle events.

        Args:
            event: Event to handle.
        """
        if event.name == "key_press_event":
            if event.key in self._keypressdict:
                f = self._keypressdict[event.key][0]
                argspec = inspect.getfullargspec(f)[0]
                if len(argspec) == 2 and argspec[1] == "event":
                    f(event)
                else:
                    f()  # this may or may not redraw everything
            if event.key in self.memoryslots._list:
                self.memoryslots.update(event.key)
        elif event.name == "close_event":  # perform cleanup
            self.cleanup()

    def cleanup(self):
        """Perform cleanup, for example, when the figure is closed."""
        for this_cid in self.cid:
            self.figure.canvas.mpl_disconnect(this_cid)
        self.mpl_restore_bindings()

    def mpl_restore_bindings(self):
        """Restore any modified default keybindings in matplotlib."""
        for param_name, key in self._bindings_removed.items():
            if key not in mpl.rcParams[param_name]:
                mpl.rcParams[param_name].append(
                    key
                )  # param names: keymap.back, keymap.forward)
            self._bindings_removed[param_name] = {}

    def __len__(self):
        if hasattr(self, "data"):  # otherwise returns None
            return len(self.data)

    def reset_axes(self, axis: str = "both", event=None):
        """
        Reframe data within matplotlib axes.

        Args:
            axis (str, optional): Axis to reset. Defaults to "both".
            event (optional): Event that triggered the reset. Defaults to None.
        """
        for ax in self.figure.axes:
            if isinstance(ax, maxes.SubplotBase):
                ax.relim()
                ax.autoscale(axis=axis)
        plt.draw()

    def add_key_binding(self, key_name: str, on_press_function: callable, description: str = None):
        """
        Add key bindings to the browser.

        Args:
            key_name (str): Key to bind.
            on_press_function (callable): Function to call when the key is pressed.
            description (str, optional): Description of the key binding. Defaults to None.
        """
        if description is None:
            description = on_press_function.__name__
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = (on_press_function, description)

    def set_default_keybindings(self):
        """Set default key bindings for navigation."""
        self.add_key_binding("left", self.decrement)
        self.add_key_binding("right", self.increment)
        self.add_key_binding(
            "up",
            (lambda s: s.increment(step=10)).__get__(self),
            description="increment by 10",
        )
        self.add_key_binding(
            "down",
            (lambda s: s.decrement(step=10)).__get__(self),
            description="decrement by 10",
        )
        self.add_key_binding(
            "shift+left",
            self.decrement_frac,
            description="step forward by 1/20 of the timeline",
        )
        self.add_key_binding(
            "shift+right",
            self.increment_frac,
            description="step backward by 1/20 of the timeline",
        )
        self.add_key_binding("shift+up", self.go_to_start)
        self.add_key_binding("shift+down", self.go_to_end)
        self.add_key_binding("ctrl+c", self.copy_to_clipboard)
        self.add_key_binding(
            "ctrl+k",
            (lambda s: s.show_key_bindings(f="new", pos="center left")).__get__(self),
            description="show key bindings",
        )
        self.add_key_binding(
            "/",
            (lambda s: s.pan(direction="right")).__get__(self),
            description="pan right",
        )
        self.add_key_binding(
            ",",
            (lambda s: s.pan(direction="left")).__get__(self),
            description="pan left",
        )
        self.add_key_binding(
            "l", (lambda s: s.pan(direction="up")).__get__(self), description="pan up"
        )
        self.add_key_binding(
            ".",
            (lambda s: s.pan(direction="down")).__get__(self),
            description="pan down",
        )
        self.add_key_binding("r", self.reset_axes)

    def increment(self, step: int = 1):
        """
        Increment the current index.

        Args:
            step (int, optional): Number of steps to increment. Defaults to 1.
        """
        self._current_idx = min(self._current_idx + step, len(self) - 1)
        self.update()

    def decrement(self, step: int = 1):
        """
        Decrement the current index.

        Args:
            step (int, optional): Number of steps to decrement. Defaults to 1.
        """
        self._current_idx = max(self._current_idx - step, 0)
        self.update()

    def go_to_start(self):
        """Go to the start of the data."""
        self._current_idx = 0
        self.update()

    def go_to_end(self):
        """Go to the end of the data."""
        self._current_idx = len(self) - 1
        self.update()

    def increment_frac(self, n_steps: int = 20):
        """
        Browse the entire dataset in n_steps. Increment the current index by a fraction of the total length.

        Args:
            n_steps (int, optional): Number of steps to divide the total length into. Defaults to 20.
        """
        self._current_idx = min(
            self._current_idx + int(len(self) / n_steps), len(self) - 1
        )
        self.update()

    def decrement_frac(self, n_steps: int = 20):
        """
        Decrement the current index by a fraction of the total length.

        Args:
            n_steps (int, optional): Number of steps to divide the total length into. Defaults to 20.
        """
        self._current_idx = max(self._current_idx - int(len(self) / n_steps), 0)
        self.update()

    def copy_to_clipboard(self):
        """
        Copy the current figure to the clipboard.
        Requires PySide2. Install this optionally after the environment is set up as it can cause problems, or live without this feature.
        """
        from PySide2.QtGui import QClipboard, QImage

        buf = io.BytesIO()
        self.figure.savefig(buf)
        QClipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def show_key_bindings(self, f: str = None, pos: str = "bottom right"):
        """
        Show the key bindings.

        Args:
            f (str, optional): Figure to show the key bindings in. Defaults to None.
            pos (str, optional): Position to show the key bindings. Defaults to "bottom right".
        """
        f = {None: self.figure, "new": plt.figure()}[f]
        text = []
        for shortcut, (_, description) in self._keypressdict.items():
            text.append(f"{shortcut:<12} - {description}")
        self._keybindingtext = utils.TextView(text, f, pos=pos)

    @staticmethod
    def _filter_sibling_axes(ax: list[maxes.Axes], share: str = "x", get_bool: bool = False):
        """
        Given a list of matplotlib axes, it will return axes to manipulate by picking one from a set of siblings.

        Args:
            ax (list[maxes.Axes]): List of axes to filter.
            share (str, optional): Axis to share. Defaults to "x".
            get_bool (bool, optional): Whether to return a boolean array. Defaults to False.

        Returns:
            list[maxes.Axes] or list[bool]: Filtered axes or boolean array representing the result of filtering relative to the input list of axes.
        """
        assert share in ("x", "y")
        if isinstance(ax, maxes.Axes):  # only one axis
            return [ax]
        ax = [tax for tax in ax if isinstance(tax, maxes.SubplotBase)]
        if not ax:  # no subplots in figure
            return
        pan_ax = [True] * len(ax)
        get_siblings = {"x": ax[0].get_shared_x_axes, "y": ax[0].get_shared_y_axes}[share]().get_siblings
        for i, ax_row in enumerate(ax):
            sib = get_siblings(ax_row)
            for j, ax_col in enumerate(ax[i + 1:]):
                if ax_col in sib:
                    pan_ax[j + i + 1] = False

        if get_bool:
            return pan_ax
        return [this_ax for this_ax, this_tf in zip(ax, pan_ax) if this_tf]

    def pan(self, direction: str = "left", frac: float = 0.2):
        """
        Pan the view.

        Args:
            direction (str, optional): Direction to pan. Defaults to "left".
            frac (float, optional): Fraction of the view to pan. Defaults to 0.2.
        """
        assert direction in ("left", "right", "up", "down")
        if direction in ("left", "right"):
            pan_ax = "x"
        else:
            pan_ax = "y"
        ax = self._filter_sibling_axes(self.figure.axes, share=pan_ax, get_bool=False)
        if ax is None:
            return
        for this_ax in ax:
            lim1, lim2 = {"x": this_ax.get_xlim, "y": this_ax.get_ylim}[pan_ax]()
            inc = (lim2 - lim1) * frac
            if direction in ("down", "right"):
                new_lim = (lim1 + inc, lim2 + inc)
            else:
                new_lim = (lim1 - inc, lim2 - inc)
            {"x": this_ax.set_xlim, "y": this_ax.set_ylim}[pan_ax](new_lim)
        plt.draw()
        self.update_without_clear()  # panning is pointless if update clears the axis!!

    def has(self, asset_type: str) -> bool:
        """
        Check if the browser has a specific asset type.

        Args:
            asset_type (str): Type of asset to check for.

        Returns:
            bool: True if the asset type is present, False otherwise.
        """
        assert asset_type in (
            "buttons",
            "selectors",
            "memoryslots",
            "statevariables",
            "events",
        )
        return len(getattr(self, asset_type)) != 0


class PlotBrowser(GenericBrowser):
    """
    Takes a list of data, and a plotting function (or a pair of setup
    and update functions) that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """

    def __init__(self, plot_data: list, plot_func: callable, figure_handle: plt.Figure = None, **plot_kwargs):
        """
        Initialize the PlotBrowser.

        Args:
            plot_data (list): List of data objects to browse.
            plot_func (callable): Plotting function or a tuple of (setup_func, update_func).
                
                plot_func can be a tuple (setup_func, update_func), or just one plotting function - update_func
                If only one function is supplied, figure axes will be cleared on each update.
                setup_func takes:
                    the first element in plot_data list as its first input
                    keyword arguments (same as plot_func)
                setup_func outputs:
                    **dictionary** of plot handles that goes as the second input to update_func

                update_func is a plot-refreshing function that accepts 3 inputs:
                    an element in the plot_data list as its first input
                    output of the setup_func if it exists, or a figure handle on which to plot
                    keyword arguments
            figure_handle (plt.Figure, optional): Matplotlib figure handle. Defaults to None. Ideally, this is handled by the setup function.
            **plot_kwargs: Additional keyword arguments to pass to the plotting function.
        """
        self.data = plot_data  # list where each element serves as input to plot_func
        self.plot_kwargs = plot_kwargs

        if isinstance(plot_func, tuple):
            assert len(plot_func) == 2
            self.setup_func, self.plot_func = plot_func
            self.plot_handles = self.setup_func(self.data[0], **self.plot_kwargs)
            plot_handle = list(self.plot_handles.values())[0]
            if "figure" in self.plot_handles:
                figure_handle = self.plot_handles["figure"]
            elif isinstance(plot_handle, list):
                figure_handle = plot_handle[0].figure
            else:
                figure_handle = plot_handle.figure  # figure_handle passed as input will be ignored
        else:
            self.setup_func, self.plot_func = None, plot_func
            self.plot_handles = None
            figure_handle = figure_handle

        # setup
        super().__init__(figure_handle)

        # initialize
        self.set_default_keybindings()
        self.buttons.add(
            text="Auto limits",
            type_="Toggle",
            action_func=self.update,
            start_state=False,
        )
        self.memoryslots.show()
        if self.__class__.__name__ == "PlotBrowser":  # if an inherited class is accessing this, then don't run the update function here
            self.update()  # draw the first instance
            self.reset_axes()
            plt.show(block=False)
        # add selectors after drawing!
        try:
            s0 = self.selectors.add(list(self.plot_handles.values())[0])
            self.buttons.add(
                text="Selector 0",
                type_="Toggle",
                action_func=s0.toggle,
                start_state=s0.is_active,
            )
        except AssertionError:
            print("Unable to add selectors")

    def get_current_data(self):
        """
        Get the current data.

        Returns:
            The current data.
        """
        return self.data[self._current_idx]

    def update(self, event=None):
        """
        Update the browser.

        Args:
            event (optional): Event that triggered the update. Defaults to None.
        """
        if self.setup_func is None:
            self.figure.clear()  # redraw the entire figure contents each time, NOT recommended
            self.memoryslots.show()
            self.plot_func(self.get_current_data(), self.figure, **self.plot_kwargs)
        else:
            self.memoryslots.update_display()
            self.plot_func(self.get_current_data(), self.plot_handles, **self.plot_kwargs)
        if self.buttons["Auto limits"].state:  # is True
            self.reset_axes()
        super().update(event)
        plt.draw()

    def udpate_without_clear(self):
        """Update the browser without clearing the axis."""
        self.memoryslots.update_display()
        plt.draw()


class SignalBrowser(GenericBrowser):
    """
    Browse an array of pysampled.Data elements, or 2D arrays.
    """

    def __init__(self, plot_data: list, titlefunc: callable = None, figure_handle: plt.Figure = None, reset_on_change: bool = False):
        """
        Initialize the SignalBrowser.

        Args:
            plot_data (list): List of data objects to browse.
            titlefunc (callable, optional): Function to generate titles. Defaults to None.
            figure_handle (plt.Figure, optional): Matplotlib figure handle. Defaults to None.
            reset_on_change (bool, optional): Whether to reset the view on change. Defaults to False.
        """
        super().__init__(figure_handle)

        self._ax = self.figure.subplots(1, 1)
        this_data = plot_data[0]
        if isinstance(this_data, pysampled.Data):
            self._plot = self._ax.plot(this_data.t, this_data())
        else:
            self._plot = self._ax.plot(this_data)

        self.data = plot_data
        if titlefunc is None:
            if hasattr(self.data[0], "name"):
                self.titlefunc = lambda s: f"{s.data[s._current_idx].name}"
            else:
                self.titlefunc = lambda s: f"Plot number {s._current_idx}"
        else:
            self.titlefunc = titlefunc

        self.reset_on_change = reset_on_change
        # initialize
        self.set_default_keybindings()
        self.buttons.add(
            text="Auto limits",
            type_="Toggle",
            action_func=self.update,
            start_state=False,
        )
        plt.show(block=False)
        self.update()

    def update(self, event=None):
        """
        Update the browser.

        Args:
            event (optional): Event that triggered the update. Defaults to None.
        """
        this_data = self.data[self._current_idx]
        if isinstance(this_data, pysampled.Data):
            data_to_plot = this_data.split_to_1d()
            for plot_handle, this_data_to_plot in zip(self._plot, data_to_plot):
                plot_handle.set_data(this_data_to_plot.t, this_data_to_plot())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
        if self.buttons["Auto limits"].state:  # is True
            self.reset_axes()
        plt.draw()
