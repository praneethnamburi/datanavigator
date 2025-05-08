"""
Module for browsing and visualizing time series data.

Classes:
    SignalBrowser: A browser for navigating through an array of `pysampled.Data` elements or 2D arrays.
"""

from __future__ import annotations

import pysampled
from matplotlib import pyplot as plt
from typing import Callable, Optional

from .core import GenericBrowser


class SignalBrowser(GenericBrowser):
    """
    Browse an array of pysampled.Data elements, or 2D arrays.
    """

    def __init__(
        self,
        plot_data: list[pysampled.Data],
        titlefunc: Optional[Callable] = None,
        figure_handle: Optional[plt.Figure] = None,
        reset_on_change: bool = False,
    ) -> None:
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
            (self._plot,) = self._ax.plot(this_data)

        self.data = plot_data
        if titlefunc is None:
            self.titlefunc = lambda s: getattr(
                s.data[s._current_idx], "name", f"Plot number {s._current_idx}"
            )
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
            self._plot.set_ydata(this_data)
        self._ax.set_title(self.titlefunc(self))
        if (
            "Auto limits" in self.buttons and self.buttons["Auto limits"].state
        ):  # is True
            self.reset_axes()
        plt.draw()
