"""
Module for browsing and visualizing data using customizable plotting functions.

Classes:
    PlotBrowser: A browser for navigating through a list of data objects and visualizing them using custom plotting functions.
"""

from __future__ import annotations
from typing import Union, Tuple, Any

from matplotlib import pyplot as plt

from .core import GenericBrowser


class PlotBrowser(GenericBrowser):
    """
    Takes a list of data, and a plotting function (or a pair of setup
    and update functions) that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """

    def __init__(
        self,
        plot_data: list,
        plot_func: Union[Tuple[callable, callable], callable],
        figure_handle: plt.Figure = None,
        **plot_kwargs,
    ):
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
                    plot_handles["figure"] is the figure handle to be used for plotting, and is required

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
            assert "figure" in self.plot_handles
            figure_handle = self.plot_handles["figure"]
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

        # if an inherited class is accessing this, then don't run the update function here
        if self.__class__.__name__ == "PlotBrowser":
            self.update()
            self.reset_axes()
            plt.show(block=False)

        # add selectors after drawing!
        if self.plot_handles is not None:
            self.add_selectors()

    def add_selectors(self):
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
        Data getter. Plotting data is a list of objects that are being
        plotted one at a time. This function returns the current object.
        """
        return self.data[self._current_idx]

    def update(self, event: Any = None):
        """Update the browser."""
        if self.setup_func is None:
            self.figure.clear()  # redraw the entire figure contents each time, NOT recommended
            self.memoryslots.show()
            self.plot_func(self.get_current_data(), self.figure, **self.plot_kwargs)
        else:
            self.memoryslots.update_display()
            self.plot_func(
                self.get_current_data(), self.plot_handles, **self.plot_kwargs
            )
        if self.buttons["Auto limits"].state:  # is True
            self.reset_axes()
        super().update(event)
        plt.draw()
