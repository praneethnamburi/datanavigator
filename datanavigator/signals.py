"""
Module for browsing and visualizing time series data.

Classes:
    SignalBrowser: A browser for navigating through an array of `pysampled.Data` elements or 2D arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pysampled
from matplotlib import pyplot as plt

from .core import GenericBrowser

if TYPE_CHECKING:
    from .assets import StateVariable


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
        signal_names: Optional[list[str]] = None,
        show_signal_dropdown: bool = True,
    ) -> None:
        """
        Initialize the SignalBrowser.

        Args:
            plot_data (list): List of data objects to browse.
            titlefunc (callable, optional): Function to generate titles. Defaults to None.
            figure_handle (plt.Figure, optional): Matplotlib figure handle. Defaults to None.
            reset_on_change (bool, optional): Whether to reset the view on change. Defaults to False.
            signal_names (list[str], optional): One display label per
                ``plot_data`` entry for the sidebar dropdown (see
                :meth:`add_signal_dropdown`), which lets the displayed
                signal be picked by name in addition to arrow-key
                navigation. Defaults to None -- labels are auto-derived
                from each entry's ``name`` attribute (falling back to
                ``"signal <i>"``).
            show_signal_dropdown (bool, optional): Whether to add the
                signal-selection dropdown. Defaults to True; pass False to
                suppress it (e.g. an embedded browser where the sidebar
                control is unwanted).
        """
        super().__init__(figure_handle)

        self.data = plot_data
        self._ax = self.figure.subplots(1, 1)
        # Pre-allocate the maximum number of 1-D traces across all entries,
        # then show only as many as the current entry needs (hiding the
        # rest). This lets entries with different sub-channel counts share
        # one stable set of Line2D handles -- no handle churn (and no color
        # reshuffle) when switching between, say, a 1-channel EMG and a
        # 3-axis accelerometer signal.
        self._n_traces_max = max(self._n_traces(d) for d in plot_data)
        self._plot = [self._ax.plot([], [])[0] for _ in range(self._n_traces_max)]

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
        if show_signal_dropdown:
            self.add_signal_dropdown(signal_names)

    def update(self, event=None):
        """
        Update the browser.

        Args:
            event (optional): Event that triggered the update. Defaults to None.
        """
        traces = self._as_traces(self.data[self._current_idx])
        for i, line in enumerate(self._plot):
            if i < len(traces):
                x, y = traces[i]
                line.set_data(x, y)
                line.set_visible(True)
            else:
                # Surplus pre-allocated handle: blank + hide it so a signal
                # with fewer sub-channels than the maximum doesn't show
                # stale data from a previously-displayed wider signal.
                line.set_data([], [])
                line.set_visible(False)
        self._ax.set_title(self.titlefunc(self))
        if "Auto limits" in self.buttons and self.buttons["Auto limits"].state:  # is True
            self.reset_axes()
        # update_assets() (the GenericBrowser seam SignalBrowser.update used
        # to skip) keeps the signal dropdown in step with arrow-key
        # navigation via GenericBrowser._sync_item_dropdown, and refreshes
        # any other shown assets. No-ops for assets that were never shown.
        self.update_assets()
        plt.draw()

    def add_signal_dropdown(
        self,
        names: Optional[list[str]] = None,
        var_name: str = "signal",
    ) -> "StateVariable":
        """Add the signal-selection dropdown (a thin wrapper).

        Delegates to :meth:`GenericBrowser.add_item_dropdown` with a
        ``"signal"`` row label. ``names=None`` derives labels from each
        entry's ``name`` (falling back to ``"signal <i>"`` via
        :meth:`_default_item_names`). See :meth:`add_item_dropdown` for the
        full contract: idempotent relabeling, Qt vs text rendering, and the
        two-way ``_current_idx`` binding.
        """
        return self.add_item_dropdown(names, var_name=var_name)

    def _default_item_names(self) -> list[str]:
        """Signal-flavored default dropdown labels (``"signal <i>"`` fallback)."""
        return [
            getattr(d, "name", None) or f"signal {i}" for i, d in enumerate(self.data)
        ]

    @staticmethod
    def _n_traces(data) -> int:
        """Number of 1-D traces a single ``data`` entry renders as."""
        if isinstance(data, pysampled.Data):
            return len(data.split_to_1d())
        arr = np.asarray(data)
        return 1 if arr.ndim == 1 else arr.shape[1]

    @staticmethod
    def _as_traces(data) -> list[tuple]:
        """Decompose a ``data`` entry into a list of ``(x, y)`` 1-D traces.

        Keeps :meth:`_n_traces` and :meth:`update` in agreement: pysampled
        signals split per channel (time as x); a 1-D array/list is a single
        index-vs-value trace; a 2-D array is one trace per column.
        """
        if isinstance(data, pysampled.Data):
            return [(d.t, d()) for d in data.split_to_1d()]
        arr = np.asarray(data)
        if arr.ndim == 1:
            return [(np.arange(len(arr)), arr)]
        x = np.arange(arr.shape[0])
        return [(x, arr[:, j]) for j in range(arr.shape[1])]
