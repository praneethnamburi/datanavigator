"""
Module for browsing and visualizing time series data.

Classes:
    SignalBrowser: A browser for navigating through an array of `pysampled.Data` elements or 2D arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

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
        # The optional signal-selection dropdown's StateVariable; None until
        # add_signal_dropdown() runs. Set before the first update() so the
        # navigation<->dropdown sync in update() can no-op safely.
        self._signal_var: Optional["StateVariable"] = None
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
        this_data = self.data[self._current_idx]
        if isinstance(this_data, pysampled.Data):
            data_to_plot = this_data.split_to_1d()
            for plot_handle, this_data_to_plot in zip(self._plot, data_to_plot):
                plot_handle.set_data(this_data_to_plot.t, this_data_to_plot())
        else:
            self._plot.set_ydata(this_data)
        self._ax.set_title(self.titlefunc(self))
        if "Auto limits" in self.buttons and self.buttons["Auto limits"].state:  # is True
            self.reset_axes()
        # Keep the signal dropdown (if present) in step with arrow-key
        # navigation: push _current_idx into the state var directly --
        # bypassing _on_signal_var_change, which handles only the reverse
        # (dropdown-pick -> index) direction -- then let update_assets
        # refresh the sidebar control. update_assets() is the GenericBrowser
        # seam SignalBrowser.update historically skipped; it no-ops for
        # assets that were never shown.
        if (
            self._signal_var is not None
            and self._signal_var._current_state_idx != self._current_idx
        ):
            self._signal_var._current_state_idx = self._current_idx
        self.update_assets()
        plt.draw()

    def add_signal_dropdown(
        self,
        names: Optional[list[str]] = None,
        var_name: str = "signal",
    ) -> "StateVariable":
        """Add a sidebar dropdown that jumps straight to a signal by name.

        A convenience over arrow-key navigation: the browser is handed one
        label per entry in ``self.data`` and renders them as a ``QComboBox``
        in the Qt sidebar (a :class:`~datanavigator.assets.StateVariable`
        with ``widget="dropdown"``). Picking an entry sets
        :attr:`_current_idx` and redraws; arrow-key navigation keeps the
        dropdown in step (see :meth:`update`).

        Calling this again with the same ``var_name`` replaces the existing
        dropdown -- so the default dropdown built at construction can be
        relabeled later (e.g. once signal addresses are known).

        Args:
            names: one display label per ``self.data`` entry. ``None``
                derives them from each entry's ``name`` attribute, falling
                back to ``"signal <i>"``. Length must match ``len(self.data)``.
            var_name: the state-variable name (the dropdown's row label).

        Returns:
            The registered :class:`StateVariable`.

        On non-Qt backends the control degrades to the read-only
        state-variables text path, but the index binding still works.
        """
        if names is None:
            names = [
                getattr(d, "name", None) or f"signal {i}" for i, d in enumerate(self.data)
            ]
        if len(names) != len(self.data):
            raise ValueError(
                f"add_signal_dropdown: got {len(names)} names for "
                f"{len(self.data)} signals; lengths must match."
            )
        names = [str(n) for n in names]

        # Idempotent: drop any prior dropdown of this name (e.g. the default
        # one from __init__) so a follow-up call relabels rather than raising
        # on the duplicate state-variable name.
        if var_name in self.statevariables:
            self.statevariables.remove(var_name)

        var = self.statevariables.add(var_name, names, widget="dropdown")
        # Mirror the current browse position into the dropdown, then wire
        # pick -> index. Direct index assignment bypasses the on-change
        # callback (not registered yet, and the value already matches).
        var._current_state_idx = self._current_idx
        var.add_on_change(self._on_signal_var_change)
        self._signal_var = var

        # Mount/refresh the sidebar control (QComboBox on Qt; TextView fallback).
        self.statevariables.show()
        return var

    def _on_signal_var_change(self) -> None:
        """Mirror a dropdown pick into the browse index.

        Fired by :meth:`StateVariable.set_state` from the QComboBox
        ``on_pick`` handler (which then calls :meth:`update` itself), so we
        only move the index here -- no redraw. Keyboard navigation takes the
        reverse path: :meth:`update` writes ``_current_state_idx`` directly,
        bypassing this callback.
        """
        if self._signal_var is not None:
            self._current_idx = self._signal_var._current_state_idx
