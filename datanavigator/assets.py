"""
This module provides classes and functions for managing various assets such as buttons, selectors, and state variables in a graphical user interface.

Classes:
    Button - Add a 'name' state to a matplotlib widget button.
    StateButton - Store a number/coordinate in a button.
    ToggleButton - Add a toggle button to a matplotlib figure.
    Selector - Select points in a plot using the lasso selection widget.
    StateVariable - Manage state variables with multiple states.

    AssetContainer - Container for managing assets such as buttons, memory slots, etc.
    
    Buttons - Manager for buttons in a matplotlib figure or GUI.
    Selectors - Manager for selector objects for picking points on line2D objects.
    MemorySlots - Manager for memory slots to store and navigate positions.
    StateVariables - Manager for state variables.
"""

from __future__ import annotations

import numpy as np
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.path import Path as mPath
from matplotlib.widgets import Button as ButtonWidget
from matplotlib.widgets import LassoSelector as LassoSelectorWidget
from typing import Any, Callable, List, Optional, Union

from .utils import TextView


class Button(ButtonWidget):
    """Add a 'name' state to a matplotlib widget button."""

    def __init__(self, ax, name: str, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.name = name


class StateButton(Button):
    """Store a number/coordinate in a button."""

    def __init__(self, ax, name: str, start_state: Any, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.state = start_state  # stores something in the state


class ToggleButton(StateButton):
    """
    Add a toggle button to a matplotlib figure.

    For example usage, see PlotBrowser.
    """

    def __init__(self, ax, name: str, start_state: bool = True, **kwargs) -> None:
        super().__init__(ax, name, start_state, **kwargs)
        self.on_clicked(self.toggle)
        self.set_text()

    def set_text(self) -> None:
        """Set the text of the toggle button."""
        self.label._text = f"{self.name}={self.state}"

    def toggle(self, event=None) -> None:
        """Toggle the state of the button."""
        self.state = not self.state
        self.set_text()

    def set_state(self, state: bool) -> None:
        """Set the state of the button."""
        assert isinstance(state, bool)
        self.state = state
        self.set_text()


class Selector:
    """
    Select points in a plot using the lasso selection widget.

    Indices of selected points are stored in self.sel.

    Example:
        f, ax = plt.subplots(1, 1)
        ph, = ax.plot(np.random.rand(20))
        plt.show(block=False)
        ls = gui.Lasso(ph)
        ls.start()
        -- play around with selecting points --
        ls.stop() -> disconnects the events
    """

    def __init__(self, plot_handle: mlines.Line2D) -> None:
        """Initialize the selector with a plot handle."""
        assert isinstance(plot_handle, mlines.Line2D)
        self.plot_handle = plot_handle
        self.xdata, self.ydata = plot_handle.get_data()
        self.ax = plot_handle.axes
        (self.overlay_handle,) = self.ax.plot([], [], ".")
        self.sel = np.zeros(self.xdata.shape, dtype=bool)
        self.is_active = False

    def get_data(self) -> np.ndarray:
        """Get the data points of the plot."""
        return np.vstack((self.xdata, self.ydata)).T

    def onselect(self, verts: List[tuple]) -> None:
        """Select if not previously selected; Unselect if previously selected."""
        selected_ind = mPath(verts).contains_points(self.get_data())
        self.sel = np.logical_xor(selected_ind, self.sel)
        sel_x = list(self.xdata[self.sel])
        sel_y = list(self.ydata[self.sel])
        self.overlay_handle.set_data(sel_x, sel_y)
        plt.draw()

    def start(self, event=None) -> None:
        """Start the lasso selection."""
        self.lasso = LassoSelectorWidget(self.plot_handle.axes, self.onselect)
        self.is_active = True

    def stop(self, event=None) -> None:
        """Stop the lasso selection."""
        self.lasso.disconnect_events()
        self.is_active = False

    def toggle(self, event=None) -> None:
        """Toggle the lasso selection."""
        if self.is_active:
            self.stop(event)
        else:
            self.start(event)


class AssetContainer:
    """
    Container for assets such as a button, memoryslot, etc.

    Args:
        parent (Any): matplotlib figure, or something that has a 'figure' attribute that is a figure.
    """

    def __init__(self, parent: Any) -> None:
        self._list: List[Any] = []  # list of assets
        self.parent = parent

    def __len__(self) -> int:
        return len(self._list)

    @property
    def names(self) -> List[str]:
        return [x.name for x in self._list]

    def __getitem__(self, key: Union[int, str]) -> Any:
        """Return an asset by the name key or by position in the list."""
        if not self._has_names():
            assert isinstance(key, int)

        if isinstance(key, int) and key not in self.names:
            return self._list[key]

        return {x.name: x for x in self._list}[key]

    def _has_names(self) -> bool:
        try:
            self.names
            return True
        except AttributeError:
            return False

    def add(self, asset: Any) -> Any:
        """Add an asset to the container."""
        if hasattr(asset, "name"):
            assert asset.name not in self.names
        self._list.append(asset)
        return asset
    
    def __contains__(self, item: str) -> bool:
        """Check if an asset with the given name exists in the container."""
        return item in self.names


class Buttons(AssetContainer):
    """Manager for buttons in a matplotlib figure or GUI (see GenericBrowser for example)."""

    def add(
        self,
        text: str = "Button",
        action_func: Optional[Union[Callable, List[Callable]]] = None,
        pos: Optional[tuple] = None,
        w: float = 0.25,
        h: float = 0.05,
        buf: float = 0.01,
        type_: str = "Push",
        **kwargs,
    ) -> Button:
        """
        Add a button to the parent figure / object.

        If pos is provided, then w, h, and buf will be ignored.
        """
        assert type_ in ("Push", "Toggle")
        nbtn = len(self)
        if pos is None:  # start adding at the top left corner
            parent_fig = self.parent.figure
            mul_factor = 6.4 / parent_fig.get_size_inches()[0]

            btn_w = w * mul_factor
            btn_h = h * mul_factor
            btn_buf = buf
            pos = (
                btn_buf,
                (1 - btn_buf) - ((btn_buf + btn_h) * (nbtn + 1)),
                btn_w,
                btn_h,
            )

        if type_ == "Toggle":
            b = ToggleButton(plt.axes(pos), text, **kwargs)
        else:
            b = Button(plt.axes(pos), text, **kwargs)

        if action_func is not None:  # more than one can be attached
            if isinstance(action_func, (list, tuple)):
                for af in action_func:
                    b.on_clicked(af)
            else:
                b.on_clicked(action_func)

        return super().add(b)


class Selectors(AssetContainer):
    """Manager for selector objects - for picking points on line2D objects."""

    def add(self, plot_handle: mlines.Line2D) -> Selector:
        """Add a selector to the container."""
        return super().add(Selector(plot_handle))


class MemorySlots(AssetContainer):
    """Manager for memory slots to store and navigate positions."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self._list = self.initialize()
        self._memtext = None

    @staticmethod
    def initialize() -> dict:
        """Initialize memory slots."""
        return {str(k): None for k in range(1, 10)}

    def disable(self) -> None:
        """Disable memory slots."""
        self._list = {}

    def enable(self) -> None:
        """Enable memory slots."""
        self._list = self.initialize()

    def show(self, pos: str = "bottom left") -> None:
        """Show memory slot text."""
        self._memtext = TextView(self._list, fax=self.parent.figure, pos=pos)

    def update(self, key: str) -> None:
        """
        Handle memory slot updates.

        Initiate when None, go to the slot if it exists, free slot if pressed when it exists.
        key is the event.key triggered by a callback.
        """
        if self._list[key] is None:
            self._list[key] = self.parent._current_idx
            self.update_display()
        elif self._list[key] == self.parent._current_idx:
            self._list[key] = None
            self.update_display()
        else:
            self.parent._current_idx = self._list[key]
            self.parent.update()

    def update_display(self) -> None:
        """Refresh memory slot text if it is not hidden."""
        if self._memtext is not None:
            self._memtext.update(self._list)

    def hide(self) -> None:
        """Hide the memory slot text."""
        if self._memtext is not None:
            self._memtext._text.remove()
        self._memtext = None

    def is_enabled(self) -> bool:
        """Check if memory slots are enabled."""
        return bool(self._list)


class StateVariable:
    """Manage state variables with multiple states."""

    def __init__(self, name: str, states: list) -> None:
        self.name = name
        self.states = list(states)
        self._current_state_idx = 0

    @property
    def current_state(self) -> Any:
        """Get the current state."""
        return self.states[self._current_state_idx]

    def n_states(self) -> int:
        """Get the number of states."""
        return len(self.states)

    def cycle(self) -> None:
        """Cycle to the next state."""
        self._current_state_idx = (self._current_state_idx + 1) % self.n_states()

    def cycle_back(self) -> None:
        """Cycle to the previous state."""
        self._current_state_idx = (self._current_state_idx - 1) % self.n_states()

    def set_state(self, state: Union[int, str]) -> None:
        """Set the state."""
        if isinstance(state, int):
            assert 0 <= state < self.n_states()
            self._current_state_idx = state
        if isinstance(state, str):
            assert state in self.states
            self._current_state_idx = self.states.index(state)


class StateVariables(AssetContainer):
    """Manager for state variables."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self._text = None

    def asdict(self) -> dict:
        """Return state variables as a dictionary."""
        return {x.name: x.states for x in self._list}

    def add(self, name: str, states: list) -> StateVariable:
        """Add a state variable to the container."""
        assert name not in self.names
        return super().add(StateVariable(name, states))

    def _get_display_text(self) -> List[str]:
        """Get the display text for state variables."""
        return ["State variables:"] + [
            f"{x.name} - {x.current_state}" for x in self._list
        ]

    def show(self, pos: str = "bottom right") -> None:
        """Show state variables text."""
        self._text = TextView(self._get_display_text(), fax=self.parent.figure, pos=pos)

    def update_display(self, draw: bool = True) -> None:
        """Update the display of state variables."""
        self._text.update(self._get_display_text())
        if draw:
            plt.draw()
