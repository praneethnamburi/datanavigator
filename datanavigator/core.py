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
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib as mpl
from matplotlib import axes as maxes
from matplotlib import pyplot as plt

from ._qt import find_qt_window
from .assets import Buttons, MemorySlots, Selectors, StateVariables
from .events import Events


@dataclass
class KeyBinding:
    """One entry in :attr:`GenericBrowser._keypressdict`.

    ``group`` drives the section header in
    :meth:`GenericBrowser.show_key_bindings` (``None`` falls into the
    "Other" section, rendered last). ``on_button`` opts the binding into
    button-face hint rendering: the cheatsheet dialog *and* a matching
    button get the shortcut shown alongside the action.

    Pre-rc2 this slot was a ``(callback, description)`` tuple; the
    dataclass is a hard cut — no tuple-style backward compat. Downstream
    callers that poke ``_keypressdict`` directly (rare) need to migrate
    to attribute access.
    """

    callback: Callable
    description: str
    group: Optional[str] = None
    on_button: bool = False


def _group_keybindings(
    keypressdict: dict,
    section_order: Optional[tuple[str, ...]] = None,
) -> list[tuple[str, list[tuple[str, str]]]]:
    """Bucket ``keypressdict`` entries by ``KeyBinding.group``.

    Within each bucket, entries follow ``keypressdict`` iteration order
    (i.e. insertion order). Between buckets:

    - If ``section_order`` is given, those group names lead, in that
      order. Groups not in ``section_order`` follow in insertion order.
    - If ``section_order`` is ``None``, all named groups follow
      insertion order.

    The ``None`` group (becomes "Other") is always pushed to the end so
    explicitly named sections lead.
    """
    buckets: dict[Optional[str], list[tuple[str, str]]] = {}
    for shortcut, kb in keypressdict.items():
        buckets.setdefault(kb.group, []).append((shortcut, kb.description))

    other = buckets.pop(None, None)
    if section_order:
        leading = [
            (g, buckets.pop(g)) for g in section_order if g in buckets
        ]
        sections = leading + list(buckets.items())
    else:
        sections = list(buckets.items())
    sections = [(g or "Other", rows) for g, rows in sections]
    if other is not None:
        sections.append(("Other", other))
    return sections


def _format_keybindings_text(
    keypressdict: dict,
    section_order: Optional[tuple[str, ...]] = None,
) -> str:
    """Render the cheatsheet as plain text. Used by the non-Qt fallback."""
    lines = []
    for group_name, rows in _group_keybindings(keypressdict, section_order):
        lines.append(f"[{group_name}]")
        for shortcut, description in rows:
            lines.append(f"  {shortcut:<14} {description}")
        lines.append("")
    return "\n".join(lines).rstrip()


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
        # Soft Qt mode (1.4.0): if matplotlib is on a Qt backend, hold a
        # reference to the QMainWindow it built around this figure. Phase 2+
        # widget migrations attach native Qt widgets to it; under non-Qt
        # backends (Agg, TkAgg, inline) this is None and the asset managers
        # fall back to their pre-1.4 mpl-axes-based rendering.
        self._qt_window = find_qt_window(figure_handle)
        self._keypressdict = {}  # managed by add_key_binding
        self._bindings_removed = {}

        # tracking variable
        self._current_idx = 0

        # Holds the modeless cheatsheet dialog so it isn't GC'd while open.
        self._keybinding_dialog = None
        # Optional tuple of group names pinning the cheatsheet section
        # order. Subclasses (e.g. VideoPointAnnotator) override this so
        # the dialog reads in workflow order regardless of when each
        # binding was registered. ``None`` -> insertion order.
        self._section_order: Optional[tuple[str, ...]] = None
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
            key = self._resolve_keypress(event.key)
            if key is not None:
                f = self._keypressdict[key].callback
                argspec = inspect.getfullargspec(f)[0]
                if len(argspec) == 2 and argspec[1] == "event":
                    f(event)
                else:
                    f()  # this may or may not redraw everything
            if event.key in self.memoryslots._list:
                self.memoryslots.update(event.key)
        elif event.name == "close_event":  # perform cleanup
            self.cleanup()

    def _resolve_keypress(self, key):
        """Look up a key in ``self._keypressdict`` with a case-insensitive
        fallback for single-letter keys.

        matplotlib reports keys as uppercase when Shift is held (or
        Sticky Keys is active, or the user inadvertently holds shift).
        For DUSTrack-shaped workflows, a user pressing what they think
        is plain ``t`` while Shift is held would get mpl ``'T'``, which
        wouldn't match the ``'t'`` binding -- silent no-op. To avoid
        that surprise, single uppercase letters that aren't explicitly
        bound fall through to the lowercase variant. Explicit uppercase
        bindings (or multi-character bindings like ``'shift+left'``)
        take precedence and aren't affected.

        Returns the resolved key string (a key into _keypressdict) or
        None if no binding matches.
        """
        if key in self._keypressdict:
            return key
        if isinstance(key, str) and len(key) == 1 and key.isalpha() and key.isupper():
            lower = key.lower()
            if lower in self._keypressdict:
                return lower
        return None

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

    def add_key_binding(
        self,
        key_name: str,
        on_press_function: callable,
        description: str = None,
        *,
        group: Optional[str] = None,
        on_button: bool = False,
    ):
        """
        Add a key binding to the browser.

        Args:
            key_name: Key to bind (matplotlib key-event string, e.g. "s",
                "ctrl+a", "shift+left").
            on_press_function: Function to call when the key is pressed.
            description: Human-readable label shown in the cheatsheet.
                Defaults to ``on_press_function.__name__``.
            group: Section header in the cheatsheet dialog. ``None``
                falls into the "Other" section (rendered last).
            on_button: When ``True``, find a button whose ``action_func``
                is *identically* ``on_press_function`` and append
                ``"  ({key_name})"`` to its label. The match is by ``is``
                comparison — pass the same callable object to both
                :meth:`Buttons.add` and this method (no intermediate
                lambdas) or the hint won't attach. Resolution is
                symmetric: if the button is added *after* the binding,
                :meth:`Buttons.add` performs the same scan.
        """
        if description is None:
            description = on_press_function.__name__
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = KeyBinding(
            callback=on_press_function,
            description=description,
            group=group,
            on_button=on_button,
        )
        if on_button:
            self._apply_button_hint(key_name, on_press_function)

    def _apply_button_hint(self, key_name: str, callback: Callable) -> None:
        """Append a shortcut hint to any already-added button matching ``callback``.

        Called from :meth:`add_key_binding` when ``on_button=True`` and
        from :meth:`Buttons.add` (via the reverse-direction scan). The
        match is by ``is`` identity against the per-button
        ``_action_funcs`` list that :meth:`Buttons.add` populates.
        """
        from .assets import apply_shortcut_hint
        for btn in getattr(self.buttons, "_list", []):
            action_funcs = getattr(btn, "_action_funcs", ())
            if any(af is callback for af in action_funcs):
                apply_shortcut_hint(btn, key_name)

    def remove_key_binding(self, key_name: str):
        """
        Remove a key binding from the browser.

        Args:
            key_name (str): Key to remove the binding for.
        """
        if key_name in self._keypressdict:
            del self._keypressdict[key_name]
            reversed_bindings_dict = {key: mpl_param_name for mpl_param_name, key in self._bindings_removed.items()}
            if key_name in reversed_bindings_dict:
                # find the key value pair with the value equal to key_name
                mpl.rcParams[reversed_bindings_dict[key_name]].append(key_name)

    def set_default_keybindings(self):
        """Set default key bindings for navigation."""
        self.add_key_binding("left", self.decrement, group="Navigation")
        self.add_key_binding("right", self.increment, group="Navigation")
        self.add_key_binding(
            "up",
            (lambda s: s.increment(step=10)).__get__(self),
            description="increment by 10",
            group="Navigation",
        )
        self.add_key_binding(
            "down",
            (lambda s: s.decrement(step=10)).__get__(self),
            description="decrement by 10",
            group="Navigation",
        )
        self.add_key_binding(
            "shift+left",
            self.decrement_frac,
            description="step forward by 1/20 of the timeline",
            group="Navigation",
        )
        self.add_key_binding(
            "shift+right",
            self.increment_frac,
            description="step backward by 1/20 of the timeline",
            group="Navigation",
        )
        self.add_key_binding("shift+up", self.go_to_end, group="Navigation")
        self.add_key_binding("shift+down", self.go_to_start, group="Navigation")
        self.add_key_binding("ctrl+c", self.copy_to_clipboard, "Copy screenshot to clipboard")
        self.add_key_binding(
            "ctrl+k",
            (lambda s: s.show_key_bindings()).__get__(self),
            description="show key bindings",
            group="View",
        )
        self.add_key_binding(
            "/",
            (lambda s: s.pan(direction="right")).__get__(self),
            description="pan right",
            group="View",
        )
        self.add_key_binding(
            ",",
            (lambda s: s.pan(direction="left")).__get__(self),
            description="pan left",
            group="View",
        )
        self.add_key_binding(
            "l",
            (lambda s: s.pan(direction="up")).__get__(self),
            description="pan up",
            group="View",
        )
        self.add_key_binding(
            ".",
            (lambda s: s.pan(direction="down")).__get__(self),
            description="pan down",
            group="View",
        )
        self.add_key_binding("r", self.reset_axes, group="View")

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
        Copy the current figure window to the clipboard.

        On a Qt backend, grabs the entire ``QMainWindow`` -- the
        matplotlib canvas *plus* every Qt-side widget parented to it
        (left-column dock with buttons / statevariables, fast_render
        image pane, downstream sidebars added by consumers such as
        DUSTrack). The pre-rc2 path ``savefig``'d the figure alone and
        missed all of them; on a DUSTrack window the clipboard image
        would show the trace canvas with nothing where the sidebar /
        image pane visually sat.

        On non-Qt backends (Agg, headless), falls back to copying just
        the matplotlib figure via ``savefig``.

        Requires a Qt binding (PyQt5 / PyQt6 / PySide2 / PySide6); imported
        lazily via ``qtpy`` so installs without Qt still get the rest of
        the package. ``pip install datanavigator[qt]`` pulls PyQt6.
        """
        from qtpy.QtGui import QImage
        from qtpy.QtWidgets import QApplication

        # Use the QApplication singleton clipboard. Direct QClipboard()
        # instantiation worked on Qt5 but is disallowed on Qt6.
        app = QApplication.instance() or QApplication([])
        qt_window = find_qt_window(self.figure)
        if qt_window is not None:
            app.clipboard().setPixmap(qt_window.grab())
            return

        buf = io.BytesIO()
        self.figure.savefig(buf)
        app.clipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def show_key_bindings(self):
        """Open the keyboard-shortcut cheatsheet.

        On a Qt-backed figure: a modeless ``QDialog`` with one section
        per ``KeyBinding.group`` (insertion order; ``None`` group
        rendered last as "Other"). Each section is a 2-column table —
        monospace shortcut on the left, description on the right.

        On non-Qt backends: prints the same content to stdout. The
        pre-rc2 ``TextView`` overlay is gone; matplotlib doesn't host a
        nice cheatsheet without the Qt path.
        """
        from ._qt import make_keybindings_dialog
        dialog = make_keybindings_dialog(
            self.figure, self._keypressdict, self._section_order
        )
        if dialog is not None:
            self._keybinding_dialog = dialog
            return
        # Non-Qt fallback: dump to stdout.
        print(_format_keybindings_text(self._keypressdict, self._section_order))

    @staticmethod
    def _filter_sibling_axes(
        ax: list[maxes.Axes], share: str = "x", get_bool: bool = False
    ):
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
        get_siblings = {"x": ax[0].get_shared_x_axes, "y": ax[0].get_shared_y_axes}[
            share
        ]().get_siblings
        for i, ax_row in enumerate(ax):
            sib = get_siblings(ax_row)
            for j, ax_col in enumerate(ax[i + 1 :]):
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
