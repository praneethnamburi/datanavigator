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


def apply_shortcut_hint(button, key_name: str) -> None:
    """Append ``"  (key_name)"`` to ``button``'s visible label.

    Used by the keybinding cheatsheet wiring: when a binding declared
    ``on_button=True`` matches a button (by ``action_func`` identity),
    we want the shortcut visible on the button face. Handles both
    backends:

    - mpl :class:`Button` / :class:`ToggleButton` carry a
      :class:`matplotlib.text.Text` label at ``.label``. Toggle buttons
      rebuild the label every state flip via :meth:`set_text`; we patch
      ``.name`` so the suffix survives across toggles.
    - The Qt-path ``_QtPushButton`` / ``_QtToggleButton`` wrappers (in
      :mod:`._qt`) expose ``.name`` + ``._qt_btn``; we mutate ``.name``
      and call ``setText`` (push) or ``set_text()`` (toggle, which
      rebuilds from ``name``).

    Idempotent on the same ``key_name``: if the suffix is already
    present, returns without re-appending.
    """
    suffix = f"  ({key_name})"
    current_name = getattr(button, "name", "")
    if current_name.endswith(suffix):
        return
    new_name = current_name + suffix
    button.name = new_name
    if isinstance(button, ToggleButton):
        button.set_text()
    elif isinstance(button, Button):
        button.label.set_text(new_name)
    elif hasattr(button, "_qt_btn"):
        if hasattr(button, "set_text") and getattr(button, "_state", None) is not None:
            button.set_text()  # toggle wrapper rebuilds from .name
        else:
            button._qt_btn.setText(new_name)


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

    def remove(self, name: str) -> Any:
        """Remove and return the asset with this ``name``.

        Counterpart to :meth:`add`; raises ``KeyError`` if no asset in
        the container carries that name. The caller is responsible for
        tearing down any plot handles / Qt widgets the asset owns --
        :class:`AssetContainer` only manages the membership list.
        """
        for i, x in enumerate(self._list):
            if getattr(x, "name", None) == name:
                return self._list.pop(i)
        raise KeyError(name)

    def __contains__(self, item: str) -> bool:
        """Check if an asset with the given name exists in the container."""
        return item in self.names


class Buttons(AssetContainer):
    """Manager for buttons in a matplotlib figure or GUI (see GenericBrowser for example).

    Phase 3 of the 1.4.0 Qt refactor (soft mode): when the parent's figure
    is on a Qt canvas AND no explicit ``pos`` is given, ``add()`` builds a
    native QPushButton in a QVBoxLayout column hosted by a QDockWidget on
    the QMainWindow's LeftDockWidgetArea (pre-rc2: QToolBar; see
    :func:`_qt._get_buttons_widget`). On every other backend, or when
    an explicit ``pos`` is given, the original matplotlib-widgets path
    runs unchanged. The returned object exposes the same public surface
    either way (``name``, ``on_clicked``, plus toggle-specific ``state``
    / ``toggle`` / ``set_text`` / ``set_state``).

    Lifecycle gotcha: ``Buttons.add()`` reads the figure's canvas at call
    time. If a Figure subclass adds buttons inside its own ``__init__``
    (as in :class:`datanavigator.examples.ButtonDemo`), this runs *before*
    matplotlib attaches a Qt canvas to the figure, so those buttons end
    up on the mpl path even under QtAgg. The buttons still function;
    they just miss the Phase 3 perf win. Browsers that take a
    figure_handle (the common case) are unaffected, because the figure
    is fully constructed before any button is added.

    rc2 row-cursor note: ``_mpl_row_cursor`` tracks the next vertical
    slot for the mpl-fallback layout independent of ``len(self)``.
    ``add()`` and ``add_separator()`` used to derive y from ``len(self)``,
    which assumed one button per row. ``add_multi()`` (one row, N
    buttons) breaks that assumption, so the cursor is bumped by 1 per
    row (regardless of how many buttons land on it) while ``len(self)``
    still grows by N. For single-add-only flows the two stay equal,
    preserving pre-rc2 placement.
    """

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self._mpl_row_cursor = 0
        # Consumer-registered style stylers keyed by tag. _resolve_style
        # checks this first, then falls back to styles.BUILTIN_STYLES, so
        # consumers can either pick a built-in tag or shadow it with a
        # bespoke palette.
        self._style_registry: dict = {}

    def add(
        self,
        text: str = "Button",
        action_func: Optional[Union[Callable, List[Callable]]] = None,
        pos: Optional[tuple] = None,
        w: float = 0.25,
        h: float = 0.05,
        buf: float = 0.01,
        type_: str = "Push",
        style_tag: Optional[str] = None,
        **kwargs,
    ) -> Button:
        """
        Add a button to the parent figure / object.

        If pos is provided, then w, h, and buf will be ignored.
        """
        assert type_ in ("Push", "Toggle")

        # Try the Qt path first. parent.figure works for both
        # GenericBrowser-shaped parents and Figure-as-parent (examples.py).
        # We only Qt-ify the default-position case; an explicit pos is a
        # request for mpl-style placement, which we can't replicate in a
        # QVBoxLayout without surprises.
        b = None
        if pos is None:
            from ._qt import make_qt_button
            parent_fig = self.parent.figure
            start_state = kwargs.get("start_state", True if type_ == "Toggle" else None)
            b = make_qt_button(
                parent_fig, text, type_=type_,
                start_state=bool(start_state) if start_state is not None else True,
            )

        if b is None:
            # mpl fallback (the pre-1.4 path).
            if pos is None:  # start adding at the top left corner
                parent_fig = self.parent.figure
                mul_factor = 6.4 / parent_fig.get_size_inches()[0]

                btn_w = w * mul_factor
                btn_h = h * mul_factor
                btn_buf = buf
                nbtn = self._mpl_row_cursor
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

        self._finalize_button(b, action_func, style_tag=style_tag)
        self._mpl_row_cursor += 1
        return super().add(b)

    def register_style(
        self, name: str, styler: Callable[[Any], None],
    ) -> None:
        """Register a per-button styler under ``name``.

        Consumers call this once at setup, then pass ``style_tag=name``
        to :meth:`add` / :meth:`add_multi`. Re-registering an existing
        name (including a :mod:`.styles` built-in) replaces the binding
        -- consumer registrations always win over built-ins.

        ``styler`` is invoked with the freshly-constructed button as its
        sole argument at the tail of :meth:`_finalize_button`. It should
        detect the Qt path via ``getattr(b, "_qt_btn", None)`` and apply
        QSS / palette manipulation; mpl-path styling is the styler's
        choice (the dnav built-ins no-op there).
        """
        self._style_registry[name] = styler

    def _resolve_style(self, tag: str) -> Callable[[Any], None]:
        """Resolve a style tag to a styler callable.

        Consumer registry first, then :data:`styles.BUILTIN_STYLES`. An
        unknown tag raises ``KeyError`` -- silent no-op would mask
        typos and "forgot to register_style before adding."
        """
        if tag in self._style_registry:
            return self._style_registry[tag]
        from .styles import BUILTIN_STYLES
        if tag in BUILTIN_STYLES:
            return BUILTIN_STYLES[tag]
        raise KeyError(
            f"unknown style_tag {tag!r}; register it via "
            f"Buttons.register_style({tag!r}, ...) or use one of the "
            f"built-in tags: {sorted(BUILTIN_STYLES)}"
        )

    def reapply_styles(self) -> None:
        """Re-run the styler for every already-added button.

        Useful after :meth:`register_style` changes a binding (e.g. a
        theme swap re-registers ``"primary"`` with a dark-mode palette
        and the existing buttons need to repaint).
        """
        for b in self._list:
            tag = getattr(b, "_style_tag", None)
            if tag is None:
                continue
            self._resolve_style(tag)(b)

    def _finalize_button(
        self,
        b: Any,
        action_func: Optional[Union[Callable, List[Callable]]],
        style_tag: Optional[str] = None,
    ) -> None:
        """Wire action callbacks + reverse-resolve any pre-registered keybinding.

        Shared post-construction tail for :meth:`add` and :meth:`add_multi`.
        Mutates ``b`` in place: sets ``b._action_funcs`` (a list, even when
        zero or one callable was passed), connects every callable via
        ``on_clicked``, and applies a shortcut-hint suffix to the label if
        the parent already has a matching ``on_button=True`` KeyBinding.
        If ``style_tag`` is given, resolves it through the consumer
        registry / built-ins and applies the styler on ``b`` (and
        records the tag on ``b._style_tag`` so :meth:`reapply_styles`
        can find it later).
        """
        # Record the action callables so add_key_binding(... on_button=True)
        # can match later (or so the reverse scan below can match an
        # already-registered binding to this newly-added button).
        b._action_funcs = []
        if action_func is not None:  # more than one can be attached
            if isinstance(action_func, (list, tuple)):
                for af in action_func:
                    b.on_clicked(af)
                    b._action_funcs.append(af)
            else:
                b.on_clicked(action_func)
                b._action_funcs.append(action_func)

        # Reverse direction: if any pre-registered KeyBinding has
        # on_button=True and points at one of this button's action_funcs,
        # apply the shortcut hint now. Handles "binding declared first,
        # button added later".
        keypressdict = getattr(self.parent, "_keypressdict", None) or {}
        for shortcut, kb in keypressdict.items():
            if not getattr(kb, "on_button", False):
                continue
            if any(af is kb.callback for af in b._action_funcs):
                apply_shortcut_hint(b, shortcut)

        # Style-tag application -- record the tag (so reapply_styles can
        # find it later) and run the styler once. KeyError on miss
        # surfaces typos / unregistered tags at add-time.
        b._style_tag = style_tag
        if style_tag is not None:
            self._resolve_style(style_tag)(b)

    def add_multi(self, *specs: dict) -> List[Any]:
        """Add N buttons side-by-side in a single row.

        Each ``spec`` is a dict of kwargs accepted by :meth:`add` --
        typically ``{text=..., action_func=..., type_=...}``. Returns the
        list of created button objects in spec order.

        On the Qt path, all N buttons are children of a single QWidget
        with a QHBoxLayout inserted into the buttons column's
        QVBoxLayout; the row therefore consumes one vertical slot in the
        sidebar regardless of N. On the mpl fallback path, the row's
        width is divided evenly across N buttons at a shared y, and
        ``_mpl_row_cursor`` is bumped by 1 (not N) so a following
        :meth:`add` lands on the next row, not N rows down.

        Edge cases:
        - ``add_multi()`` with no specs is a no-op and returns ``[]``.
        - ``add_multi(one_spec)`` is equivalent to ``add(**one_spec)`` --
          full-width single button -- and returns a one-element list.
        - Per-spec ``pos=`` is rejected (row layout is the whole point;
          explicit ``pos`` is a single-button-only escape hatch).
        """
        if not specs:
            return []
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise TypeError(
                    f"add_multi: spec {i} is {type(spec).__name__!r}; "
                    f"expected dict of add() kwargs."
                )
            if "pos" in spec and spec["pos"] is not None:
                raise ValueError(
                    f"add_multi: spec {i} carries pos={spec['pos']!r}; "
                    f"per-spec pos= is incompatible with row layout. "
                    f"Use add() for explicit-position single buttons."
                )

        # Single-spec degenerate case: fall through to add() so geometry,
        # Qt-vs-mpl branching, and finalization are byte-identical to the
        # one-button path. Wrap the lone return in a list for API totality.
        if len(specs) == 1:
            return [self.add(**specs[0])]

        # Qt path -- one row widget hosts N buttons.
        from ._qt import make_qt_button_row
        parent_fig = self.parent.figure
        row_buttons = make_qt_button_row(parent_fig, specs)

        if row_buttons is None:
            # mpl fallback -- divide one row's width across N buttons at a
            # shared y. Width / height / outer buf come from spec 0 (matches
            # the convention of "the row is one logical add() call").
            spec0 = specs[0]
            w = spec0.get("w", 0.25)
            h = spec0.get("h", 0.05)
            buf = spec0.get("buf", 0.01)
            mul_factor = 6.4 / parent_fig.get_size_inches()[0]
            btn_w = w * mul_factor
            btn_h = h * mul_factor
            btn_buf = buf
            n = len(specs)
            # Inter-button horizontal gap: reuse btn_buf for a uniform look.
            x_gap = btn_buf
            per_btn_w = (btn_w - (n - 1) * x_gap) / n
            nbtn = self._mpl_row_cursor
            y = (1 - btn_buf) - ((btn_buf + btn_h) * (nbtn + 1))

            row_buttons = []
            for i, spec in enumerate(specs):
                text = spec.get("text", "Button")
                type_ = spec.get("type_", "Push")
                assert type_ in ("Push", "Toggle")
                spec_extra = {
                    k: v for k, v in spec.items()
                    if k not in (
                        "text", "action_func", "pos", "w", "h", "buf",
                        "type_", "style_tag",
                    )
                }
                x = btn_buf + i * (per_btn_w + x_gap)
                pos = (x, y, per_btn_w, btn_h)
                if type_ == "Toggle":
                    b = ToggleButton(plt.axes(pos), text, **spec_extra)
                else:
                    b = Button(plt.axes(pos), text, **spec_extra)
                row_buttons.append(b)

        # Finalize each button (action_func wiring + keybinding-hint reverse
        # scan + per-spec style_tag) and register with the container.
        # One row -> cursor += 1.
        out = []
        for spec, b in zip(specs, row_buttons):
            self._finalize_button(
                b, spec.get("action_func"), style_tag=spec.get("style_tag"),
            )
            super().add(b)
            out.append(b)
        self._mpl_row_cursor += 1
        return out

    def add_separator(
        self, name: Optional[str] = None, style: str = "single",
    ) -> None:
        """Add a visual group boundary between buttons.

        On the Qt path (figure on a Qt canvas), inserts a sunken
        ``QFrame.HLine`` into the buttons-column ``QVBoxLayout`` -- a
        thin line marking a group boundary. ``style="double"`` inserts
        two stacked HLines for a stronger section break, used by
        DUSTrack to mark major button groups in its rc2 sidebar. On
        the mpl path, inserts one (or two, for ``style="double"``)
        invisible buttons that occupy layout slots so subsequent
        buttons are pushed down. (Pre-rc2 the Qt-path inserted a
        ``QToolBar.addSeparator()`` QAction; single-HLine only.)

        Promoted to a first-class API for downstream consumers (DUSTrack)
        that previously hand-rolled invisible spacers by mutating
        :class:`matplotlib.widgets.Button` internals (``.ax``, ``.label``,
        ``.ax.patch``) -- internals that don't exist on the Qt path.

        Args:
            name: Internal name for the mpl-path spacer slot;
                auto-generated if None. Ignored on the Qt path.
            style: ``"single"`` (default) or ``"double"``. See above.
        """
        from ._qt import add_qt_separator
        if add_qt_separator(self.parent.figure, style=style):
            return  # Qt path -- column layout owns the slot

        # mpl fallback: invisible button(s) at the next vertical slot.
        # For style="double" we add two slots so the cumulative spacing
        # matches the Qt-path visual rhythm.
        n_slots = 2 if style == "double" else 1
        for slot_i in range(n_slots):
            slot_name = name if (n_slots == 1) else f"{name}_{slot_i}" if name else None
            if slot_name is None:
                slot_name = f"__separator_{len(self)}"
            nbtn = self._mpl_row_cursor
            parent_fig = self.parent.figure
            mul_factor = 6.4 / parent_fig.get_size_inches()[0]
            btn_w = 0.25 * mul_factor
            btn_h = 0.05 * mul_factor
            btn_buf = 0.01
            pos = (
                btn_buf,
                (1 - btn_buf) - ((btn_buf + btn_h) * (nbtn + 1)),
                btn_w,
                btn_h,
            )
            spacer = Button(plt.axes(pos), slot_name)
            spacer.ax.patch.set_visible(False)
            spacer.label.set_visible(False)
            spacer.ax.axis("off")
            super().add(spacer)
            self._mpl_row_cursor += 1


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
            if self._memtext._overlay is not None:
                self._memtext._overlay.hide()
            elif self._memtext._text is not None:
                self._memtext._text.remove()
        self._memtext = None

    def is_enabled(self) -> bool:
        """Check if memory slots are enabled."""
        return bool(self._list)


class StateVariable:
    """Manage state variables with multiple states.

    The ``widget`` hint is metadata read by the rc2 Qt sidebar to choose
    a control surface for this state variable: ``"label"`` (read-only
    text line; default), ``"dropdown"`` (QComboBox), or ``"toggle"``
    (mutually-exclusive row of checkable QToolButtons). On non-Qt
    backends the hint is ignored and the value renders as plain text
    via the legacy TextView path.
    """

    _ALLOWED_WIDGETS = ("label", "dropdown", "toggle")

    def __init__(self, name: str, states: list, widget: str = "label") -> None:
        assert widget in self._ALLOWED_WIDGETS, (
            f"widget={widget!r} not in {self._ALLOWED_WIDGETS}"
        )
        self.name = name
        self.states = list(states)
        self._current_state_idx = 0
        self.widget = widget
        # 1.4.0rc2: callbacks fire after every set_state / cycle /
        # cycle_back regardless of whether the index actually changed.
        # Covers both keyboard (cycle / set_state via __call__) and Qt
        # dropdown / toggle (set_state via _QtStatevarsWidget on_pick /
        # on_click) entry points -- the two surfaces that mutate state.
        # Direct ``_current_state_idx = ...`` assignments (e.g.
        # ``select_label_with_mouse``) intentionally bypass; callers
        # that do this and need a notification must fire one themselves
        # or follow up with a ``set_state`` on a downstream var.
        self._on_change_callbacks: list[Callable[[], None]] = []

    @property
    def current_state(self) -> Any:
        """Get the current state."""
        return self.states[self._current_state_idx]

    def n_states(self) -> int:
        """Get the number of states."""
        return len(self.states)

    def add_on_change(self, callback: Callable[[], None]) -> None:
        """Register a no-arg callback fired after state mutations.

        Fires on every :meth:`set_state` / :meth:`cycle` /
        :meth:`cycle_back` call, regardless of whether the index
        actually changed -- consumers that need a real-change guard
        should track the previous value themselves (e.g.
        :meth:`VideoPointAnnotator._on_active_label_change`).
        """
        self._on_change_callbacks.append(callback)

    def _notify_change(self) -> None:
        for cb in self._on_change_callbacks:
            cb()

    def cycle(self) -> None:
        """Cycle to the next state."""
        self._current_state_idx = (self._current_state_idx + 1) % self.n_states()
        self._notify_change()

    def cycle_back(self) -> None:
        """Cycle to the previous state."""
        self._current_state_idx = (self._current_state_idx - 1) % self.n_states()
        self._notify_change()

    def set_state(self, state: Union[int, str]) -> None:
        """Set the state."""
        if isinstance(state, int):
            assert 0 <= state < self.n_states()
            self._current_state_idx = state
        if isinstance(state, str):
            assert state in self.states
            self._current_state_idx = self.states.index(state)
        self._notify_change()


class StateVariables(AssetContainer):
    """Manager for state variables."""

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self._text = None

    def asdict(self) -> dict:
        """Return state variables as a dictionary."""
        return {x.name: x.states for x in self._list}

    def add(self, name: str, states: list, widget: str = "label") -> StateVariable:
        """Add a state variable to the container.

        Args:
            name: identifier; must be unique within the container.
            states: the rotation of values this variable can take.
            widget: rc2 Qt-sidebar control surface hint -- one of
                ``"label"`` (read-only text; default; matches pre-rc2
                behavior), ``"dropdown"`` (QComboBox), or ``"toggle"``
                (mutually-exclusive QToolButton row). Ignored on non-Qt
                backends.
        """
        assert name not in self.names
        return super().add(StateVariable(name, states, widget=widget))

    def _get_display_text(self) -> List[str]:
        """Get the display text for state variables."""
        return ["State variables:"] + [
            f"{x.name} - {x.current_state}" for x in self._list
        ]

    def show(self, pos: str = "bottom right", fax=None) -> None:
        """Show state variables.

        rc2: tries the interactive Qt widget first (dropdowns / toggles
        / labels per each :class:`StateVariable`'s ``widget`` hint),
        mounted in the QDockWidget column beneath the buttons. Falls
        back to the pre-rc2 :class:`utils.TextView` path on non-Qt
        backends or if no Qt window is found. The ``pos`` / ``fax``
        arguments are only consulted on the TextView fallback; on the
        Qt path they're ignored (the widget's position is dock-managed).
        """
        from ._qt import make_qt_statevars_widget
        widget = make_qt_statevars_widget(self.parent.figure, self)
        if widget is not None:
            self._text = widget
            return
        if fax is None:
            fax = self.parent.figure
        self._text = TextView(self._get_display_text(), fax=fax, pos=pos)

    def update_display(self, draw: bool = True) -> None:
        """Update the display of state variables."""
        self._text.update(self._get_display_text())
        if draw:
            plt.draw()
