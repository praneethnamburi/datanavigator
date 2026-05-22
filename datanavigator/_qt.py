"""
Qt-binding-touching internals.

Imports here are deliberately lazy / behind a string check so that
``import datanavigator`` works on machines that have no Qt binding
installed -- 1.4.0 is a *soft* Qt requirement: when matplotlib is on a
Qt backend, Qt features activate; on Agg or other backends, datanavigator
falls back to its pre-1.4 matplotlib-native rendering.

Phase 1 contributes :func:`find_qt_window`; Phase 2 adds
:class:`QtTextOverlay` (a QLabel parented to the matplotlib canvas,
used by :class:`utils.TextView` when running under QtAgg). Phase 3
adds :func:`make_qt_button` -- a QPushButton replacement for the
matplotlib-widgets-based Button stack used by :class:`assets.Buttons`.
Pre-rc2 those buttons lived in a ``QToolBar`` on ``LeftToolBarArea``;
rc2 moves them into a ``QDockWidget``-hosted ``QVBoxLayout`` so a
sibling state-variables widget can stack beneath them in the same
column (see :func:`_get_buttons_widget`).
"""

from __future__ import annotations

from typing import List, Optional, Tuple


def find_qt_window(figure) -> Optional["QMainWindow"]:  # noqa: F821 (forward ref)
    """Return the QMainWindow hosting ``figure`` on a Qt backend, else None.

    matplotlib's ``QtAgg`` backend builds a ``QMainWindow`` (its
    ``FigureManagerQT``) around every figure it creates; the canvas is the
    central widget, and ``figure.canvas.manager.window`` is the
    ``QMainWindow``. On non-Qt backends (Agg, TkAgg, %matplotlib inline,
    ...) those attributes either don't exist or aren't a QMainWindow,
    and this function returns ``None``.

    The class-name string check is intentional: it avoids importing qtpy
    (and therefore failing) on machines that have no Qt binding installed,
    while still correctly identifying every matplotlib Qt canvas
    (FigureCanvasQTAgg, FigureCanvasQT, FigureCanvasQTCairo, ...).
    """
    canvas = getattr(figure, "canvas", None)
    if canvas is None:
        return None
    if "qt" not in type(canvas).__name__.lower():
        return None
    manager = getattr(canvas, "manager", None)
    if manager is None:
        return None
    return getattr(manager, "window", None)


def _make_qt_text_overlay_class():
    """Build :class:`QtTextOverlay` lazily so importing this module never
    touches qtpy (and therefore never fails on a no-Qt-binding machine).

    The class inherits from ``QObject`` so it can install itself as an
    event filter on the canvas to catch resize events; the QLabel itself
    is parented to the canvas, so it's destroyed automatically when the
    canvas (and figure) go away.
    """
    from qtpy.QtCore import QEvent, QObject, Qt
    from qtpy.QtWidgets import QLabel

    class QtTextOverlay(QObject):
        """QLabel overlay on a matplotlib Qt canvas, positioned mpl-style.

        ``pos`` is the 4-tuple ``(x_frac, y_frac, va, ha)`` produced by
        :func:`datanavigator.utils._parse_pos` -- the same convention
        matplotlib's ``Axes.text`` uses, so callers don't need to know a
        Qt path exists.
        """

        # Translucent white box, monospace, small padding -- mimics today's
        # mpl text overlay closely enough that the swap is visually quiet.
        _STYLE = (
            "QLabel { "
            "color: black; "
            "background-color: rgba(255, 255, 255, 200); "
            "padding: 2px 4px; "
            "}"
        )

        def __init__(self, canvas, pos: Tuple[float, float, str, str],
                     text_lines: List[str]):
            # Parent the QObject to the canvas so it's auto-cleaned up
            # when the canvas is destroyed.
            super().__init__(canvas)
            self._canvas = canvas
            self._pos = pos

            self._label = QLabel(canvas)
            self._label.setStyleSheet(self._STYLE)
            self._label.setTextFormat(Qt.PlainText)

            font = self._label.font()
            font.setFamily("Courier New")
            self._label.setFont(font)

            # Watch the canvas for Resize so we can reposition on window
            # resize. Without this the label would stick to a fixed
            # pixel coordinate while the canvas grew/shrank around it.
            canvas.installEventFilter(self)

            self.update(text_lines)

        def update(self, text_lines: List[str]) -> None:
            """Replace the displayed text and reposition."""
            self._label.setText("\n".join(text_lines))
            self._label.adjustSize()
            self._reposition()
            self._label.raise_()  # keep above sibling widgets
            self._label.show()

        def hide(self) -> None:
            """Hide and dispose of the overlay (irreversible)."""
            self._canvas.removeEventFilter(self)
            self._label.hide()
            self._label.deleteLater()

        def _reposition(self) -> None:
            x_frac, y_frac, va, ha = self._pos
            cw, ch = self._canvas.width(), self._canvas.height()
            lw, lh = self._label.width(), self._label.height()

            # Horizontal: ha is matplotlib's text alignment ('left' / 'right'
            # / 'center'). For a QLabel we want to compute the top-left
            # corner that places the label so its `ha` edge sits at x_frac*cw.
            if ha == "left":
                x = int(x_frac * cw)
            elif ha == "right":
                x = int(x_frac * cw - lw)
            else:  # "center"
                x = int(x_frac * cw - lw / 2)

            # Vertical: mpl's y axis runs bottom-to-top (y=0 bottom, y=1 top);
            # Qt's runs top-to-bottom. Convert, then adjust by va.
            y_top_origin = (1 - y_frac) * ch
            if va == "bottom":
                y = int(y_top_origin - lh)
            elif va == "top":
                y = int(y_top_origin)
            else:  # "center"
                y = int(y_top_origin - lh / 2)

            self._label.move(x, y)

        def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
            if obj is self._canvas and event.type() == QEvent.Resize:
                self._reposition()
            return False  # don't consume; let the event keep propagating

    return QtTextOverlay


def make_text_overlay(figure, pos: Tuple[float, float, str, str],
                      text_lines: List[str]):
    """Build a Qt overlay over ``figure``'s canvas, or return None on non-Qt.

    Lazy import of qtpy: returns None (without raising) if Qt isn't
    available, so callers can keep their mpl fallback inline.
    """
    canvas = getattr(figure, "canvas", None)
    if canvas is None or "qt" not in type(canvas).__name__.lower():
        return None
    try:
        cls = _make_qt_text_overlay_class()
    except ImportError:
        return None
    return cls(canvas, pos, text_lines)


# ---------------------------------------------------------------------------
# Phase 3 -- Qt buttons stacked in a QVBoxLayout inside a QDockWidget on
# the QMainWindow's LeftDockWidgetArea.
#
# Pre-rc2 (Phase 3 original): buttons lived in a QToolBar. rc2 swaps
# the host to a QDockWidget+QVBoxLayout so the same column can hold
# the rc2 state-variables widget (dropdowns / toggles) beneath the
# buttons -- QToolBar's auto-layout doesn't compose with heterogeneous
# children cleanly. Public surface (Buttons.add / add_separator) is
# unchanged.
# ---------------------------------------------------------------------------


def _accepts_event_arg(func) -> bool:
    """True if ``func`` can be called with one positional argument.

    Today's :class:`matplotlib.widgets.Button` passes a ``MouseEvent`` to
    every registered callback. Qt's ``QPushButton.clicked`` signal emits
    a bool. To preserve "transition unnoticeable to consumers", we look
    at ``func``'s signature: if it can take one positional, we pass
    ``None`` (mpl-style event-like, all known datanavigator callbacks
    only ever use ``event=None`` defaults or ignore it). If it can't,
    we call it with no args (``lambda: None`` shape, used in tests).
    """
    import inspect
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        # Some C-implemented callables don't expose a signature; assume
        # they accept an event (mpl-compatible default).
        return True
    for p in sig.parameters.values():
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY,
                      inspect.Parameter.VAR_POSITIONAL):
            return True
    return False


def _make_qt_button_classes():
    """Lazy: build the Qt button wrapper classes (qtpy import deferred)."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QPushButton

    class _QtPushButton:
        """Public-API match for :class:`assets.Button` (mpl path).

        Attributes:
            name: button label / identity (used by AssetContainer.__getitem__).
            _qt_btn: the underlying QPushButton (Phase 3+ internals).
        """

        def __init__(self, container, name: str, **_kwargs):
            self.name = name
            self._qt_btn = QPushButton(name)
            # Don't steal keyboard focus when clicked. The matplotlib
            # canvas owns the keyboard for "hover over plot, press a key"
            # workflows (mpl key_press_event with .xdata / .ydata
            # populated from the last cursor position). With Qt's default
            # StrongFocus on the button, the FIRST click moves focus to
            # the button, and every subsequent keypress routes there
            # instead of the canvas -- silently breaking every mpl key
            # binding the consumer set up via add_key_binding(). NoFocus
            # keeps mouse clicks working while leaving the canvas's
            # focus undisturbed.
            self._qt_btn.setFocusPolicy(Qt.NoFocus)
            # No trailing stretch in the buttons sub-widget (the outer
            # left-column layout owns the stretch), so plain addWidget
            # produces a top-down stack.
            container.layout().addWidget(self._qt_btn)

        def on_clicked(self, action_func) -> None:
            """Register a callback. See module-level :func:`_accepts_event_arg`."""
            takes_event = _accepts_event_arg(action_func)

            def slot(*_args):
                if takes_event:
                    action_func(None)
                else:
                    action_func()

            self._qt_btn.clicked.connect(slot)

    class _QtToggleButton(_QtPushButton):
        """Public-API match for :class:`assets.ToggleButton` (mpl path)."""

        def __init__(self, container, name: str, start_state: bool = True, **_kwargs):
            super().__init__(container, name)
            self._qt_btn.setCheckable(True)
            self._state = bool(start_state)
            self._qt_btn.setChecked(self._state)
            # User clicks -> Qt toggles -> our state -> label refresh
            self._qt_btn.toggled.connect(self._on_toggled)
            self.set_text()

        @property
        def state(self) -> bool:
            return self._state

        @state.setter
        def state(self, value: bool) -> None:
            self._state = bool(value)
            # blockSignals avoids re-entering _on_toggled while we mutate
            self._qt_btn.blockSignals(True)
            self._qt_btn.setChecked(self._state)
            self._qt_btn.blockSignals(False)
            self.set_text()

        def _on_toggled(self, checked: bool) -> None:
            self._state = bool(checked)
            self.set_text()

        def set_text(self) -> None:
            self._qt_btn.setText(f"{self.name}={self._state}")

        def toggle(self, event=None) -> None:  # event kept for mpl parity
            self.state = not self._state

        def set_state(self, state: bool) -> None:
            assert isinstance(state, bool)
            self.state = state

    return _QtPushButton, _QtToggleButton


#: Minimum width (px) of the left-column dock host. Sized a touch
#: above the pre-rc2 fast_render ``sidebar_width=280`` default so the
#: column is unconditionally wide enough to show full DLC layer names
#: (e.g. ``dlc_iteration-3_250000``) in the statevars dropdowns, and
#: so combos that hold *short* values (e.g. ``select`` / ``place``)
#: don't end "somewhere in the middle of the sidebar". Tuned to 300
#: empirically 2026-05-18 during DUSTrack 1.1.0rc2 testing.
_LEFT_COLUMN_MIN_WIDTH = 300


def _get_left_column(qt_window):
    """Build (or return the cached) two-section left column on ``qt_window``.

    Layout, top-to-bottom in the host widget's outer QVBoxLayout:

      [ buttons_widget (QWidget + inner QVBoxLayout)   ]  -- buttons land here
      [ statevars_slot (initially empty)               ]  -- statevars widget
      [ addStretch(1)                                  ]  -- bottom filler

    The two sections are separate QWidgets so a button added AFTER
    ``statevariables.show()`` still lands above the statevars panel
    (motivating example: VideoPointAnnotator adds its "Refresh UI"
    button in ``set_key_bindings``, which runs after the statevars
    setup).

    Width contract: ``host`` carries a minimum width of
    :data:`_LEFT_COLUMN_MIN_WIDTH` so the column doesn't shrink to fit
    short button labels / short statevar values. Combined with the
    statevars widget's ``Preferred`` (not ``Fixed``) horizontal size
    policy, this means each statevar control fills the full column
    width regardless of its current content.

    Cached on the QMainWindow as ``_dnav_left_column``. The legacy
    ``_dnav_buttons_widget`` attribute still points to the buttons
    sub-widget so pre-existing code paths (and the rc2 smoke test in
    ``tests/qt_learning/07_phase4a_smoke.py``) keep working.
    """
    col = getattr(qt_window, "_dnav_left_column", None)
    if col is not None:
        return col
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QDockWidget, QSizePolicy, QVBoxLayout, QWidget,
    )
    dock = QDockWidget("datanavigator", qt_window)
    dock.setTitleBarWidget(QWidget())
    dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

    host = QWidget(dock)
    outer = QVBoxLayout(host)
    outer.setContentsMargins(4, 4, 4, 4)
    outer.setSpacing(8)

    buttons_widget = QWidget(host)
    buttons_layout = QVBoxLayout(buttons_widget)
    buttons_layout.setContentsMargins(0, 0, 0, 0)
    buttons_layout.setSpacing(4)
    outer.addWidget(buttons_widget)

    # statevars slot index is captured at construction time; the
    # widget itself is inserted lazily by make_qt_statevars_widget.
    statevars_slot_index = outer.count()  # currently 1
    outer.addStretch(1)

    host.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
    host.setMinimumWidth(_LEFT_COLUMN_MIN_WIDTH)
    dock.setWidget(host)
    qt_window.addDockWidget(Qt.LeftDockWidgetArea, dock)

    col = _LeftColumn(
        dock=dock, host=host, outer_layout=outer,
        buttons_widget=buttons_widget,
        statevars_slot_index=statevars_slot_index,
    )
    qt_window._dnav_left_column = col
    qt_window._dnav_buttons_widget = buttons_widget  # legacy alias
    qt_window._dnav_buttons_dock = dock              # legacy alias
    return col


class _LeftColumn:
    """Plain-struct holder for left-column references.

    Defined at module scope (not nested inside _get_left_column) so the
    instance survives across calls and its attribute names show up in
    debuggers / repr without depending on a closure.
    """
    __slots__ = (
        "dock", "host", "outer_layout",
        "buttons_widget", "statevars_widget", "statevars_slot_index",
    )

    def __init__(self, *, dock, host, outer_layout,
                 buttons_widget, statevars_slot_index):
        self.dock = dock
        self.host = host
        self.outer_layout = outer_layout
        self.buttons_widget = buttons_widget
        self.statevars_widget = None
        self.statevars_slot_index = statevars_slot_index


def _get_buttons_widget(qt_window):
    """Return the cached QWidget hosting datanavigator buttons.

    Pre-rc2 (the QToolBar era) this returned a ``QToolBar``; Commit 1
    of rc2 swapped to a ``QWidget`` + ``QVBoxLayout``; Commit 2 makes
    it a sub-widget of a two-section left column. The public contract
    is unchanged: buttons added via :class:`_QtPushButton` land in the
    widget's layout.
    """
    return _get_left_column(qt_window).buttons_widget


def _make_qt_separator_widget(parent, style: str = "single"):
    """Build a separator widget (single or double HLine) for the Qt path.

    ``style="single"``: a sunken ``QFrame.HLine`` (matches the pre-rc2
    visual one-for-one).
    ``style="double"``: a ``QWidget`` containing two sunken
    ``QFrame.HLine``\\ s stacked vertically with a small gap, used to
    mark a stronger section boundary (e.g. between buttons that belong
    to different functional groups in DUSTrack's sidebar).

    The double-separator returns a single ``QWidget`` so callers can
    add it to a layout the same way they add a single ``QFrame`` --
    no double-insertion bookkeeping.
    """
    from qtpy.QtWidgets import QFrame, QVBoxLayout, QWidget
    if style == "single":
        sep = QFrame(parent)
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        return sep
    if style == "double":
        host = QWidget(parent)
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 2, 0, 2)
        lay.setSpacing(3)
        for _ in range(2):
            line = QFrame(host)
            line.setFrameShape(QFrame.HLine)
            line.setFrameShadow(QFrame.Sunken)
            lay.addWidget(line)
        return host
    raise ValueError(
        f"style={style!r} not in ('single', 'double')"
    )


def add_qt_separator(figure, style: str = "single") -> bool:
    """Add a separator to the buttons column on ``figure``'s window.

    ``style="single"`` (default) inserts a single sunken ``QFrame.HLine``
    -- visual continuity with the pre-rc2 ``QToolBar.addSeparator()``
    behavior. ``style="double"`` inserts two stacked HLines with a small
    gap (via :func:`_make_qt_separator_widget`), used for a stronger
    visual break between groups of buttons.

    Returns ``True`` if the separator was added (Qt path active), or
    ``False`` if ``figure`` is not on a Qt canvas so the caller should
    fall back to its mpl-side spacer hack. Lazy-creates the buttons
    widget if it doesn't exist yet (matches :func:`make_qt_button`'s
    caching).
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return False
    try:
        container = _get_buttons_widget(qt_window)
    except ImportError:
        return False
    sep = _make_qt_separator_widget(container, style=style)
    container.layout().addWidget(sep)
    return True


# ---------------------------------------------------------------------------
# rc2 -- Qt-native state-variables widget (dropdowns / toggles / labels).
# ---------------------------------------------------------------------------
#
# Pre-rc2 statevariables rendered via :class:`utils.TextView` -- a QLabel
# overlay on the canvas (Phase 2 Qt path) or an ``Axes.text`` artist
# (mpl fallback). Both were read-only. rc2 promotes them to interactive
# controls in the same column as the buttons: each StateVariable's
# ``widget`` hint picks QLabel / QComboBox / QButtonGroup-of-QToolButtons.
#
# On any user interaction the widget calls ``state.set_state(value)``
# and triggers ``statevars_container.parent.update()`` -- the same
# generic redraw every keybind handler already invokes after a
# ``cycle()``. No new callback API is exposed to consumers; the only
# new surface is the ``widget=`` kwarg on ``StateVariables.add()``.


def _make_qt_statevars_widget_class():
    """Build :class:`_QtStatevarsWidget` lazily so importing this module
    never touches qtpy on a no-Qt-binding machine."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QButtonGroup, QComboBox, QFrame, QHBoxLayout, QLabel,
        QSizePolicy, QToolButton, QVBoxLayout, QWidget,
    )

    class _QtStatevarsWidget(QWidget):
        """rc2 interactive sidebar widget for state variables.

        Builds one row per state variable, picking the control type from
        :attr:`StateVariable.widget`:

        - ``"label"``    -- QLabel showing ``"<name>: <value>"`` (read-only;
          parity with the pre-rc2 text path).
        - ``"dropdown"`` -- QLabel name *above* a QComboBox (vertical
          stack so long state names don't get squeezed and the combo
          gets the full column width).
        - ``"toggle"``   -- QLabel name *above* a horizontal QButtonGroup
          of mutually-exclusive checkable QToolButtons.

        Rows are separated by a sunken ``QFrame.HLine`` for visual
        grouping.

        Duck-typed compatibility with the pre-rc2 ``statevariables._text``
        slot:

        - ``.update(text=None)`` -- ignored arg; resyncs each control
          from its model. Called by ``StateVariables.update_display()``.
        - ``._pos`` -- settable no-op (DUSTrack used to write this to
          move the canvas-overlay TextView; meaningless once the
          controls live in a layout-managed column, but writes are
          harmless).
        - ``.text`` -- last-rendered "name: value" strings, surfaced
          for parity tests that snapshot the prior text path.
        """

        def __init__(self, statevars_container, parent=None):
            super().__init__(parent)
            # Don't steal keyboard focus -- the matplotlib canvas owns
            # focus for hover-and-press bindings (same reason as
            # _QtPushButton).
            self.setFocusPolicy(Qt.NoFocus)

            # Slightly darker background than the parent dock so the
            # statevars section reads as a visually distinct group from
            # the buttons column above it. Theme-adaptive: derive the
            # tint from the current palette rather than hard-coding an
            # RGB so the widget still looks reasonable under both light
            # and dark themes (DUSTrack runs dark by default; other
            # consumers don't).
            self.setAutoFillBackground(True)
            pal = self.palette()
            base = pal.color(self.backgroundRole())
            pal.setColor(self.backgroundRole(), base.darker(120))
            self.setPalette(pal)

            self._container = statevars_container
            self._layout = QVBoxLayout(self)
            self._layout.setContentsMargins(4, 4, 4, 4)
            self._layout.setSpacing(4)

            # Per-statevar control rows. _sync maps name -> sync_fn(state)
            # that resyncs the control to state.current_state. Rows are
            # joined by sunken HLine separators for visual grouping so
            # the eye doesn't have to parse where one state variable
            # ends and the next begins -- particularly useful once the
            # controls stack (name-above-control), which makes adjacent
            # rows visually similar.
            self._sync = {}
            n = len(self._container._list)
            for i, state in enumerate(self._container._list):
                row, sync_fn = self._build_row(state)
                self._layout.addWidget(row)
                self._sync[state.name] = sync_fn
                if i < n - 1:
                    sep = QFrame(self)
                    sep.setFrameShape(QFrame.HLine)
                    sep.setFrameShadow(QFrame.Sunken)
                    self._layout.addWidget(sep)

            # Trailing double separator marks the visual end of the
            # state-variables section. DUSTrack asked for an
            # "after state variables" group boundary 2026-05-18; the
            # double separator matches the style the buttons column
            # uses for major group breaks (see
            # :meth:`assets.Buttons.add_separator` ``style="double"``).
            if n > 0:
                trailing = _make_qt_separator_widget(self, style="double")
                self._layout.addWidget(trailing)

            # Horizontal: ``Preferred`` (not ``Fixed``) so the statevars
            # widget fills the column when the column's effective width
            # is wider than the widget's own sizeHint -- which is the
            # common case once _LEFT_COLUMN_MIN_WIDTH bumps the column
            # past the widest combo content. Combined with each combo's
            # ``Expanding`` horizontal policy, every dropdown then
            # reaches the column edge regardless of how short its
            # current state value is.
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

            # Pre-rc2 text-sink protocol parity (see class docstring).
            self.text = []
            self._pos = None

        def _build_row(self, state):
            # rc2 polish: stack name above control (QVBoxLayout) instead
            # of side-by-side. Side-by-side starved the combo of width
            # so long state-values (e.g. ``dlc_iteration-3_250000``)
            # truncated to ``dlc_...250000``. Stacked, the combo gets
            # the full column width and shows the whole name.
            kind = getattr(state, "widget", "label")
            row = QWidget(self)
            v = QVBoxLayout(row)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(2)

            if kind == "label":
                lbl = QLabel(_fmt_label(state), row)
                lbl.setWordWrap(True)
                v.addWidget(lbl)

                def sync(s, lbl=lbl):
                    lbl.setText(_fmt_label(s))

                return row, sync

            name_lbl = QLabel(f"{state.name}:", row)
            v.addWidget(name_lbl)

            if kind == "dropdown":
                combo = QComboBox(row)
                combo.setFocusPolicy(Qt.NoFocus)
                # Let the combo grow to fit the widest item so long
                # state values aren't truncated in the closed state.
                combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
                combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                _populate_combo(combo, state)
                v.addWidget(combo)

                container = self._container

                def on_pick(idx, s=state, c=combo):
                    if idx < 0 or idx == s._current_state_idx:
                        return
                    s.set_state(int(idx))
                    container.parent.update()

                combo.currentIndexChanged.connect(on_pick)

                def sync(s, c=combo):
                    # States list might have been mutated externally
                    # (e.g. add_annotation_layers extends the layer
                    # rotation). Re-populate rather than just moving
                    # the current index.
                    _populate_combo(c, s)

                return row, sync

            if kind == "toggle":
                bgroup_w = QWidget(row)
                bgroup_h = QHBoxLayout(bgroup_w)
                bgroup_h.setContentsMargins(0, 0, 0, 0)
                bgroup_h.setSpacing(2)
                group = QButtonGroup(bgroup_w)
                group.setExclusive(True)
                buttons = []
                for i, val in enumerate(state.states):
                    btn = QToolButton(bgroup_w)
                    btn.setText(str(val))
                    btn.setCheckable(True)
                    btn.setFocusPolicy(Qt.NoFocus)
                    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                    group.addButton(btn, i)
                    bgroup_h.addWidget(btn)
                    buttons.append(btn)
                if 0 <= state._current_state_idx < len(buttons):
                    buttons[state._current_state_idx].setChecked(True)

                container = self._container

                def on_click(idx, s=state):
                    if idx < 0 or idx == s._current_state_idx:
                        return
                    s.set_state(int(idx))
                    container.parent.update()

                group.idClicked.connect(on_click)

                v.addWidget(bgroup_w)

                def sync(s, btns=buttons):
                    # blockSignals so the programmatic setChecked
                    # doesn't fire idClicked -> recursive update.
                    for i, b in enumerate(btns):
                        b.blockSignals(True)
                        b.setChecked(i == s._current_state_idx)
                        b.blockSignals(False)

                return row, sync

            # Unknown widget kind: fall through to label.
            lbl = QLabel(_fmt_label(state), row)
            v.addWidget(lbl)
            return row, (lambda s, lbl=lbl: lbl.setText(_fmt_label(s)))

        def update(self, text=None) -> None:
            """Duck-typed match for :meth:`utils.TextView.update`.

            The ``text`` positional is ignored: each control re-reads
            its value from the StateVariable model. We also refresh the
            ``self.text`` snapshot so parity tests that compare
            statevars._text.text across the pre-rc2 / rc2 paths keep
            working.
            """
            for state in self._container._list:
                sync_fn = self._sync.get(state.name)
                if sync_fn is not None:
                    sync_fn(state)
            self.text = [_fmt_label(s) for s in self._container._list]

        def hide(self) -> None:
            super().hide()

    def _fmt_label(state) -> str:
        return f"{state.name}: {state.current_state}"

    def _populate_combo(combo, state) -> None:
        combo.blockSignals(True)
        combo.clear()
        for val in state.states:
            combo.addItem(str(val))
        if 0 <= state._current_state_idx < combo.count():
            combo.setCurrentIndex(state._current_state_idx)
        combo.blockSignals(False)

    return _QtStatevarsWidget


def make_qt_statevars_widget(figure, statevars_container):
    """Build a Qt-native statevars widget and mount under the buttons column.

    Returns ``None`` if ``figure`` isn't on a Qt canvas (caller falls
    back to the pre-rc2 :class:`utils.TextView` path). Inserts the
    widget into the left-column dock's statevars slot, replacing any
    previously-mounted widget at that index.
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        cls = _make_qt_statevars_widget_class()
    except ImportError:
        return None
    col = _get_left_column(qt_window)
    widget = cls(statevars_container, parent=col.host)

    # If a previous statevars widget was mounted (re-show, or test
    # re-init), drop it before installing the replacement.
    if col.statevars_widget is not None:
        col.outer_layout.removeWidget(col.statevars_widget)
        col.statevars_widget.setParent(None)
        col.statevars_widget.deleteLater()

    col.outer_layout.insertWidget(col.statevars_slot_index, widget)
    col.statevars_widget = widget
    # Snapshot once so consumers can observe the rendered labels.
    widget.update()
    return widget


def make_qt_button(figure, name: str, type_: str = "Push", start_state: bool = True):
    """Build a Qt-backed Button or ToggleButton if ``figure`` is on a Qt canvas.

    Returns ``None`` if no Qt window is found (caller falls back to mpl).
    On the Qt path, returns an object with the same public surface as
    :class:`assets.Button` / :class:`assets.ToggleButton` (``name``,
    ``on_clicked``, and for toggles ``state`` / ``toggle`` / ``set_state``
    / ``set_text``).
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        push_cls, toggle_cls = _make_qt_button_classes()
    except ImportError:
        return None
    container = _get_buttons_widget(qt_window)
    if type_ == "Toggle":
        return toggle_cls(container, name, start_state=start_state)
    return push_cls(container, name)


def make_qt_button_row(figure, specs):
    """Build N Qt buttons side-by-side in one row of the buttons column.

    Counterpart to :func:`make_qt_button` for :meth:`assets.Buttons.add_multi`.
    Creates a child ``QWidget`` + ``QHBoxLayout`` (one row), adds it to
    the column's ``QVBoxLayout`` -- so the row occupies a single
    vertical slot regardless of how many buttons it carries -- then
    instantiates one ``_QtPushButton`` / ``_QtToggleButton`` per spec
    with that row widget as the container. Each wrapper's ``__init__``
    appends its underlying ``QPushButton`` to the row's QHBoxLayout
    (the same ``container.layout().addWidget`` line that makes the
    single-button path work, just with a horizontal layout this time).

    Returns the list of wrapper objects (in spec order), or ``None`` if
    no Qt window / qtpy binding is available so the caller can fall
    back to mpl placement.

    Each spec is a dict of kwargs from :meth:`assets.Buttons.add_multi`;
    we read ``text`` / ``type_`` / ``start_state`` here and ignore the
    rest (the caller's ``_finalize_button`` does action wiring).
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        push_cls, toggle_cls = _make_qt_button_classes()
        from qtpy.QtWidgets import QHBoxLayout, QWidget
    except ImportError:
        return None
    column = _get_buttons_widget(qt_window)
    row_widget = QWidget(column)
    row_layout = QHBoxLayout(row_widget)
    # Match the column's inner-row rhythm: zero outer margins, a small
    # gap between buttons. spacing=4 mirrors the column's inter-row
    # spacing set in _get_left_column (outer.setSpacing(8) is between
    # *sections*; the buttons sub-layout uses 4 -- see _qt.py:353).
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.setSpacing(4)
    column.layout().addWidget(row_widget)

    out = []
    for spec in specs:
        text = spec.get("text", "Button")
        type_ = spec.get("type_", "Push")
        if type_ == "Toggle":
            start_state = spec.get("start_state", True)
            out.append(toggle_cls(row_widget, text, start_state=bool(start_state)))
        else:
            out.append(push_cls(row_widget, text))
    return out


# ---------------------------------------------------------------------------
# 1.5.0 fast_render -- Qt-native video + scatter pane.
# ---------------------------------------------------------------------------
#
# Probe 13 (BENCHMARKING.md / tests/qt_learning/13_*) showed that the
# ~82 ms / frame canvas pixmap upload that dominated 1.4.0-qt's update
# cost can be bypassed by rendering the decoded frame Qt-side
# (QGraphicsView + QPixmapItem) and moving the annotation scatter to a
# QGraphicsItemGroup. matplotlib then only has to rasterize the trace
# axes -- a much smaller widget area.
#
# Three components live below:
#   _QtImagePane       -- video pane (pixmap item + marker group + title)
#   _QtScatterArtist   -- mpl PathCollection facade over a marker group
#   _QtPickAdapter     -- mouse-event filter that fires synthetic
#                         mpl-shaped pick / button_press events
#
# Each is wrapped in a lazy class-factory so importing datanavigator on
# a machine with no Qt binding never touches qtpy (same pattern as
# QtTextOverlay above).


def _ndarray_to_qpixmap(arr):
    """Build a QPixmap from an H x W x 3 uint8 ndarray.

    The QImage holds a view of ``arr``'s buffer, so the caller must keep
    ``arr`` alive until ``QPixmap.fromImage`` finishes -- it copies
    synchronously, so a single-call site (set_image) is safe.
    """
    from qtpy.QtGui import QImage, QPixmap
    h, w, _ = arr.shape
    img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img)


def _as_rgb_uint8(arr):
    """Coerce a decoded frame ndarray into contiguous H x W x 3 uint8."""
    import numpy as np
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.shape[2] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr)


def _make_qt_image_pane_class():
    """Build :class:`_QtImagePane` lazily so importing this module never
    touches qtpy on a no-Qt-binding machine.

    The pane is a QWidget holding a title QLabel above a QGraphicsView.
    The view's scene contains exactly one QGraphicsPixmapItem (the
    video frame) plus zero-or-more annotation marker groups added via
    :meth:`_QtImagePane.add_marker_group`. Image-pixel coordinates are
    used throughout -- QGraphicsView's y-down orientation matches the
    matplotlib ``imshow(origin='upper')`` convention we're replacing,
    so no transform inversion is needed.
    """
    from qtpy.QtCore import QRectF, Qt
    from qtpy.QtGui import QBrush, QColor, QPainter, QTransform
    from qtpy.QtWidgets import (
        QFrame,
        QGraphicsItemGroup,
        QGraphicsPixmapItem,
        QGraphicsScene,
        QGraphicsView,
        QLabel,
        QVBoxLayout,
        QWidget,
    )

    class _PanZoomGraphicsView(QGraphicsView):
        """QGraphicsView with built-in wheel-zoom + middle-button pan/reset.

        Zoom is anchored under the cursor (matches every modern image
        viewer). Middle-mouse *drag* pans the scene; middle-mouse
        *click* (no drag, < 4 px movement total) resets the transform
        and re-fits the scene rect. Left and right mouse buttons pass
        through unchanged so :class:`_QtPickAdapter`'s pick / place
        callbacks keep firing for annotation work.

        ``user_adjusted`` flips True on wheel-zoom or a real pan drag
        (not on the initial auto-fit, which leaves a scale transform
        but is not a user adjustment). :class:`_QtImagePane.resizeEvent`
        keys off this flag to decide whether to re-fit on window resize.
        """

        _ZOOM_STEP = 1.25
        _CLICK_VS_DRAG_PX = 4

        def __init__(self, scene, image_pane):
            super().__init__(scene, image_pane)
            self._image_pane = image_pane
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
            self._pan_anchor = None
            self._pan_total = 0
            self.user_adjusted = False

        def wheelEvent(self, event):  # noqa: N802 (Qt naming)
            angle = event.angleDelta().y()
            if angle == 0:
                event.ignore()
                return
            factor = self._ZOOM_STEP if angle > 0 else 1.0 / self._ZOOM_STEP
            self.scale(factor, factor)
            self.user_adjusted = True
            event.accept()

        def mousePressEvent(self, event):  # noqa: N802 (Qt naming)
            if event.button() == Qt.MiddleButton:
                self._pan_anchor = event.pos()
                self._pan_total = 0
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event):  # noqa: N802 (Qt naming)
            if self._pan_anchor is not None:
                delta = event.pos() - self._pan_anchor
                self._pan_anchor = event.pos()
                self._pan_total += abs(delta.x()) + abs(delta.y())
                hbar = self.horizontalScrollBar()
                vbar = self.verticalScrollBar()
                hbar.setValue(hbar.value() - delta.x())
                vbar.setValue(vbar.value() - delta.y())
                event.accept()
                return
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event):  # noqa: N802 (Qt naming)
            if event.button() == Qt.MiddleButton and self._pan_anchor is not None:
                self.unsetCursor()
                self._pan_anchor = None
                if self._pan_total < self._CLICK_VS_DRAG_PX:
                    # Click, not drag -- reset image zoom.
                    self._image_pane.reset_view()
                else:
                    # Real drag -- user moved the view, stop auto-re-fit
                    # on window resize.
                    self.user_adjusted = True
                event.accept()
                return
            super().mouseReleaseEvent(event)

    class _QtImagePane(QWidget):
        """Qt-native replacement for ``ax.imshow`` + ``ax.scatter`` overlays.

        Per-frame contract:
        - :meth:`set_image` swaps the QPixmap on the QGraphicsPixmapItem
          (one Qt-side copy; no Agg raster)
        - :meth:`set_title` updates the title QLabel
        - Annotation marker groups update themselves via
          :class:`_QtScatterArtist` -- the pane does not push offsets.
        """

        def __init__(self, parent=None, picker_radius: float = 5.0):
            super().__init__(parent)
            self._picker_radius = float(picker_radius)
            # Don't steal focus on mouse press -- the matplotlib canvas
            # owns keyboard focus for hover-key bindings (mirrors the
            # _QtPushButton pattern above).
            self.setFocusPolicy(Qt.NoFocus)

            self._title = QLabel("", self)
            self._title.setStyleSheet(
                "QLabel { color: black; background-color: transparent;"
                " padding: 2px 4px; }"
            )

            self._scene = QGraphicsScene(self)
            self._view = _PanZoomGraphicsView(self._scene, self)
            self._view.setFocusPolicy(Qt.NoFocus)
            self._view.setRenderHint(QPainter.SmoothPixmapTransform, False)
            self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._view.setFrameShape(QFrame.NoFrame)
            self._view.setMouseTracking(True)

            self._pixmap_item = QGraphicsPixmapItem()
            self._scene.addItem(self._pixmap_item)

            # Marker groups are added on top of the pixmap; stacked on
            # higher Z so they always render above it regardless of
            # insertion order.
            self._marker_groups = []
            self._scatter_artists = []  # populated by _QtScatterArtist init

            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self._title)
            layout.addWidget(self._view, stretch=1)

            self._bg_color = QColor(0, 0, 0)
            self._view.setBackgroundBrush(QBrush(self._bg_color))
            self._scene_rect_set = False
            # Track the (w, h) the scene rect was last sized to; a
            # follow-up :meth:`set_image` whose array doesn't match
            # rebuilds the scene rect + refits. Without this, swapping
            # to a video of different dimensions (DUSTrack 1.2.0a3
            # seed-modal swap, future cross-camera projects) leaves
            # the new image rendered in the old scene rect -- the
            # symptom is "the loaded video looks zoomed in and reset
            # doesn't help" because reset_view fits to the stale rect.
            self._scene_rect_dims = None  # type: tuple[int, int] | None

            # Pick adapter is opt-in via :meth:`install_pick_adapter`.
            self._pick_adapter = None

            # When the cursor enters the pane, redirect keyboard focus
            # to ``_focus_target`` (typically the matplotlib canvas).
            # Without this, key presses while hovering over the pane
            # never reach mpl's key_press_event handler -- silently
            # breaking every hover-and-press binding in Tier 2.
            self._focus_target = None
            self.setMouseTracking(True)
            self._view.setMouseTracking(True)

        def set_image(self, arr) -> None:
            """Push an H x W x {1,3,4} ndarray to the pixmap item."""
            rgb = _as_rgb_uint8(arr)
            # Keep a reference to the buffer until the synchronous
            # QPixmap.fromImage copy completes inside _ndarray_to_qpixmap.
            self._last_rgb_buf = rgb
            pixmap = _ndarray_to_qpixmap(rgb)
            self._pixmap_item.setPixmap(pixmap)
            h, w = rgb.shape[:2]
            if not self._scene_rect_set:
                self._scene.setSceneRect(QRectF(0, 0, w, h))
                self._fit_view()
                self._scene_rect_set = True
                self._scene_rect_dims = (w, h)
            elif self._scene_rect_dims != (w, h):
                # Dimensions changed across a swap. The prior transform
                # is geometrically tied to the old scene rect, so any
                # restored view state is now invalid -- rebuild the
                # scene rect + reset transform + refit. user_adjusted
                # is cleared so resizeEvent picks the refit path again.
                self._scene.setSceneRect(QRectF(0, 0, w, h))
                self._view.resetTransform()
                self._view.user_adjusted = False
                self._fit_view()
                self._scene_rect_dims = (w, h)

        def _fit_view(self) -> None:
            self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        def reset_view(self) -> None:
            """Clear any user zoom/pan and re-fit the scene to the viewport.

            Bound to middle-click on the pane and to the 'r' key on
            VideoPointAnnotator (Tier 2). The transform reset is
            necessary because ``fitInView`` operates on top of the
            current transform; without ``resetTransform`` first, a
            zoomed-in user clicking reset would partially un-zoom.
            """
            self._view.resetTransform()
            self._view.user_adjusted = False
            self._fit_view()

        def get_view_state(self) -> Optional[dict]:
            """Snapshot the current transform + scrollbar positions.

            Returned dict round-trips through :meth:`set_view_state`.
            Returns ``None`` when no image has been set yet (no scene
            rect = nothing to snapshot) -- callers treat that as "no
            saved viewport, defer to fit-to-frame on restore".

            Used by DUSTrack 1.2.0a3's multi-video swap to remember
            each bundle's image-pane viewport across swap-out /
            swap-in cycles so a returning swap lands the user back on
            exactly the zoom region they left.
            """
            if not self._scene_rect_set:
                return None
            t = self._view.transform()
            return {
                "transform": (
                    float(t.m11()), float(t.m12()),
                    float(t.m21()), float(t.m22()),
                    float(t.dx()), float(t.dy()),
                ),
                "h_scroll": int(self._view.horizontalScrollBar().value()),
                "v_scroll": int(self._view.verticalScrollBar().value()),
                "user_adjusted": bool(self._view.user_adjusted),
            }

        def set_view_state(self, state: Optional[dict]) -> None:
            """Restore a previously-snapshotted view state.

            ``None`` (or a state captured before any image was set)
            falls back to :meth:`reset_view`. Otherwise applies the
            saved transform + scrollbar positions verbatim, and
            restores the ``user_adjusted`` flag so subsequent
            :meth:`resizeEvent` calls respect the user's prior zoom
            intent (without this, the next window resize would
            silently clobber the restored viewport).
            """
            if state is None or not self._scene_rect_set:
                self.reset_view()
                return
            m11, m12, m21, m22, dx, dy = state["transform"]
            self._view.setTransform(QTransform(m11, m12, m21, m22, dx, dy))
            self._view.horizontalScrollBar().setValue(int(state["h_scroll"]))
            self._view.verticalScrollBar().setValue(int(state["v_scroll"]))
            self._view.user_adjusted = bool(state.get("user_adjusted", False))

        def resizeEvent(self, event):  # noqa: N802 (Qt naming)
            super().resizeEvent(event)
            if self._scene_rect_set and not self._view.user_adjusted:
                # Auto-re-fit on window resize so the image scales with
                # the window. Once the user has wheel-zoomed or pan-
                # dragged (``user_adjusted`` flag set by the view), we
                # leave their view alone -- a re-fit would clobber it.
                # ``transform().isIdentity()`` does NOT work as the
                # gate here: the initial _fit_view() leaves a non-
                # identity scale transform, so an identity check would
                # block every subsequent re-fit.
                self._view.resetTransform()
                self._fit_view()

        def set_title(self, text: str) -> None:
            self._title.setText(text)

        def set_background_color(self, color) -> None:
            """Set the QGraphicsView background brush.

            Accepts any QColor-compatible argument (e.g. '#2a2a2a',
            matplotlib RGB tuple, named CSS color). Used by DUSTrack's
            dark-theme code path via VideoPointAnnotator's shim.
            """
            self._bg_color = _coerce_qcolor(color)
            self._view.setBackgroundBrush(QBrush(self._bg_color))

        def add_marker_group(self) -> QGraphicsItemGroup:
            """Allocate and return a fresh marker group on top of the pixmap.

            The returned group carries a back-reference to its owning
            ``_QtImagePane`` as ``group._image_pane`` so downstream
            scatter-artist factories can register with the pane for
            hit-testing without an extra parameter threading through
            ``setup_display_scatter``.
            """
            group = QGraphicsItemGroup()
            group.setZValue(1.0 + len(self._marker_groups))
            self._scene.addItem(group)
            self._marker_groups.append(group)
            group._image_pane = self
            return group

        def set_focus_target(self, widget) -> None:
            """Widget to focus on mouse-enter (typically the mpl canvas)."""
            self._focus_target = widget

        def enterEvent(self, event):  # noqa: N802 (Qt naming)
            if self._focus_target is not None:
                self._focus_target.setFocus()
            super().enterEvent(event)

        def install_pick_adapter(self):
            """Lazily attach a :class:`_QtPickAdapter` (callable container).

            Idempotent -- repeat calls return the cached adapter.
            """
            if self._pick_adapter is None:
                adapter_cls = _make_qt_pick_adapter_class()
                self._pick_adapter = adapter_cls(self)
                self._view.viewport().installEventFilter(self._pick_adapter)
            return self._pick_adapter

    return _QtImagePane


def make_image_pane(figure, picker_radius: float = 5.0,
                    sidebar_width: int = 280):
    """Build and install a :class:`_QtImagePane` above ``figure``'s canvas.

    The QMainWindow's central widget is replaced with a two-row
    container:

      Top row    : image_pane (stretch 2)
      Bottom row : trace canvas (stretch 1)

    Pre-rc2 the top row also held a fixed-width word-wrapping QLabel
    sidebar (the Tier 2 statevariables text sink); rc2 retired that
    sink in favor of the QDockWidget left column built by
    :func:`_get_left_column` (which hosts buttons + statevariables for
    both Tier 1 and Tier 2), so the sidebar QLabel and its
    ``pane.sidebar`` attribute are gone. ``sidebar_width`` is kept on
    the signature for API stability but is now ignored.

    Buttons + state-variables live in the QDockWidget on
    ``LeftDockWidgetArea``; spatially Qt manages them outside the
    central widget.

    Returns the new pane, or ``None`` if ``figure`` isn't on a Qt
    canvas.
    """
    _ = sidebar_width  # retired; kept on signature for backwards compat
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        pane_cls = _make_qt_image_pane_class()
    except ImportError:
        return None
    from qtpy.QtWidgets import QVBoxLayout, QWidget
    canvas = figure.canvas

    container = QWidget(qt_window)
    v_layout = QVBoxLayout(container)
    v_layout.setContentsMargins(0, 0, 0, 0)
    v_layout.setSpacing(0)

    pane = pane_cls(container, picker_radius=picker_radius)
    v_layout.addWidget(pane, stretch=2)
    v_layout.addWidget(canvas, stretch=1)

    pane.set_focus_target(canvas)
    qt_window.setCentralWidget(container)
    # Resize the host window so the image pane gets a fair share of
    # vertical real estate. The mpl figure for Tier 2 is sized for the
    # trace canvas only (e.g. figsize=(12, 3) in VideoPointAnnotator),
    # so the QMainWindow's auto-fit-to-canvas size would leave the
    # image pane ~stretch_2 / 3 of a too-short window. Match the
    # layout's 2:1 stretch hint: total height = canvas natural height
    # x 3, image pane lands at ~2x canvas height.
    canvas_h = canvas.height()
    canvas_w = canvas.width()
    if canvas_h > 0 and canvas_w > 0:
        qt_window.resize(canvas_w, canvas_h * 3)
    # Stash on the window so other code can find it (e.g. parity tests,
    # status overlays) and so subsequent calls can short-circuit.
    qt_window._dnav_image_pane = pane
    return pane


# rc2 retired :class:`_QtSidebarTextSink` / :func:`make_sidebar_text_sink`.
# That sink was the fast_render Tier 2 path that wrote a flat newline-
# joined string into the image_pane's QLabel sidebar. It's been replaced
# by :func:`make_qt_statevars_widget`, which renders the same
# statevariables as live QComboBox / QButtonGroup / QLabel controls in
# the QDockWidget left column (one column for both Tier 1 and Tier 2).


def _coerce_qcolor(color):
    """Best-effort QColor builder from mpl-shaped or string inputs."""
    from qtpy.QtGui import QColor
    if isinstance(color, QColor):
        return color
    if isinstance(color, str):
        return QColor(color)
    if isinstance(color, (tuple, list)) and len(color) in (3, 4):
        # mpl rgb(a) tuples are floats in 0..1.
        def _u8(v):
            return max(0, min(255, int(round(float(v) * 255))))
        if len(color) == 3:
            return QColor(_u8(color[0]), _u8(color[1]), _u8(color[2]))
        return QColor(_u8(color[0]), _u8(color[1]), _u8(color[2]), _u8(color[3]))
    raise TypeError(f"Cannot coerce {color!r} to QColor.")


def _make_qt_scatter_artist_class():
    """Build :class:`_QtScatterArtist` lazily.

    The class is a duck-typed match for the subset of
    :class:`matplotlib.collections.PathCollection` that
    :class:`VideoAnnotation` actually calls per the audit:
    ``set_offsets``, ``set_visible``, ``set_alpha``, ``set_facecolor``,
    ``set_sizes``, ``remove``. The handle is what gets stashed in
    ``plot_handles[f"labels_in_ax{ax_cnt}"]``.
    """
    from qtpy.QtCore import QRectF
    from qtpy.QtGui import QBrush, QPen
    from qtpy.QtWidgets import QGraphicsEllipseItem

    import numpy as np

    class _QtScatterArtist:
        """Qt-native scatter artist over a QGraphicsItemGroup."""

        # Default marker radius in scene (image-pixel) units. mpl's
        # default scatter point is rcParams['lines.markersize']**2 = 36
        # pt^2 -- a 6 pt diameter dot. On a typical ~700px-tall image
        # at default zoom, 6 image pixels reads as a similar size.
        _DEFAULT_RADIUS_PX = 6.0

        def __init__(self, group, palette, picker_radius: float = 5.0,
                     image_pane=None):
            self._group = group
            self._palette = list(palette)
            self._items = []  # parallel to palette; created lazily by set_offsets
            self._offsets = np.full((len(self._palette), 2), np.nan)
            self._radius = self._DEFAULT_RADIUS_PX
            self._visible = True
            self._alpha = 1.0
            self._picker_radius = float(picker_radius)
            self._image_pane = image_pane
            if image_pane is not None:
                image_pane._scatter_artists.append(self)

        # -- per-item rebuild on palette / size change ---------------------
        def _ensure_items(self, n_pts: int) -> None:
            while len(self._items) < n_pts:
                idx = len(self._items)
                item = QGraphicsEllipseItem(
                    QRectF(-self._radius, -self._radius,
                           2 * self._radius, 2 * self._radius)
                )
                color = self._palette[idx % len(self._palette)]
                qcolor = _coerce_qcolor(color)
                qcolor.setAlphaF(self._alpha)
                item.setBrush(QBrush(qcolor))
                item.setPen(QPen(qcolor, 0))
                item.setVisible(False)  # NaN until first set_offsets
                self._group.addToGroup(item)
                self._items.append(item)
            while len(self._items) > n_pts:
                item = self._items.pop()
                self._group.removeFromGroup(item)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)

        # -- mpl-PathCollection-shaped API --------------------------------
        def set_offsets(self, offsets) -> None:
            arr = np.asarray(offsets, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(f"set_offsets expects (N, 2); got {arr.shape}")
            self._offsets = arr
            self._ensure_items(arr.shape[0])
            for i, (x, y) in enumerate(arr):
                item = self._items[i]
                if np.isnan(x) or np.isnan(y):
                    item.setVisible(False)
                    continue
                item.setPos(float(x), float(y))
                item.setVisible(self._visible)

        def get_offsets(self):
            """mpl-shaped accessor used by the parity test."""
            return self._offsets.copy()

        def set_visible(self, visible: bool) -> None:
            self._visible = bool(visible)
            for i, item in enumerate(self._items):
                if self._visible and i < len(self._offsets):
                    x, y = self._offsets[i]
                    item.setVisible(not (np.isnan(x) or np.isnan(y)))
                else:
                    item.setVisible(False)

        def set_alpha(self, alpha) -> None:
            if alpha is None:
                alpha = 1.0
            self._alpha = float(alpha)
            for idx, item in enumerate(self._items):
                color = self._palette[idx % len(self._palette)]
                qcolor = _coerce_qcolor(color)
                qcolor.setAlphaF(self._alpha)
                item.setBrush(QBrush(qcolor))
                item.setPen(QPen(qcolor, 0))

        def set_facecolor(self, colors) -> None:
            """Accepts a single color or one-per-point list (mpl shape)."""
            if isinstance(colors, str) or (
                isinstance(colors, (tuple, list)) and len(colors) in (3, 4)
                and not isinstance(colors[0], (tuple, list))
            ):
                colors = [colors] * len(self._items)
            for item, color in zip(self._items, colors):
                qcolor = _coerce_qcolor(color)
                qcolor.setAlphaF(self._alpha)
                item.setBrush(QBrush(qcolor))
                item.setPen(QPen(qcolor, 0))

        def set_sizes(self, sizes) -> None:
            """Match mpl: ``sizes`` is per-point area in points^2."""
            sizes = list(sizes)
            for i, item in enumerate(self._items):
                s = float(sizes[i % len(sizes)])
                # Half-width of the bounding square approximates the
                # markersize -> radius mapping mpl uses internally.
                r = (s ** 0.5) / 2.0
                item.setRect(QRectF(-r, -r, 2 * r, 2 * r))

        def remove(self) -> None:
            for item in self._items:
                self._group.removeFromGroup(item)
                scene = item.scene()
                if scene is not None:
                    scene.removeItem(item)
            self._items.clear()
            if self._image_pane is not None:
                try:
                    self._image_pane._scatter_artists.remove(self)
                except ValueError:
                    pass

        # -- hit-testing for _QtPickAdapter -------------------------------
        def hit_test(self, x: float, y: float):
            """Return the offset index whose center is within
            ``self._picker_radius`` of ``(x, y)`` in scene coords, or
            ``None`` if no marker is close enough. Ties broken by
            nearest distance.
            """
            best_i = None
            best_d2 = self._picker_radius ** 2
            for i, (mx, my) in enumerate(self._offsets):
                if np.isnan(mx) or np.isnan(my):
                    continue
                d2 = (mx - x) ** 2 + (my - y) ** 2
                if d2 <= best_d2:
                    best_d2 = d2
                    best_i = i
            return best_i

    return _QtScatterArtist


def make_scatter_artist(image_pane, palette, picker_radius: float = 5.0):
    """Allocate a marker group + :class:`_QtScatterArtist` on ``image_pane``."""
    if image_pane is None:
        return None
    cls = _make_qt_scatter_artist_class()
    group = image_pane.add_marker_group()
    return cls(group, palette, picker_radius=picker_radius, image_pane=image_pane)


def _make_qt_pick_adapter_class():
    """Build :class:`_QtPickAdapter` lazily.

    Installed as an event filter on the image pane's viewport. On a
    mouse press, it maps the viewport position to scene coords, hit-
    tests every :class:`_QtScatterArtist` registered with the pane, and
    fires:
    - Pick callbacks (matching matplotlib's ``pick_event``) with a
      synthetic event exposing ``.mouseevent.button.name`` and ``.ind``
      (a one-element list); only when the hit-test finds a marker.
    - Button-press callbacks (matching ``button_press_event``) with
      ``.inaxes``, ``.button.name``, ``.xdata``, ``.ydata``; on every
      press over the image pane regardless of hit-test outcome.

    The ``.inaxes`` sentinel is the image pane itself, which the
    Tier 2 wiring also stores as ``VideoPointAnnotator._ax_image`` and
    ``._ax``, so the existing ``event.inaxes == self._ax_image`` /
    ``== self._ax`` identity checks in pointtracking.py keep firing.
    """
    from types import SimpleNamespace

    from qtpy.QtCore import QEvent, QObject, Qt

    # Middle button is reserved for the pan/reset gesture handled by
    # _PanZoomGraphicsView; skip it here so we don't fire spurious
    # synthetic mpl events for pan drags.
    _QT_TO_MPL = {
        Qt.LeftButton: "LEFT",
        Qt.RightButton: "RIGHT",
    }

    class _QtPickAdapter(QObject):
        def __init__(self, image_pane):
            super().__init__(image_pane)
            self._image_pane = image_pane
            self._pick_callbacks = []
            self._button_press_callbacks = []

        def connect_pick(self, callback) -> None:
            self._pick_callbacks.append(callback)

        def connect_button_press(self, callback) -> None:
            self._button_press_callbacks.append(callback)

        def eventFilter(self, obj, event):  # noqa: N802 (Qt naming)
            if event.type() != QEvent.MouseButtonPress:
                return False
            if obj is not self._image_pane._view.viewport():
                return False

            qt_button = event.button()
            button_name = _QT_TO_MPL.get(qt_button)
            if button_name is None:
                return False

            view = self._image_pane._view
            # ``position()`` is QPointF (Qt6); ``pos()`` is QPoint
            # (Qt5). Try the modern API first, fall back.
            try:
                vp = event.position()
                vp_x, vp_y = vp.x(), vp.y()
            except AttributeError:  # pragma: no cover (older Qt5)
                p = event.pos()
                vp_x, vp_y = float(p.x()), float(p.y())

            scene_pos = view.mapToScene(int(vp_x), int(vp_y))
            x_data, y_data = scene_pos.x(), scene_pos.y()

            # ``event.button`` in matplotlib is a MouseButton enum
            # whose .name is "LEFT"/"RIGHT"/"MIDDLE" -- mirror with a
            # SimpleNamespace so the consumer's ``.button.name`` keeps
            # working.
            button_obj = SimpleNamespace(name=button_name)

            press_event = SimpleNamespace(
                name="button_press_event",
                inaxes=self._image_pane,
                button=button_obj,
                xdata=x_data,
                ydata=y_data,
                x=vp_x,
                y=vp_y,
            )
            for cb in self._button_press_callbacks:
                cb(press_event)

            # Pick: ask each scatter artist if it has a marker near
            # the click. Fire callbacks per hit (order = artist
            # registration), passing one event each.
            for artist in self._image_pane._scatter_artists:
                idx = artist.hit_test(x_data, y_data)
                if idx is None:
                    continue
                pick_event = SimpleNamespace(
                    name="pick_event",
                    mouseevent=press_event,
                    artist=artist,
                    ind=[int(idx)],
                )
                for cb in self._pick_callbacks:
                    cb(pick_event)

            # Don't consume -- let Qt continue propagation so the
            # canvas can also react if needed (e.g. focus changes).
            return False

    return _QtPickAdapter


# ---------------------------------------------------------------------------
# Keybindings cheatsheet dialog (rc2). Replaces the pre-rc2
# matplotlib TextView overlay rendered by GenericBrowser.show_key_bindings.
# ---------------------------------------------------------------------------


def make_keybindings_dialog(figure, keypressdict, section_order=None):
    """Open a modeless Qt dialog showing the cheatsheet for ``keypressdict``.

    Returns the dialog instance (caller should keep a reference; the
    dialog is non-modal and would be garbage-collected otherwise), or
    ``None`` on a non-Qt backend so :meth:`GenericBrowser.show_key_bindings`
    can fall back to stdout.

    Layout: one ``QGroupBox`` per ``KeyBinding.group``. Section order
    comes from ``section_order`` (tuple of group names, leading); any
    groups not listed there follow in insertion order. The ``None``
    group is rendered last as "Other". Each section is a 2-column
    borderless ``QTableWidget`` (shortcut · description) with monospace
    right-aligned shortcuts. The whole stack lives in a ``QScrollArea``
    so long lists don't overflow the screen.
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        cls = _make_keybindings_dialog_class()
    except ImportError:
        return None
    from .core import _group_keybindings
    sections = _group_keybindings(keypressdict, section_order)
    dialog = cls(qt_window, sections)
    dialog.show()
    dialog.raise_()
    return dialog


def _make_keybindings_dialog_class():
    """Lazy: qtpy import deferred so the module imports without Qt."""
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFont
    from qtpy.QtWidgets import (
        QAbstractItemView,
        QDialog,
        QGroupBox,
        QHeaderView,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )

    class _KeybindingsDialog(QDialog):
        """Modeless cheatsheet dialog parented to a matplotlib QMainWindow."""

        def __init__(self, parent, sections):
            super().__init__(parent)
            self.setWindowTitle("Keyboard shortcuts")
            self.setModal(False)
            # Don't steal focus from the matplotlib canvas -- otherwise
            # the dialog grabs key events the user means for the figure.
            self.setFocusPolicy(Qt.NoFocus)

            outer = QVBoxLayout(self)
            outer.setContentsMargins(8, 8, 8, 8)

            scroll = QScrollArea(self)
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.NoFrame)
            outer.addWidget(scroll)

            content = QWidget()
            scroll.setWidget(content)
            inner = QVBoxLayout(content)
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(8)

            mono = QFont("Consolas")
            mono.setStyleHint(QFont.Monospace)

            total_rows = 0
            for group_name, rows in sections:
                box = QGroupBox(group_name, content)
                box_layout = QVBoxLayout(box)
                box_layout.setContentsMargins(8, 4, 8, 4)

                table = QTableWidget(len(rows), 2, box)
                table.horizontalHeader().hide()
                table.verticalHeader().hide()
                table.setShowGrid(False)
                table.setSelectionMode(QAbstractItemView.NoSelection)
                table.setEditTriggers(QAbstractItemView.NoEditTriggers)
                table.setFocusPolicy(Qt.NoFocus)
                table.setFrameShape(QTableWidget.NoFrame)

                for row, (shortcut, description) in enumerate(rows):
                    sc_item = QTableWidgetItem(shortcut)
                    sc_item.setFont(mono)
                    sc_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    sc_item.setFlags(Qt.ItemIsEnabled)
                    table.setItem(row, 0, sc_item)

                    desc_item = QTableWidgetItem(description)
                    desc_item.setFlags(Qt.ItemIsEnabled)
                    table.setItem(row, 1, desc_item)

                table.resizeColumnsToContents()
                table.resizeRowsToContents()
                table.horizontalHeader().setSectionResizeMode(
                    0, QHeaderView.ResizeToContents
                )
                table.horizontalHeader().setSectionResizeMode(
                    1, QHeaderView.Stretch
                )
                # Fix the table height to its content; the QScrollArea
                # outside handles overflow for the whole dialog.
                row_h = table.verticalHeader().defaultSectionSize()
                table.setFixedHeight(row_h * len(rows) + 4)

                box_layout.addWidget(table)
                inner.addWidget(box)
                total_rows += len(rows)

            inner.addStretch(1)

            # Sized for ~all-shortcuts-visible without scrolling on a
            # typical DUSTrack session (>50 entries); QScrollArea kicks
            # in past that.
            self.resize(480, min(720, 80 + total_rows * 22))

    return _KeybindingsDialog
