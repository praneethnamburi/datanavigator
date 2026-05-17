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
adds :func:`make_qt_button` -- a QPushButton-in-QToolBar replacement
for the matplotlib-widgets-based Button stack used by
:class:`assets.Buttons`.
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
# Phase 3 -- Qt buttons in a QToolBar attached to the QMainWindow.
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
    from qtpy.QtWidgets import QPushButton

    class _QtPushButton:
        """Public-API match for :class:`assets.Button` (mpl path).

        Attributes:
            name: button label / identity (used by AssetContainer.__getitem__).
            _qt_btn: the underlying QPushButton (Phase 3+ internals).
        """

        def __init__(self, toolbar, name: str, **_kwargs):
            self.name = name
            self._qt_btn = QPushButton(name)
            toolbar.addWidget(self._qt_btn)

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

        def __init__(self, toolbar, name: str, start_state: bool = True, **_kwargs):
            super().__init__(toolbar, name)
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


def _get_buttons_toolbar(qt_window):
    """Return the cached QToolBar for datanavigator buttons on ``qt_window``.

    First call attaches a new QToolBar to the left side of the window and
    caches it under ``qt_window._dnav_buttons_toolbar``. Subsequent calls
    reuse it so all Buttons.add() calls land in the same toolbar.
    """
    tb = getattr(qt_window, "_dnav_buttons_toolbar", None)
    if tb is None:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QToolBar
        tb = QToolBar("datanavigator", qt_window)
        qt_window.addToolBar(Qt.LeftToolBarArea, tb)
        qt_window._dnav_buttons_toolbar = tb
    return tb


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
    tb = _get_buttons_toolbar(qt_window)
    if type_ == "Toggle":
        return toggle_cls(tb, name, start_state=start_state)
    return push_cls(tb, name)
