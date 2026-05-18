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
    from qtpy.QtCore import Qt
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


def add_qt_separator(figure) -> bool:
    """Add a separator to the datanavigator QToolBar on ``figure``'s window.

    Returns True if the separator was added (Qt path active), False if
    ``figure`` is not on a Qt canvas so the caller should fall back to
    its mpl-side spacer hack. Lazy-creates the toolbar if it doesn't
    exist yet (matches :func:`make_qt_button`'s caching).
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return False
    try:
        tb = _get_buttons_toolbar(qt_window)
    except ImportError:
        return False
    tb.addSeparator()
    return True


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
    from qtpy.QtGui import QBrush, QColor, QPainter
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
            self._view = QGraphicsView(self._scene, self)
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

        def _fit_view(self) -> None:
            self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

        def resizeEvent(self, event):  # noqa: N802 (Qt naming)
            super().resizeEvent(event)
            if self._scene_rect_set:
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
    container mirroring Tier 1's spatial layout:

      Top row    : ``[sidebar | image_pane]`` (stretch 2)
      Bottom row : trace canvas (stretch 1, full width)

    The sidebar is a fixed-width word-wrapping QLabel used by Tier 2
    statevariables rendering. Fixed width (default 280 px ~ 40
    monospace chars) means a long annotation-layer name wraps to the
    next line instead of reflowing the whole layout -- the image and
    trace canvas widths stay stable across updates.

    The full-width trace canvas in the bottom row preserves the
    image-to-trace time alignment users build a mental model around
    when scrubbing long videos (the same column-width the imshow
    occupies above).

    Buttons are *not* in this layout -- they live in the QToolBar
    attached to ``LeftToolBarArea`` by the 1.4.0-qt
    ``_get_buttons_toolbar`` path. Conceptually they sit beside the
    sidebar; spatially Qt manages them outside the central widget.

    Returns the new pane (with ``pane.sidebar`` exposed), or ``None``
    if ``figure`` isn't on a Qt canvas.
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return None
    try:
        pane_cls = _make_qt_image_pane_class()
    except ImportError:
        return None
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import (
        QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
    )
    canvas = figure.canvas

    container = QWidget(qt_window)
    v_layout = QVBoxLayout(container)
    v_layout.setContentsMargins(0, 0, 0, 0)
    v_layout.setSpacing(0)

    top_row = QWidget(container)
    h_layout = QHBoxLayout(top_row)
    h_layout.setContentsMargins(0, 0, 0, 0)
    h_layout.setSpacing(0)

    sidebar = QLabel("", top_row)
    sidebar.setAlignment(Qt.AlignTop | Qt.AlignLeft)
    sidebar.setTextFormat(Qt.PlainText)
    # Word-wrap inside a fixed-width column: long labels wrap to the
    # next line, so the whole layout doesn't reflow on every state
    # change. Sidebar height grows down (shrinks the image slightly
    # via the top-row's vertical stretch); width is invariant.
    sidebar.setWordWrap(True)
    sidebar.setFixedWidth(int(sidebar_width))
    sidebar.setStyleSheet(
        "QLabel { color: black; background-color: rgba(245, 245, 245, 240);"
        " padding: 4px 8px; border-right: 1px solid #d0d0d0; }"
    )
    font = sidebar.font()
    font.setFamily("Courier New")
    sidebar.setFont(font)
    # Vertical sizing: minimum height = wrapped content height; can
    # expand if the layout has slack.
    sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.MinimumExpanding)
    # Hidden until populated -- some browsers don't use statevariables.
    sidebar.hide()

    pane = pane_cls(top_row, picker_radius=picker_radius)
    h_layout.addWidget(sidebar)
    h_layout.addWidget(pane, stretch=1)
    v_layout.addWidget(top_row, stretch=2)

    v_layout.addWidget(canvas, stretch=1)

    pane.set_focus_target(canvas)
    pane.sidebar = sidebar
    qt_window.setCentralWidget(container)
    # Stash on the window so other code can find it (e.g. parity tests,
    # status overlays) and so subsequent calls can short-circuit.
    qt_window._dnav_image_pane = pane
    return pane


def _make_qt_sidebar_text_sink_class():
    """Build :class:`_QtSidebarTextSink` lazily.

    A duck-typed match for :class:`utils.TextView` from
    :class:`StateVariables`' point of view: exposes ``.update(text)``
    and ``.text``. Writes into a layout-managed QLabel (the sidebar
    built by :func:`make_image_pane`) instead of an
    overlay positioned in canvas-fraction coords. The label's width
    auto-grows with content, so the text can never overlap the plot.
    """
    class _QtSidebarTextSink:
        def __init__(self, label):
            self._label = label
            self.text = []
            # ``_pos`` is a settable no-op for parity with
            # :class:`utils.TextView` -- DUSTrack writes
            # ``self.statevariables._text._pos`` to re-anchor the
            # canvas overlay, which is meaningless once the text
            # lives in a layout-managed sidebar.
            self._pos = None

        def update(self, text):
            if isinstance(text, dict):
                lines = [f"{k} - {v}" for k, v in text.items()]
            elif isinstance(text, (list, tuple)):
                lines = list(text)
            else:
                lines = [str(text)]
            self.text = lines
            self._label.setText("\n".join(lines))
            # No adjustSize() -- the label's width is fixed
            # (setFixedWidth) and word-wrap handles overflow vertically.
            # Calling adjustSize would override the fixed width.
            self._label.show()

        def hide(self):
            self._label.hide()

    return _QtSidebarTextSink


def make_sidebar_text_sink(image_pane):
    """Return a :class:`_QtSidebarTextSink` bound to ``image_pane.sidebar``.

    Returns ``None`` if the pane has no sidebar (e.g. built before the
    1.5.0 layout-managed slot was added).
    """
    sidebar = getattr(image_pane, "sidebar", None)
    if sidebar is None:
        return None
    cls = _make_qt_sidebar_text_sink_class()
    return cls(sidebar)


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

    _QT_TO_MPL = {
        Qt.LeftButton: "LEFT",
        Qt.RightButton: "RIGHT",
        Qt.MiddleButton: "MIDDLE",
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
