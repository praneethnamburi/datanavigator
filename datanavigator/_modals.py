"""Canonical datanavigator modal overlays: blocking confirm + text prompt.

The house modal style — a dark, translucent, *blocking* ``QFrame`` parented to the
figure's ``QMainWindow``, covering it (so clicks can't reach the underlying canvas)
with role-styled buttons. Promoted from DUSTrack's bespoke ``ConfirmOverlay`` so
every browser (DUSTrack, delsys, …) shares one modal vocabulary instead of
re-implementing it.

Both public helpers — :func:`confirm` and :func:`prompt_text` — resolve the Qt
window via :func:`datanavigator._qt.find_qt_window` and **no-op gracefully off-Qt**
(returning the supplied default), so headless / Agg callers never block. qtpy is
imported lazily inside the class factories, mirroring the rest of ``_qt``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ._qt import find_qt_window

#: Severity -> title color (shared by both overlays).
_SEVERITY_COLOR = {"info": "white", "warning": "#7cdb7c", "destructive": "#ff7c7c"}

#: Per-button-role QSS (``primary`` / ``destructive`` / ``neutral``).
_ROLE_QSS = {
    "primary": (
        "QPushButton { background-color: #3a86ff; color: white; "
        "  border: 1px solid #2a76ef; padding: 6px 24px; "
        "  font-size: 11pt; font-weight: bold; }"
        "QPushButton:hover { background-color: #4a96ff; }"
        "QPushButton:pressed { background-color: #2a76ef; }"
    ),
    "destructive": (
        "QPushButton { background-color: #ff7c7c; color: white; "
        "  border: 1px solid #df5c5c; padding: 6px 24px; "
        "  font-size: 11pt; font-weight: bold; }"
        "QPushButton:hover { background-color: #ff9c9c; }"
        "QPushButton:pressed { background-color: #df5c5c; }"
    ),
    "neutral": (
        "QPushButton { background-color: #555555; color: white; "
        "  border: 1px solid #444444; padding: 6px 24px; "
        "  font-size: 11pt; }"
        "QPushButton:hover { background-color: #666666; }"
        "QPushButton:pressed { background-color: #444444; }"
    ),
}


def confirm(
    figure,
    *,
    title: str,
    message: str,
    buttons: List[Tuple[str, str]],
    default: Optional[str] = None,
    severity: str = "info",
) -> Optional[str]:
    """Show the blocking confirm overlay; return the clicked button's label.

    Args:
        figure: A matplotlib figure whose ``QMainWindow`` hosts the overlay.
        title: Heading (tinted by ``severity``).
        message: Body text (word-wrapped).
        buttons: Ordered ``[(label, role), ...]``; ``role`` is ``"primary"`` /
            ``"destructive"`` / ``"neutral"``.
        default: Label that receives focus / is the implicit Enter target, and the
            value returned on the non-Qt fallback. Defaults to the last button.
        severity: ``"info"`` / ``"warning"`` / ``"destructive"`` (title color).

    Returns:
        The clicked label. Off a Qt backend (no window), returns ``default``.
    """
    if default is None and buttons:
        default = buttons[-1][0]
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return default
    try:
        cls = _make_confirm_overlay_class()
    except ImportError:
        return default
    return cls(
        qt_window,
        title=title,
        message=message,
        buttons=buttons,
        default=default,
        severity=severity,
    ).exec_()


def prompt_text(
    figure, *, title: str, prompt: str = "", default: str = ""
) -> Optional[str]:
    """Show the blocking single-line text-input overlay; return the entered text.

    Returns the trimmed text on OK (empty input is rejected — OK stays disabled),
    or ``None`` on Cancel. Off a Qt backend (no window), returns ``default``.
    """
    qt_window = find_qt_window(figure)
    if qt_window is None:
        return default
    try:
        cls = _make_text_prompt_overlay_class()
    except ImportError:
        return default
    return cls(qt_window, title=title, prompt=prompt, default=default).exec_()


class _ModalOverlayBase:
    """Shared backdrop-frame + reposition + event-filter + ``exec_`` scaffolding.

    Subclasses build content into the layout returned by :meth:`_install_frame`
    and drive a local ``QEventLoop``. Mixed with ``QObject`` in the lazy factories
    so qtpy stays import-on-demand.
    """

    def _install_frame(self, main_window, object_name: str, extra_qss: str = ""):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QFrame, QVBoxLayout

        self._mw = main_window
        self._frame = QFrame(main_window)
        self._frame.setObjectName(object_name)
        self._frame.setStyleSheet(
            f"#{object_name} {{ background-color: rgba(0, 0, 0, 200); }}"
            "QLabel { color: white; }" + extra_qss
        )
        self._frame.setFocusPolicy(Qt.StrongFocus)
        layout = QVBoxLayout(self._frame)
        layout.setAlignment(Qt.AlignCenter)
        main_window.installEventFilter(self)
        return layout

    def eventFilter(self, obj, event):  # noqa: N802 (Qt API)
        from qtpy.QtCore import QEvent

        if obj is self._mw and event.type() == QEvent.Resize:
            self._reposition()
        return False

    def _reposition(self):
        self._frame.setGeometry(0, 0, self._mw.width(), self._mw.height())
        self._frame.raise_()

    def _dismiss(self):
        try:
            self._mw.removeEventFilter(self)
        except Exception:
            pass
        self._frame.hide()
        self._frame.deleteLater()

    def exec_(self):
        """Block on a local event loop until a button resolves; return the result."""
        self._loop.exec_()
        return self._result


def _make_confirm_overlay_class():
    """Lazy: the blocking confirm overlay (qtpy imported here)."""
    from qtpy.QtCore import QEventLoop, QObject, Qt
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton

    class _ConfirmOverlay(_ModalOverlayBase, QObject):
        def __init__(
            self, main_window, *, title, message, buttons, default=None, severity="info"
        ):
            super().__init__(main_window)
            self._result = None
            self._loop = QEventLoop()
            title_color = _SEVERITY_COLOR.get(severity, "white")
            layout = self._install_frame(
                main_window,
                "datanavigator_confirm_overlay",
                f"#dn_confirm_title {{ color: {title_color}; "
                "  font-size: 22pt; font-weight: bold; }"
                "#dn_confirm_message { font-size: 12pt; }",
            )
            layout.addStretch(1)

            title_lbl = QLabel(title)
            title_lbl.setObjectName("dn_confirm_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_lbl)

            message_lbl = QLabel(message)
            message_lbl.setObjectName("dn_confirm_message")
            message_lbl.setAlignment(Qt.AlignCenter)
            message_lbl.setWordWrap(True)
            message_lbl.setMaximumWidth(720)
            layout.addWidget(message_lbl, alignment=Qt.AlignCenter)

            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            default_btn = None
            for label, role in buttons:
                btn = QPushButton(label)
                btn.setMinimumWidth(160)
                btn.setStyleSheet(_ROLE_QSS.get(role, _ROLE_QSS["neutral"]))
                btn.clicked.connect(lambda _checked=False, lbl=label: self._on_clicked(lbl))
                button_row.addWidget(btn)
                if default is not None and label == default:
                    default_btn = btn
            layout.addLayout(button_row)
            layout.addStretch(1)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            (default_btn or self._frame).setFocus()

        def _on_clicked(self, label):
            self._result = label
            self._dismiss()
            self._loop.quit()

    return _ConfirmOverlay


def _make_text_prompt_overlay_class():
    """Lazy: the blocking single-line text-input overlay (qtpy imported here)."""
    from qtpy.QtCore import QEventLoop, QObject, Qt
    from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget

    class _TextPromptOverlay(_ModalOverlayBase, QObject):
        def __init__(self, main_window, *, title, prompt="", default=""):
            super().__init__(main_window)
            self._result = None
            self._loop = QEventLoop()
            layout = self._install_frame(
                main_window,
                "datanavigator_prompt_overlay",
                "#dn_prompt_title { color: white; font-size: 22pt; font-weight: bold; }"
                "#dn_prompt_label { font-size: 12pt; }"
                "QLineEdit { font-size: 12pt; padding: 4px; min-width: 360px; }",
            )
            layout.addStretch(1)

            title_lbl = QLabel(title)
            title_lbl.setObjectName("dn_prompt_title")
            title_lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(title_lbl)

            if prompt:
                prompt_lbl = QLabel(prompt)
                prompt_lbl.setObjectName("dn_prompt_label")
                prompt_lbl.setAlignment(Qt.AlignCenter)
                layout.addWidget(prompt_lbl, alignment=Qt.AlignCenter)

            content = QWidget()
            content_row = QHBoxLayout(content)
            self._edit = QLineEdit(default)
            self._edit.returnPressed.connect(self._on_ok)
            self._edit.textChanged.connect(self._revalidate)
            content_row.addWidget(self._edit)
            layout.addWidget(content, alignment=Qt.AlignCenter)

            button_row = QHBoxLayout()
            button_row.setAlignment(Qt.AlignCenter)
            self._ok_btn = QPushButton("OK")
            self._ok_btn.setMinimumWidth(140)
            self._ok_btn.setStyleSheet(_ROLE_QSS["primary"])
            self._ok_btn.clicked.connect(self._on_ok)
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setMinimumWidth(140)
            cancel_btn.setStyleSheet(_ROLE_QSS["neutral"])
            cancel_btn.clicked.connect(self._on_cancel)
            button_row.addWidget(self._ok_btn)
            button_row.addWidget(cancel_btn)
            layout.addLayout(button_row)
            layout.addStretch(1)

            self._frame.show()
            self._reposition()
            self._frame.raise_()
            self._edit.setFocus()
            self._edit.selectAll()
            self._revalidate()

        def _revalidate(self, *_):
            self._ok_btn.setEnabled(bool(self._edit.text().strip()))

        def _on_ok(self, *_):
            text = self._edit.text().strip()
            if not text:
                return
            self._result = text
            self._dismiss()
            self._loop.quit()

        def _on_cancel(self, *_):
            self._result = None
            self._dismiss()
            self._loop.quit()

    return _TextPromptOverlay
