"""Built-in button style tags shipped by datanavigator.

Each styler is a ``Callable[[Button], None]`` registered under a
semantic name in :data:`BUILTIN_STYLES`. Consumers reach the built-ins
through ``Buttons.add(..., style_tag="primary")`` (etc.); a consumer
can shadow any of them by re-registering the same name through
:meth:`Buttons.register_style`.

Semantic (role-based) names so a future palette swap doesn't break
consumer call-sites:

- ``primary``   -- saturated accent for the headline action.
- ``secondary`` -- muted accent for supporting actions.
- ``neutral``   -- passthrough, returns native rendering.
- ``warn``      -- warm tone for "use sparingly" actions.

The Qt path is detected via ``getattr(b, "_qt_btn", None)`` and styled
via QSS. On the mpl path each built-in is a no-op (the matplotlib
``Button`` paint is left alone), matching DUSTrack's pre-refactor
behavior where its sidebar palette only landed on the Qt path.

Stylers consult the live ``QApplication`` palette for dark / light
mode so the defaults behave reasonably in either; consumers that
want bespoke palettes register their own stylers and don't touch this
module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def _is_dark_mode() -> bool:
    """Heuristically detect dark mode from the live QApplication palette.

    Returns ``False`` when no Qt binding / application is available --
    callers should treat that as "light mode" since the styler will
    no-op on the mpl path anyway.
    """
    try:
        from qtpy.QtWidgets import QApplication
        from qtpy.QtGui import QPalette
    except ImportError:
        return False
    app = QApplication.instance()
    if app is None:
        return False
    c = app.palette().color(QPalette.Window)
    return (c.red() + c.green() + c.blue()) / 3 < 128


def _qss(bg: str, fg: str, border: str, hover: str, pressed: str) -> str:
    return (
        f"QPushButton {{ background-color: {bg}; color: {fg}; "
        f"border: 1px solid {border}; padding: 4px; }} "
        f"QPushButton:hover {{ background-color: {hover}; }} "
        f"QPushButton:pressed {{ background-color: {pressed}; }}"
    )


def _apply_primary(b: Any) -> None:
    qbtn = getattr(b, "_qt_btn", None)
    if qbtn is None:
        return
    if _is_dark_mode():
        qbtn.setStyleSheet(_qss(
            bg="#3b6db5", fg="#ffffff", border="#2a4f80",
            hover="#4a7fc8", pressed="#2a4f80",
        ))
    else:
        qbtn.setStyleSheet(_qss(
            bg="#cfdef3", fg="#2c3e50", border="#a8c0dd",
            hover="#bccfea", pressed="#a8c0dd",
        ))


def _apply_secondary(b: Any) -> None:
    qbtn = getattr(b, "_qt_btn", None)
    if qbtn is None:
        return
    if _is_dark_mode():
        qbtn.setStyleSheet(_qss(
            bg="#3a3a3a", fg="#e0e0e0", border="#555555",
            hover="#4a4a4a", pressed="#2e2e2e",
        ))
    else:
        qbtn.setStyleSheet(_qss(
            bg="#e0e4e8", fg="#2c3e50", border="#c0c5cb",
            hover="#d0d4d9", pressed="#c0c5cb",
        ))


def _apply_neutral(b: Any) -> None:
    # Explicit reset so reapply_styles can swing a button back to
    # native rendering after a prior styled pass.
    qbtn = getattr(b, "_qt_btn", None)
    if qbtn is None:
        return
    qbtn.setStyleSheet("")


def _apply_warn(b: Any) -> None:
    qbtn = getattr(b, "_qt_btn", None)
    if qbtn is None:
        return
    if _is_dark_mode():
        qbtn.setStyleSheet(_qss(
            bg="#7a4a2e", fg="#fbe6c8", border="#5a3520",
            hover="#8c5a3a", pressed="#5a3520",
        ))
    else:
        qbtn.setStyleSheet(_qss(
            bg="#f5d9c0", fg="#2c3e50", border="#d9b88a",
            hover="#eaca9f", pressed="#d9b88a",
        ))


BUILTIN_STYLES: Dict[str, Callable[[Any], None]] = {
    "primary": _apply_primary,
    "secondary": _apply_secondary,
    "neutral": _apply_neutral,
    "warn": _apply_warn,
}
