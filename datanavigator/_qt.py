"""
Qt-binding-touching internals.

Imports here are deliberately lazy / behind a string check so that
``import datanavigator`` works on machines that have no Qt binding
installed -- 1.4.0 is a *soft* Qt requirement: when matplotlib is on a
Qt backend, Qt features activate; on Agg or other backends, datanavigator
falls back to its pre-1.4 matplotlib-native rendering.

Phase 1 contributes only :func:`find_qt_window`; Phase 2+ will add the
widget-replacement helpers (QLabel-backed TextView, QPushButton-backed
Buttons, etc.) used by the asset managers.
"""

from __future__ import annotations

from typing import Optional


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
