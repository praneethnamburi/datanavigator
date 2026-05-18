"""
Sandbox #7 (Phase 4a smoke).

Covers the two API additions/fixes that came out of the first DUSTrack
external smoke:

  1. Buttons.add_separator() -- promoted to a first-class API. On Qt
     (post-rc2) it inserts a QFrame.HLine into the buttons-column
     QVBoxLayout; on mpl it inserts an invisible spacer button to
     occupy a layout slot. (Pre-rc2 the Qt path was a
     QToolBar.addSeparator() QAction.)
  2. TextView.update() now re-reads self._pos on the Qt path, so
     post-construction mutations (DUSTrack does this to move
     statevariables from "bottom right" to "bottom left") apply.

Run with:
    QT_API=pyqt5 QT_QPA_PLATFORM=offscreen python tests/qt_learning/07_phase4a_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))
import datanavigator as dnav  # noqa: E402


def test_add_separator_qt_path():
    from qtpy.QtWidgets import QFrame, QPushButton
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.add(text="first")
    # Buttons widget should exist now (lazily created by first add).
    container = b._qt_window._dnav_buttons_widget
    layout = container.layout()
    # widgets_before = total children excluding the trailing addStretch().
    # countWidgets walks layout items because addStretch produces a
    # QSpacerItem, not a QWidget.
    def count_widgets(lay):
        n = 0
        for i in range(lay.count()):
            if lay.itemAt(i).widget() is not None:
                n += 1
        return n
    widgets_before = count_widgets(layout)
    # add_separator should add a QFrame.HLine.
    result = b.buttons.add_separator()
    assert result is None, "add_separator returns None on Qt path"
    widgets_after = count_widgets(layout)
    assert widgets_after == widgets_before + 1, \
        f"widget count: before={widgets_before} after={widgets_after}"
    # Find the most recently inserted QFrame and verify it's an HLine.
    frames = [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if isinstance(layout.itemAt(i).widget(), QFrame)
        and not isinstance(layout.itemAt(i).widget(), QPushButton)
    ]
    assert frames, "no QFrame separator found"
    assert frames[-1].frameShape() == QFrame.HLine, \
        f"separator frameShape != HLine: {frames[-1].frameShape()}"
    # Adding a follow-up button should still work after the separator.
    b.buttons.add(text="after_sep")
    assert "after_sep" in b.buttons
    plt.close(fig)
    print("add_separator Qt path: HLine separator added, button after it works")


def test_textview_pos_propagates_to_overlay():
    fig = plt.figure()
    tv = dnav.TextView(["initial"], fax=fig, pos="bottom right")
    overlay = tv._overlay
    assert overlay is not None, "expected Qt overlay path"
    initial_pos = overlay._pos
    # DUSTrack-style: mutate TextView._pos directly, then call update.
    from datanavigator.utils import _parse_pos
    tv._pos = _parse_pos("bottom left")
    tv.update()
    assert overlay._pos == tv._pos, \
        f"overlay _pos didn't propagate: {overlay._pos} != {tv._pos}"
    assert overlay._pos != initial_pos
    print(f"TextView._pos propagation: {initial_pos} -> {overlay._pos}")
    plt.close(fig)


def main():
    test_add_separator_qt_path()
    test_textview_pos_propagates_to_overlay()
    print("Phase 4a smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
