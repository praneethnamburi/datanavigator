"""
Sandbox #7 (Phase 4a smoke).

Covers the two API additions/fixes that came out of the first DUSTrack
external smoke:

  1. Buttons.add_separator() -- promoted to a first-class API. On Qt it
     calls QToolBar.addSeparator(); on mpl it inserts an invisible
     spacer button to occupy a layout slot.
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
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.add(text="first")
    # Toolbar should exist now (lazily created by first add).
    toolbar = b._qt_window._dnav_buttons_toolbar
    actions_before = len(toolbar.actions())
    # add_separator should add a QAction marked isSeparator()
    result = b.buttons.add_separator()
    assert result is None, "add_separator returns None on Qt path"
    actions_after = len(toolbar.actions())
    assert actions_after == actions_before + 1, \
        f"toolbar action count: before={actions_before} after={actions_after}"
    last_action = toolbar.actions()[-1]
    assert last_action.isSeparator(), "last action should be a separator"
    # Adding a follow-up button should still work after the separator.
    b.buttons.add(text="after_sep")
    assert "after_sep" in b.buttons
    plt.close(fig)
    print("add_separator Qt path: separator added, button after it works")


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
