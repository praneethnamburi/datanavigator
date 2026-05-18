"""
Regression smoke for the "Keyboard shortcuts button does nothing" bug.

Phase 4b's TextView Qt path skipped self.setup() (which calls
plt.show(block=False)). That broke any callback that creates a *new*
plt.figure() and renders text on it via TextView -- the new figure
existed but never got shown. Most visible offender:
GenericBrowser.show_key_bindings("new", ...), wired to DUSTrack's
"Keyboard shortcuts" button.

This smoke creates a GenericBrowser under QtAgg, then calls
show_key_bindings("new", ...) and asserts:

  1. A *new* figure was added beyond the original (registered in
     matplotlib's figure manager).
  2. The new figure's QMainWindow is .isVisible() True after a
     processEvents() drain.

Run:
    QT_API=pyqt5 python tests/qt_learning/12_show_key_bindings_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from matplotlib._pylab_helpers import Gcf
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from datanavigator.core import GenericBrowser
from datanavigator._qt import find_qt_window


def main():
    fig = plt.figure()
    b = GenericBrowser(figure_handle=fig)
    b.set_default_keybindings()

    n_before = len(Gcf.get_all_fig_managers())
    print(f"figures before show_key_bindings: {n_before}")

    b.show_key_bindings(f="new", pos="center left")

    for _ in range(5):
        app.processEvents()

    n_after = len(Gcf.get_all_fig_managers())
    print(f"figures after show_key_bindings:  {n_after}")
    assert n_after == n_before + 1, (
        f"expected one new figure registered, got {n_after - n_before}"
    )

    # The new figure is the one b._keybindingtext attached to.
    new_fig = b._keybindingtext.figure
    new_window = find_qt_window(new_fig)
    assert new_window is not None, "new figure has no Qt window"
    print(f"new figure's QMainWindow.isVisible() = {new_window.isVisible()}")
    assert new_window.isVisible(), (
        "new figure's QMainWindow should be visible after show_key_bindings"
    )

    # Overlay should also be present, with the keybinding text in it.
    overlay = b._keybindingtext._overlay
    assert overlay is not None, "TextView should be on Qt overlay path"
    assert overlay._label.isVisible(), "overlay QLabel should be visible"
    # Keybinding text should mention 'left' / 'right' / 'ctrl+k'.
    label_text = overlay._label.text()
    for needle in ("left", "right", "ctrl+k"):
        assert needle in label_text, (
            f"expected keybinding text to mention {needle!r}, got: {label_text!r}"
        )

    print(f"keybinding label has {len(label_text.splitlines())} lines")
    print("show_key_bindings smoke OK")
    plt.close(new_fig)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
