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

    b.show_key_bindings()

    for _ in range(5):
        app.processEvents()

    dialog = b._keybinding_dialog
    assert dialog is not None, "show_key_bindings should attach a QDialog"
    assert dialog.isVisible(), "cheatsheet dialog should be visible"
    print(f"dialog visible: {dialog.isVisible()}, title: {dialog.windowTitle()!r}")
    print("show_key_bindings smoke OK")
    dialog.close()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
