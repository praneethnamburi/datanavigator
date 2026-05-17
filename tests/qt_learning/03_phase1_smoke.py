"""
Sandbox #3 (Phase 1 smoke test).

Verifies the central claim of Phase 1: when matplotlib is on the QtAgg
backend, every GenericBrowser instance discovers and holds a reference to
the QMainWindow matplotlib already built around its figure -- the
foundation Phase 2+ uses to attach native Qt widgets.

Run with:
    QT_API=pyqt5 python tests/qt_learning/03_phase1_smoke.py

Expected output:
    backend = QtAgg
    figure.canvas type = FigureCanvasQTAgg
    b._qt_window type = <class '...QMainWindow' or subclass>
    Phase 1 smoke OK
"""

import sys
import os

# Order matters: matplotlib.use() must be called before any pyplot import,
# and qtpy must be importable to pick a Qt binding. In real datanavigator
# use the user (or the env) chooses the backend; this script does it
# explicitly so the smoke is self-contained.
os.environ.setdefault("QT_API", "pyqt5")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication, QMainWindow

# We need a QApplication before creating a Qt-backed figure, otherwise
# matplotlib's FigureManagerQT will create one for us. Creating it
# explicitly here gives us control over its lifetime.
app = QApplication.instance() or QApplication([])

from datanavigator.core import GenericBrowser


def main():
    print(f"backend = {matplotlib.get_backend()}")

    fig = plt.figure()
    print(f"figure.canvas type = {type(fig.canvas).__name__}")

    b = GenericBrowser(figure_handle=fig)
    print(f"b._qt_window type = {type(b._qt_window).__name__}")

    assert b._qt_window is not None, "expected a QMainWindow, got None"
    assert isinstance(b._qt_window, QMainWindow), (
        f"expected a QMainWindow subclass, got {type(b._qt_window).__name__}"
    )

    # Phase 2+ will attach widgets here. For smoke, just show the window
    # exists and is functional by toggling the title.
    b._qt_window.setWindowTitle("Phase 1 smoke -- datanavigator 1.4.0-qt")

    print("Phase 1 smoke OK")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
