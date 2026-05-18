"""
Investigation script for "hover + key press stopped working".

User reports: under 1.4.0-qt (with Phase 4e fix), hovering over a
plot and pressing a key doesn't trigger the registered handler.
ALSO not working on master. Need to figure out where this regressed.

This script probes three configurations on the SAME matplotlib /
Qt stack:

  (a) Bare matplotlib (no datanavigator) -- baseline: does mpl
      key_press_event fire at all when the canvas has focus and a
      Qt-level keyPressEvent is delivered?
  (b) Whatever datanavigator is on sys.path -- does GenericBrowser's
      key dispatch still see the event after wrapping the canvas?
  (c) The "hover" angle -- if focus is NOT explicitly set (just a
      mouse move event), does the key event reach the handler?

Run:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\14_investigate_hover_key.py
"""

import os
import sys
import logging

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import QPoint, Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


def banner(text):
    print()
    print("=" * 60)
    print(text)
    print("=" * 60)


def main():
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"backend   : {matplotlib.get_backend()}")
    import qtpy
    print(f"qtpy api  : {qtpy.API_NAME}")

    # ------------------------------------------------------------------
    # (a) Bare matplotlib -- no datanavigator wrapping anywhere
    # ------------------------------------------------------------------
    banner("(a) Bare matplotlib + mpl_connect key_press_event")
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4])

    fired_a = []

    def on_key_a(event):
        fired_a.append((event.key, event.xdata, event.ydata, event.inaxes))

    cid = fig.canvas.mpl_connect("key_press_event", on_key_a)
    fig.canvas.draw()
    fig.show()
    for _ in range(5):
        app.processEvents()

    print(f"canvas size      : {fig.canvas.size().width()}x{fig.canvas.size().height()}")
    print(f"canvas visible   : {fig.canvas.isVisible()}")
    print(f"canvas focusPol  : {fig.canvas.focusPolicy()}")

    # Variant 1: no setFocus, no mouse move -- raw key event to canvas.
    QTest.keyClick(fig.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()
    print(f"after raw keyClick (no focus, no hover): fired={len(fired_a)}")

    # Variant 2: simulate mouse-enter via QTest.mouseMove (just synthetic
    # cursor position over the canvas center) and try again. This is
    # closest to "hover".
    center = QPoint(fig.canvas.width() // 2, fig.canvas.height() // 2)
    QTest.mouseMove(fig.canvas, center)
    for _ in range(3):
        app.processEvents()
    QTest.keyClick(fig.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()
    print(f"after mouseMove + keyClick (hover-like): fired={len(fired_a)}")
    print(f"after mouseMove, canvas.hasFocus()      : {fig.canvas.hasFocus()}")

    # Variant 3: explicit setFocus + keyClick. This is what my smoke 13 does.
    fig.canvas.setFocus()
    for _ in range(3):
        app.processEvents()
    print(f"after setFocus, canvas.hasFocus()       : {fig.canvas.hasFocus()}")
    QTest.keyClick(fig.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()
    print(f"after setFocus + keyClick               : fired={len(fired_a)}")

    fig.canvas.mpl_disconnect(cid)
    plt.close(fig)

    # ------------------------------------------------------------------
    # (b) datanavigator GenericBrowser
    # ------------------------------------------------------------------
    banner("(b) datanavigator GenericBrowser + add_key_binding")
    import datanavigator
    print(f"datanavigator source: {datanavigator.__file__}")

    fig2 = plt.figure()
    b = datanavigator.GenericBrowser(figure_handle=fig2)
    fired_b = []
    b.add_key_binding("z", lambda: fired_b.append("z"))
    fig2.show()
    for _ in range(5):
        app.processEvents()

    print(f"canvas focusPol  : {fig2.canvas.focusPolicy()}")
    fig2.canvas.setFocus()
    for _ in range(3):
        app.processEvents()
    print(f"canvas.hasFocus(): {fig2.canvas.hasFocus()}")

    QTest.keyClick(fig2.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()
    print(f"after setFocus + keyClick: fired={len(fired_b)}")

    # Now simulate hover only (no setFocus) -- which is what user does.
    # Move focus elsewhere first to make the test meaningful.
    fig2.canvas.clearFocus()
    for _ in range(3):
        app.processEvents()
    print(f"after clearFocus, canvas.hasFocus(): {fig2.canvas.hasFocus()}")
    QTest.mouseMove(fig2.canvas, center)
    for _ in range(3):
        app.processEvents()
    print(f"after mouseMove, canvas.hasFocus(): {fig2.canvas.hasFocus()}")
    QTest.keyClick(fig2.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()
    print(f"after hover-only keyClick : fired={len(fired_b)} "
          f"(expected: stays at 1 if matplotlib doesn't auto-focus on hover; "
          f"2 if it does)")

    plt.close(fig2)

    banner("Summary")
    print(f"matplotlib version       : {matplotlib.__version__}")
    print(f"(a) bare mpl, hover only : fired={fired_a[:1] if len(fired_a) >= 1 else 'no'}")
    print(f"(a) bare mpl, total fires: {len(fired_a)}")
    print(f"(b) dnav GenericBrowser  : fired={len(fired_b)} z presses recorded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
