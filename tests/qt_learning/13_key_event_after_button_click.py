"""
Regression smoke for the "hover-over-plot + key-press stops working
after button click" bug.

The pre-1.4 mpl Button widgets lived inside matplotlib axes; they
couldn't take Qt keyboard focus, so the canvas always had the focus
and ``mpl_connect("key_press_event", ...)`` handlers fired on every
keypress (with ``.xdata``/``.ydata`` populated from the last cursor
position).

Phase 3's QPushButton replacement inherited Qt's default StrongFocus,
which meant the FIRST button click moved focus to the button and
silently broke every key binding. Reported by DUSTrack interactive
test: pressing "z" while hovering over a plot stopped working.

Fix: ``self._qt_btn.setFocusPolicy(Qt.NoFocus)`` in _QtPushButton.

This smoke verifies the fix end-to-end:

  1. Build a GenericBrowser with a key binding for "z" (set a flag).
  2. Add a Qt-path button and click it -- this would steal focus
     pre-fix.
  3. Send a Qt-level keyPressEvent for "z" to the canvas.
  4. Assert the mpl key_press_event handler ran -- i.e., the
     canvas still has focus.

Run:
    QT_API=pyqt5 python tests/qt_learning/13_key_event_after_button_click.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import Qt
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from datanavigator.core import GenericBrowser


def main():
    fig = plt.figure()
    b = GenericBrowser(figure_handle=fig)

    # Register a key binding for "z" that bumps a counter so we can
    # tell whether the mpl handler fired.
    z_fires = [0]

    def on_z():
        z_fires[0] += 1

    b.add_key_binding("z", on_z, description="bump counter")

    # The browser uses self.figure.canvas.mpl_connect to route key
    # events through GenericBrowser.__call__, which dispatches to
    # entries in self._keypressdict. So an mpl key_press_event for
    # "z" should land in on_z.

    # Add a Qt-path button and click it. Pre-fix, this would move
    # keyboard focus to the QPushButton and break every key binding.
    push = b.buttons.add(text="probe", action_func=lambda: None)
    assert type(push).__name__ == "_QtPushButton", (
        f"expected _QtPushButton on the Qt path, got {type(push).__name__}"
    )

    # The actual focus-policy check is the load-bearing assertion:
    assert push._qt_btn.focusPolicy() == Qt.NoFocus, (
        f"button has focusPolicy={push._qt_btn.focusPolicy()}, "
        "expected Qt.NoFocus -- the fix didn't take"
    )

    # Show the QMainWindow -- Qt won't accept focus on a hidden widget.
    b._qt_window.show()
    fig.canvas.setFocus()
    for _ in range(5):
        app.processEvents()
    assert fig.canvas.hasFocus(), (
        f"canvas should start with focus; got focusWidget="
        f"{QApplication.focusWidget()}"
    )

    # Click the button. Under StrongFocus this would transfer focus
    # to the button. Under NoFocus the canvas keeps focus.
    push._qt_btn.click()
    for _ in range(3):
        app.processEvents()

    canvas_kept_focus = fig.canvas.hasFocus()
    print(f"canvas.hasFocus() after button click = {canvas_kept_focus}")

    # Synthesize a Qt-level key press for "z" on the canvas. This is
    # the path a real user's keystroke takes: Windows -> Qt event ->
    # canvas keyPressEvent -> matplotlib KeyEvent -> our __call__.
    QTest.keyClick(fig.canvas, Qt.Key_Z)
    for _ in range(3):
        app.processEvents()

    print(f"on_z fire count after Qt key event = {z_fires[0]}")
    assert z_fires[0] == 1, (
        f"expected on_z to fire once after QTest.keyClick, got {z_fires[0]}. "
        "If 0, the canvas lost focus to the button (focus-stealing bug). "
        "If >1, the event multiplied somewhere."
    )

    print("Key-event-after-button-click smoke OK")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
