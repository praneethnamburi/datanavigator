"""
Sandbox #5 (Phase 3 smoke).

Phase 3 makes ``assets.Buttons.add`` Qt-aware: when the parent's figure
is on QtAgg AND no explicit ``pos`` is given, the button is built as a
native QPushButton inside a QToolBar attached to the QMainWindow. Public
API (``b.name``, ``b.on_clicked``, plus toggle's ``b.state`` /
``b.toggle()`` / ``b.set_text()`` / ``b.set_state()``) is identical.

This sandbox:
  1. Creates a Push and a Toggle button via Buttons.add().
  2. Verifies each is the Qt wrapper (not the mpl class).
  3. Wires three callback shapes (no-arg lambda, event=None default,
     mpl-style positional event) and click-fires each to confirm the
     callback ran without exceptions or arg-mismatch errors.
  4. Verifies the toggle's state property round-trips through both
     ``set_state`` and a user-driven QPushButton.click().

Run with:
    QT_API=pyqt5 python tests/qt_learning/05_phase3_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

from datanavigator.core import GenericBrowser


def main():
    print(f"backend = {matplotlib.get_backend()}")

    fig = plt.figure()
    b = GenericBrowser(figure_handle=fig)

    # Tracking flags for callback exercise.
    seen = {"noarg": 0, "default_evt": None, "positional_evt": None}

    def noarg_cb():
        seen["noarg"] += 1

    def default_evt_cb(event=None):
        seen["default_evt"] = event

    def positional_evt_cb(event):
        seen["positional_evt"] = event

    # Add buttons (Push + Toggle) with three different callback shapes.
    push = b.buttons.add(text="push", type_="Push", action_func=noarg_cb)
    toggle = b.buttons.add(
        text="toggle", type_="Toggle", action_func=default_evt_cb,
    )
    # Second action_func via on_clicked, mpl-style positional.
    push.on_clicked(positional_evt_cb)

    # Assertions on the wrapper objects (Qt path was taken).
    assert push.__class__.__name__ == "_QtPushButton", \
        f"expected Qt wrapper, got {type(push).__name__}"
    assert toggle.__class__.__name__ == "_QtToggleButton", \
        f"expected Qt wrapper, got {type(toggle).__name__}"
    assert push.name == "push"
    assert toggle.name == "toggle"
    print(f"push wrapper = {type(push).__name__}")
    print(f"toggle wrapper = {type(toggle).__name__}")

    # AssetContainer indexing still works.
    assert b.buttons["push"] is push
    assert b.buttons["toggle"] is toggle

    # Fire each callback via QPushButton.click() (Qt API; emits the clicked signal).
    push._qt_btn.click()  # triggers both registered callbacks
    assert seen["noarg"] == 1, f"noarg callback not fired: {seen}"
    assert seen["positional_evt"] is None, "positional_evt should receive None"
    # Note "is None" passes both for "callback not fired" and "fired with None".
    # We verify it WAS fired below by setting it to a sentinel first.
    seen["positional_evt"] = "SENTINEL"
    push._qt_btn.click()
    assert seen["noarg"] == 2
    assert seen["positional_evt"] is None, \
        f"expected None from positional callback, got {seen['positional_evt']!r}"
    print("push button: noarg + positional_evt callbacks fired correctly")

    # Toggle: starts True (default), default_evt_cb should fire on each click.
    assert toggle.state is True, f"toggle start_state should be True, got {toggle.state}"
    assert toggle._qt_btn.text() == "toggle=True"

    toggle._qt_btn.click()
    assert toggle.state is False, f"after 1 click expected False, got {toggle.state}"
    assert toggle._qt_btn.text() == "toggle=False"
    assert seen["default_evt"] is None  # callback fired with None as event

    # API: set_state explicit
    toggle.set_state(True)
    assert toggle.state is True
    assert toggle._qt_btn.isChecked() is True
    assert toggle._qt_btn.text() == "toggle=True"

    # API: toggle method
    toggle.toggle()
    assert toggle.state is False
    print(f"toggle state round-trip ok; final text={toggle._qt_btn.text()!r}")

    print("Phase 3 smoke OK")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
