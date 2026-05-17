"""
Sandbox #6 (Phase 4 -- downstream consumer smoke).

Verifies the 1.4.0 Qt refactor is genuinely API-unnoticeable to the
real consumers: datanavigator's own examples (ButtonDemo, SelectorDemo,
EventPickerDemo), pn-utilities' re-export shim (pntools.gui), and the
DUSTrack package (canonical downstream).

The Tremor01Editor / Tremor02Editor in pn-projects/projects/wobble are
NOT importable as a package from anywhere we ship (pn-projects has no
pyproject.toml / setup.py), so they need a manual smoke test --
documented at the bottom of this script's output.

Run with:
    QT_API=pyqt5 QT_QPA_PLATFORM=offscreen python tests/qt_learning/06_downstream_smoke.py
"""

import os
import sys

# Offscreen Qt: Qt windows are created and laid out but never shown,
# so the script runs cleanly in a headless / unattended session.
os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

# Make sure we're testing against THIS branch's datanavigator, not the
# 1.3.0 PyPI install in b4.
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))
import datanavigator as dnav  # noqa: E402


def check(name: str, fn) -> bool:
    """Run fn() with name banner; report pass/fail."""
    print(f"--- {name} ---")
    try:
        fn()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  FAIL: {type(e).__name__}: {e}")
        return False
    print("  ok")
    return True


def test_examples_ButtonDemo():
    # Lifecycle gotcha: ButtonDemo subclasses plt.Figure and calls
    # buttons.add() inside Figure.__init__, BEFORE matplotlib attaches
    # a Qt canvas. find_qt_window correctly returns None at that point
    # and the mpl path runs. The buttons still work; they just miss the
    # Qt path perf win. Documented in assets.Buttons.add() docstring.
    from datanavigator.assets import Button, ToggleButton
    fig = plt.figure(FigureClass=dnav.ButtonDemo)
    assert "test" in fig.buttons
    assert "push button" in fig.buttons
    toggle = fig.buttons["test"]
    push = fig.buttons["push button"]
    # Expect the mpl path here, NOT the Qt path -- this is the canvas-attach
    # timing edge case for Figure subclasses with buttons in __init__.
    assert isinstance(push, Button), f"expected mpl Button, got {type(push).__name__}"
    assert isinstance(toggle, ToggleButton), \
        f"expected mpl ToggleButton, got {type(toggle).__name__}"
    # And confirm the buttons still functionally work via the mpl on_clicked:
    initial_state = toggle.state
    toggle.toggle()
    assert toggle.state != initial_state, "toggle.toggle() didn't flip state"
    plt.close(fig)


def test_examples_SelectorDemo():
    demo = dnav.SelectorDemo()
    assert type(demo.buttons["Start selection"]).__name__ == "_QtPushButton"
    assert type(demo.buttons["Stop selection"]).__name__ == "_QtPushButton"
    # Fire both buttons to exercise the LassoSelector start/stop path.
    demo.buttons["Stop selection"]._qt_btn.click()
    demo.buttons["Start selection"]._qt_btn.click()
    plt.close(demo.ax.figure)


def test_examples_EventPickerDemo():
    # EventPickerDemo extends SignalBrowser. SignalBrowser does NOT call
    # memoryslots.show() by default, so demo.memoryslots._memtext is None
    # -- we only verify the buttons SignalBrowser adds ("Auto limits").
    demo = dnav.EventPickerDemo()
    assert "Auto limits" in demo.buttons
    for name in demo.buttons.names:
        b = demo.buttons[name]
        assert type(b).__name__.startswith("_Qt"), (
            f"button {name!r} on mpl path under QtAgg: {type(b).__name__}"
        )
    print(f"  EventPickerDemo buttons (all Qt-path): {demo.buttons.names}")
    # Also confirm event-picker added events properly:
    assert len(demo.events) > 0, "EventPickerDemo expected to register events"
    plt.close(demo.figure)


def test_pntools_gui_shim():
    # pntools.gui does `from datanavigator import *`. Verify the shim still
    # exposes the public surface unchanged.
    import pntools.gui as gui
    for name in ("GenericBrowser", "SignalBrowser", "PlotBrowser",
                 "VideoBrowser", "Button", "ToggleButton", "Buttons",
                 "VideoAnnotation"):
        assert hasattr(gui, name), f"pntools.gui re-export missing {name!r}"
    # Verify the re-exported GenericBrowser IS the one from 1.4.0-qt
    # (has the Phase 1 _qt_window attribute).
    fig = plt.figure()
    b = gui.GenericBrowser(figure_handle=fig)
    assert hasattr(b, "_qt_window"), \
        "pntools.gui.GenericBrowser missing _qt_window -- shim shadowed local source?"
    assert b._qt_window is not None, \
        "running under QtAgg but _qt_window is None"
    plt.close(fig)


def test_dustrack_import():
    # Don't actually instantiate dustrack -- it needs a video. Just verify
    # it imports cleanly against this branch's datanavigator (the 1.4.0-qt
    # changes didn't break any symbol DUSTrack depends on).
    import dustrack
    # DUSTrack's main class:
    assert hasattr(dustrack, "DUSTrack"), "dustrack.DUSTrack symbol missing"
    # Verify it's pulling from local datanavigator (the one with _qt_window):
    import datanavigator
    assert hasattr(datanavigator.GenericBrowser, "_qt_window") is False  # class attr
    # _qt_window is an instance attribute, so check on a fresh instance:
    fig = plt.figure()
    b = datanavigator.GenericBrowser(figure_handle=fig)
    assert hasattr(b, "_qt_window"), "wrong datanavigator on sys.path"
    plt.close(fig)


def test_immersionlab_import():
    # immersionlab uses pntools and datanavigator transitively. Quick
    # import smoke to catch any namespace breakage.
    import immersionlab  # noqa: F401


def main():
    results = []
    for name, fn in [
        ("examples.ButtonDemo", test_examples_ButtonDemo),
        ("examples.SelectorDemo", test_examples_SelectorDemo),
        ("examples.EventPickerDemo", test_examples_EventPickerDemo),
        ("pntools.gui re-export shim", test_pntools_gui_shim),
        ("dustrack import + datanavigator path", test_dustrack_import),
        ("immersionlab import", test_immersionlab_import),
    ]:
        results.append((name, check(name, fn)))

    print()
    print("=" * 60)
    print("Downstream consumer smoke summary")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print()
    print("Manual smoke still required (not auto-testable here):")
    print("  - pn-projects: Tremor01Editor / Tremor02Editor under")
    print("    C:/dev/pn-projects/projects/wobble/__init__.py:2494, 2582")
    print("    (pn-projects has no installable package; run from notebooks)")
    print("  - DUSTrack interactive: open a video, scrub frames, exercise")
    print("    point annotation. Verify perceived responsiveness.")

    all_pass = all(ok for _, ok in results)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
