"""
Diagnose: why don't DUSTrack buttons fire callbacks when the user
clicks them in the live UI?

Three-stage test on each button in a real DUSTrack instance:

  (a) Inspect: wrapper class, signal connection count, action_func
      identity, whether the QPushButton is enabled / visible / has
      a non-zero size.
  (b) Programmatic .click(): direct signal emission, same as what
      Phase 3 smoke 05 already covers. Should always fire the slot.
  (c) QTest.mouseClick(): synthesize a real QMouseEvent at the
      button's center. Same path a user's hardware mouse takes.

If (b) fires the callback and (c) doesn't, it's an event-delivery
issue (the toolbar / parent widget is intercepting or the QPushButton
isn't receiving the mouse event). If (b) doesn't fire either, wiring
itself is broken.

Run:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\10_diagnose_buttons.py
"""

import os
import sys
import logging

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import Qt, QtMsgType, qInstallMessageHandler
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication


def _silence(msg_type, _ctx, message):
    if msg_type == QtMsgType.QtWarningMsg:
        for n in ("Cannot find font directory", "does not support propagateSizeHints", "does not support raise"):
            if n in message:
                return
    sys.stderr.write(message + "\n")


qInstallMessageHandler(_silence)
app = QApplication.instance() or QApplication([])

CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


def main():
    from dustrack import DLCProject

    print("opening DLCProject...")
    g = DLCProject(CONFIG)
    ret = g.annotate(video_index=0)
    for _ in range(10):
        app.processEvents()

    print(f"\n{len(ret.buttons.names)} buttons registered:")
    print(f"  {ret.buttons.names}\n")

    # CRITICAL SAFETY: DUSTrack's buttons have real, expensive side
    # effects ("Create DLC Project" creates a project; "Train DLC model"
    # starts a multi-hour training run). We MUST disconnect the real
    # slots before any click, real or synthetic. Replace with a sniffer
    # that just bumps a counter. The DUSTrack instance is thrown away
    # at the end of this script, so the wiring damage is local.
    fired = {name: 0 for name in ret.buttons.names}

    def make_sniffer(name):
        def sniff(*_args):
            fired[name] += 1
        return sniff

    for name in ret.buttons.names:
        b = ret.buttons[name]
        if not hasattr(b, "_qt_btn"):
            continue
        # Drop ALL slots currently connected to .clicked (including any
        # real action_funcs DUSTrack wired up). Then connect only our
        # sniffer. After this, clicking the button -- real or programmatic
        # -- runs ONLY sniff(), never the real action_func.
        try:
            b._qt_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            # PySide6 raises if no slots were connected. Harmless.
            pass
        b._qt_btn.clicked.connect(make_sniffer(name))

    # (a) Inspect
    print("--- (a) Inspect ---")
    for name in ret.buttons.names:
        b = ret.buttons[name]
        qbtn = getattr(b, "_qt_btn", None)
        if qbtn is None:
            print(f"  {name:35s}  NO _qt_btn (mpl path?)  type={type(b).__name__}")
            continue
        print(f"  {name:35s}  wrapper={type(b).__name__:18s} "
              f"enabled={qbtn.isEnabled()} visible={qbtn.isVisible()} "
              f"size={qbtn.size().width()}x{qbtn.size().height()}")

    # (b) Programmatic click via QPushButton.click()
    print("\n--- (b) Programmatic .click() ---")
    before = dict(fired)
    for name in ret.buttons.names:
        b = ret.buttons[name]
        if hasattr(b, "_qt_btn"):
            b._qt_btn.click()
    app.processEvents()
    for name in ret.buttons.names:
        delta = fired[name] - before[name]
        status = "FIRED" if delta else "no-fire"
        print(f"  {name:35s}  {status}")

    # (c) Synthesized mouse event via QTest
    print("\n--- (c) QTest.mouseClick() ---")
    before = dict(fired)
    for name in ret.buttons.names:
        b = ret.buttons[name]
        if hasattr(b, "_qt_btn"):
            # QTest.mouseClick(widget, button, modifier=None, pos=None)
            # Center of the widget is the default.
            QTest.mouseClick(b._qt_btn, Qt.LeftButton)
    app.processEvents()
    for name in ret.buttons.names:
        delta = fired[name] - before[name]
        status = "FIRED" if delta else "no-fire"
        print(f"  {name:35s}  {status}")

    print("\nDone. Close DUSTrack window manually or this process.")
    plt.close(ret.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
