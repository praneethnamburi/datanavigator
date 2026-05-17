"""
Interactive focus / key-event probe.

Run this in dlc env, then interact with the window:

  1. Hover the cursor over the plot area -- doesn't print anything by
     itself.
  2. Click on the plot area once.
  3. Press 'z', 'left', 'right', or any key.
  4. Click on a button in the toolbar.
  5. Hover back over the plot, press 'z' again.
  6. Close the window.

Throughout, the script:
  - Watches QApplication.focusChanged -- prints when focus moves
    between widgets, and which widget gained focus.
  - Watches the canvas's keyPressEvent (via subclassing the canvas) --
    prints every Qt-level key press the canvas receives.
  - Watches mpl_connect("key_press_event") -- prints every mpl-level
    key event the handler is given (with .xdata / .ydata).

This produces the data we need to know exactly which step is failing
in the OS -> Qt -> matplotlib path. Possible findings:

  A. Click sets focus, key press reaches Qt canvas, mpl handler fires
     normally -- in which case the bug is elsewhere.
  B. Click sets focus, key press reaches Qt canvas, but mpl handler
     never sees it -- matplotlib's key dispatch is broken.
  C. Click sets focus, but key press doesn't reach Qt canvas --
     something is eating the key event upstream.
  D. Click does NOT transfer focus to canvas -- the canvas's focus
     policy is somehow ineffective.

Run:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe -u \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\15_interactive_focus_probe.py
"""

import os
import sys
import time
import logging

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import QtMsgType, qInstallMessageHandler
from qtpy.QtWidgets import QApplication


def _silence(msg_type, _ctx, message):
    if msg_type == QtMsgType.QtWarningMsg:
        for n in ("Cannot find font directory",
                  "does not support propagateSizeHints",
                  "does not support raise"):
            if n in message:
                return
    sys.stderr.write(message + "\n")


qInstallMessageHandler(_silence)
app = QApplication.instance() or QApplication([])


def main():
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"qtpy api  : {__import__('qtpy').API_NAME}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([1, 2, 3, 4, 5], [2, 4, 3, 5, 4])
    ax.set_title("Interactive focus probe -- hover, click, press keys")

    # Monkey-patch the canvas's keyPressEvent to log every Qt key event.
    canvas = fig.canvas
    orig_keyPressEvent = canvas.keyPressEvent

    def logged_keyPressEvent(event):
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] Qt-canvas keyPressEvent  key={event.key():#x} "
              f"text={event.text()!r} hasFocus={canvas.hasFocus()}",
              flush=True)
        return orig_keyPressEvent(event)

    canvas.keyPressEvent = logged_keyPressEvent

    # mpl_connect("key_press_event") -- the matplotlib-level dispatch.
    def on_mpl_key(event):
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] mpl key_press_event     key={event.key!r} "
              f"xdata={event.xdata} ydata={event.ydata} "
              f"inaxes={event.inaxes is not None}",
              flush=True)

    canvas.mpl_connect("key_press_event", on_mpl_key)

    # Focus changes globally -- print whenever focus moves.
    def on_focus_changed(old, new):
        t = time.strftime("%H:%M:%S")
        old_name = type(old).__name__ if old is not None else "None"
        new_name = type(new).__name__ if new is not None else "None"
        print(f"[{t}] focusChanged: {old_name} -> {new_name}", flush=True)

    app.focusChanged.connect(on_focus_changed)

    # Also dump initial canvas focus state.
    print(f"\ncanvas.focusPolicy()  = {canvas.focusPolicy()}")
    print(f"canvas.isVisible()    = {canvas.isVisible()}")

    print("\n" + "=" * 60)
    print("READY. Try the following sequence:")
    print("  1. Hover over the plot. Press 'z'. -> watch output.")
    print("  2. Click on the plot area. -> focusChanged should print.")
    print("  3. Press 'z'. -> Qt-canvas keyPressEvent AND mpl key_press_event")
    print("     should both fire.")
    print("  4. Hover over the title bar / window chrome. Press 'z' again.")
    print("  5. Close the window when done.")
    print("=" * 60 + "\n")

    plt.show()
    print("\nWindow closed. Exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
