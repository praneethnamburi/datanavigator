"""
Live, interactive probe for the DUSTrack button-callback bug.

Run this script from a regular shell (not background). It opens
DLCProject, calls annotate(), and attaches an ADDITIONAL sniffer slot
to each button that prints a timestamped line whenever the button's
.clicked signal fires. The real action_funcs stay connected and
unchanged -- the sniffer is additive.

You (the user) then click buttons with your real mouse, watch the
console, and tell me:

  - "Keyboard shortcuts" -> sniffer prints + new window opens   = healthy
  - "Keyboard shortcuts" -> sniffer prints, no window           = wiring fine,
                                                                  action_func
                                                                  failing silently
  - "Keyboard shortcuts" -> sniffer does NOT print              = real mouse
                                                                  events not
                                                                  reaching the
                                                                  QPushButton

Close the DUSTrack window when done; the script exits.

Run:
    C:\\Users\\praneeth\\anaconda3\\envs\\dlc\\python.exe \\
        C:\\dev\\datanavigator\\tests\\qt_learning\\11_live_button_probe.py
"""

import os
import sys
import logging
import time

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


CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


def main():
    from dustrack import DLCProject

    print("opening DLCProject...")
    g = DLCProject(CONFIG)
    print("calling annotate() (opens DUSTrack UI)...")
    ret = g.annotate(video_index=0)
    for _ in range(5):
        app.processEvents()

    print("\nAttaching sniffer to each button (additive; real callbacks "
          "remain connected).")
    print("Buttons to test:")
    for name in ret.buttons.names:
        print(f"  - {name}")

    def make_sniffer(name):
        def sniff(*args):
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] SNIFFER FIRED: {name!r} (signal args={args})",
                  flush=True)
        return sniff

    for name in ret.buttons.names:
        b = ret.buttons[name]
        if hasattr(b, "_qt_btn"):
            b._qt_btn.clicked.connect(make_sniffer(name))

    print("\n" + "=" * 60)
    print("READY. Click buttons in the DUSTrack window with your mouse.")
    print("Each click should print: [HH:MM:SS] SNIFFER FIRED: '<name>' ...")
    print("Close the DUSTrack window when done to exit.")
    print("=" * 60)
    print()

    # Use matplotlib's plt.show() which enters the Qt event loop and
    # blocks until the figure is closed. The sniffer runs as a normal
    # Qt slot whenever .clicked fires from any source (mouse or other).
    plt.show()
    print("\nDUSTrack window closed. Exiting probe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
