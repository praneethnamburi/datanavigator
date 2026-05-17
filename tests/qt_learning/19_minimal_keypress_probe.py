"""
Minimal interactive keypress probe -- bare matplotlib + qtpy only.

Goal: isolate whether the "hover + 't' stops working" symptom comes from
matplotlib / Qt / OS at the env level, or from datanavigator's wrapping.

NO datanavigator / dustrack / dlc imports. Runs in any env with
matplotlib + qtpy + a Qt binding installed. That means dustrack1a1
(decord-era, datanavigator 1.1.4) can run it without dependency
errors -- decord is never imported.

What it does:
  - Creates a 2x2 figure (mimicking DUSTrack's multi-axes layout).
  - Each subplot draws a different curve so you can visually
    identify them.
  - Registers an mpl key_press_event handler that logs every keypress:
      event.key (with shift state visible)
      event.inaxes (which subplot, or None)
      event.xdata, event.ydata
      canvas.hasFocus()
      QApplication.focusWidget() type
  - Also logs motion_notify_event axis crossings.
  - Blocks on plt.show() until you close the window.

Sequence to run in BOTH envs:
  1. CLICK on one of the subplots to give the canvas focus.
  2. HOVER over the top-left subplot (no click). Press 't'.
  3. HOVER over the top-right. Press 't'.
  4. HOVER over the bottom-left. Press 't'.
  5. HOVER outside the axes (in the figure margin). Press 't'.
  6. Try other keys: 'a', 'z', 'left', 'right'.
  7. Close the window.

If both envs show the same output for the same sequence, the bug is
NOT in matplotlib / Qt / focus / event delivery -- it's in
datanavigator's setup or in DUSTrack's wrapping. Then we look there.

If outputs differ (e.g., dustrack1a1 shows key='t' inaxes=ax0 and dlc
shows key='t' inaxes=None), the bug is at the matplotlib / Qt event
delivery layer in that env.

Run:
    conda activate <env>
    python -u C:\\dev\\datanavigator\\tests\\qt_learning\\19_minimal_keypress_probe.py
"""

import os
import sys
import time
import logging
import warnings

# Make this script's env contract minimal: only mpl + qtpy.
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="shibokensupport.signature.parser")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication


def main():
    import qtpy
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"qtpy api  : {qtpy.API_NAME}")
    print(f"python    : {sys.version.split()[0]}")
    print()

    app = QApplication.instance() or QApplication([])

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    axes = axes.flatten()
    titles = ["ax0 top-left", "ax1 top-right", "ax2 bottom-left", "ax3 bottom-right"]
    for ax, title, n in zip(axes, titles, range(4)):
        ax.plot([1, 2, 3, 4, 5], [n + i for i in range(5)])
        ax.set_title(title)
    fig.suptitle("Hover over each subplot and press 't' / 'a' / 'z'")
    fig.tight_layout()

    canvas = fig.canvas

    def fmt_axis(ax):
        if ax is None:
            return "None"
        for i, target in enumerate(axes):
            if ax is target:
                return f"ax{i}"
        return f"ax<{id(ax) & 0xffffff:06x}>"

    last_axis = [None]

    def on_key(event):
        t = time.strftime("%H:%M:%S")
        fw = QApplication.focusWidget()
        fw_name = type(fw).__name__ if fw else "None"
        xy = f"({event.xdata}, {event.ydata})"
        print(f"[{t}] KEY    key={event.key!r:10s} inaxes={fmt_axis(event.inaxes):8s} "
              f"xy={xy:30s} hasFocus={canvas.hasFocus()} fw={fw_name}",
              flush=True)

    def on_motion(event):
        # Only log when axis hit changes, to avoid noise.
        ax = event.inaxes
        if ax is not last_axis[0]:
            t = time.strftime("%H:%M:%S")
            print(f"[{t}] motion inaxes={fmt_axis(ax)} "
                  f"(was {fmt_axis(last_axis[0])})", flush=True)
            last_axis[0] = ax

    canvas.mpl_connect("key_press_event", on_key)
    canvas.mpl_connect("motion_notify_event", on_motion)

    print("=" * 70)
    print("READY. Interact with the figure:")
    print("  1. Click on one of the subplots.")
    print("  2. Hover over each subplot in turn, press 't' each time.")
    print("  3. Try 'a', 'z', 'left', 'right' too.")
    print("  4. Close the window when done.")
    print("=" * 70)
    print()

    plt.show()
    print("\nExiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
