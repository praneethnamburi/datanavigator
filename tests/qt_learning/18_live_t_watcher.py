"""
Live probe: when the user presses 't' (or any key) interactively,
print the full mpl event state. Same script can be run in dustrack1a1
(known-working) and dlc (broken) -- the diff between outputs will
isolate the failure mode.

Goal:
  1. Open DLCProject + DUSTrack.
  2. Disconnect every real action_func so nothing fires destructively.
  3. Attach a logger to mpl_connect("key_press_event") that prints,
     for every keypress:
       - event.key (the literal mpl reports)
       - event.inaxes (which axis, or None)
       - event.inaxes is self._ax_image (T/F)
       - event.xdata, event.ydata (data coordinates)
       - event.x, event.y (pixel coordinates)
       - canvas.hasFocus()
       - QApplication.focusWidget() (which widget currently has focus)
  4. Also log every motion_notify_event with the same fields so we
     can see mpl's cursor tracking.

Sequence to run interactively (do this in BOTH dustrack1a1 and dlc envs):
  1. Click on the image area in DUSTrack once.
  2. Hover over the image (no click).
  3. Press 't' (plain, NO shift, NO caps lock).
  4. Hover over the trace area, press 't'.
  5. Close the window.

Compare the two envs' outputs. The line where keyPressEvent fires but
inaxes_is_ax_image=False (when it SHOULD be True) is the smoking gun.

Run:
    conda activate <env>  # dustrack1a1 OR dlc
    python -u C:\\dev\\datanavigator\\tests\\qt_learning\\18_live_t_watcher.py
"""

import os
import sys
import logging
import warnings
import time

os.environ.setdefault("QT_API", "pyside6")
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="shibokensupport.signature.parser")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))


CONFIG = r"M:\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml"


def main():
    from dustrack import DLCProject
    import datanavigator

    print(f"datanavigator: {datanavigator.__version__} ({datanavigator.__file__})")
    print(f"matplotlib   : {matplotlib.__version__}")
    print(f"qtpy api     : {__import__('qtpy').API_NAME}")
    print()

    print("opening DLCProject + annotate()...")
    g = DLCProject(CONFIG)
    ret = g.annotate(video_index=0)
    for _ in range(10):
        app.processEvents()
    print()

    # SAFETY: disconnect every real button slot.
    for name in ret.buttons.names:
        b = ret.buttons[name]
        if hasattr(b, "_qt_btn"):
            try:
                b._qt_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass

    # SAFETY: replace add_annotation, save, remove_annotation with sniffers
    # so nothing fires destructively when keys are pressed.
    for method_name in ("add_annotation", "remove_annotation", "save",
                        "_add_annotation"):
        def make_no_op(name):
            def no_op(*args, **kwargs):
                pass
            no_op.__name__ = f"sniffer_{name}"
            return no_op
        if hasattr(ret, method_name):
            setattr(ret, method_name, make_no_op(method_name))
    # Also clear all keypress dispatch entries so we observe but don't act.
    # But keep the lookup mechanism intact so we can verify what would fire.
    keypress_record = dict(ret._keypressdict)  # save for inspection
    ret._keypressdict = {}  # empty -> nothing actually runs on keypress

    print(f"original key bindings ({len(keypress_record)}):")
    for k in sorted(keypress_record.keys()):
        print(f"  {k}")
    print()

    canvas = ret.figure.canvas

    def fmt_axis(ax):
        if ax is None:
            return "None"
        if ax is ret._ax_image:
            return "ax_IMAGE"
        if ax is ret._ax_trace_x:
            return "ax_TRACE_X"
        if ax is ret._ax_trace_y:
            return "ax_TRACE_Y"
        return f"ax<{id(ax) & 0xffffff:06x}>"

    def on_key(event):
        t = time.strftime("%H:%M:%S")
        fw = QApplication.focusWidget()
        fw_name = type(fw).__name__ if fw else "None"
        would_fire = "WOULD_FIRE" if event.key in keypress_record else "no-binding"
        print(f"[{t}] key={event.key!r:8s} inaxes={fmt_axis(event.inaxes):12s} "
              f"xdata={str(event.xdata)[:7]:8s} ydata={str(event.ydata)[:7]:8s} "
              f"hasFocus={canvas.hasFocus()} focus_widget={fw_name} "
              f"{would_fire}",
              flush=True)

    last_motion_fmt = [""]
    def on_motion(event):
        # Only log when the axis-hit changes -- otherwise too noisy.
        fmt = f"inaxes={fmt_axis(event.inaxes)}"
        if fmt != last_motion_fmt[0]:
            t = time.strftime("%H:%M:%S")
            xy = (str(event.xdata)[:7], str(event.ydata)[:7]) if event.xdata else (None, None)
            print(f"[{t}] motion {fmt:18s} xy={xy}", flush=True)
            last_motion_fmt[0] = fmt

    canvas.mpl_connect("key_press_event", on_key)
    canvas.mpl_connect("motion_notify_event", on_motion)

    print("=" * 70)
    print("READY. Interact with the DUSTrack window:")
    print("  1. CLICK on the IMAGE area.")
    print("  2. HOVER over the image (no click), press PLAIN 't' (no shift).")
    print("  3. HOVER over the trace area, press 't'.")
    print("  4. Try other keys: 'r', 's' (those are bound too).")
    print("  5. Close the window when done.")
    print("=" * 70)
    print()

    plt.show()
    print("\nExiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
