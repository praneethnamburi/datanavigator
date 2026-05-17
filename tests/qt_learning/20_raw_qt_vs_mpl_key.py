"""
Side-by-side probe of (raw Qt keyPressEvent) vs (matplotlib's
interpretation). Same env, same keystroke, two views.

User has reported behavior inversion between PyQt6 (dustrack1a1) and
PySide6 (dlc) for the same physical keystrokes -- 't' reports as 'T',
shift+t reports as 't', etc. This script reveals whether Qt itself is
reporting weird modifier state or whether matplotlib is misinterpreting
Qt's event in the PySide6 path.

For each keystroke, it logs TWO lines:

  RAW Qt:  key int, text, modifiers flags, modifier names, native
           scan code, native virtual key. This is exactly what Qt
           sees from the OS, no interpretation.

  mpl:     event.key (matplotlib's interpreted string).

Compare:
  - dustrack1a1 (PyQt6) vs dlc (PySide6)
  - Press the SAME physical sequence in each, see how Qt + mpl diverge.

Suggested sequence (press these in order with Caps Lock visibly OFF):
  1. plain t
  2. shift+t
  3. plain a
  4. shift+a
  5. Caps Lock ON, plain t
  6. Caps Lock ON, shift+t
  7. Caps Lock OFF, then close window

Run:
    conda activate <env>
    python -u C:\\dev\\datanavigator\\tests\\qt_learning\\20_raw_qt_vs_mpl_key.py
"""

import os
import sys
import time
import logging
import warnings

logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        module="shibokensupport.signature.parser")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication


def _mods_to_int(mods):
    """int(Qt.KeyboardModifier) works on PyQt6 but not PySide6 6.4.x.
    PySide6 needs .value. This wrapper tries both."""
    if hasattr(mods, "value"):
        return int(mods.value)
    try:
        return int(mods)
    except TypeError:
        return -1  # unknown


def _modifier_names(mods):
    """Decode Qt.KeyboardModifier flags into readable names."""
    mods_int = _mods_to_int(mods)
    flags = []
    for name in ("ShiftModifier", "ControlModifier", "AltModifier",
                 "MetaModifier", "KeypadModifier", "GroupSwitchModifier"):
        flag = getattr(Qt.KeyboardModifier, name, None)
        if flag is None:
            flag = getattr(Qt, name, None)
        if flag is None:
            continue
        flag_int = _mods_to_int(flag)
        if mods_int & flag_int:
            flags.append(name.replace("Modifier", ""))
    return ",".join(flags) if flags else "None"


def main():
    import qtpy
    print(f"matplotlib: {matplotlib.__version__}")
    print(f"qtpy api  : {qtpy.API_NAME}")
    print(f"python    : {sys.version.split()[0]}")

    app = QApplication.instance() or QApplication([])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([1, 2, 3], [1, 2, 3])
    ax.set_title("Press: plain t, shift+t, plain a, shift+a (twice each)")
    canvas = fig.canvas

    # Subclass-style monkeypatch: wrap keyPressEvent so we see the raw
    # Qt event before mpl handles it.
    orig_keyPressEvent = canvas.keyPressEvent
    raw_log = []

    def wrapped_keyPressEvent(event):
        t = time.strftime("%H:%M:%S")
        key_int = event.key()
        if hasattr(key_int, "value"):  # PySide6 returns Qt.Key enum
            key_int = key_int.value
        raw = {
            "key_int": key_int,
            "key_hex": f"0x{key_int:x}",
            "text_repr": repr(event.text()),
            "mods_int": _mods_to_int(event.modifiers()),
            "mods_names": _modifier_names(event.modifiers()),
            "scan_code": event.nativeScanCode(),
            "virtual_key": event.nativeVirtualKey(),
            "is_auto_repeat": event.isAutoRepeat(),
        }
        raw_log.append(raw)
        print(f"[{t}] RAW Qt  key={raw['key_hex']:>10s}  "
              f"text={raw['text_repr']:5s}  "
              f"mods={raw['mods_names']:18s}  "
              f"scan={raw['scan_code']:>3d}  vk={raw['virtual_key']:>3d}",
              flush=True)
        return orig_keyPressEvent(event)

    canvas.keyPressEvent = wrapped_keyPressEvent

    def on_mpl_key(event):
        t = time.strftime("%H:%M:%S")
        print(f"[{t}] mpl     key={event.key!r:>15s}",
              flush=True)

    canvas.mpl_connect("key_press_event", on_mpl_key)

    print()
    print("=" * 70)
    print("CHECK YOUR CAPS LOCK FIRST. Then press the following IN ORDER:")
    print("  1. plain t")
    print("  2. shift+t")
    print("  3. plain a")
    print("  4. shift+a")
    print("Then (optional): toggle Caps Lock and repeat to test that state.")
    print("Close the window when done.")
    print("=" * 70)
    print()

    plt.show()
    print(f"\nTotal Qt key events logged: {len(raw_log)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
