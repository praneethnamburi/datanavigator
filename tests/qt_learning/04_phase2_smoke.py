"""
Sandbox #4 (Phase 2 smoke).

Phase 2 makes ``datanavigator.utils.TextView`` Qt-aware: when its figure
lives on a QtAgg canvas, the text is rendered as a QLabel overlay
parented to the canvas instead of an mpl Text artist. Public API is
unchanged.

This sandbox creates a TextView at each of the 5 mpl-style positions
ComponentBrowser uses, then asserts each one resolved to a Qt overlay
(no mpl artist). It also exercises ``MemorySlots.show`` and
``StateVariables.show`` end-to-end so we see the asset managers driving
overlays without any code change of their own.

Run with:
    QT_API=pyqt5 python tests/qt_learning/04_phase2_smoke.py

Expected output:
    backend = QtAgg
    pos="bottom left"   overlay=ok  mpl_artist=None
    pos="bottom right"  overlay=ok  mpl_artist=None
    pos="top left"      overlay=ok  mpl_artist=None
    pos="center left"   overlay=ok  mpl_artist=None
    pos="bottom center" overlay=ok  mpl_artist=None
    memoryslots overlay=ok
    statevariables overlay=ok
    Phase 2 smoke OK
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
from datanavigator.utils import TextView


def main():
    print(f"backend = {matplotlib.get_backend()}")

    fig = plt.figure()

    positions = [
        "bottom left",
        "bottom right",
        "top left",
        "center left",
        "bottom center",
    ]
    for pos in positions:
        tv = TextView([f"hello from {pos}"], fax=fig, pos=pos)
        ok = tv._overlay is not None and tv._text is None and tv._ax is None
        print(f"pos={pos!r:20s}  overlay={'ok' if ok else 'MISSING'}  "
              f"mpl_artist={tv._text}  spacer_axes={tv._ax}")
        assert ok, f"expected Qt overlay + no mpl spacer axes at pos={pos}"

    # Drive the asset managers end-to-end. They construct TextView under
    # the hood; we never touch the overlay class directly.
    b = GenericBrowser(figure_handle=fig)
    b.memoryslots.show(pos="bottom left")
    b.memoryslots._list["1"] = 42
    b.memoryslots.update_display()
    ok = b.memoryslots._memtext._overlay is not None
    print(f"memoryslots overlay={'ok' if ok else 'MISSING'}")
    assert ok

    b.statevariables.add(name="mode", states=["a", "b"])
    b.statevariables.show(pos="bottom right")
    b.statevariables.update_display(draw=False)
    ok = b.statevariables._text._overlay is not None
    print(f"statevariables overlay={'ok' if ok else 'MISSING'}")
    assert ok

    print("Phase 2 smoke OK")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
