"""
Sandbox #8 -- frame-update responsiveness benchmark.

Times N successive ``browser.update()`` calls on a synthetic-frame browser
that mimics the visual complexity of DUSTrack: an imshow + memoryslot
display + state variable display + 8 buttons. Reports per-update wall
time stats so we can compare the same scene across datanavigator
branches (main / 1.4.0-qt).

Methodology:
  - Each iteration: set _current_idx, call browser.update(), call
    app.processEvents() to drain Qt's paint queue. Time the pair.
    processEvents is essential because matplotlib's canvas.draw_idle()
    only *schedules* a repaint; timing update() alone would miss the
    actual rasterization that happens on Qt's next idle tick.
  - QT_QPA_PLATFORM=offscreen so it runs in a headless / unattended
    shell. Qt still rasterizes; only the OS compositor step is skipped.
  - First few iterations discarded as warm-up (font cache, lazy Qt
    widget realization, etc.).

Run:
    QT_API=pyqt5 QT_QPA_PLATFORM=offscreen python tests/qt_learning/08_benchmark.py

Output is a single block of stats so two runs can be diffed by eye or
piped through grep.
"""

import os
import sys
import statistics
import time

os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

# Force the local working tree's datanavigator (whichever branch is checked
# out), not whatever's installed in the env.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)
import datanavigator as dnav  # noqa: E402


def make_synthetic_frames(n: int = 200, h: int = 480, w: int = 640) -> np.ndarray:
    """Animated checker pattern -- different from frame to frame so
    set_data has actual work to do (no cache hits)."""
    frames = np.empty((n, h, w, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    for i in range(n):
        for c, phase in enumerate((0, 2, 4)):
            frames[i, ..., c] = (
                127 + 127 * np.sin(0.02 * (x + y + 4 * i + phase * 10))
            ).astype(np.uint8)
    return frames


class BenchBrowser(dnav.GenericBrowser):
    """Synthetic browser with DUSTrack-like visual complexity."""

    def __init__(self, frames):
        fig = plt.figure(figsize=(8, 6))
        super().__init__(fig)
        self.frames = frames

        self._ax = self.figure.subplots(1, 1)
        self._im = self._ax.imshow(self.frames[0])
        self._ax.axis("off")

        # Asset load representative of a real annotation browser:
        self.memoryslots.show(pos="bottom left")
        self.statevariables.add(name="mode", states=["correction", "annotation"])
        self.statevariables.add(name="label", states=[str(i) for i in range(10)])
        self.statevariables.show(pos="bottom right")
        for i in range(8):
            self.buttons.add(text=f"action_{i}",
                             action_func=lambda *a: None)

    def __len__(self):
        return len(self.frames)

    def update(self, event=None):
        self._im.set_data(self.frames[self._current_idx])
        self._ax.set_title(f"frame {self._current_idx}/{len(self) - 1}")
        super().update()
        # plt.draw() == canvas.draw_idle() on modern matplotlib; both
        # schedule a coalesced repaint. processEvents() in the bench
        # loop is what actually drains the paint.
        plt.draw()


def main():
    n_frames = 200
    n_warmup = 10

    print("=" * 60)
    print(f"datanavigator frame-update benchmark")
    print(f"  source : {dnav.__file__}")
    print(f"  version: {dnav.__version__}")
    print(f"  backend: {matplotlib.get_backend()}")
    print(f"  qt_api : {os.environ.get('QT_API', '?')}")
    print(f"  Qt plat: {os.environ.get('QT_QPA_PLATFORM', 'default')}")
    print(f"  frames : {n_frames} ({n_warmup} warmup, "
          f"{n_frames - n_warmup} measured)")
    print("=" * 60)

    frames = make_synthetic_frames(n_frames)
    b = BenchBrowser(frames)
    b.figure.show()
    # Realize widgets / drain initial paint before timing.
    for _ in range(3):
        app.processEvents()

    times_ms = []
    for i in range(n_frames):
        b._current_idx = i
        t0 = time.perf_counter()
        b.update()
        app.processEvents()
        t1 = time.perf_counter()
        if i >= n_warmup:
            times_ms.append((t1 - t0) * 1000)

    times_sorted = sorted(times_ms)
    n = len(times_ms)
    p95 = times_sorted[int(n * 0.95)]
    p99 = times_sorted[int(n * 0.99)] if n > 100 else times_sorted[-1]

    print(f"N measured  = {n}")
    print(f"min         = {min(times_ms):7.2f} ms")
    print(f"median      = {statistics.median(times_ms):7.2f} ms")
    print(f"mean        = {statistics.mean(times_ms):7.2f} ms")
    print(f"p95         = {p95:7.2f} ms")
    print(f"p99         = {p99:7.2f} ms")
    print(f"max         = {max(times_ms):7.2f} ms")
    print(f"effective fps (median) = "
          f"{1000.0 / statistics.median(times_ms):6.1f}")
    print(f"effective fps (p95)    = {1000.0 / p95:6.1f}")
    print("=" * 60)

    plt.close(b.figure)
    return 0


if __name__ == "__main__":
    sys.exit(main())
