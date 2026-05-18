"""
Sandbox #2: embed a matplotlib figure inside a QMainWindow.

Run with:
    python tests/qt_learning/02_matplotlib_in_qt.py

This is the exact scaffolding pattern Phase 1 of the 1.4.0 refactor will use:

    +--------------------------------------------------+
    |  [Prev] [Next]   <-- QToolBar of QPushButtons    |  <-- toolbar
    +--------------------------------------------------+
    |                                                  |
    |    matplotlib FigureCanvasQTAgg                  |
    |    (a QWidget that wraps the mpl Agg renderer)   |  <-- central widget
    |                                                  |
    +--------------------------------------------------+
    |  Frame 0/49                                      |  <-- status bar
    +--------------------------------------------------+

Three key things you'll learn here that didn't show up in sandbox #1:

    1. How to build a matplotlib Figure *without pyplot*.
       (pyplot creates its own QMainWindow behind the scenes; for embedding
       we want to own that wrapper ourselves.)

    2. FigureCanvasQTAgg -- the matplotlib widget. It's a QWidget; you
       drop it into a layout like any other widget. Its `.figure` attribute
       is a regular `matplotlib.figure.Figure`, so all your existing
       `set_data` / `set_title` / `Axes` code works unchanged.

    3. `canvas.draw_idle()` vs `plt.draw()`. `draw_idle` schedules a repaint
       on Qt's next idle moment and coalesces multiple requests into one --
       this is what kills the redraw-the-whole-window-per-frame cost.
       (`plt.draw()` still works if pyplot is imported, but `draw_idle` is
       the embedded-canvas idiom.)
"""

import sys

import numpy as np

# -- matplotlib imports for embedded use -------------------------------------
#
# Note: we are NOT importing pyplot. When you `import matplotlib.pyplot`,
# matplotlib picks a backend and pyplot's `figure()` creates a top-level
# window for you. For embedded use we want to own the window, so we:
#
#   - import `Figure` directly from `matplotlib.figure`
#   - import `FigureCanvasQTAgg` directly from the backend module
#
# This is the canonical "matplotlib in Qt" pattern documented at
#   https://matplotlib.org/stable/gallery/user_interfaces/embedding_in_qt_sgskip.html
#
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QToolBar,
    QShortcut,
)


# -- A tiny synthetic "video" so we have something to scrub through ----------
def make_frames(n=50, h=240, w=320):
    """Return a (n, h, w, 3) uint8 array: animated checker pattern."""
    frames = np.zeros((n, h, w, 3), dtype=np.uint8)
    for i in range(n):
        # Moving diagonal stripes; channels phase-shifted so it looks colored.
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        for c, phase in enumerate((0, 2, 4)):
            frames[i, ..., c] = 127 + 127 * np.sin(0.05 * (x + y + 4 * i + phase * 10))
    return frames


class FrameBrowser(QMainWindow):
    """A miniature VideoBrowser. Same structure 1.4.0's GenericBrowser will have."""

    def __init__(self, frames):
        super().__init__()
        self.setWindowTitle("datanavigator -- matplotlib in Qt sandbox")
        self.resize(700, 500)

        self.frames = frames
        self.idx = 0

        # -- 1. Build the matplotlib Figure ---------------------------------
        #
        # `Figure(...)` from matplotlib.figure is a plain Figure object,
        # unattached to any backend / window. It's just the data model:
        # axes, artists, text, etc.
        #
        # `figsize` is in inches; `dpi` is dots-per-inch. The actual pixel
        # size of the rendered canvas = figsize * dpi. The Qt widget will
        # rescale on window resize regardless.
        #
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self._ax = self.figure.add_subplot(1, 1, 1)
        self._ax.axis("off")
        self._im = self._ax.imshow(self.frames[0])
        # Notice: `self.figure` and `self._ax` are *exactly* what GenericBrowser
        # exposes today. Consumer code that touches them won't notice the
        # backend swap.

        # -- 2. Wrap the figure in a Qt widget ------------------------------
        #
        # `FigureCanvasQTAgg(figure)` returns a QWidget. From Qt's perspective
        # it's just another widget you can put in a layout, give a size policy,
        # and connect to signals.
        #
        # From matplotlib's perspective it's the backend that knows how to
        # paint the figure onto a QWidget. The canvas drives matplotlib's
        # event system, so `canvas.mpl_connect(...)` still works for things
        # like `button_press_event` (we'll use that later for click-to-seek).
        #
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.setCentralWidget(self.canvas)

        # -- 3. Toolbar with Prev / Next buttons ----------------------------
        #
        # QToolBar is the standard place to put action buttons. addToolBar
        # docks it to the top of the QMainWindow by default. (You can also
        # use addWidget on a manually-positioned QWidget, but a toolbar is
        # the conventional, OS-themed choice.)
        #
        # These QPushButtons render natively -- no matplotlib redraw is
        # triggered when you hover or click them. This is the architectural
        # win over `matplotlib.widgets.Button` (which lives inside an
        # mpl Axes and therefore participates in every canvas redraw).
        #
        toolbar = QToolBar("Navigation")
        self.addToolBar(toolbar)

        prev_btn = QPushButton("Prev")
        next_btn = QPushButton("Next")
        toolbar.addWidget(prev_btn)
        toolbar.addWidget(next_btn)

        prev_btn.clicked.connect(self._on_prev)
        next_btn.clicked.connect(self._on_next)

        # -- 4. Keyboard shortcuts ------------------------------------------
        #
        # `QShortcut` binds a key sequence (any Qt-recognized name) to a slot,
        # scoped to a parent widget. With `self` (the QMainWindow) as parent,
        # the shortcut fires whenever the main window has focus -- regardless
        # of which child widget is focused.
        #
        # This replaces `matplotlib.figure.canvas.mpl_connect("key_press_event", ...)`
        # in 1.4.0 because mpl key events only fire when the canvas itself
        # has keyboard focus, which is fiddly. QShortcut is rock-solid.
        #
        QShortcut(QKeySequence("Right"), self, activated=self._on_next)
        QShortcut(QKeySequence("Left"), self, activated=self._on_prev)

        # -- 5. Status bar for frame index ----------------------------------
        self._update_status()

    def _on_next(self):
        if self.idx < len(self.frames) - 1:
            self.idx += 1
            self._update()

    def _on_prev(self):
        if self.idx > 0:
            self.idx -= 1
            self._update()

    def _update(self):
        # -- The hot path -- this is what runs every frame.
        #
        # Compare to VideoBrowser.update() today:
        #   self._im.set_data(...)        <-- same
        #   self._ax.set_title(...)       <-- we use the Qt status bar instead
        #   super().update()              <-- updates TextView overlays (mpl text!)
        #   plt.draw()                    <-- full canvas redraw
        #
        # Here:
        self._im.set_data(self.frames[self.idx])
        self.canvas.draw_idle()  # schedule one Qt repaint; coalesces with others
        self._update_status()    # native QLabel update -- no canvas involvement

    def _update_status(self):
        self.statusBar().showMessage(f"Frame {self.idx}/{len(self.frames) - 1}")


def main():
    frames = make_frames()
    app = QApplication(sys.argv)
    window = FrameBrowser(frames)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
