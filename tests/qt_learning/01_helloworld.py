"""
Qt hello-world for datanavigator 1.4.0.

Run with:
    python tests/qt_learning/01_helloworld.py

This file is intentionally tiny but covers the five Qt concepts you'll need
to recognize when we start hosting matplotlib inside Qt:

    1. QApplication       -- the event loop. One per process.
    2. QMainWindow        -- a top-level window with reserved areas
                             (central widget, toolbars, status bar, dock widgets).
    3. QWidget + layouts  -- how widgets stack inside a window.
    4. Signals & slots    -- how Qt wires UI events to Python callbacks.
    5. exec()             -- the blocking call that runs the event loop.

We import through `qtpy` so the same code works on PyQt5 / PyQt6 / PySide2 / PySide6
without changes. `qtpy` picks whichever binding is already installed.
"""

import sys

# -- 1. The imports ----------------------------------------------------------
#
# qtpy is a thin shim. `from qtpy.QtWidgets import QPushButton` resolves to
# whichever Qt binding is installed; the API is identical across bindings.
#
# Qt is split into several modules. The ones you'll see most:
#   QtCore     -- non-GUI fundamentals (signals, threads, timers, Qt enums)
#   QtGui      -- low-level GUI (QImage, QPainter, fonts, clipboards)
#   QtWidgets  -- the actual widgets (QPushButton, QLabel, QMainWindow, ...)
#
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)


# -- 2. Subclass QMainWindow -------------------------------------------------
#
# QMainWindow is the standard top-level window. It has a fixed structure:
#
#     +-----------------------------------------+
#     | menu bar (optional)                     |
#     +-----------------------------------------+
#     | toolbars (top, left, right, or bottom)  |
#     +-----------------------------------------+
#     |                                         |
#     |   CENTRAL WIDGET                        |
#     |   (you set this with setCentralWidget;  |
#     |    exactly one. In datanavigator 1.4    |
#     |    it will be the matplotlib canvas.)   |
#     |                                         |
#     +-----------------------------------------+
#     | status bar (a strip at the bottom)      |
#     +-----------------------------------------+
#
# We'll use the central widget for our content and the status bar for the
# memoryslot display (same role TextView plays today).
#
class HelloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("datanavigator Qt hello-world")
        self.resize(480, 240)

        # -- 3. Build the central widget --------------------------------------
        #
        # A QWidget is the base class for everything visible. It can be a
        # leaf (a button) or a container (a panel that holds other widgets).
        # A container arranges its children using a *layout*.
        #
        # Layouts you'll see most often:
        #   QVBoxLayout  -- stack children vertically
        #   QHBoxLayout  -- stack children horizontally
        #   QGridLayout  -- rows × columns
        #
        # Pattern: create container -> create layout -> add widgets to layout
        # -> setLayout on container.
        #
        central = QWidget()
        layout = QVBoxLayout()
        central.setLayout(layout)
        self.setCentralWidget(central)

        # A label is a static (or programmatically updatable) text widget.
        # We keep a reference (self.counter_label) because we'll mutate it
        # from the button handler below.
        self.counter = 0
        self.counter_label = QLabel("Clicks: 0")
        self.counter_label.setAlignment(Qt.AlignCenter)  # `Qt` is an enum module
        layout.addWidget(self.counter_label)

        # Two buttons side-by-side, in their own horizontal sub-layout.
        button_row = QHBoxLayout()
        layout.addLayout(button_row)  # layouts can nest

        increment_btn = QPushButton("Increment")
        reset_btn = QPushButton("Reset")
        button_row.addWidget(increment_btn)
        button_row.addWidget(reset_btn)

        # -- 4. Signals and slots ---------------------------------------------
        #
        # Every Qt widget has signals (things it announces) and slots
        # (functions that can receive announcements). A QPushButton has a
        # `clicked` signal. We connect it to a Python method using
        # `.connect(...)`.
        #
        # `clicked` actually emits a `bool` argument (the checked state, for
        # toggle buttons). Most of the time you don't care; a no-arg handler
        # works because Qt does the right thing if the slot accepts fewer
        # arguments than the signal emits.
        #
        increment_btn.clicked.connect(self._on_increment)
        reset_btn.clicked.connect(self._on_reset)

        # -- 5. Status bar ---------------------------------------------------
        #
        # `statusBar()` lazy-creates one if you haven't already. We'll use
        # it for the memoryslot display in the real refactor.
        #
        self.statusBar().showMessage("Ready. Press buttons to update the counter.")

    def _on_increment(self):
        self.counter += 1
        # Updating a QLabel is just `.setText(...)`. Qt schedules the repaint
        # automatically -- no equivalent of `plt.draw()` is needed.
        self.counter_label.setText(f"Clicks: {self.counter}")
        self.statusBar().showMessage(f"Incremented to {self.counter}", 1500)
        # The 2nd arg to showMessage is a timeout in ms (0 = persistent).

    def _on_reset(self):
        self.counter = 0
        self.counter_label.setText("Clicks: 0")
        self.statusBar().showMessage("Reset.", 1500)


# -- 6. The event loop -------------------------------------------------------
#
# Every Qt program needs exactly one QApplication. It owns the event loop.
# `sys.argv` is conventionally passed in so Qt can parse Qt-specific CLI
# flags (e.g. -style fusion); harmless if there are none.
#
# `window.show()` makes the window visible.
# `app.exec()` enters the event loop and blocks until the last window closes.
# (In PyQt5 the method is `exec_` because `exec` was a keyword in Py2; qtpy
# normalizes both spellings -- `app.exec()` works on every binding.)
#
def main():
    app = QApplication(sys.argv)
    window = HelloWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
