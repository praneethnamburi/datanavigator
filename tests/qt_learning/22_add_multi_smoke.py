"""
Sandbox #22 (Buttons.add_multi smoke).

Covers the 1.4.0rc2-additive Buttons.add_multi(*specs) API: N buttons
in a single horizontal row hosted by a child QWidget inside the buttons
column's QVBoxLayout. Motivated by DUSTrack's "Trace: line / Trace: dot"
and "Freeze plot axes / Unfreeze plot axes" pairs reclaiming one
vertical sidebar slot each.

Run with:
    QT_API=pyqt5 QT_QPA_PLATFORM=offscreen python tests/qt_learning/22_add_multi_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication, QHBoxLayout, QPushButton

app = QApplication.instance() or QApplication([])

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))
import datanavigator as dnav  # noqa: E402


def test_add_multi_row_layout_qt():
    """A solo + 2-button row + solo sequence should yield 4 buttons in
    _list, with the row's two buttons sharing a single QHBoxLayout host
    that's a sibling of the two solo QPushButtons in the column."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)

    b.buttons.add(text="solo_top")
    clicked = {"a": 0, "b": 0}
    row_buttons = b.buttons.add_multi(
        dict(text="row_a", action_func=lambda ev: clicked.__setitem__("a", clicked["a"] + 1)),
        dict(text="row_b", action_func=lambda ev: clicked.__setitem__("b", clicked["b"] + 1)),
    )
    b.buttons.add(text="solo_bot")

    assert len(b.buttons) == 4, f"expected 4 buttons in _list, got {len(b.buttons)}"
    assert [bn.name for bn in b.buttons._list] == ["solo_top", "row_a", "row_b", "solo_bot"]
    assert len(row_buttons) == 2
    assert [bn.name for bn in row_buttons] == ["row_a", "row_b"]

    # Sibling parent of the two row buttons should be one row QWidget,
    # itself sitting in the column QVBoxLayout. The two solo buttons
    # should be direct children of the column instead.
    qhost_a = row_buttons[0]._qt_btn.parentWidget()
    qhost_b = row_buttons[1]._qt_btn.parentWidget()
    assert qhost_a is qhost_b, "row buttons should share a parent QWidget"
    assert isinstance(qhost_a.layout(), QHBoxLayout), \
        f"row parent's layout should be QHBoxLayout, got {type(qhost_a.layout()).__name__}"

    column = b._qt_window._dnav_buttons_widget
    # row_widget should be a child of `column`, while the two solos should
    # have their QPushButton parented directly to `column`.
    assert qhost_a.parentWidget() is column, \
        f"row widget parent should be column buttons widget, got {qhost_a.parentWidget()}"
    solo_top_btn = b.buttons["solo_top"]._qt_btn
    solo_bot_btn = b.buttons["solo_bot"]._qt_btn
    assert solo_top_btn.parentWidget() is column
    assert solo_bot_btn.parentWidget() is column

    # Click each row button via the underlying QPushButton.
    row_buttons[0]._qt_btn.click()
    row_buttons[1]._qt_btn.click()
    row_buttons[1]._qt_btn.click()
    assert clicked == {"a": 1, "b": 2}, f"click dispatch broken: {clicked}"

    plt.close(fig)
    print("add_multi Qt path: row widget hosts both buttons, clicks dispatch, len(_list)==4")


def test_add_multi_empty_and_single():
    """Edge cases: empty -> []; single-spec -> behaves like add() (one
    QPushButton parented to the column, not in a row widget)."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)

    # Empty.
    out = b.buttons.add_multi()
    assert out == [], f"empty add_multi should return [], got {out!r}"
    assert len(b.buttons) == 0

    # Single spec. The buttons column is lazily created here.
    out = b.buttons.add_multi(dict(text="lonely"))
    assert len(out) == 1 and out[0].name == "lonely"
    assert len(b.buttons) == 1
    # Single-spec falls through to add(); its QPushButton should be a
    # direct child of the column QWidget, not wrapped in a row QWidget.
    column = b._qt_window._dnav_buttons_widget
    assert out[0]._qt_btn.parentWidget() is column

    plt.close(fig)
    print("add_multi edge cases: empty -> [], single-spec -> direct column child")


def test_add_multi_rejects_pos():
    """Per-spec pos= is rejected -- row layout is incompatible with explicit
    absolute placement."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    try:
        b.buttons.add_multi(dict(text="ok"), dict(text="bad", pos=(0.1, 0.1, 0.2, 0.05)))
    except ValueError as e:
        assert "pos" in str(e), f"unexpected ValueError message: {e}"
        plt.close(fig)
        print("add_multi pos= rejection: ValueError raised as expected")
        return
    plt.close(fig)
    raise AssertionError("add_multi should have raised ValueError for per-spec pos=")


def main():
    test_add_multi_row_layout_qt()
    test_add_multi_empty_and_single()
    test_add_multi_rejects_pos()
    print("add_multi smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
