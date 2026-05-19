"""
Sandbox #23 (Buttons style_tag smoke).

Covers the 1.4.0rc2-additive Buttons.register_style + style_tag= API:
consumer registers a styler under a tag, then add(... style_tag=tag) /
add_multi(specs with per-spec style_tag) auto-applies the styler to
the freshly built QPushButton (its QSS, palette, or any other Qt-side
mutation lands inside _finalize_button, before the button is returned).

Motivated by DUSTrack's sidebar palette refactor (1.1.0rc2): the
per-group _btns_workflow / _btns_display lists + _style_sidebar_buttons
batch pass dissolve into inline style_tag=... on each add() call.

Run with:
    QT_API=pyqt5 QT_QPA_PLATFORM=offscreen python tests/qt_learning/23_style_tag_smoke.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import QApplication

app = QApplication.instance() or QApplication([])

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))
import datanavigator as dnav  # noqa: E402


_TEST_QSS = (
    "QPushButton { background-color: #ff00aa; color: #ffffff; "
    "border: 2px solid #aa0066; padding: 4px; }"
)


def _apply_test_style(b):
    b._qt_btn.setStyleSheet(_TEST_QSS)


def test_style_tag_lands_qss_on_qpushbutton():
    """register_style + add(style_tag=...) should plumb QSS through to the
    underlying QPushButton at add-time -- no separate batch pass."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.register_style("test", _apply_test_style)
    bn = b.buttons.add(text="hero", style_tag="test")
    qss = bn._qt_btn.styleSheet()
    assert qss == _TEST_QSS, f"styler didn't reach QPushButton: {qss!r}"
    assert bn._style_tag == "test"
    plt.close(fig)
    print("style_tag single add: QSS landed on QPushButton via register_style")


def test_style_tag_via_add_multi_per_spec():
    """Each spec in add_multi carries its own style_tag; per-button styler
    runs in spec order. A spec without style_tag stays unstyled."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.register_style("pink", _apply_test_style)
    row = b.buttons.add_multi(
        dict(text="row_a", style_tag="pink"),
        dict(text="row_b", style_tag="pink"),
        dict(text="row_c"),
    )
    assert row[0]._qt_btn.styleSheet() == _TEST_QSS
    assert row[1]._qt_btn.styleSheet() == _TEST_QSS
    assert row[2]._qt_btn.styleSheet() == "", \
        f"row_c (no style_tag) shouldn't be styled: {row[2]._qt_btn.styleSheet()!r}"
    assert [bn._style_tag for bn in row] == ["pink", "pink", None]
    plt.close(fig)
    print("style_tag add_multi: per-spec tagging applies the styler on the right buttons")


def test_unknown_style_tag_raises():
    """A tag that's neither consumer-registered nor a built-in should
    fail loud at add-time."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    try:
        b.buttons.add(text="x", style_tag="not_a_real_tag")
    except KeyError as e:
        assert "not_a_real_tag" in str(e)
        plt.close(fig)
        print("style_tag unknown-tag: KeyError raised as expected")
        return
    plt.close(fig)
    raise AssertionError("add(style_tag='not_a_real_tag') should have raised KeyError")


def test_reapply_styles_after_registry_swap():
    """Re-registering an existing tag and calling reapply_styles() should
    re-paint every already-added button that carried the tag."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.register_style("v", _apply_test_style)
    bn = b.buttons.add(text="hero", style_tag="v")
    assert bn._qt_btn.styleSheet() == _TEST_QSS

    new_qss = "QPushButton { background-color: #00aaff; padding: 4px; }"
    def repaint(btn):
        btn._qt_btn.setStyleSheet(new_qss)
    b.buttons.register_style("v", repaint)
    b.buttons.reapply_styles()
    assert bn._qt_btn.styleSheet() == new_qss, \
        f"reapply_styles didn't repaint: {bn._qt_btn.styleSheet()!r}"
    plt.close(fig)
    print("reapply_styles: registry swap propagates to existing buttons")


def test_builtin_primary_lands_qss():
    """The dnav-shipped 'primary' built-in applies *some* QSS on the Qt
    path (theme-aware -- light or dark, doesn't matter, just non-empty)."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    bn = b.buttons.add(text="hero", style_tag="primary")
    qss = bn._qt_btn.styleSheet()
    assert "QPushButton" in qss and "background-color" in qss, \
        f"builtin 'primary' didn't land QSS: {qss!r}"
    plt.close(fig)
    print("style_tag built-in 'primary': QSS landed on QPushButton")


def main():
    test_style_tag_lands_qss_on_qpushbutton()
    test_style_tag_via_add_multi_per_spec()
    test_unknown_style_tag_raises()
    test_reapply_styles_after_registry_swap()
    test_builtin_primary_lands_qss()
    print("style_tag smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
