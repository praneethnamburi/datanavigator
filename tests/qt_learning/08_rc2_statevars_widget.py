"""
Sandbox #8 (rc2 statevariables widget).

Covers the rc2 promotion of state-variables from a TextView text sink
to a layout-managed Qt widget (QComboBox / QButtonGroup / QLabel)
mounted under the buttons column. Exercises the three branches of
StateVariable.widget plus the round-trip dropdown-pick -> set_state ->
parent.update() interaction.

Run with:
    QT_API=pyqt6 QT_QPA_PLATFORM=offscreen python tests/qt_learning/08_rc2_statevars_widget.py
"""

import os
import sys

os.environ.setdefault("QT_API", "pyqt5")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib
matplotlib.use("QtAgg")

from matplotlib import pyplot as plt
from qtpy.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QLabel, QToolButton,
)

app = QApplication.instance() or QApplication([])

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
))
import datanavigator as dnav  # noqa: E402


def _find_widgets(qwidget, qcls):
    """Recursively collect all descendants of qwidget that are qcls."""
    found = []
    for child in qwidget.findChildren(qcls):
        found.append(child)
    return found


def test_widget_label_renders_as_qlabel():
    """widget='label' produces a single QLabel showing 'name: value'."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("mode", ["a", "b"], widget="label")
    b.statevariables.show()

    sv = b.statevariables._text
    assert hasattr(sv, "_container"), "expected rc2 widget"
    labels = _find_widgets(sv, QLabel)
    # Title label + 1 per state-variable
    label_texts = [lb.text() for lb in labels]
    assert any("mode: a" in t for t in label_texts), \
        f"no 'mode: a' label in {label_texts}"
    plt.close(fig)
    print("widget=label: QLabel renders 'mode: a'")


def test_widget_dropdown_renders_as_qcombobox():
    """widget='dropdown' produces a QComboBox populated with the states."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("layer", ["alpha", "beta", "gamma"], widget="dropdown")
    b.statevariables.show()

    sv = b.statevariables._text
    combos = _find_widgets(sv, QComboBox)
    assert len(combos) == 1, f"expected 1 QComboBox, got {len(combos)}"
    combo = combos[0]
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["alpha", "beta", "gamma"], items
    assert combo.currentIndex() == 0
    plt.close(fig)
    print("widget=dropdown: QComboBox populated with 3 items")


def test_widget_toggle_renders_as_button_group():
    """widget='toggle' produces a QButtonGroup of checkable QToolButtons."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("mode", ["select", "place"], widget="toggle")
    b.statevariables.show()

    sv = b.statevariables._text
    buttons = _find_widgets(sv, QToolButton)
    assert len(buttons) == 2, f"expected 2 QToolButtons, got {len(buttons)}"
    assert {bn.text() for bn in buttons} == {"select", "place"}
    # The button matching state index 0 starts checked, the other doesn't.
    checked = [bn for bn in buttons if bn.isChecked()]
    assert len(checked) == 1
    assert checked[0].text() == "select"
    plt.close(fig)
    print("widget=toggle: QButtonGroup of 2 checkable QToolButtons, "
          "'select' checked")


def test_dropdown_pick_triggers_parent_update():
    """User picks an item in the QComboBox -> set_state on the model +
    parent.update() called once (matching the keybind-cycle pathway)."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("layer", ["alpha", "beta", "gamma"], widget="dropdown")
    b.statevariables.show()

    updates = {"count": 0}
    orig_update = b.update

    def counting_update(*a, **kw):
        updates["count"] += 1
        return orig_update(*a, **kw)

    b.update = counting_update

    combo = _find_widgets(b.statevariables._text, QComboBox)[0]
    combo.setCurrentIndex(2)

    assert b.statevariables["layer"].current_state == "gamma", \
        b.statevariables["layer"].current_state
    assert updates["count"] == 1, updates
    plt.close(fig)
    print("dropdown pick: set_state('gamma') + parent.update() fired once")


def test_toggle_click_triggers_parent_update():
    """Click the unchecked QToolButton -> set_state + parent.update()."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("mode", ["select", "place"], widget="toggle")
    b.statevariables.show()

    updates = {"count": 0}
    b.update = lambda *a, **kw: updates.__setitem__("count", updates["count"] + 1)

    buttons = _find_widgets(b.statevariables._text, QToolButton)
    place_btn = next(bn for bn in buttons if bn.text() == "place")
    place_btn.click()  # simulates user click

    assert b.statevariables["mode"].current_state == "place"
    assert updates["count"] == 1, updates
    plt.close(fig)
    print("toggle click: set_state('place') + parent.update() fired once")


def test_external_cycle_resyncs_widget():
    """Code-side cycle() + statevariables.update_display() repositions the
    QComboBox / QToolButton check state. Verifies the .update() duck-typed
    method on the widget."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.statevariables.add("layer", ["alpha", "beta"], widget="dropdown")
    b.statevariables.add("mode", ["select", "place"], widget="toggle")
    b.statevariables.show()

    combo = _find_widgets(b.statevariables._text, QComboBox)[0]
    buttons = _find_widgets(b.statevariables._text, QToolButton)

    b.statevariables["layer"].cycle()
    b.statevariables["mode"].cycle()
    b.statevariables.update_display(draw=False)

    assert combo.currentIndex() == 1
    place_btn = next(bn for bn in buttons if bn.text() == "place")
    assert place_btn.isChecked()
    plt.close(fig)
    print("external cycle + update_display: combo/toggle resynced")


def test_widget_mounts_in_left_column_under_buttons():
    """The statevars widget lives in the QDockWidget left column,
    inserted below the buttons sub-widget."""
    fig = plt.figure()
    b = dnav.GenericBrowser(figure_handle=fig)
    b.buttons.add(text="first")
    b.statevariables.add("layer", ["a", "b"], widget="dropdown")
    b.statevariables.show()
    # A button added AFTER show() must still land in the buttons sub-
    # widget, not below the statevars panel (regression guard for the
    # pre-rc2 fix to the insertWidget(count-1) issue).
    b.buttons.add(text="late")

    col = b._qt_window._dnav_left_column
    assert col.statevars_widget is not None
    # outer layout: [buttons_widget, statevars_widget, addStretch]
    outer = col.outer_layout
    items = [outer.itemAt(i).widget() for i in range(outer.count())]
    assert items[0] is col.buttons_widget
    assert items[1] is col.statevars_widget
    # Late button is inside buttons_widget, not below statevars.
    btn_layout = col.buttons_widget.layout()
    btn_texts = []
    for i in range(btn_layout.count()):
        w = btn_layout.itemAt(i).widget()
        if w is not None and hasattr(w, "text"):
            btn_texts.append(w.text())
    assert "first" in btn_texts and "late" in btn_texts, btn_texts
    plt.close(fig)
    print("left column: buttons_widget on top, statevars_widget below; "
          "late buttons still in buttons_widget")


def main():
    test_widget_label_renders_as_qlabel()
    test_widget_dropdown_renders_as_qcombobox()
    test_widget_toggle_renders_as_button_group()
    test_dropdown_pick_triggers_parent_update()
    test_toggle_click_triggers_parent_update()
    test_external_cycle_resyncs_widget()
    test_widget_mounts_in_left_column_under_buttons()
    print("rc2 statevars-widget smoke OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
