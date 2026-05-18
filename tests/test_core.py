import pytest
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from unittest.mock import MagicMock
from datanavigator.core import GenericBrowser

from tests.conftest import simulate_key_press


def test_browser_initialization():
    GenericBrowser()
    GenericBrowser(figure_handle=plt.figure())


def test_case_insensitive_key_dispatch_fallback():
    """When mpl emits an uppercase single letter that isn't explicitly bound,
    dispatch falls back to the lowercase variant. Fixes the Sticky-Keys /
    Caps Lock / accidental-shift class of "binding fired silently" bugs.
    """
    b = GenericBrowser(figure_handle=plt.figure())
    fired = {"t_lower": 0, "t_upper": 0}

    def on_t():
        fired["t_lower"] += 1

    def on_T():
        fired["t_upper"] += 1

    # Only the lowercase is bound; uppercase should fall back to it.
    b.add_key_binding("t", on_t)

    # Direct match.
    event = simulate_key_press(b.figure, key="t")
    b(event)
    assert fired["t_lower"] == 1, "plain t didn't fire 't' binding"

    # Uppercase (Shift+t or Caps Lock-modified t) falls back.
    event = simulate_key_press(b.figure, key="T")
    b(event)
    assert fired["t_lower"] == 2, "Shift+T didn't fall back to 't' binding"

    # Explicit uppercase binding takes precedence over the fallback.
    b.add_key_binding("T", on_T)
    event = simulate_key_press(b.figure, key="T")
    b(event)
    assert fired["t_upper"] == 1
    assert fired["t_lower"] == 2, "lowercase shouldn't fire when 'T' is bound"

    # Lowercase still works independently.
    event = simulate_key_press(b.figure, key="t")
    b(event)
    assert fired["t_lower"] == 3
    assert fired["t_upper"] == 1


def test_shift_letter_is_uppercase_not_prefixed():
    """matplotlib reports shift+<letter> as the uppercase letter, NOT as
    'shift+t'. shift+<arrow> is reported with the 'shift+' prefix. The
    Phase 4i guard preserves both conventions: independent t / T
    bindings work as expected for shift+letter; shift+arrow bindings
    work via direct match on their literal string.

    This test documents the contract for anyone wondering "how do I bind
    shift+t to a different action than t?": bind 'T', not 'shift+t'.
    """
    b = GenericBrowser(figure_handle=plt.figure())
    log = []
    b.add_key_binding("t", lambda: log.append("t"))
    b.add_key_binding("T", lambda: log.append("T"))
    b.add_key_binding("shift+left", lambda: log.append("shift+left"))
    b.add_key_binding("left", lambda: log.append("left"))

    # Pressing plain t fires 't'.
    b(simulate_key_press(b.figure, key="t"))
    # Pressing Shift+t (mpl emits 'T') fires 'T', NOT fallback to 't'.
    b(simulate_key_press(b.figure, key="T"))
    # Pressing Shift+left (mpl emits 'shift+left') fires 'shift+left'.
    b(simulate_key_press(b.figure, key="shift+left"))
    # Pressing plain left fires 'left'.
    b(simulate_key_press(b.figure, key="left"))

    assert log == ["t", "T", "shift+left", "left"], (
        f"expected ['t', 'T', 'shift+left', 'left'], got {log}"
    )


def test_case_insensitive_fallback_doesnt_affect_special_keys():
    """The fallback applies only to single alphabetic letters. shift+left,
    ctrl+c, etc. are unaffected (they're multi-character keys that mpl
    emits as-is).
    """
    b = GenericBrowser(figure_handle=plt.figure())

    fired_decrement = []
    fired_decrement_frac = []
    b._keypressdict["left"] = (lambda: fired_decrement.append(1), "decrement")
    b._keypressdict["shift+left"] = (
        lambda: fired_decrement_frac.append(1), "decrement frac"
    )

    # Pressing 'left' fires 'left' only, not 'shift+left'.
    event = simulate_key_press(b.figure, key="left")
    b(event)
    assert len(fired_decrement) == 1
    assert len(fired_decrement_frac) == 0

    # Pressing 'shift+left' fires 'shift+left' only.
    event = simulate_key_press(b.figure, key="shift+left")
    b(event)
    assert len(fired_decrement) == 1
    assert len(fired_decrement_frac) == 1


def test_buttons_add_separator_mpl_path():
    """Buttons.add_separator() under Agg adds an invisible spacer button.

    The spacer takes a layout slot so the next add() is pushed down,
    matching today's DUSTrack hand-rolled-spacer behavior.
    """
    from datanavigator.assets import Button
    b = GenericBrowser(figure_handle=plt.figure())
    b.buttons.add(text="first")
    n_before_sep = len(b.buttons)
    b.buttons.add_separator()
    n_after_sep = len(b.buttons)
    assert n_after_sep == n_before_sep + 1, "separator should occupy a slot"
    # The spacer is a Button (mpl path) with an auto-generated name and
    # its visible bits switched off.
    spacer = b.buttons._list[-1]
    assert isinstance(spacer, Button)
    assert spacer.name.startswith("__separator_")
    assert spacer.ax.patch.get_visible() is False
    assert spacer.label.get_visible() is False
    # And add_separator() returns None on both paths.
    assert b.buttons.add_separator() is None


def test_buttons_use_mpl_path_under_agg():
    """Soft Qt mode: under Agg, Buttons.add returns mpl widgets, not Qt wrappers.

    The Qt-path classes are named _QtPushButton / _QtToggleButton (private)
    and only appear when the parent's figure is on a Qt canvas. Under the
    Agg backend the test suite uses, the returned objects are the public
    mpl-inheriting Button / ToggleButton from assets.py.
    """
    from datanavigator.assets import Button, ToggleButton
    assert mpl.get_backend().lower() == "agg"
    b = GenericBrowser(figure_handle=plt.figure())
    push = b.buttons.add(text="push")
    toggle = b.buttons.add(text="toggle", type_="Toggle")
    assert isinstance(push, Button)
    assert isinstance(toggle, ToggleButton)
    assert not type(push).__name__.startswith("_Qt")
    assert not type(toggle).__name__.startswith("_Qt")


def test_qt_window_is_none_under_agg():
    """Soft Qt mode: on the Agg backend the test suite uses, _qt_window is None.

    Phase 1 of the 1.4.0 refactor exposes ``self._qt_window``: the
    QMainWindow matplotlib builds around the figure under QtAgg, or None
    on any non-Qt backend. The formal pytest suite runs on Agg (see
    conftest.py), so the discovery helper must return None here. The
    QtAgg path is exercised by tests/qt_learning/03_phase1_smoke.py.
    """
    assert mpl.get_backend().lower() == "agg"
    b = GenericBrowser(figure_handle=plt.figure())
    assert b._qt_window is None


class TestGenericBrowser:
    @pytest.fixture
    def browser(self, matplotlib_figure):
        figure, ax = matplotlib_figure
        b = GenericBrowser(figure_handle=figure)
        b.events.add(
            name="test_event",
            size=2,
            fname="test.json",
            data_id_func=lambda: "test_id",
            color="blue",
        )
        b.statevariables.add(name="test_state", states=["state1", "state2"])
        b.statevariables.show()
        b.buttons.add(text="test_button", type_="Push", action_func=lambda: None)
        return b

    def test_initialization(self, browser):
        assert isinstance(browser.figure, plt.Figure)
        assert browser._current_idx == 0
        assert browser._keypressdict == {}
        assert browser._bindings_removed == {}

    def test_add_key_binding(self, browser):
        def dummy_function():
            pass

        browser.add_key_binding("alt+1", dummy_function, "test description")
        assert "alt+1" in browser._keypressdict
        assert browser._keypressdict["alt+1"] == (dummy_function, "test description")

    def test_set_default_keybindings(self, browser):
        browser.set_default_keybindings()
        assert "left" in browser._keypressdict
        assert "right" in browser._keypressdict
        assert "up" in browser._keypressdict
        assert "down" in browser._keypressdict

    def test_call_key_press_event(self, browser):
        event = simulate_key_press(browser.figure, key="left")
        browser.set_default_keybindings()
        browser._keypressdict["left"] = (MagicMock(), "decrement")
        browser._keypressdict["left"][0].assert_not_called()
        browser(event)
        browser._keypressdict["left"][0].assert_called_once()

    def test_call_close_event(self, browser):
        event = MagicMock()
        event.name = "close_event"
        browser.cleanup = MagicMock()
        browser.cleanup.assert_not_called()
        browser(event)
        browser.cleanup.assert_called_once()

    def test_update(self, browser):
        browser.update_assets = MagicMock()
        browser.update()
        browser.update_assets.assert_called_once()

    def test_update_without_clear(self, browser):
        browser.update_assets = MagicMock()
        browser.update_without_clear()
        browser.update_assets.assert_called_once()

    def test_reset_axes(self, browser):
        ax = browser.figure.get_axes()[0]
        ax.plot([0, 1], [0, 1])
        browser.reset_axes()
        assert np.allclose(ax.get_xlim(), (-0.05, 1.05))
        assert np.allclose(ax.get_ylim(), (-0.05, 1.05))

    def test_increment(self, browser):
        browser.data = [1, 2, 3, 4, 5]
        browser.update = MagicMock()
        browser.increment()
        assert browser._current_idx == 1
        browser.update.assert_called_once()

    def test_decrement(self, browser):
        browser.data = [1, 2, 3, 4, 5]
        browser.update = MagicMock()
        browser._current_idx = 5
        browser.decrement()
        assert browser._current_idx == 4
        browser.update.assert_called_once()

    def test_go_to_start(self, browser):
        browser.update = MagicMock()
        browser._current_idx = 5
        browser.go_to_start()
        assert browser._current_idx == 0
        browser.update.assert_called_once()

    def test_go_to_end(self, browser):
        browser.data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        browser.update = MagicMock()
        browser.go_to_end()
        assert browser._current_idx == 9
        browser.update.assert_called_once()

    def test_increment_frac(self, browser):
        browser.data = list(range(1000))
        browser.go_to_start()
        browser.update = MagicMock()
        browser.increment_frac()
        assert browser._current_idx == 50  # 20 steps to browse 1000 items
        browser.update.assert_called_once()
        browser._current_idx = 980
        browser.increment_frac()
        assert browser._current_idx == 999

    def test_decrement_frac(self, browser):
        browser.data = list(range(1000))
        browser.update = MagicMock()
        browser._current_idx = 10
        browser.decrement_frac()
        assert browser._current_idx == 0
        browser.update.assert_called_once()
        browser.go_to_end()
        browser.decrement_frac()
        assert browser._current_idx == 949

    def test_pan(self, browser):
        ax = browser.figure.get_axes()[0]
        ax.plot([0, 1], [0, 1])
        ax.set_xlim(0, 1)
        browser.pan(direction="left")
        assert np.allclose(ax.get_xlim(), (-0.2, 0.8))
        ax.set_ylim(-1, 1)
        browser.pan(direction="down")
        assert np.allclose(ax.get_ylim(), (-0.6, 1.4))

    def test_cleanup(self, browser):
        browser.figure.canvas.mpl_disconnect = MagicMock()
        browser.mpl_restore_bindings = MagicMock()
        browser.cleanup()
        browser.figure.canvas.mpl_disconnect.assert_called()
        browser.mpl_restore_bindings.assert_called_once()

    def test_mpl_restore_bindings(self, browser):
        browser._bindings_removed = {"keymap.back": "left"}
        browser.mpl_restore_bindings()
        assert "left" in mpl.rcParams["keymap.back"]

    def test_has(self, browser):
        browser.buttons = [1]
        assert browser.has("buttons")
        browser.buttons = []
        assert not browser.has("buttons")

    def test_filter_sibling_axes(self, browser):
        ax1 = browser.figure.add_subplot(211)
        ax2 = browser.figure.add_subplot(212, sharex=ax1)
        result = browser._filter_sibling_axes([ax1, ax2], share="x")
        assert result == [ax1]

    def test_memoryslots(self, browser):
        event = simulate_key_press(browser.figure, key="1")
        browser.data = [1, 2, 3, 4, 5]
        browser._current_idx = 2
        browser(event)
        assert browser.memoryslots._list["1"] == 2

    def test_show_key_bindings(self, browser):
        browser.show_key_bindings("new")


class TestStateVariableWidgetHint:
    """Coverage for the rc2 ``widget=`` kwarg on the StateVariable model.

    The hint is consumed by the Qt sidebar; on the Agg backend the test
    suite uses, the value just rides along as an attribute. These tests
    pin the model contract -- legal values, default, and propagation
    through StateVariables.add().
    """

    def test_default_is_label(self):
        from datanavigator.assets import StateVariable

        sv = StateVariable("mode", ["a", "b"])
        assert sv.widget == "label"

    def test_explicit_dropdown(self):
        from datanavigator.assets import StateVariable

        sv = StateVariable("mode", ["a", "b"], widget="dropdown")
        assert sv.widget == "dropdown"

    def test_explicit_toggle(self):
        from datanavigator.assets import StateVariable

        sv = StateVariable("mode", ["a", "b"], widget="toggle")
        assert sv.widget == "toggle"

    def test_unknown_widget_rejected(self):
        from datanavigator.assets import StateVariable

        with pytest.raises(AssertionError):
            StateVariable("mode", ["a", "b"], widget="slider")

    def test_propagation_through_container_add(self):
        b = GenericBrowser(figure_handle=plt.figure())
        b.statevariables.add("layer", ["a", "b"], widget="dropdown")
        b.statevariables.add("number_keys", ["select", "place"], widget="toggle")
        b.statevariables.add("plain", ["x", "y"])  # default

        assert b.statevariables["layer"].widget == "dropdown"
        assert b.statevariables["number_keys"].widget == "toggle"
        assert b.statevariables["plain"].widget == "label"

    def test_show_falls_back_to_textview_on_agg(self):
        """On the Agg backend the Qt-widget path is unavailable; show()
        must transparently fall back to the legacy TextView so existing
        consumers keep working. The widget hints are silently ignored.
        """
        from datanavigator.utils import TextView

        b = GenericBrowser(figure_handle=plt.figure())
        b.statevariables.add("layer", ["a", "b"], widget="dropdown")
        b.statevariables.show(pos="bottom left")
        assert isinstance(b.statevariables._text, TextView)
