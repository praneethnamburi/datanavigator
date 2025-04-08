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

class TestGenericBrowser:
    @pytest.fixture
    def browser(self, matplotlib_figure):
        figure, ax = matplotlib_figure
        b = GenericBrowser(figure_handle=figure)
        b.events.add(name="test_event", size=2, fname="test.json", data_id_func=lambda: "test_id", color="blue")
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
        assert browser._current_idx == 50 # 20 steps to browse 1000 items
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
