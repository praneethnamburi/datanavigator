import pytest

from datanavigator.signals import SignalBrowser
from tests.conftest import simulate_key_press, press_browser_button


def test_signal_browser_init_with_pysampled_data(signal_list, mock_figure):
    """Test initialization with pysampled.Data objects."""
    browser = SignalBrowser(plot_data=signal_list, figure_handle=mock_figure)

    assert browser.data == signal_list
    assert callable(browser.titlefunc)
    assert browser.reset_on_change is False
    assert browser.titlefunc(browser) == "Plot number 0"
    event = simulate_key_press(browser.figure, key="right")
    browser(event)
    assert browser._current_idx == 1
    assert browser.titlefunc(browser) == "Plot number 1"


def test_signal_browser_init_with_2d_array(mock_figure):
    """Test initialization with 2D array."""
    plot_data = [[10, 20, 30, 40], [1, 5, 4, 3]]
    browser = SignalBrowser(plot_data=plot_data, figure_handle=mock_figure)

    assert browser.titlefunc(browser) == "Plot number 0"
    event = simulate_key_press(browser.figure, key="right")
    browser(event)
    assert browser._current_idx == 1
    assert browser.titlefunc(browser) == "Plot number 1"


def test_signal_browser_init_with_titlefunc(signal_list, mock_figure):
    """Test initialization with a custom title function."""
    custom_titlefunc = lambda s: "Custom Title"
    browser = SignalBrowser(
        plot_data=signal_list, titlefunc=custom_titlefunc, figure_handle=mock_figure
    )

    assert browser.titlefunc == custom_titlefunc
    assert browser.titlefunc(browser) == "Custom Title"


def test_signal_browser_init_with_reset_on_change(signal_list, mock_figure):
    """Test initialization with reset_on_change set to True."""
    browser = SignalBrowser(
        plot_data=signal_list, reset_on_change=True, figure_handle=mock_figure
    )

    assert browser.reset_on_change is True


def test_signal_browser_buttons_added(signal_list, mock_figure):
    """Test that buttons are added during initialization and simulate a mouse click on the 'Right' button."""
    browser = SignalBrowser(plot_data=signal_list, figure_handle=mock_figure)

    # Add the "Right" button
    browser.buttons.add(
        text="Right",
        type_="Push",
        action_func=(lambda s, event: s.increment(step=1)).__get__(browser),
    )
    browser.update()

    # Add the "Right" button
    browser.buttons.add(
        text="Left",
        type_="Push",
        action_func=(lambda s, event: s.decrement(step=1)).__get__(browser),
    )
    browser.update()

    # Assert that the increment method was called
    press_browser_button(browser.buttons["Right"])
    assert browser._current_idx == 1
    assert browser.titlefunc(browser) == "Plot number 1"

    assert browser.buttons["Auto limits"].state is False
    press_browser_button(browser.buttons["Auto limits"])
    assert browser.buttons["Auto limits"].state is True

    press_browser_button(browser.buttons["Left"])
    assert browser._current_idx == 0
    assert browser.titlefunc(browser) == "Plot number 0"


def test_signal_dropdown_on_by_default(signal_list, mock_figure):
    """A signal-selection dropdown is added by default with auto-derived names."""
    browser = SignalBrowser(plot_data=signal_list, figure_handle=mock_figure)

    assert browser.statevariables.names == ["signal"]
    assert browser._signal_var is not None
    assert browser._signal_var.widget == "dropdown"
    assert browser._signal_var.states == ["signal 0", "signal 1", "signal 2"]
    assert browser._signal_var._current_state_idx == browser._current_idx == 0


def test_signal_dropdown_explicit_names(signal_list, mock_figure):
    """signal_names= supplies the dropdown labels."""
    names = ["EMG_L", "EMG_R", "ACC"]
    browser = SignalBrowser(
        plot_data=signal_list, figure_handle=mock_figure, signal_names=names
    )
    assert browser._signal_var.states == names


def test_signal_dropdown_can_be_suppressed(signal_list, mock_figure):
    """show_signal_dropdown=False suppresses the dropdown entirely."""
    browser = SignalBrowser(
        plot_data=signal_list, figure_handle=mock_figure, show_signal_dropdown=False
    )
    assert browser._signal_var is None
    assert browser.statevariables.names == []


def test_signal_dropdown_relabel_replaces(signal_list, mock_figure):
    """Re-calling add_signal_dropdown relabels in place (no duplicate var)."""
    browser = SignalBrowser(plot_data=signal_list, figure_handle=mock_figure)
    assert browser._signal_var.states == ["signal 0", "signal 1", "signal 2"]

    var = browser.add_signal_dropdown(["a", "b", "c"])
    assert browser.statevariables.names == ["signal"]
    assert var.states == ["a", "b", "c"]
    assert browser._signal_var is var


def test_signal_dropdown_pick_moves_index(signal_list, mock_figure):
    """Picking an entry moves the browse index.

    Mirrors the QComboBox on_pick path: set_state(idx) (which fires the
    on-change callback) followed by the browser's own update()."""
    browser = SignalBrowser(
        plot_data=signal_list, figure_handle=mock_figure, signal_names=["a", "b", "c"]
    )
    browser._signal_var.set_state(2)
    browser.update()
    assert browser._current_idx == 2

    browser._signal_var.set_state(1)
    browser.update()
    assert browser._current_idx == 1


def test_signal_dropdown_follows_keyboard_navigation(signal_list, mock_figure):
    """Arrow-key navigation keeps the dropdown's selected index in step."""
    browser = SignalBrowser(
        plot_data=signal_list, figure_handle=mock_figure, signal_names=["a", "b", "c"]
    )
    assert browser._signal_var._current_state_idx == 0

    browser(simulate_key_press(browser.figure, key="right"))
    assert browser._current_idx == 1
    assert browser._signal_var._current_state_idx == 1

    browser(simulate_key_press(browser.figure, key="left"))
    assert browser._current_idx == 0
    assert browser._signal_var._current_state_idx == 0


def test_signal_dropdown_length_mismatch_raises(signal_list, mock_figure):
    """A names list whose length != number of signals is rejected."""
    browser = SignalBrowser(
        plot_data=signal_list, figure_handle=mock_figure, show_signal_dropdown=False
    )
    with pytest.raises(ValueError):
        browser.add_signal_dropdown(["only", "two"])
