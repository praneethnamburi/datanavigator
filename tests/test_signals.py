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
