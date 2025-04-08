import pytest
from unittest.mock import MagicMock
from unittest.mock import Mock, patch
from matplotlib.backend_bases import MouseEvent

from datanavigator.signals import SignalBrowser
from tests.conftest import simulate_key_press


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
    browser = SignalBrowser(plot_data=signal_list, titlefunc=custom_titlefunc, figure_handle=mock_figure)

    assert browser.titlefunc == custom_titlefunc
    assert browser.titlefunc(browser) == "Custom Title"

def test_signal_browser_init_with_reset_on_change(signal_list, mock_figure):
    """Test initialization with reset_on_change set to True."""
    browser = SignalBrowser(plot_data=signal_list, reset_on_change=True, figure_handle=mock_figure)

    assert browser.reset_on_change is True

def test_signal_browser_buttons_added(signal_list, mock_figure):
    """Test that buttons are added during initialization and simulate a mouse click on the 'Right' button."""
    browser = SignalBrowser(plot_data=signal_list, figure_handle=mock_figure)
    
    # Add the "Right" button
    browser.buttons.add(
        text="Right",
        type_="Push",
        action_func=browser.increment,
    )
    browser.update()
    
    # Get the "Right" button's Axes
    right_button = browser.buttons["Right"]
    button_ax = right_button.ax

    # Simulate a mouse click on the button
    x, y = button_ax.get_position().x0 + 0.5 * button_ax.get_position().width, \
           button_ax.get_position().y0 + 0.5 * button_ax.get_position().height
    canvas = button_ax.figure.canvas
    canvas_width, canvas_height = canvas.get_width_height()

    # Create and process a button press event
    press_event = MouseEvent(
        name="button_press_event",
        canvas=canvas,
        x=canvas_width * x,
        y=canvas_height * y,
        button=1,
        key=None,
    )
    browser(press_event)

    # Create and process a button release event
    release_event = MouseEvent(
        name="button_release_event",
        canvas=canvas,
        x=canvas_width * x,
        y=canvas_height * y,
        button=1,
        key=None,
    )
    canvas.callbacks.process("button_release_event", release_event)

    # Assert that the increment method was called
    assert browser._current_idx == 1
    assert browser.titlefunc(browser) == "Plot number 1"

