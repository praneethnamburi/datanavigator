import pytest
import pysampled

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.backend_bases import MouseEvent

import datanavigator


@pytest.fixture(scope="module")
def matplotlib_figure():
    # used by test_events and test_core
    # Set up: create the figure and axis
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig, ax
    # Tear down: close the figure
    plt.close(fig)


@pytest.fixture
def mock_figure():
    """Fixture to mock a matplotlib figure."""
    fig = plt.figure()
    yield fig
    plt.close(fig)


@pytest.fixture
def signal_list():
    """Fixture to mock pysampled.Data objects."""
    return [
        pysampled.generate_signal("white_noise"),
        pysampled.generate_signal("sine_wave"),
        pysampled.generate_signal("three_sine_waves"),
    ]


def simulate_key_press(figure, key="a"):
    """
    Simulate a key press event on the given axis.

    Args:
        fax (tuple): A tuple containing the figure and axis (fig, ax).
        key (str, optional): The key to press. Defaults to 'a'.
    """
    # Create a KeyEvent
    event = KeyEvent(
        name="key_press_event",
        canvas=figure.canvas,
        key=key,
        guiEvent=None,
    )
    return event


def simulate_key_press_at_xy(fax, key="1", xdata=0.5, ydata=0.5):
    """
    Simulate a key press event on the given axis with the mouse positioned
    at a specific data coordinate. Adds 'xdata' and 'ydata' attributes
    to the event object.

    Args:
        fax (tuple): A tuple containing the figure and axis (fig, ax).
        key (str, optional): The key to press. Defaults to '1'.
        xdata (float, optional): The x-coordinate in data space. Defaults to 0.5.
        ydata (float, optional): The y-coordinate in data space. Defaults to 0.5.
    """
    fig, ax = fax
    # Convert the data coordinates to canvas coordinates
    x_pixel, y_pixel = ax.transData.transform((xdata, ydata))
    # Create a KeyEvent with pixel position information
    event = KeyEvent(
        name="key_press_event",
        canvas=fig.canvas,
        key=key,
        x=x_pixel,
        y=y_pixel,
        guiEvent=None,
    )
    # Manually add the data coordinates to the event object
    event.xdata = xdata
    event.ydata = ydata
    # Add the axes attribute, similar to MouseEvent
    event.inaxes = ax
    return event


def simulate_mouse_click(fax, xdata=0.5, ydata=0.5, button=1):
    """
    Simulate a mouse click event on the given axis.

    Args:
        ax (matplotlib.axes.Axes): The axis to click on.
        xdata (float): The x-coordinate of the click.
        ydata (float): The y-coordinate of the click.
        button (int, optional): The mouse button to use (1 for left, 2 for middle, 3 for right). Defaults to 1.
    """
    fig, ax = fax
    # Create a MouseEvent
    event = MouseEvent(
        name="button_press_event",
        canvas=ax.figure.canvas,
        x=ax.transData.transform((xdata, ydata))[0],
        y=ax.transData.transform((xdata, ydata))[1],
        button=button,
        key=None,
        step=0,
        dblclick=False,
        guiEvent=None,
    )
    return event


def press_browser_button(button: datanavigator.Button):
    """
    Simulates a mouse click on a given button in a Matplotlib-based datanavigator GUI.
    This function triggers both a button press and a button release event
    on the specified button, effectively simulating a full mouse click.

    Args:
        button (datanavigator.Button): The button object to be clicked.
            It must have an `ax` attribute representing the Matplotlib Axes
            associated with the button.

    Raises:
        AttributeError: If the `button` object does not have the required `ax` attribute.
        ValueError: If the button's Axes or its associated canvas is not properly configured.
    """
    if not hasattr(button, "ax"):
        raise AttributeError(
            "The button object must have an 'ax' attribute representing its Matplotlib Axes."
        )

    button_ax = button.ax
    if button_ax is None or button_ax.figure is None or button_ax.figure.canvas is None:
        raise ValueError(
            "The button's Axes or its associated canvas is not properly configured."
        )

    # Calculate the center position of the button in canvas coordinates
    bbox = button_ax.get_position()
    x = bbox.x0 + 0.5 * bbox.width
    y = bbox.y0 + 0.5 * bbox.height
    canvas = button_ax.figure.canvas
    canvas_width, canvas_height = canvas.get_width_height()

    # Convert normalized coordinates to canvas coordinates
    canvas_x = canvas_width * x
    canvas_y = canvas_height * y

    # Create and process a button press event
    press_event = MouseEvent(
        name="button_press_event",
        canvas=canvas,
        x=canvas_x,
        y=canvas_y,
        button=1,
        key=None,
    )
    canvas.callbacks.process("button_press_event", press_event)

    # Create and process a button release event
    release_event = MouseEvent(
        name="button_release_event",
        canvas=canvas,
        x=canvas_x,
        y=canvas_y,
        button=1,
        key=None,
    )
    canvas.callbacks.process("button_release_event", release_event)
