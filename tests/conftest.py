import pytest
import pysampled

import matplotlib.pyplot as plt
from matplotlib.backend_bases import KeyEvent
from matplotlib.backend_bases import MouseEvent

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

def simulate_key_press(figure, key='a'):
    """
    Simulate a key press event on the given axis.

    Args:
        fax (tuple): A tuple containing the figure and axis (fig, ax).
        key (str, optional): The key to press. Defaults to 'a'.
    """
    # Create a KeyEvent
    event = KeyEvent(
        name='key_press_event',
        canvas=figure.canvas,
        key=key,
        guiEvent=None,
    )
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
        name='button_press_event',
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
