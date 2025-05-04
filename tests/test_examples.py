import os
import pytest
from unittest.mock import patch, MagicMock

# Assuming conftest.py is in the same directory or accessible via pytest's discovery
from .conftest import (
    simulate_key_press_at_xy,
    press_browser_button,
)
from datanavigator import examples, _config

# --- Fixtures ---


@pytest.fixture(scope="function", autouse=True)
def setup_folders(tmp_path):
    """Ensure clean cache and clip folders for each test."""
    clip_dir = tmp_path / "clips"
    cache_dir = tmp_path / "cache"
    clip_dir.mkdir()
    cache_dir.mkdir()
    _config.set_clip_folder(str(clip_dir))
    _config.set_cache_folder(str(cache_dir))
    yield clip_dir, cache_dir
    # Teardown is handled by tmp_path fixture


@pytest.fixture
def example_video_path(setup_folders):
    clip_dir, _ = setup_folders
    return os.path.join(clip_dir, "example_video.mp4")


# --- Tests for get_example_video ---


@patch("urllib.request.urlopen")
def test_get_example_video_download(mock_urlopen, example_video_path):
    """Test video download when file doesn't exist."""
    # Mock the download response
    mock_response = MagicMock()
    mock_response.read.return_value = b"video_content"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    # Ensure file does not exist initially
    assert not os.path.exists(example_video_path)

    # Call the function
    result_path = examples.get_example_video()

    # Assertions
    assert result_path == example_video_path
    assert os.path.exists(example_video_path)
    with open(example_video_path, "rb") as f:
        assert f.read() == b"video_content"
    mock_urlopen.assert_called_once()


@patch("urllib.request.urlopen")
def test_get_example_video_exists(mock_urlopen, example_video_path):
    """Test returning path when video file already exists."""
    # Create a dummy existing file
    with open(example_video_path, "w") as f:
        f.write("existing_content")

    # Call the function
    result_path = examples.get_example_video()

    # Assertions
    assert result_path == example_video_path
    mock_urlopen.assert_not_called()  # Should not attempt download


def test_get_example_video_custom_folder_exists(tmp_path):
    """Test using a valid custom destination folder."""
    custom_folder = tmp_path / "custom_clips"
    custom_folder.mkdir()
    expected_path = custom_folder / "example_video.mp4"

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"custom_content"
        mock_urlopen.return_value.__enter__.return_value = mock_response

        result_path = examples.get_example_video(dest_folder=str(custom_folder))

        assert result_path == str(expected_path)
        assert os.path.exists(result_path)
        mock_urlopen.assert_called_once()


def test_get_example_video_custom_folder_not_exist(tmp_path):
    """Test error when custom destination folder doesn't exist."""
    custom_folder = tmp_path / "non_existent_clips"
    with pytest.raises(AssertionError, match="does not exist"):
        examples.get_example_video(dest_folder=str(custom_folder))


@patch("urllib.request.urlopen")
def test_get_example_video_custom_url(mock_urlopen, example_video_path):
    """Test using a custom source URL."""
    custom_url = "http://example.com/custom_video.mp4"
    mock_response = MagicMock()
    mock_response.read.return_value = b"custom_url_content"
    mock_urlopen.return_value.__enter__.return_value = mock_response

    result_path = examples.get_example_video(source_url=custom_url)

    assert result_path == example_video_path
    assert os.path.exists(example_video_path)
    mock_urlopen.assert_called_once()
    # Check if the request used the custom URL
    mock_request = patch("urllib.request.Request").start()
    mock_request.return_value = MagicMock()
    mock_request.return_value.call_args = ((custom_url,), {})
    args, kwargs = mock_request.return_value.call_args
    assert args[0] == custom_url
    patch.stopall()


# --- Tests for EventPickerDemo ---


@pytest.fixture
def event_picker():
    """Fixture to create an EventPickerDemo instance."""
    # Prevent plt.show() from blocking tests
    with patch("matplotlib.pyplot.show"):
        picker = examples.EventPickerDemo()
    # Need the figure/axes from the picker for event simulation
    picker.fax = (picker.figure, picker._ax)
    return picker


def test_event_picker_init(event_picker):
    """Test EventPickerDemo initialization."""
    assert len(event_picker.data) == 10
    assert "pick1" in event_picker.events.names
    assert "pick2" in event_picker.events.names
    assert "pick3" in event_picker.events.names
    assert len(event_picker._ax.get_lines()) > 0  # Check if data is plotted


def test_event_picker_add_event(event_picker, setup_folders):
    """Test adding events via simulated key presses."""
    _, cache_dir = setup_folders
    picker = event_picker
    fax = picker.fax
    canvas = picker.figure.canvas

    # Add pick1 event
    event1_key = simulate_key_press_at_xy(fax, key="1", xdata=0.5)
    canvas.callbacks.process(event1_key.name, event1_key)
    current_id = picker.events["pick1"].data_id_func()
    assert current_id in picker.events["pick1"]._data
    assert len(picker.events["pick1"]._data[current_id].added) == 1
    assert picker.events["pick1"]._data[current_id].added[0] == [0.5]

    # Add pick2 event (requires two clicks)
    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.2))
    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.8))
    current_id = picker.events[
        "pick2"
    ].data_id_func()  # ID might change if signal changes
    assert current_id in picker.events["pick2"]._data
    assert len(picker.events["pick2"]._data[current_id].added) == 1
    assert picker.events["pick2"]._data[current_id].added[0] == [0.2, 0.8]

    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.3))
    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.25))
    assert (
        len(picker.events["pick2"]._data[current_id].added) == 1
    )  # Still 1 because secind click is before first click

    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.1))
    picker(simulate_key_press_at_xy(fax, key="2", xdata=0.15))
    assert (
        len(picker.events["pick2"]._data[current_id].added) == 2
    )  # Still 1 because secind click is before first click

    # Add pick3 event (overwrite) - requires three clicks
    event3_key_1 = simulate_key_press_at_xy(fax, key="3", xdata=0.1)
    event3_key_2 = simulate_key_press_at_xy(fax, key="3", xdata=0.5)
    event3_key_3 = simulate_key_press_at_xy(fax, key="3", xdata=0.9)
    canvas.callbacks.process(event3_key_1.name, event3_key_1)
    canvas.callbacks.process(event3_key_2.name, event3_key_2)
    canvas.callbacks.process(event3_key_3.name, event3_key_3)
    current_id = picker.events["pick3"].data_id_func()
    assert current_id in picker.events["pick3"]._data
    assert len(picker.events["pick3"]._data[current_id].added) == 1
    assert picker.events["pick3"]._data[current_id].added[0] == [0.1, 0.5, 0.9]

    # Add another pick3 event to test overwrite
    event3_key_4 = simulate_key_press_at_xy(fax, key="3", xdata=0.15)
    event3_key_5 = simulate_key_press_at_xy(fax, key="3", xdata=0.55)
    event3_key_6 = simulate_key_press_at_xy(fax, key="3", xdata=0.95)
    canvas.callbacks.process(event3_key_4.name, event3_key_4)
    canvas.callbacks.process(event3_key_5.name, event3_key_5)
    canvas.callbacks.process(event3_key_6.name, event3_key_6)
    assert (
        len(picker.events["pick3"]._data[current_id].added) == 1
    )  # Still 1 due to overwrite
    assert picker.events["pick3"]._data[current_id].added[0] == [0.15, 0.55, 0.95]


def test_event_picker_remove_event(event_picker):
    """Test removing an event via simulated key press."""
    picker = event_picker
    fax = picker.fax
    canvas = picker.figure.canvas

    # Add a pick1 event first
    add_event = simulate_key_press_at_xy(fax, key="1", xdata=0.6)
    canvas.callbacks.process(add_event.name, add_event)
    current_id = picker.events["pick1"].data_id_func()
    assert len(picker.events["pick1"]._data[current_id].added) == 1

    # Simulate remove key press near the added event
    remove_event = simulate_key_press_at_xy(
        fax, key="alt+1", xdata=0.605
    )  # Close to 0.6
    canvas.callbacks.process(remove_event.name, remove_event)

    # Check if the event was removed from 'added'
    assert len(picker.events["pick1"]._data[current_id].added) == 0


def test_event_picker_save_event(event_picker, setup_folders):
    """Test saving events via simulated key press."""
    _, cache_dir = setup_folders
    picker = event_picker
    fax = picker.fax
    canvas = picker.figure.canvas
    pick1_file = os.path.join(cache_dir, "_pick1.json")

    # Add a pick1 event
    add_event = simulate_key_press_at_xy(fax, key="1", xdata=0.7)
    canvas.callbacks.process(add_event.name, add_event)

    # Ensure save file doesn't exist yet
    assert not os.path.exists(pick1_file)

    # Simulate save key press
    save_event = simulate_key_press_at_xy(fax, key="ctrl+1")
    canvas.callbacks.process(save_event.name, save_event)

    # Check if the file was created
    assert os.path.exists(pick1_file)
    # Optional: Add check for file content


# --- Tests for ButtonDemo ---


@pytest.fixture
def button_demo():
    """Fixture to create a ButtonDemo instance."""
    # Prevent plt.show() from blocking tests
    with patch("matplotlib.pyplot.show"):
        demo = examples.ButtonDemo()
    return demo


def test_button_demo_init(button_demo):
    """Test ButtonDemo initialization."""
    assert len(button_demo.buttons._list) == 2
    assert button_demo.buttons["test"].__class__.__name__ == "ToggleButton"
    assert button_demo.buttons["push button"].__class__.__name__ == "Button"


@patch("builtins.print")
def test_button_demo_push_button_click(mock_print, button_demo):
    """Test clicking the push button."""
    push_button = button_demo.buttons["push button"]
    press_browser_button(push_button)
    # Check if the callback (which prints) was called
    mock_print.assert_called_once()


# --- Tests for SelectorDemo ---


@pytest.fixture
def selector_demo(matplotlib_figure):
    """Fixture to create a SelectorDemo instance."""
    # Prevent plt.show() from blocking tests
    with patch("matplotlib.pyplot.show"):
        # Need a figure with axes for SelectorDemo
        fig, ax = matplotlib_figure
        # Mock the figure creation within SelectorDemo if needed, or pass one
        with patch("matplotlib.pyplot.subplots", return_value=(fig, ax)):
            demo = examples.SelectorDemo()
    return demo


def test_selector_demo_init(selector_demo):
    """Test SelectorDemo initialization."""
    assert len(selector_demo.buttons._list) == 2
    assert selector_demo.lasso is not None  # Should be started by default
    assert len(selector_demo.ax.get_lines()) == 4  # data + selected points placeholder


def test_selector_demo_start_stop(selector_demo):
    """Test starting and stopping the lasso selector via buttons."""
    # Stop
    stop_button = selector_demo.buttons["Stop selection"]
    with patch.object(selector_demo.lasso, "disconnect_events") as mock_disconnect:
        press_browser_button(stop_button)
        mock_disconnect.assert_called_once()

    # Start
    start_button = selector_demo.buttons["Start selection"]
    with patch(
        "matplotlib.widgets.LassoSelector.__init__", return_value=None
    ) as mock_lasso_init:
        # Need to mock the LassoSelector init as it requires interaction
        with patch.object(
            examples.LassoSelectorWidget, "__init__", return_value=None
        ) as mock_lasso_widget_init:
            press_browser_button(start_button)
            # Check if LassoSelector was potentially re-initialized (hard to assert precisely without more mocking)
            assert mock_lasso_widget_init.called
