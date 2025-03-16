import pytest
import os
from copy import deepcopy
import numpy as np
from unittest.mock import Mock
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt

from datanavigator.events import EventData, Event, Events, _find_nearest_idx_val
from datanavigator.utils import portion


@pytest.fixture(scope="module")
def matplotlib_figure():
    # Set up: create the figure and axis
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    yield fig, ax
    # Tear down: close the figure
    plt.close(fig)


@pytest.fixture(scope="module")
def event_file_path(tmp_path_factory):
    return tmp_path_factory.mktemp("data") / "picked_event.json"


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


def test_event_data_initialization():
    event_data = EventData()
    assert event_data.default == []
    assert event_data.added == []
    assert event_data.removed == []
    assert event_data.tags == []
    assert event_data.algorithm_name == ""
    assert event_data.params == {}

def test_event_data_asdict():
    event_data = EventData(default=[1], added=[2], removed=[3], tags=["tag"], algorithm_name="algo", params={"param": "value"})
    expected_dict = {
        "default": [1],
        "added": [2],
        "removed": [3],
        "tags": ["tag"],
        "algorithm_name": "algo",
        "params": {"param": "value"},
    }
    assert event_data.asdict() == expected_dict

def test_event_data_len():
    event_data = EventData(default=[[1, 2]], added=[[3, 4]])
    assert len(event_data) == 2

def test_event_data_get_times():
    event_data = EventData(default=[[3, 4]], added=[[1, 2]])
    assert event_data.get_times() == [[1, 2], [3, 4]]

def test_event_data_to_portions():
    event_data = EventData(default=[[1, 2], [3, 4]])
    expected_portion = portion.closed(1, 2) | portion.closed(3, 4)
    assert event_data.to_portions() == expected_portion

def test_event_data_and():
    event_data1 = EventData(default=[[1, 2], [3, 4]])
    event_data2 = EventData(default=[[2.5, 3.5], [4, 5]])
    expected_portion = portion.closed(3, 3.5)
    # Strictly speaking, since I'm using closed intervals, [4] should also be part of the intersection
    # But I'm only using portions for representing intervals, so I'm not too concerned about this
    assert (event_data1 & event_data2).to_portions() == expected_portion

def test_event_data_contains():
    event_data = EventData(default=[[1, 2], [3, 4]])
    assert 1.5 in event_data
    assert 2.5 not in event_data

def test_event_initialization():
    data_id_func = Mock(return_value="test_id")
    event = Event(name="test_event", size=2, fname="test.json", data_id_func=data_id_func)
    assert event.name == "test_event"
    assert event.size == 2
    assert event.fname == "test.json"
    assert event.data_id_func == data_id_func

def test_event_add_remove(matplotlib_figure, event_file_path):
    data_id_func = Mock(return_value="test_id")
    event = Event(name="test_event", size=2, pick_action="overwrite", fname=event_file_path, data_id_func=data_id_func)
    
    # add an event
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.5))
    assert np.allclose(event._buffer, [0.5])
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.7))
    assert event._buffer == []
    assert np.allclose(event._data["test_id"].get_times(), [[0.5, 0.7]])

    # this event should not be added because end > start
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.6))
    assert np.allclose(event._data["test_id"].get_times(), [[0.5, 0.7]])

    # this should overwrite the first event
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.85))
    assert np.allclose(event._data["test_id"].get_times(), [[0.8, 0.85]])

    # add an event when there is one default event
    event = Event(name="test_event", size=2, pick_action="overwrite", fname=event_file_path, data_id_func=data_id_func)
    event._data["test_id"] = EventData(default=[[0.2, 0.25]])
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.85))
    assert np.allclose(event._data["test_id"].get_times(), [[0.8, 0.85]])
    assert np.allclose(event._data["test_id"].removed, [[0.2, 0.25]])

    # add an event when pick_action is append
    event = Event(name="test_event", size=2, pick_action="append", fname=event_file_path, data_id_func=data_id_func)
    event._data["test_id"] = EventData(default=[[0.2, 0.25]])
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.85))
    assert np.allclose(event.to_dict()["test_id"], [[0.2, 0.25], [0.8, 0.85]])

    # delete an event from added, as opposed to removing it
    event.remove(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    assert event._data["test_id"].removed == []

    # remove an event - default events are stored in a "removed" list in the EventData object to keep track of what was removed
    event.remove(simulate_mouse_click(matplotlib_figure, xdata=0.2))
    assert event._data["test_id"].default == []
    assert event._data["test_id"].added == []
    assert np.allclose(event._data["test_id"].removed, [[0.2, 0.25]])
    assert event.to_dict() == {"test_id": []}

def test_event_to_dict():
    data_id_func = Mock(return_value="test_id")
    event = Event(name="test_event", size=2, fname="test.json", data_id_func=data_id_func)
    assert event.to_dict() == {} # test empty event
    event._data["test_id"] = EventData(default=[[1, 2]])
    assert event.to_dict() == {"test_id": [[1, 2]]}

    event = Event(name="test_event", size=2, pick_action="append", fname="test.json", data_id_func=data_id_func)
    event._data["test_id"] = EventData(default=[[1, 2]])
    assert event.to_dict() == {"test_id": [[1, 2]]}

def test_event_save(matplotlib_figure, event_file_path):
    # add an event when pick_action is append
    data_id_func = Mock(return_value="test_id")
    event = Event(name="test_event", size=2, pick_action="append", fname=event_file_path, data_id_func=data_id_func)
    event._data["test_id"] = EventData(default=[[0.2, 0.25], [0.1, 0.17], [0.45, 0.55]])
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.8))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.85))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.4))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.6))
    event.remove(simulate_mouse_click(matplotlib_figure, xdata=0.45))
    event.save()

def test_event_from_file(event_file_path):
    assert os.path.exists(event_file_path)
    event = Event.from_file(fname=event_file_path)
    # to_dict will sort the event entries
    assert np.allclose(event.to_dict()["test_id"], [[0.1, 0.17], [0.2, 0.25], [0.4, 0.6], [0.8, 0.85]])
    # events in added, default, and removed will remain in the sequence they were added
    assert np.allclose(event._data["test_id"].default, [[0.2, 0.25], [0.1, 0.17]])
    assert np.allclose(event._data["test_id"].added, [[0.8, 0.85], [0.4, 0.6]])
    assert np.allclose(event._data["test_id"].removed, [[0.45, 0.55]])
    os.remove(event_file_path)

def test_event_from_data(matplotlib_figure):
    data = {
        (1,1): [[0.1, 0.2], [0.3, 0.4]], 
        (1,2): [[0.5, 0.6]],
        (2,1): [[0.7, 0.8], [0.9, 1.0], [0.85, 0.87]],
        (2,2): [],
        }
    event = Event.from_data(deepcopy(data), name="test_event")
    # only the last events are kept because pick_action is overwrite by default
    assert np.allclose(event._data[(2,1)].default, [[0.85, 0.87]])
    assert np.allclose(event.to_dict()[(2,1)], [[0.85, 0.87]]) # because pick_action is overwrite

    event.data_id_func = Mock(return_value=(1,2))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.75))
    event.add(simulate_mouse_click(matplotlib_figure, xdata=0.82))
    assert np.allclose(event._data[(1,2)].default, [])
    assert np.allclose(event._data[(1,2)].removed, [[0.5, 0.6]])
    assert np.allclose(event._data[(1,2)].added, [[0.75, 0.82]])
    # added events will be treated as being added after default
    assert np.allclose(event.to_dict()[(1,2)], [[0.75, 0.82]])

    event = Event.from_data(deepcopy(data), name="test_event", pick_action="append")
    # to_dict will sort the event entries
    assert np.allclose(event.to_dict()[(2,1)], [[0.7, 0.8], [0.85, 0.87], [0.9, 1.0]])

    data = event._data # dict[tuple, EventData]
    event = Event.from_data(data, name="test_event")

def test_events_initialization():
    parent = Mock()
    events = Events(parent)
    assert events.parent == parent

def test_events_add():
    parent = Mock()
    events = Events(parent)
    data_id_func = Mock(return_value="test_id")
    event = events.add(name="test_event", size=2, fname="test.json", data_id_func=data_id_func, color="blue")
    assert event.name == "test_event"
    assert event.size == 2

# def test_events_add_from_file():
#     parent = Mock()
#     events = Events(parent)
#     data_id_func = Mock(return_value="test_id")
#     with open("test.json", "w") as f:
#         f.write('{"name": "test_event", "size": 2, "fname": "test.json", "color": "blue", "pick_action": "overwrite", "win_remove": [-0.1, 0.1], "win_add": [-0.25, 0.25], "plot_kwargs": {}}')
#     event = events.add_from_file(fname="test.json", data_id_func=data_id_func)
#     assert event.name == "test_event"
#     assert event.size == 2

def test_find_nearest_idx_val():
    # Test with a simple array
    array = [1.0, 2.0, 3.0, 4.0, 5.0]
    value = 3.3
    idx, val = _find_nearest_idx_val(array, value)
    assert idx == 2
    assert val == 3.0

    # Test with an array containing negative values
    array = [-5.0, -3.0, -1.0, 1.0, 3.0, 5.0]
    value = -2.5
    idx, val = _find_nearest_idx_val(array, value)
    assert idx == 1
    assert val == -3.0

    # Test with an array containing duplicate values
    array = [1.0, 2.0, 2.0, 3.0, 4.0]
    assert np.allclose(_find_nearest_idx_val(array, 2.4), (1, 2.0))
    assert np.allclose(_find_nearest_idx_val(array, 2.6), (3, 3.0))

    # Test with an empty array
    array = []
    value = 1.0
    with pytest.raises(ValueError):
        _find_nearest_idx_val(array, value)

    # Test with a single-element array
    array = [1.0]
    value = 0.5
    idx, val = _find_nearest_idx_val(array, value)
    assert idx == 0
    assert val == 1.0
