"""
This module provides classes and functions for managing events and event data.

Classes:
    :py:class:`EventData` - Manage the data from one event type in one trial.
    :py:class:`Event` - Manage selection of a sequence of events.
    :py:class:`Events` - Manager for event objects.
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt

from . import utils
from .assets import AssetContainer

PLOT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class EventData:
    """
    Manage the data from one event type in one trial.

    Attributes:
        default (list): Default events created by an algorithm.
        added (list): Manually added events. Note that if an 'added' point is removed, then it will simply be deleted. There will be no record of it.
        removed (list): Events removed from the default list.
        tags (list): Tags associated with the events.
        algorithm_name (str): Name of the algorithm used to generate the default list.
        params (dict): Parameters used to generate the default list.
    """

    def __init__(
        self,
        default: Optional[List] = None,
        added: Optional[List] = None,
        removed: Optional[List] = None,
        tags: Optional[List] = None,
        algorithm_name: str = "",
        params: Optional[Dict] = None,
    ) -> None:
        _to_list = lambda x: [] if x is None else x

        self.default = _to_list(default)
        self.added = _to_list(added)
        self.removed = _to_list(removed)
        self.tags = _to_list(tags)
        self.algorithm_name = algorithm_name
        self.params = params if params is not None else {}

    def asdict(self) -> Dict:
        """Convert the event data to a dictionary."""
        return dict(
            default=self.default,
            added=self.added,
            removed=self.removed,
            tags=self.tags,
            algorithm_name=self.algorithm_name,
            params=self.params,
        )

    def __len__(self) -> int:
        """Return the number of events."""
        return len(self.get_times())

    def get_times(self) -> List:
        """Get the times of all events."""
        x = self.default + self.added
        x.sort()
        return x

    def to_portions(self) -> "utils.portion.Interval":
        """Convert the event times to portions."""
        return functools.reduce(
            lambda a, b: a | b,
            [
                utils.portion.closed(*interval_limits)
                for interval_limits in self.get_times()
            ],
        )

    def __and__(self, other: EventData) -> EventData:
        """Return the intersection of two EventData objects."""
        try:
            x = self.to_portions() & other.to_portions()
        except TypeError:
            raise TypeError("Intersection is only implemented for size-2 events (start and stop times).")
        return EventData(default=[[xi.lower, xi.upper] for xi in x if xi.upper != xi.lower]) # only return non-zero duration events

    def __contains__(self, item: Any) -> bool:
        """Check if an item is in the event data."""
        return item in self.to_portions()

    @staticmethod
    def _process_inp(
        other: Union["utils.portion.Interval", Tuple]
    ) -> utils.portion.Interval:
        """Process the input to ensure it is an interval."""
        if not isinstance(other, utils.portion.Interval):
            assert len(other) == 2
            other = utils.portion.closed(*other)
        return other

    def overlap_duration(self, other: Union["utils.portion.Interval", Tuple]) -> float:
        """Calculate the duration of overlap with another interval."""
        other = self._process_inp(other)
        return (self.to_portions() & other).duration

    def overlap_frac(self, other: Union["utils.portion.Interval", Tuple]) -> float:
        """Calculate the fraction of overlap with another interval."""
        other = self._process_inp(other)
        return self.overlap_duration(other) / other.duration


class Event:
    """
    Manage selection of a sequence of events (of length >= 1).

    Attributes:
        name (str): Name of the event.
        size (int): Length of the sequence.
        fname (str): File name to load and save events.
        data_id_func (Callable): Function to get the current data ID from the parent UI.
        color (str): Color of the event. Defaults to a random color.
        pick_action (str): Action to take when picking an event ('overwrite' or 'append'). Use overwrite if there can only be one sequence per 'signal'. If there can be multiple, use 'append'. Defaults to overwrite
        ax_list (List): List of axes on which to show the event.
        win_remove (Tuple): Window relative to the mouse position to search for removing an event. Defaults to (-0.1, 0.1).
        win_add (Tuple): Window relative to the mouse position to search for adding an event. Defaults to (-0.25, 0.25).
        data_func (Callable): Function to process the data.
        plot_kwargs (Dict): Keyword arguments for plotting. Commonly used keys are 'display_type' (line or fill) and 'alpha' to set the transparency level.
    """

    def __init__(
        self,
        name: str,
        size: int,
        fname: str,
        data_id_func: Optional[Callable] = None,
        color: Union[str, int] = "random",
        pick_action: str = "overwrite",
        ax_list: Optional[List] = None,
        win_remove: Tuple[float, float] = (-0.1, 0.1),
        win_add: Tuple[float, float] = (-0.25, 0.25),
        data_func: Callable = float,
        **plot_kwargs,
    ) -> None:
        self.name = name
        assert isinstance(size, int) and size > 0
        self.size = size
        self.fname = fname
        self.data_id_func = data_id_func
        if isinstance(color, int):
            color = PLOT_COLORS[color]
        elif color == "random":
            color = np.random.choice(PLOT_COLORS)
        self.color = color
        assert pick_action in ("overwrite", "append")
        self.pick_action = pick_action

        self._buffer = []
        _, self._data = self.load()

        if ax_list is None:
            ax_list = []
        self.ax_list = ax_list
        self.plot_handles = []

        self.win_remove = win_remove
        self.win_add = win_add
        self.plot_kwargs = plot_kwargs
        self._hide = False
        self.data_func = data_func

    def initialize_event_data(self, data_id_list: List) -> None:
        """Initialize event data for a list of data IDs."""
        for data_id in data_id_list:
            if data_id not in self._data:
                self._data[data_id] = EventData()

    @classmethod
    def _from_existing_file(
        cls, fname: str, data_id_func: Optional[Callable] = None
    ) -> Event:
        """Create an Event object by reading an existing JSON file."""
        h, _ = cls._read_json_file(fname)
        return cls(
            h["name"],
            h["size"],
            fname,
            data_id_func,
            h["color"],
            h["pick_action"],
            None,
            h["win_remove"],
            h["win_add"],
            **h["plot_kwargs"],
        )

    @classmethod
    def from_file(cls, fname: str, **kwargs) -> Event:
        """
        Create an empty events file with the given file name (fname) and any parameters.
        Assigns best-guess defaults.
        """
        if not os.path.exists(fname):
            kwargs["name"] = kwargs.get("name", Path(fname).stem)
            kwargs["size"] = kwargs.get("size", 1)
            kwargs["data_id_func"] = kwargs.get("data_id_func", None)

            ret = cls(fname=fname, **kwargs)
            ret.save()
            return ret
        return cls._from_existing_file(fname, kwargs.get("data_id_func", None))

    @classmethod
    def from_data(
        cls,
        data: Dict,
        name: str = "Event",
        fname: str = "",
        overwrite: bool = False,
        **kwargs,
    ) -> Event:
        """
        Create an event file by filling in the 'default' events extracted by an algorithm.

        Args:
            data (dict): Data to create the event file.
            name (str): Name of the event.
            fname (str): File name to save the event.
            overwrite (bool): Whether to overwrite the existing file.
            **kwargs: Additional keyword arguments.
                tags, algorithm_name, and params will be passed to :py:class:`EventData`.
                All other keyword arguments will be passed to :py:class:`Event`.

        Returns:
            Event: The created Event object.
        """
        algorithm_info = dict(
            tags=kwargs.pop("tags", []),
            algorithm_name=kwargs.pop("algorithm_name", ""),
            params=kwargs.pop("params", {}),
        )

        size = []
        for key, val in data.items():
            if isinstance(val, EventData):
                continue
            v = np.asarray(val)
            if v.ndim == 1:  # passing in a list events of size 1
                v = v[:, np.newaxis]
                val = [list(x) for x in v]
            data[key] = EventData(default=val, **algorithm_info)
            if len(data[key]) > 0:
                # when there are no events, size cannot be inferred for that trial. Note that this process will fail if there are no events in ANY of the trials. size has to be passed in with kwargs.
                size.append(v.shape[-1])

        if not size:
            # if there were no events in the data that was passed!
            assert "size" in kwargs
            size = kwargs["size"]
        else:
            size = list(set(size))
            assert len(size) == 1  # make sure all the events are of the same type
            size = size[0]

        if "size" in kwargs:
            assert kwargs["size"] == size
            del kwargs["size"]

        ret = cls(name, size, fname, **kwargs)
        ret._data = data
        if utils.is_path_exists_or_creatable(fname):
            if (not os.path.exists(fname)) or overwrite:
                ret.save()
            else:
                # if the file exists and we decided not to overwrite, then append new data to the file
                assert os.path.exists(fname) and (not overwrite)
                ret_existing = cls.from_file(fname, **kwargs)
                new_keys = set(ret._data.keys()) - set(ret_existing._data.keys())
                if len(new_keys) > 0:
                    # if there is new data, then add it to the event file
                    print(f"Appending new data to the event file {fname}:")
                    print(new_keys)
                    ret_existing._data = {**ret._data, **ret_existing._data}
                    ret_existing.save()
        return ret

    def all_keys_are_tuples(self) -> bool:
        """Check if all keys are tuples."""
        return all([type(x) == tuple for x in self._data.keys()])

    def get_header(self) -> Dict:
        """Get the header information of the event."""
        return dict(
            name=self.name,
            size=self.size,
            fname=self.fname,
            color=self.color,
            pick_action=self.pick_action,
            win_remove=self.win_remove,
            win_add=self.win_add,
            plot_kwargs=self.plot_kwargs,
            all_keys_are_tuples=self.all_keys_are_tuples(),
        )

    @staticmethod
    def _read_json_file(fname: str) -> Tuple[Dict, Dict]:
        """Read a JSON file and return the header and data."""
        with open(fname, "r") as f:
            header, data = json.load(f)
        if header["all_keys_are_tuples"]:
            data = {eval(k): EventData(**v) for k, v in data.items()}
        else:
            data = {k: EventData(**v) for k, v in data.items()}
        return header, data

    def load(self) -> Tuple[Dict, Dict]:
        """Load the event data from the file."""
        if os.path.exists(self.fname):
            header, data = self._read_json_file(self.fname)
            return header, data
        return {}, {}

    def save(self) -> None:
        """Save the event data to the file."""
        action_str = "Updated" if os.path.exists(self.fname) else "Created"
        with open(self.fname, "w") as f:
            header = self.get_header()
            if header["all_keys_are_tuples"]:
                data = {str(k): v.asdict() for k, v in self._data.items()}
            else:
                data = {k: v.asdict() for k, v in self._data.items()}
            json.dump((header, data), f, indent=4)
        print(action_str + " " + self.fname)

    def add(self, event: Any) -> None:
        """
        Pick the time points of an interval and associate it with a supplied ID.
        If the first selection is outside the axis, then select the first available time point.
        If the last selection is outside the axis, then select the last available time point.
        If the selections are not monotonically increasing, then empty the buffer.
        If any of the 'middle' picks (i.e. not first or last in the sequence) are outside the axes, then empty the buffer.

        The parent UI would invoke this method.

        Args:
            event (Any): The event to add.
        """

        def strictly_increasing(_list: List) -> bool:
            return all(x < y for x, y in zip(_list, _list[1:]))

        def _get_lines() -> List:
            """Return non-empty lines in the axis where event was invoked, or else in all lines in the figure."""
            if event.inaxes is not None:
                return [
                    line
                    for line in event.inaxes.get_lines()
                    if len(line.get_xdata()) > 0
                ]
            return [
                line
                for ax in event.canvas.figure.axes
                for line in ax.get_lines()
                if len(line.get_xdata()) > 0
            ]

        def _get_first_available_timestamp() -> float:
            return min(
                [
                    np.nanmin(l.get_xdata())
                    for l in _get_lines()
                    if len(l.get_xdata()) > 0
                ]
            )

        def _get_last_available_timestamp() -> float:
            return max(
                [
                    np.nanmax(l.get_xdata())
                    for l in _get_lines()
                    if len(l.get_xdata()) > 0
                ]
            )

        def clamp(n: float) -> float:
            smallest = _get_first_available_timestamp()
            largest = _get_last_available_timestamp()
            return max(smallest, min(n, largest))

        if event.xdata is None:
            if not self._buffer:
                inferred_timestamp = _get_first_available_timestamp()
            else:
                assert len(self._buffer) == self.size - 1
                inferred_timestamp = _get_last_available_timestamp()
        else:
            inferred_timestamp = clamp(self.data_func(event.xdata))

        self._buffer.append(inferred_timestamp)

        if not strictly_increasing(self._buffer):
            self._buffer = []

        if len(self._buffer) < self.size:
            return

        assert len(self._buffer) == self.size

        sequence = self._buffer.copy()

        data_id = self.data_id_func()
        if data_id not in self._data:
            self._data[data_id] = EventData()
        if self.pick_action == "append":
            self._data[data_id].added.append(sequence)
        else: # overwrite
            self._data[data_id].added = [sequence]
            assert len(self._data[data_id].default) in (0, 1)
            if len(self._data[data_id].default) == 1:
                # remove the event in default and add the new one to added
                self._data[data_id].removed.append(self._data[data_id].default[0])
                self._data[data_id].default = []

        print(self.name, "add", data_id, sequence)
        self._buffer = []
        self.update_display()

    def remove(self, event: Any) -> None:
        """
        Remove an event.

        Args:
            event (Any): The event to remove.
        """
        if event.xdata is None:
            return
        t_marked = float(event.xdata)
        data_id = self.data_id_func()
        if data_id not in self._data:
            return
        ev = self._data[data_id]

        added_start_times = [x[0] for x in ev.added]
        default_start_times = [x[0] for x in ev.default]
        sequence = None
        _removed = False
        _deleted = False
        if len(ev.added) > 0 and len(ev.default) > 0:
            idx_add, val_add = _find_nearest_idx_val(added_start_times, t_marked)
            idx_def, val_def = _find_nearest_idx_val(default_start_times, t_marked)

            add_dist = np.abs(val_add - t_marked)
            def_dist = np.abs(val_def - t_marked)
            if (add_dist <= def_dist) and (
                self.win_remove[0] < add_dist < self.win_remove[1]
            ):
                sequence = ev.added.pop(idx_add)
                _deleted = True
            if (def_dist < add_dist) and (
                self.win_remove[0] < def_dist < self.win_remove[1]
            ):
                sequence = ev.default.pop(idx_def)
                ev.removed.append(sequence)
                _removed = True
        elif len(ev.added) > 0 and len(ev.default) == 0:
            idx_add, val_add = _find_nearest_idx_val(added_start_times, t_marked)
            add_dist = np.abs(val_add - t_marked)
            if self.win_remove[0] < add_dist < self.win_remove[1]:
                sequence = ev.added.pop(idx_add)
                _deleted = True
        elif len(ev.added) == 0 and len(ev.default) > 0:
            idx_def, val_def = _find_nearest_idx_val(default_start_times, t_marked)
            def_dist = np.abs(val_def - t_marked)
            if self.win_remove[0] < def_dist < self.win_remove[1]:
                sequence = ev.default.pop(idx_def)
                ev.removed.append(sequence)
                _removed = True
        else:
            return

        if sequence is None:
            return

        assert _removed is not _deleted
        # removed moves data from default (i.e. auto-detected) to removed, and delete expunges a manually added event
        print(self.name, {True: "remove", False: "delete"}[_removed], data_id, sequence)
        self.update_display()

    def get_current_event_times(self) -> List:
        """Get the current event times."""
        return list(
            np.array(
                self._data.get(self.data_id_func(), EventData()).get_times()
            ).flatten()
        )

    def _get_display_funcs(self) -> Tuple[Callable, Callable]:
        """Get the display functions."""
        display_type = self.plot_kwargs.get("display_type", "line")
        assert display_type in ("line", "fill")
        if display_type == "fill":
            assert self.size == 2
        if display_type == "line":
            return self._setup_display_line, self._update_display_line
        return self._setup_display_fill, self._update_display_fill

    def setup_display(self) -> None:
        """Setup event display on one or more axes."""
        setup_func, _ = self._get_display_funcs()
        setup_func()

    def _setup_display_line(self) -> None:
        """Setup line display for the event."""
        plot_kwargs = {**{"label": f"event:{self.name}"}, ** self.plot_kwargs}
        plot_kwargs.pop("display_type", None)
        for ax in self.ax_list:
            (this_plot,) = ax.plot([], [], color=self.color, **plot_kwargs)
            self.plot_handles.append(this_plot)

    def _setup_display_fill(self) -> None:
        """Setup fill display for the event. Everything is redrawm currently for fill display. So, don't do setup."""
        return

    def update_display(self, draw: bool = True) -> None:
        """Update the event display."""
        _, update_func = self._get_display_funcs()
        update_func(draw)

    def _get_ylim(self, this_ax: plt.Axes, type: str = "data") -> Tuple[float, float]:
        """Get the y-axis limits."""
        if type == "data":
            try:
                x = np.asarray(
                    [
                        (np.nanmin(line.get_ydata()), np.nanmax(line.get_ydata()))
                        for line in this_ax.get_lines()
                        if not line.get_label().startswith("event:")
                    ]
                )
                return np.nanmin(x[:, 0]), np.nanmax(x[:, 1])
            except ValueError:
                return this_ax.get_ylim()
        return this_ax.get_ylim()

    def _update_display_line(self, draw: bool) -> None:
        """Update the display for 'line' type events."""
        for ax, plot_handle in zip(self.ax_list, self.plot_handles):
            yl = self._get_ylim(ax)
            plot_handle.set_data(
                *utils.ticks_from_times(self.get_current_event_times(), yl)
            )
        if draw:
            plt.draw()

    def _update_display_fill(self, draw: bool) -> None:
        """Update the display for 'fill' type events."""
        if self._hide:
            return
        for plot_handle in self.plot_handles:
            plot_handle.remove()
        self.plot_handles = []
        plot_kwargs = {**dict(alpha=0.2, edgecolor=None), **self.plot_kwargs}
        plot_kwargs.pop("display_type", None)
        for ax in self.ax_list:
            yl = self._get_ylim(ax)
            x = np.asarray(
                [
                    this_times + [np.nan]
                    for this_times in self._data.get(
                        self.data_id_func(), EventData()
                    ).get_times()
                ]
            ).flatten()
            y1 = np.asarray(
                [[yl[0]] * 2 + [np.nan] for _ in range(int(len(x) / 3))]
            ).flatten()
            y2 = np.asarray(
                [[yl[1]] * 2 + [np.nan] for _ in range(int(len(x) / 3))]
            ).flatten()
            this_collection = ax.fill_between(
                x, y1, y2, color=self.color, **plot_kwargs
            )
            self.plot_handles.append(this_collection)
        if draw:
            plt.draw()

    def to_dict(self) -> Dict:
        """Convert the event data to a dictionary."""
        event_data = self._data
        if self.pick_action == "overwrite":
            ret = {k: v.get_times()[:1] for k, v in event_data.items()}
        else:
            ret = {k: v.get_times() for k, v in event_data.items()}
        return ret

    def to_portions(self) -> Dict:
        """Convert the event data to portions."""
        assert self.size == 2
        P = utils.portion
        ret = {}
        for signal_id, signal_events in self.to_dict().items():
            ret[signal_id] = functools.reduce(
                lambda a, b: a | b,
                [P.closed(*interval_limits) for interval_limits in signal_events],
            )
        return ret


class Events(AssetContainer):
    """
    Manager for :py:class:`Event` objects.
    """

    def __init__(self, parent: Any) -> None:
        super().__init__(parent)
        self._text = None

    def add(
        self,
        name: str,
        size: int,
        fname: str,
        data_id_func: Callable,
        color: Union[str, int],
        pick_action: str = "overwrite",
        ax_list: Optional[List] = None,
        win_remove: Tuple[float, float] = (-0.1, 0.1),
        win_add: Tuple[float, float] = (-0.25, 0.25),
        add_key: Optional[str] = None,
        remove_key: Optional[str] = None,
        save_key: Optional[str] = None,
        show: bool = True,
        data_func: Callable = float,
        **plot_kwargs,
    ) -> Event:
        """
        Add an event.

        Args:
            name (str): Name of the event.
            size (int): Length of the sequence. For example, if this is 2, then the event is a pair of start and end times.
            fname (str): File name to save the event.
            data_id_func (Callable): Function to get the current data ID from the parent UI.
            color (Union[str, int]): Color used to display the event
            pick_action (str, optional): Action to take when picking an event ('overwrite' or 'append'). Use overwrite if there can only be one sequence per 'signal'. If there can be multiple, use 'append'. Defaults to overwrite. Defaults to 'overwrite'.
            ax_list (Optional[List], optional): List of axes on which to show the event. Defaults to None.
            win_remove (Tuple[float, float], optional): Window relative to the mouse position to search for removing an event. Defaults to (-0.1, 0.1).
            win_add (Tuple[float, float], optional): Window relative to the mouse position to search for adding an event. Defaults to (-0.25, 0.25). For future use.
            add_key (Optional[str], optional): Keyboard shortcut for adding the event. Defaults to None.
            remove_key (Optional[str], optional): Keyboard shortcut for removing the event. Defaults to None.
            save_key (Optional[str], optional): Keyboard shortcut for saving the events to a JSON file. Defaults to None.
            show (bool, optional): Event visibility. Defaults to True.
            data_func (Callable, optional): Function to apply on the incoming event. Intended use is for type casting. Defaults to float.

        Returns:
            Event: _description_
        """
        assert name not in self.names
        this_ev = Event(
            name,
            size,
            fname,
            data_id_func,
            color,
            pick_action,
            ax_list,
            win_remove,
            win_add,
            data_func,
            **plot_kwargs,
        )
        super().add(this_ev)
        if add_key is not None:
            self.parent.add_key_binding(add_key, this_ev.add, f"Add {name}")
        if remove_key is not None:
            self.parent.add_key_binding(remove_key, this_ev.remove, f"Remove {name}")
        if save_key is not None:
            self.parent.add_key_binding(save_key, this_ev.save, f"Save {name}")
        if show:
            this_ev.setup_display()
        else:
            this_ev._hide = True  # This is for fill displays
        return this_ev

    def add_from_file(
        self,
        fname: str,
        data_id_func: Callable,
        ax_list: Optional[List] = None,
        add_key: Optional[str] = None,
        remove_key: Optional[str] = None,
        save_key: Optional[str] = None,
        show: bool = True,
        data_func: Callable = float,
        **plot_kwargs,
    ) -> Event:
        """Add events from an existing file.
        Intended use case - events are created by another algorithm  meant to be edited using a browser in the :py:mod:`datanavigator`.

        Args:
            fname (str): File name to load the events.
            data_id_func (Callable): Function to get the current data ID from the parent UI.
            ax_list (Optional[List], optional): List of axes on which to show the event. Defaults to None.
            add_key (Optional[str], optional): Keyboard shortcut for adding the event. Defaults to None.
            remove_key (Optional[str], optional): Keyboard shortcut for removing the event. Defaults to None.
            save_key (Optional[str], optional): Keyboard shortcut for saving the events to a JSON file. Defaults to None.
            show (bool, optional): Event visibility. Defaults to True.
            data_func (Callable, optional): Function to apply on the incoming event. Intended use is for type casting. Defaults to float.

        Returns:
            Event: The created Event object.
        """
        assert os.path.exists(fname)
        ev = Event._from_existing_file(fname)
        hdr = ev.get_header()
        del hdr["all_keys_are_tuples"]
        plot_kwargs = {**hdr["plot_kwargs"], **plot_kwargs}
        del hdr["plot_kwargs"]
        return self.add(
            data_id_func=data_id_func,
            ax_list=ax_list,
            add_key=add_key,
            remove_key=remove_key,
            save_key=save_key,
            show=show,
            data_func=data_func,
            **{**hdr, **plot_kwargs},
        )

    def setup_display(self) -> None:
        """Setup display for all events."""
        for ev in self._list:
            ev.setup_display()

    def update_display(self, draw: bool = True) -> None:
        """Update display for all events"""
        for ev in self._list:
            ev.update_display(draw=False)
        if draw:
            plt.draw()


def _find_nearest_idx_val(array: List[float], value: float) -> Tuple[int, float]:
    """Find the index and value of the nearest element in the array to the given value.

    Args:
        array (list[float]): List of values.
        value (float): Value to find the nearest element for.

    Returns:
        Tuple[int, float]: Index and value of the nearest element.
    """    
    """"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return idx, val
