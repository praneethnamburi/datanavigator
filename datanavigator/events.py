from __future__ import annotations

import functools
import json
import os
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from . import utils
from .assets import AssetContainer

PLOT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

class EventData:
    """
    Manage the data from one event type in one trial.
    """
    def __init__(self, default=None, added=None, removed=None, tags=None, algorithm_name:str='', params:dict=None) -> None:
        _to_list = lambda x: [] if x is None else x
        self.default = _to_list(default) # e.g. created by an algorithm
        self.added = _to_list(added)     # Manually added events, e.g. through a UI. if an 'added' point is removed, then it will simply be deleted. There will be no record of it.
        self.removed = _to_list(removed) # anything that is removed from default will be stored here
        self.tags = _to_list(tags)
        self.algorithm_name = algorithm_name
        self.params = params if params is not None else {} # params used to generate the default list 

    def asdict(self):
        return dict(
            default         = self.default,
            added           = self.added,
            removed         = self.removed,
            tags            = self.tags,
            algorithm_name  = self.algorithm_name,
            params          = self.params,
        )
    
    def __len__(self): # number of events
        return len(self.get_times())
    
    def get_times(self):
        x = self.default + self.added
        x.sort()
        return x

    def to_portions(self):
        return functools.reduce(lambda a,b: a|b, [utils.portion.closed(*interval_limits) for interval_limits in self.get_times()])
    
    def __and__(self, other: EventData):
        return EventData(default=list(self.to_portions() & other.to_portions()))
    
    def __contains__(self, item):
        return item in self.to_portions()
    
    @staticmethod
    def _process_inp(other):
        if not isinstance(other, utils.portion.Interval):
            assert len(other) == 2
            other = utils.portion.closed(*other)
        return other

    def overlap_duration(self, other) -> float:
        """Duration of 'other' that overlaps with self"""
        other = self._process_inp(other)
        return (self.to_portions() & other).duration
    
    def overlap_frac(self, other) -> float:
        """Fraction of 'other' that overlaps with self"""
        other = self._process_inp(other)
        return self.overlap_duration(other)/other.duration


class Event:
    """
    Manage selection of a sequence of events (of length >= 1)
    """
    def __init__(self, name, size, fname, data_id_func=None, color='random', pick_action='overwrite', ax_list=None, win_remove=(-0.1, 0.1), win_add=(-0.25, 0.25), data_func=float, **plot_kwargs):
        self.name = name
        assert isinstance(size, int) and size > 0
        self.size = size # length of the sequence
        self.fname = fname # load and save events to this file
        self.data_id_func = data_id_func # gets the current data_id from the parent ui when executed
        if isinstance(color, int):
            color = PLOT_COLORS[color]
        elif color == 'random':
            color = np.random.choice(PLOT_COLORS)
        self.color = color
        assert pick_action in ('overwrite', 'append') # overwrite if there can only be one sequence per 'signal'. For multiple, use 'append'
        self.pick_action = pick_action

        self._buffer = []
        _, self._data = self.load()

        self.ax_list = ax_list # list of axes on which to show the event
        self.plot_handles = []

        # self.win_add = win_add # seconds, search to add peak within this window, in the peak or valley modes
        self.win_remove = win_remove # seconds, search to remove an event within this window
        self.win_add = win_add # seconds, search to add an event within this window in peak or valley mode
        self.plot_kwargs = plot_kwargs # tune the style of the plot using this
        self._hide = False
        self.data_func = data_func

    def initialize_event_data(self, data_id_list):
        """Useful for initializing an event"""
        for data_id in data_id_list:
            if data_id not in self._data:
                self._data[data_id] = EventData()
    
    @classmethod
    def _from_existing_file(cls, fname, data_id_func=None):
        """Create an Event object by reading an existing json file."""
        h, _ = cls._read_json_file(fname)
        return cls(h['name'], h['size'], fname, data_id_func, h['color'], h['pick_action'], None, h['win_remove'], h['win_add'], **h['plot_kwargs'])
    
    @classmethod
    def from_file(cls, fname, **kwargs):
        """
        Create an empty events file with the given file name (fname) and any parameters.
        Assigns best-guess defaults
        """
        if not os.path.exists(fname):
            kwargs['name'] = kwargs.get('name', Path(fname).stem)
            kwargs['size'] = kwargs.get('size', 1)
            kwargs['data_id_func'] = kwargs.get('data_id_func', None) # this is irrelevant

            ret = cls(fname=fname, **kwargs)
            ret.save() # this has a print message
            return ret
        return cls._from_existing_file(fname, kwargs.get('data_id_func', None))

    @classmethod
    def from_data(cls, data:dict, name:str='Event', fname:str='', overwrite:bool=False, **kwargs):
        """Create an event file by filling in the 'default' events extracted by an algorithm.
        kwargs 
            - tags, algorithm_name, and params will be passed to gui.EventData
            - all other kwargs will be passed to gui.Event
        """
        algorithm_info = dict(
            tags = kwargs.pop('tags', []),
            algorithm_name = kwargs.pop('algorithm_name', ''),
            params = kwargs.pop('params', {})
        )
        
        size = []
        for key, val in data.items():
            if isinstance(val, EventData):
                continue
            v = np.asarray(val)
            if v.ndim == 1: # passing in a list events of size 1
                v = v[:, np.newaxis]
                val = [list(x) for x in v]
            data[key] = EventData(default=val, **algorithm_info)
            if len(data[key]) > 0: # when there are no events, size cannot be inferred for that trial. Note that this process will fail if there are no events in ANY of the trials. size has to be passed in with kwargs.
                size.append(v.shape[-1])
        
        if not size: # if there were no events in the data that was passed!
            assert 'size' in kwargs
            size = kwargs['size']
        else:
            size = list(set(size))
            assert len(size) == 1 # make sure we have the same type of events
            size = size[0]
        
        if 'size' in kwargs:
            assert kwargs['size'] == size
            del kwargs['size']
        
        ret = cls(name, size, fname, **kwargs)
        ret._data = data
        if utils.is_path_exists_or_creatable(fname):
            if (not os.path.exists(fname)) or overwrite:
                ret.save()
            else: # don't overwrite and exists, then append new data to the file if it exists
                assert os.path.exists(fname) and (not overwrite)
                ret_existing = cls.from_file(fname, **kwargs)
                new_keys = set(ret._data.keys()) - set(ret_existing._data.keys())
                if len(new_keys) > 0: # if there is new data, then add it to the event file
                    print(f'Appending new data to the event file {fname}:')
                    print(new_keys)
                    ret_existing._data = ret._data | ret_existing._data
                    ret_existing.save()
        return ret
    
    def all_keys_are_tuples(self) -> bool:
        return all([type(x) == tuple for x in self._data.keys()])

    def get_header(self):
        return dict(
            name  = self.name,
            size  = self.size,
            fname = self.fname,
            color = self.color,
            pick_action = self.pick_action,
            win_remove = self.win_remove,
            win_add = self.win_add,
            plot_kwargs = self.plot_kwargs,
            all_keys_are_tuples = self.all_keys_are_tuples(),
        )

    @staticmethod
    def _read_json_file(fname):
        with open(fname, 'r') as f:
            header, data = json.load(f)
        if header['all_keys_are_tuples']:
            data = {eval(k):EventData(**v) for k,v in data.items()}
        else:
            data = {k:EventData(**v) for k,v in data.items()}
        return header, data

    def load(self):
        if os.path.exists(self.fname):
            header, data = self._read_json_file(self.fname)
            return header, data
        return {}, {}

    def save(self):
        action_str = 'Updated' if os.path.exists(self.fname) else 'Created'
        with open(self.fname, 'w') as f:
            header = self.get_header()
            if header['all_keys_are_tuples']:
                data = {str(k):v.asdict() for k,v in self._data.items()}
            else:
                data = {k:v.asdict() for k,v in self._data.items()}
            json.dump((header, data), f, indent=4)
        print(action_str + ' ' + self.fname)
    
    def add(self, event): # the parent UI would invoke this
        """
        Pick the time points of an interval and associate it with a supplied ID
        If the first selection is outside the axis, then select the first available time point.
        If the last selection is outside the axis, then select the last available time point.
        If the selections are not monotonically increasing, then empty the buffer.
        If any of the 'middle' picks (i.e. not first or last in the sequence) are outside the axes, then empty the buffer.
        """
        def strictly_increasing(_list):
            return all(x<y for x, y in zip(_list, _list[1:]))

        def _get_lines():
            """Return non-empty lines in the axis where event was invoked, or else in all lines in the figure"""
            if event.inaxes is not None:
                return [line for line in event.inaxes.get_lines() if len(line.get_xdata()) > 0]
            return [line for ax in event.canvas.figure.axes for line in ax.get_lines() if len(line.get_xdata()) > 0] # return ALL lines in the figure
        
        def _get_first_available_timestamp():
            return min([np.nanmin(l.get_xdata()) for l in _get_lines() if len(l.get_xdata()) > 0])
            # return self.parent.data[self._current_idx].t[0]
        
        def _get_last_available_timestamp():
            return max([np.nanmax(l.get_xdata()) for l in _get_lines() if len(l.get_xdata()) > 0])
            # return self.parent.data[self._current_idx].t[-1]
        
        def clamp(n): 
            smallest = _get_first_available_timestamp()
            largest = _get_last_available_timestamp()
            return max(smallest, min(n, largest))
        
        # add picks to the buffer until the length is equal to the size
        if event.xdata is None: # pick is outside the axes
            if not self._buffer: # first in the sequence
                inferred_timestamp = _get_first_available_timestamp()
            else:
                assert len(self._buffer) == self.size-1 # last in the sequence
                inferred_timestamp = _get_last_available_timestamp()
        else:
            inferred_timestamp = clamp(self.data_func(event.xdata))

        self._buffer.append(inferred_timestamp)

        if not strictly_increasing(self._buffer):
            self._buffer = [] # reset buffer
        
        if len(self._buffer) < self.size:
            return
        
        assert len(self._buffer) == self.size

        sequence = self._buffer.copy()

        # add data to store
        data_id = self.data_id_func()
        if data_id not in self._data:
            self._data[data_id] = EventData()
        if self.pick_action == 'append':
            self._data[data_id].added.append(sequence)
        else: # overwrite => one event per trial
            self._data[data_id].added = [sequence]

        print(self.name, 'add', data_id, sequence)
        self._buffer = []
        self.update_display()
    
    def remove(self, event):
        """
        For events of length > 1, remove by removing the first element in that sequence.
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
        sequence = None # data that was removed
        _removed = False
        _deleted = False
        if len(ev.added) > 0 and len(ev.default) > 0:
            idx_add, val_add = utils.find_nearest(added_start_times, t_marked)
            idx_def, val_def = utils.find_nearest(default_start_times, t_marked)
            
            add_dist = np.abs(val_add-t_marked)
            def_dist = np.abs(val_def-t_marked)
            if (add_dist <= def_dist) and (self.win_remove[0] < add_dist < self.win_remove[1]):
                sequence = ev.added.pop(idx_add)
                _deleted = True
            if (def_dist < add_dist) and (self.win_remove[0] < def_dist < self.win_remove[1]):
                sequence = ev.default.pop(idx_def)
                ev.removed.append(sequence)
                _removed = True
        elif len(ev.added) > 0 and len(ev.default) == 0:
            idx_add, val_add = utils.find_nearest(added_start_times, t_marked)
            add_dist = np.abs(val_add-t_marked)
            if self.win_remove[0] < add_dist < self.win_remove[1]:
                sequence = ev.added.pop(idx_add)
                _deleted = True
        elif len(ev.added) == 0 and len(ev.default) > 0:
            idx_def, val_def = utils.find_nearest(default_start_times, t_marked)
            def_dist = np.abs(val_def-t_marked)
            if self.win_remove[0] < def_dist < self.win_remove[1]:
                sequence = ev.default.pop(idx_def)
                ev.removed.append(sequence)
                _removed = True
        else:
            return
        
        if sequence is None:
            return
        
        assert _removed is not _deleted # removed moves data from default (i.e. auto-detected) to removed, and delete expunges a manually added event
        print(self.name, {True: 'remove', False: 'delete'}[_removed], data_id, sequence)
        self.update_display()
    
    def get_current_event_times(self):
        return list(np.array(self._data.get(self.data_id_func(), EventData()).get_times()).flatten())
    
    def _get_display_funcs(self):
        display_type = self.plot_kwargs.get('display_type', 'line')
        assert display_type in ('line', 'fill')
        if display_type == 'fill':
            assert self.size == 2
        if display_type == 'line':
            return self._setup_display_line, self._update_display_line
        return self._setup_display_fill, self._update_display_fill

    def setup_display(self): # setup event display this event on one or more axes
        setup_func, _ = self._get_display_funcs()
        setup_func()

    def _setup_display_line(self):
        plot_kwargs = {'label':f'event:{self.name}'} | self.plot_kwargs
        plot_kwargs.pop('display_type', None)
        for ax in self.ax_list:
            this_plot, = ax.plot([], [], color=self.color, **plot_kwargs)
            self.plot_handles.append(this_plot)

    def _setup_display_fill(self):
        return # everything is redrawm currently for fill display. So, don't do setup.

    def update_display(self, draw=True):
        _, update_func = self._get_display_funcs()
        update_func(draw)
    
    def _get_ylim(self, this_ax, type='data'):
        if type == 'data':
            try:
                x = np.asarray([(np.nanmin(line.get_ydata()), np.nanmax(line.get_ydata())) for line in this_ax.get_lines() if not line.get_label().startswith('event:')])
                return np.nanmin(x[:, 0]), np.nanmax(x[:, 1])
            except ValueError:
                return this_ax.get_ylim()
        return this_ax.get_ylim()

    def _update_display_line(self, draw):
        for ax, plot_handle in zip(self.ax_list, self.plot_handles):
            yl = self._get_ylim(ax)
            plot_handle.set_data(*utils.ticks_from_times(self.get_current_event_times(), yl))
        if draw:
            plt.draw()
    
    def _update_display_fill(self, draw):
        if self._hide:
            return
        for plot_handle in self.plot_handles:
            plot_handle.remove()
        self.plot_handles = []
        plot_kwargs = dict(alpha=0.2, edgecolor=None) | self.plot_kwargs
        plot_kwargs.pop('display_type', None)
        for ax in self.ax_list:
            yl = self._get_ylim(ax)
            x = np.asarray([this_times + [np.nan] for this_times in self._data.get(self.data_id_func(), EventData()).get_times()]).flatten()
            y1 = np.asarray([[yl[0]]*2 + [np.nan] for _ in range(int(len(x)/3))]).flatten()
            y2 = np.asarray([[yl[1]]*2 + [np.nan] for _ in range(int(len(x)/3))]).flatten()
            this_collection = ax.fill_between(x, y1, y2, color=self.color, **plot_kwargs)
            self.plot_handles.append(this_collection)
        if draw:
            plt.draw()
    
    def to_dict(self):
        event_data = self._data
        if self.pick_action == 'overwrite':
            ret = {k:v.get_times()[0] for k,v in event_data.items()}
        else:
            ret = {k:v.get_times() for k,v in event_data.items()}
        return ret
    
    def to_portions(self):
        assert self.size == 2
        P = utils.portion
        ret = {}
        for signal_id, signal_events in self.to_dict().items():
            ret[signal_id] = functools.reduce(lambda a,b: a|b, [P.closed(*interval_limits) for interval_limits in signal_events])
        return ret
    

class Events(AssetContainer):
    def __init__(self, parent):
        super().__init__(parent)
        self._text = None

    def add(self, 
            name,
            size, 
            fname, 
            data_id_func, 
            color, 
            pick_action='overwrite', 
            ax_list=None, 
            win_remove=(-0.1, 0.1),
            win_add=(-0.25, 0.25),
            add_key=None,
            remove_key=None,
            save_key=None,
            show=True,
            data_func=float,
            **plot_kwargs):
        assert name not in self.names
        this_ev = Event(name, size, fname, data_id_func, color, pick_action, ax_list, win_remove, win_add, data_func,  **plot_kwargs)
        super().add(this_ev)
        if add_key is not None:
            self.parent.add_key_binding(add_key, this_ev.add, f'Add {name}')
        if remove_key is not None:
            self.parent.add_key_binding(remove_key, this_ev.remove, f'Remove {name}')
        if save_key is not None:
            self.parent.add_key_binding(save_key, this_ev.save, f'Save {name}')
        if show:
            this_ev.setup_display()
        else:
            this_ev._hide = True # This is for fill displays
        return this_ev
    
    def add_from_file(self, fname, data_id_func, ax_list=None, add_key=None, remove_key=None, save_key=None, show=True, data_func=float, **plot_kwargs):
        """Easier than using add for adding events that are created by another algorithm, and meant to be edited using the gui module."""
        assert os.path.exists(fname)
        ev = Event._from_existing_file(fname)
        hdr = ev.get_header()
        del hdr['all_keys_are_tuples']
        plot_kwargs = hdr['plot_kwargs'] | plot_kwargs
        del hdr['plot_kwargs']
        return self.add(data_id_func=data_id_func, ax_list=ax_list, add_key=add_key, remove_key=remove_key, save_key=save_key, show=show, data_func=data_func, **(hdr | plot_kwargs))
    
    def setup_display(self):
        for ev in self._list:
            ev.setup_display()

    def update_display(self, draw=True):
        for ev in self._list:
            ev.update_display(draw=False)
        if draw:
            plt.draw()
