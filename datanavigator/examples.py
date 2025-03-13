import os
from pathlib import Path
import numpy as np
import pysampled
import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector as LassoSelectorWidget

from . import _config
from .core import SignalBrowser
from .core import Buttons

class EventPicker(SignalBrowser):
    def __init__(self):
        plot_data = [pysampled.Data(np.random.rand(100), sr=10, meta={'id': f'sig{sig_count:02d}'}) for sig_count in range(10)]
        super().__init__(plot_data)
        self.memoryslots.disable()
        data_id_func = (lambda s: s.data[s._current_idx].meta['id']).__get__(self)
        self.events.add(
            name='pick1',
            size=1,
            fname=os.path.join(_config.get_cache_folder(), "_pick1.json"),
            data_id_func = data_id_func,
            color = 'tab:red',
            pick_action = 'append',
            ax_list = [self._ax],
            add_key='1',
            remove_key='4',
            save_key='ctrl+1',
            linewidth=1.5,
        )
        self.events.add(
            name='pick2',
            size=2,
            fname=os.path.join(_config.get_cache_folder(), "_pick2.json"),
            data_id_func = data_id_func,
            color = 'tab:green',
            pick_action = 'append',
            ax_list = [self._ax],
            add_key='2',
            remove_key='5',
            save_key='ctrl+2',
            linewidth=1.5,
        )
        self.events.add(
            name='pick3',
            size=3,
            fname=os.path.join(_config.get_cache_folder(), "_pick3.json"),
            data_id_func = data_id_func,
            color = 'tab:blue',
            pick_action = 'overwrite',
            ax_list = [self._ax],
            add_key='3',
            remove_key='6',
            save_key='ctrl+3',
            linewidth=1.5,
        )
        self.update()

    def update(self, event=None):
        self.events.update_display()
        return super().update(event)

class ButtonDemo(plt.Figure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.buttons = Buttons(parent=self)
        self.buttons.add(text='test', type_='Toggle')
        self.buttons.add(text='push button', type_='Push', action_func=self.test_callback)
        plt.show(block=False)
    
    def test_callback(self, event=None):
        print(event)


class SelectorDemo:
    def __init__(self):
        f, ax = plt.subplots(1, 1)
        self.buttons = Buttons(parent=f)
        self.buttons.add(text='Start selection', type_='Push', action_func=self.start)
        self.buttons.add(text='Stop selection', type_='Push', action_func=self.stop)
        self.ax = ax
        self.x = np.random.rand(10)
        self.t = np.r_[:1:0.1]
        self.plot_handles = {}
        self.plot_handles['data'], = ax.plot(self.t, self.x)
        self.plot_handles['selected'], = ax.plot([], [], '.')
        plt.show(block=False)
        self.start()
        self.ind = set()
    
    def get_points(self):
        return np.vstack((self.t, self.x)).T

    def onselect(self, verts):
        """Select if not previously selected; Unselect if previously selected"""
        path = Path(verts)
        selected_ind = set(np.nonzero(path.contains_points(self.get_points()))[0])
        existing_ind = self.ind.intersection(selected_ind)
        new_ind = selected_ind - existing_ind
        self.ind = (self.ind - existing_ind).union(new_ind)
        idx = list(self.ind)
        if idx:
            self.plot_handles['selected'].set_data(self.t[idx], self.x[idx])
        else:
            self.plot_handles['selected'].set_data([], [])
        plt.draw()

    def start(self, event=None):
        self.lasso = LassoSelectorWidget(self.ax, onselect=self.onselect)

    def stop(self, event=None):
        self.lasso.disconnect_events()



class SignalBrowserKeyPress(SignalBrowser):
    """Wrapper around plot_sync with key press features to make manual alignment process easier"""
    def __init__(self, plot_data, titlefunc=None, figure_handle=None, reset_on_change=False):
        super().__init__(plot_data, titlefunc, figure_handle, reset_on_change)
        self.event_keys = {'1': [], '2':[], '3':[], 't':[], 'd':[]}
    def __call__(self, event):
        from pprint import pprint
        super().__call__(event)
        if event.name == 'key_press_event':
            sr = self.data[self._current_idx].sr
            if event.key in self.event_keys:
                if event.key == '1':
                    self.first = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.first)
                    print(f'first: {self.first}')
                elif event.key == '2':
                    self.second = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.second)
                    print(f'second: {self.second}')
                elif event.key == '3':
                    self.third = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.third)
                    print(f'third: {self.third}')
                elif event.key == 't':
                    pprint(self.event_keys, width=1)
                    self.export = self.event_keys   
                elif event.key == 'd':
                    for key in self.event_keys:
                        self.event_keys[key].clear()
                    pprint(self.event_keys, width=1)
