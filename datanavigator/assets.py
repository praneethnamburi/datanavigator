from __future__ import annotations

import numpy as np
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.path import Path as mPath
from matplotlib.widgets import Button as ButtonWidget
from matplotlib.widgets import LassoSelector as LassoSelectorWidget

### Extended widget classes
class Button(ButtonWidget):
    """Add a 'name' state to a matplotlib widget button"""
    def __init__(self, ax, name:str, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.name = name

class StateButton(Button): # store a number/coordinate
    def __init__(self, ax, name: str, start_state, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.state = start_state # stores something in the state

class ToggleButton(StateButton):
    """
    Add a toggle button to a matplotlib figure

    For example usage, see PlotBrowser
    """
    def __init__(self, ax, name:str, start_state:bool=True, **kwargs) -> None:
        super().__init__(ax, name, start_state, **kwargs)
        self.on_clicked(self.toggle)
        self.set_text()
    
    def set_text(self):
        self.label._text = f'{self.name}={self.state}'

    def toggle(self, event=None):
        self.state = not self.state
        self.set_text()
    
    def set_state(self, state:bool):
        assert isinstance(state, bool)
        self.state = state
        self.set_text()


class Selector:
    """
    Select points in a plot using the lasso selection widget
    Indices of selected points are stored in self.sel

    Example:
        f, ax = plt.subplots(1, 1)
        ph, = ax.plot(np.random.rand(20))
        plt.show(block=False)
        ls = gui.Lasso(ph)
        ls.start()
        -- play around with selecting points --
        ls.stop() -> disconnects the events
    """
    def __init__(self, plot_handle) -> None:
        """plot_handle - matplotlib.lines.Line2D object returned by plt.plot function"""
        assert isinstance(plot_handle, mlines.Line2D)
        self.plot_handle = plot_handle
        self.xdata, self.ydata = plot_handle.get_data()
        self.ax = plot_handle.axes
        self.overlay_handle, = self.ax.plot([], [], ".")
        self.sel = np.zeros(self.xdata.shape, dtype=bool)
        self.is_active = False

    def get_data(self):
        return np.vstack((self.xdata, self.ydata)).T

    def onselect(self, verts):
        """Select if not previously selected; Unselect if previously selected"""
        selected_ind = mPath(verts).contains_points(self.get_data())
        self.sel = np.logical_xor(selected_ind, self.sel)
        sel_x = list(self.xdata[self.sel])
        sel_y = list(self.ydata[self.sel])
        self.overlay_handle.set_data(sel_x, sel_y)
        plt.draw()
    
    def start(self, event=None): # split callbacks when using start and stop buttons
        self.lasso = LassoSelectorWidget(self.plot_handle.axes, self.onselect)
        self.is_active = True

    def stop(self, event=None):
        self.lasso.disconnect_events()
        self.is_active = False
    
    def toggle(self, event=None): # one callback when activated using a toggle button
        if self.is_active:
            self.stop(event)
        else:
            self.start(event)


### Managers for extended widget classes defined here (used by Generic browser)
class AssetContainer:
    """Container for assets such as a button, memoryslot, etc

    Args:
        parent (Any): matplotlib figure, or something that has a 'figure' attribute that is a figure.
    """        
    def __init__(self, parent):
        self._list: list = [] # list of assets
        self.parent = parent
    
    def __len__(self):
        return len(self._list)
    
    @property
    def names(self):
        return [x.name for x in self._list]

    def __getitem__(self, key):
        """Return an asset by the name key or by position in the list."""
        if not self._has_names():
            assert isinstance(key, int)
            
        if isinstance(key, int) and key not in self.names:
            return self._list[key]
        
        return {x.name: x for x in self._list}[key]

    def _has_names(self):
        try:
            self.names
            return True
        except AttributeError:
            return False
        
    def add(self, asset):
        if hasattr(asset, 'name'):
            assert asset.name not in self.names
        self._list.append(asset)
        return asset


class Buttons(AssetContainer):
    """Manager for buttons in a matplotlib figure or GUI (see GenericBrowser for example)"""
    def add(self, text='Button', action_func=None, pos=None, w=0.25, h=0.05, buf=0.01, type_='Push', **kwargs):
        """
        Add a button to the parent figure / object
        If pos is provided, then w, h, and buf will be ignored
        """
        assert type_ in ('Push', 'Toggle')
        nbtn = len(self)
        if pos is None: # start adding at the top left corner
            parent_fig = self.parent.figure
            mul_factor = 6.4/parent_fig.get_size_inches()[0]
            
            btn_w = w*mul_factor
            btn_h = h*mul_factor
            btn_buf = buf
            pos = (btn_buf, (1-btn_buf)-((btn_buf+btn_h)*(nbtn+1)), btn_w, btn_h)
        
        if type_ == 'Toggle':
            b = ToggleButton(plt.axes(pos), text, **kwargs)
        else:
            b = Button(plt.axes(pos), text, **kwargs)

        if action_func is not None: # more than one can be attached
            if isinstance(action_func, (list, tuple)):
                for af in action_func:
                    b.on_clicked(af)
            else:
                b.on_clicked(action_func)
        
        return super().add(b)


class Selectors(AssetContainer):
    """Manager for selector objects - for picking points on line2D objects"""
    def add(self, plot_handle):
        return super().add(Selector(plot_handle))
        
class MemorySlots(AssetContainer):
    def __init__(self, parent):
        super().__init__(parent)
        self._list = self.initialize()
        self._memtext = None

    @staticmethod
    def initialize():
        return {str(k):None for k in range(1, 10)}

    def disable(self):
        self._list = {}
    
    def enable(self):
        self._list = self.initialize()

    def show(self, pos='bottom left'):
        self._memtext = utils.TextView(self._list, fax=self.parent.figure, pos=pos)
    
    def update(self, key):
        """
        memory slot handling - Initiate when None, Go to the slot if it exists, frees slot if pressed when it exists
        key is the event.key triggered by a callback
        """
        if self._list[key] is None:
            self._list[key] = self.parent._current_idx
            self.update_display()
        elif self._list[key] == self.parent._current_idx:
            self._list[key] = None
            self.update_display()
        else:
            self.parent._current_idx = self._list[key]
            self.parent.update()
    
    def update_display(self):
        """Refresh memory slot text if it is not hidden"""
        if self._memtext is not None:
            self._memtext.update(self._list)

    def hide(self):
        """Hide the memory slot text"""
        if self._memtext is not None:
            self._memtext._text.remove()
        self._memtext = None
    
    def is_enabled(self):
        return bool(self._list)

class StateVariable:
    def __init__(self, name:str, states:list):
        self.name = name
        self.states = list(states)
        self._current_state_idx = 0
    
    @property
    def current_state(self):
        return self.states[self._current_state_idx]
    
    def n_states(self):
        return len(self.states)
    
    def cycle(self):
        self._current_state_idx = (self._current_state_idx+1)%self.n_states()
    
    def cycle_back(self):
        self._current_state_idx = (self._current_state_idx-1)%self.n_states()
    
    def set_state(self, state):
        if isinstance(state, int):
            assert 0 <= state < self.n_states()
            self._current_state_idx = state
        if isinstance(state, str):
            assert state in self.states
            self._current_state_idx = self.states.index(state)

class StateVariables(AssetContainer):
    def __init__(self, parent):
        super().__init__(parent)
        self._text = None
    
    def asdict(self):
        return {x.name:x.states for x in self._list}
    
    def add(self, name:str, states:list):
        assert name not in self.names
        return super().add(StateVariable(name, states))
    
    def _get_display_text(self):
        return ['State variables:'] + [f'{x.name} - {x.current_state}' for x in self._list]
    
    def show(self, pos='bottom right'):
        self._text = utils.TextView(self._get_display_text(), fax=self.parent.figure, pos=pos)

    def update_display(self, draw=True):
        self._text.update(self._get_display_text())
        if draw:
            plt.draw()
