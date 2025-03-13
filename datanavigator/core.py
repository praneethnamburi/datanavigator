from __future__ import annotations

import functools
import inspect
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union, Mapping, Callable

import cv2 as cv
from decord import VideoReader
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import axes as maxes
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FFMpegWriter

import pysampled

from . import utils
from . import _config
from .assets import Buttons, Selectors, MemorySlots, StateVariables
from .events import Events

PLOT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']

class GenericBrowser:
    """
    Generic class to browse data. Meant to be extended before use.
    Features:
        Navigate using arrow keys.
        Store positions in memory using number keys (e.g. for flipping between positions when browsing a video).
        Quickly add toggle and push buttons.
        Design custom functions and assign hotkeys to them (add_key_binding)

    Default Navigation (arrow keys):
        ctrl+k      - show all keybindings
        right       - forward one frame
        left        - back one frame
        up          - forward 10 frames
        down        - back 10 frames
        shift+left  - first frame
        shift+right - last frame
        shift+up    - forward nframes/20 frames
        shift+down  - back nframes/20 frames
    """ 
    def __init__(self, figure_handle=None):
        if figure_handle is None:
            figure_handle = plt.figure()
        assert isinstance(figure_handle, plt.Figure)
        self.figure = figure_handle
        self._keypressdict = {} # managed by add_key_binding
        self._bindings_removed = {}
        
        # tracking variable
        self._current_idx = 0

        self._keybindingtext = None
        self.buttons = Buttons(parent=self)
        self.selectors = Selectors(parent=self)
        self.memoryslots = MemorySlots(parent=self)
        self.statevariables = StateVariables(parent=self)
        self.events = Events(parent=self)

        # for cleanup
        self.cid = []
        self.cid.append(self.figure.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self.figure.canvas.mpl_connect('close_event', self))
    
    def update_assets(self):
        if self.has('memoryslots'):
            self.memoryslots.update_display()
        if self.has('events'):
            self.events.update_display()
        if self.has('statevariables'):
            self.statevariables.update_display()

    def update(self, event=None): # extended classes are expected to implement their update function!
        self.update_assets()
    
    def update_without_clear(self, event=None):
        self.update_assets()
        # I did this for browsers that clear the axis each time! Those classes will need to re-implement this method

    def mpl_remove_bindings(self, key_list):
        """If the existing key is bound to something in matplotlib, then remove it"""
        for key in key_list:
            this_param_name = [k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v]
            if this_param_name: # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key
    
    def __call__(self, event):
        # print(event.__dict__) # for debugging
        if event.name == 'key_press_event':
            if event.key in self._keypressdict:
                f = self._keypressdict[event.key][0]
                argspec = inspect.getfullargspec(f)[0]
                if len(argspec) == 2 and argspec[1] == 'event':
                    f(event)
                else:
                    f() # this may or may not redraw everything
            if event.key in self.memoryslots._list:
                self.memoryslots.update(event.key)
        elif event.name == 'close_event': # perform cleanup
            self.cleanup()
    
    def cleanup(self):
        """Perform cleanup, for example, when the figure is closed."""
        for this_cid in self.cid:
            self.figure.canvas.mpl_disconnect(this_cid)
        self.mpl_restore_bindings()

    def mpl_restore_bindings(self):
        """Restore any modified default keybindings in matplotlib"""
        for param_name, key in self._bindings_removed.items():
            if key not in mpl.rcParams[param_name]:
                mpl.rcParams[param_name].append(key) # param names: keymap.back, keymap.forward)
            self._bindings_removed[param_name] = {}

    def __len__(self):
        if hasattr(self, 'data'): # otherwise returns None
            return len(self.data)

    def reset_axes(self, axis='both', event=None): # event in case it is used as a callback function
        """Reframe data within matplotlib axes."""
        for ax in self.figure.axes:
            if isinstance(ax, maxes.SubplotBase):
                ax.relim()
                ax.autoscale(axis=axis)
        plt.draw()

    ## select plots

    # Event responses - useful to pair with add_key_binding
    # These capabilities can be assigned to different key bindings
    def add_key_binding(self, key_name, on_press_function, description=None):
        """
        This is useful to add key-bindings in classes that inherit from this one, or on the command line.
        See usage in the __init__ function
        """
        if description is None:
            description = on_press_function.__name__
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = (on_press_function, description)

    def set_default_keybindings(self):
        self.add_key_binding('left', self.decrement)
        self.add_key_binding('right', self.increment)
        self.add_key_binding('up', (lambda s: s.increment(step=10)).__get__(self), description='increment by 10')
        self.add_key_binding('down', (lambda s: s.decrement(step=10)).__get__(self), description='decrement by 10')
        self.add_key_binding('shift+left', self.decrement_frac, description='step forward by 1/20 of the timeline')
        self.add_key_binding('shift+right', self.increment_frac, description='step backward by 1/20 of the timeline')
        self.add_key_binding('shift+up', self.go_to_start)
        self.add_key_binding('shift+down', self.go_to_end)
        self.add_key_binding('ctrl+c', self.copy_to_clipboard)
        self.add_key_binding('ctrl+k', (lambda s: s.show_key_bindings(f='new', pos='center left')).__get__(self), description='show key bindings')
        self.add_key_binding('/', (lambda s: s.pan(direction='right')).__get__(self), description='pan right')
        self.add_key_binding(',', (lambda s: s.pan(direction='left')).__get__(self), description='pan left')
        self.add_key_binding('l', (lambda s: s.pan(direction='up')).__get__(self), description='pan up')
        self.add_key_binding('.', (lambda s: s.pan(direction='down')).__get__(self), description='pan down')
        self.add_key_binding('r', self.reset_axes)
    
    def increment(self, step=1):
        self._current_idx = min(self._current_idx+step, len(self)-1)
        self.update()

    def decrement(self, step=1):
        self._current_idx = max(self._current_idx-step, 0)
        self.update()
    
    def go_to_start(self): # default: shift+left
        self._current_idx = 0
        self.update()
    
    def go_to_end(self):
        self._current_idx = len(self)-1
        self.update()
    
    def increment_frac(self, n_steps=20):
        # browse entire dataset in n_steps
        self._current_idx = min(self._current_idx+int(len(self)/n_steps), len(self)-1)
        self.update()
    
    def decrement_frac(self, n_steps=20):
        self._current_idx = max(self._current_idx-int(len(self)/n_steps), 0)
        self.update()
    
    def copy_to_clipboard(self):
        from PySide2.QtGui import QClipboard, QImage
        buf = io.BytesIO()
        self.figure.savefig(buf)
        QClipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def show_key_bindings(self, f=None, pos='bottom right'):
        f = {None: self.figure, 'new': plt.figure()}[f]
        text = []
        for shortcut, (_, description) in self._keypressdict.items():
            text.append(f'{shortcut:<12} - {description}')
        self._keybindingtext = utils.TextView(text, f, pos=pos)
    
    @staticmethod
    def _filter_sibling_axes(ax, share='x', get_bool=False):
        """Given a list of matplotlib axes, it will return axes to manipulate by picking one from a set of siblings"""
        assert share in ('x', 'y')
        if isinstance(ax, maxes.Axes): # only one axis
            return [ax]
        ax = [tax for tax in ax if isinstance(tax, maxes.SubplotBase)]
        if not ax: # no subplots in figure
            return
        pan_ax = [True]*len(ax)
        get_siblings = {'x': ax[0].get_shared_x_axes, 'y': ax[0].get_shared_y_axes}[share]().get_siblings
        for i, ax_row in enumerate(ax):
            sib = get_siblings(ax_row)
            for j, ax_col in enumerate(ax[i+1:]):
                if ax_col in sib:
                    pan_ax[j+i+1] = False

        if get_bool:
            return pan_ax
        return [this_ax for this_ax, this_tf in zip(ax, pan_ax) if this_tf]
        
    def pan(self, direction='left', frac=0.2):
        assert direction in ('left', 'right', 'up', 'down')
        if direction in ('left', 'right'):
            pan_ax='x'
        else:
            pan_ax='y'
        ax = self._filter_sibling_axes(self.figure.axes, share=pan_ax, get_bool=False)
        if ax is None:
            return
        for this_ax in ax:
            lim1, lim2 = {'x': this_ax.get_xlim, 'y': this_ax.get_ylim}[pan_ax]()
            inc = (lim2-lim1)*frac
            if direction in ('down', 'right'):
                new_lim = (lim1+inc, lim2+inc)
            else:
                new_lim = (lim1-inc, lim2-inc)
            {'x': this_ax.set_xlim, 'y': this_ax.set_ylim}[pan_ax](new_lim)
        plt.draw()
        self.update_without_clear() # panning is pointless if update clears the axis!!
    
    def has(self, asset_type): # e.g. has('events')
        assert asset_type in ('buttons', 'selectors', 'memoryslots', 'statevariables', 'events')
        return len(getattr(self, asset_type)) != 0


class PlotBrowser(GenericBrowser):
    """
    Takes a list of data, and a plotting function (or a pair of setup
    and update functions) that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """
    def __init__(self, plot_data, plot_func, figure_handle=None, **plot_kwargs):
        """
            plot_data - list of data objects to browse

            plot_func can be a tuple (setup_func, update_func), or just one plotting function - update_func
                If only one function is supplied, figure axes will be cleared on each update.
                setup_func takes:
                    the first element in plot_data list as its first input
                    keyword arguments (same as plot_func)
                setup_func outputs:
                    **dictionary** of plot handles that goes as the second input to update_func

                update_func is a plot-refreshing function that accepts 3 inputs:
                    an element in the plot_data list as its first input
                    output of the setup_func if it exists, or a figure handle on which to plot
                    keyword arguments
            
            figure_handle - (default: None) matplotlib figure handle within which to instantiate the browser
                Ideally, the setup function will handle this

            plot_kwargs - these keyword arguments will be passed to plot_function after data and figure
        """
        self.data = plot_data # list where each element serves as input to plot_func
        self.plot_kwargs = plot_kwargs

        if isinstance(plot_func, tuple):
            assert len(plot_func) == 2
            self.setup_func, self.plot_func = plot_func            
            self.plot_handles = self.setup_func(self.data[0], **self.plot_kwargs)
            plot_handle = list(self.plot_handles.values())[0]
            if 'figure' in self.plot_handles:
                figure_handle = self.plot_handles['figure']
            elif isinstance(plot_handle, list):
                figure_handle = plot_handle[0].figure
            else:
                figure_handle = plot_handle.figure # figure_handle passed as input will be ignored
        else:
            self.setup_func, self.plot_func = None, plot_func
            self.plot_handles = None
            figure_handle = figure_handle

        # setup
        super().__init__(figure_handle)

        # initialize
        self.set_default_keybindings()
        self.buttons.add(text='Auto limits', type_='Toggle', action_func=self.update, start_state=False)
        self.memoryslots.show()
        if self.__class__.__name__ == 'PlotBrowser': # if an inherited class is accessing this, then don't run the update function here
            self.update() # draw the first instance
            self.reset_axes()
            plt.show(block=False)
        # add selectors after drawing!
        try:
            s0 = self.selectors.add(list(self.plot_handles.values())[0])
            self.buttons.add(text='Selector 0', type_='Toggle', action_func=s0.toggle, start_state=s0.is_active)
        except AssertionError:
            print('Unable to add selectors')

    def get_current_data(self):
        return self.data[self._current_idx]

    def update(self, event=None): # event = None lets this function be attached as a callback
        if self.setup_func is None:
            self.figure.clear() # redraw the entire figure contents each time, NOT recommended
            self.memoryslots.show()
            self.plot_func(self.get_current_data(), self.figure, **self.plot_kwargs)
        else:
            self.memoryslots.update_display()
            self.plot_func(self.get_current_data(), self.plot_handles, **self.plot_kwargs)
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        super().update(event)
        plt.draw()
    
    def udpate_without_clear(self):
        self.memoryslots.update_display()
        plt.draw()


class SignalBrowser(GenericBrowser):
    """
    Browse an array of pysampled.Data elements, or 2D arrays
    """
    def __init__(self, plot_data, titlefunc=None, figure_handle=None, reset_on_change=False):
        super().__init__(figure_handle)

        self._ax = self.figure.subplots(1, 1)
        this_data = plot_data[0]
        if isinstance(this_data, pysampled.Data):
            self._plot = self._ax.plot(this_data.t, this_data())
        else:
            self._plot = self._ax.plot(this_data)

        self.data = plot_data
        if titlefunc is None:
            if hasattr(self.data[0], 'name'):
                self.titlefunc=lambda s: f'{s.data[s._current_idx].name}'
            else:
                self.titlefunc = lambda s: f'Plot number {s._current_idx}'
        else:
            self.titlefunc = titlefunc

        self.reset_on_change = reset_on_change
        # initialize
        self.set_default_keybindings()
        self.buttons.add(text='Auto limits', type_='Toggle', action_func=self.update, start_state=False)
        plt.show(block=False)
        self.update()
    
    def update(self, event=None):
        this_data = self.data[self._current_idx]
        if isinstance(this_data, pysampled.Data):
            data_to_plot = this_data.split_to_1d()
            for plot_handle, this_data_to_plot in zip(self._plot, data_to_plot):
                plot_handle.set_data(this_data_to_plot.t, this_data_to_plot())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        plt.draw()


class VideoBrowser(GenericBrowser):
    """Scroll through images of a video

        If figure_handle is an axis handle, the video will be plotted in that axis.
        
        Future Enhancements:
            - Extend VideoBrowser to play, pause, and extract clips using hotkeys.
            - Show timeline in VideoBrowser.
            - Add clickable navigation.
    """
    def __init__(self, vid_name, titlefunc=None, figure_or_ax_handle=None, image_process_func=lambda im:im):
        assert isinstance(figure_or_ax_handle, (plt.Axes, plt.Figure, type(None)))
        if isinstance(figure_or_ax_handle, plt.Axes):
            figure_handle = figure_or_ax_handle.figure
            ax_handle = figure_or_ax_handle
        else: # this is the same if figure_or_ax_handle is none or a figure handle
            figure_handle = figure_or_ax_handle
            ax_handle = None
        super().__init__(figure_handle)

        if not os.path.exists(vid_name): # try looking in the CLIP FOLDER
            vid_name = os.path.join(_config.get_clip_folder(), os.path.split(vid_name)[-1])        
        assert os.path.exists(vid_name)
        self.fname = vid_name
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, 'rb') as f:
            self.data = VideoReader(f)
        
        if ax_handle is None:
            self._ax = self.figure.subplots(1, 1)
        else:
            assert isinstance(ax_handle, plt.Axes)
            self._ax = ax_handle
        this_data = self.data[0]
        self._im = self._ax.imshow(this_data.asnumpy())
        self._ax.axis('off')

        self.fps = self.data.get_avg_fps()
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'
        
        self.image_process_func = image_process_func
        
        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self.memoryslots.show(pos='bottom left')

        if self.__class__.__name__ == 'VideoBrowser': # if an inherited class is accessing this, then don't run the update function here
            plt.show(block=False)
            self.update()
    
    def increment_frac(self, n_steps=100):
        # browse entire dataset in n_steps
        self._current_idx = min(self._current_idx+int(len(self)/n_steps), len(self)-1)
        self.update()

    def decrement_frac(self, n_steps=100):
        self._current_idx = max(self._current_idx-int(len(self)/n_steps), 0)
        self.update()

    def update(self):
        self._im.set_data(self.image_process_func(self.data[self._current_idx].asnumpy()))
        self._ax.set_title(self.titlefunc(self))
        super().update() # updates memory slots
        plt.draw()

    def extract_clip(self, start_frame=None, end_frame=None, fname_out=None, out_rate=None):
        #TODO: For musicrunning, grab the corresponding audio and add the audio track to the video clip?
        try:
            import ffmpeg
            use_subprocess = False
        except ModuleNotFoundError:
            import subprocess
            use_subprocess = True

        if start_frame is None:
            start_frame = self.memoryslots._list['1']
        if end_frame is None:
            end_frame = self.memoryslots._list['2']
        assert end_frame > start_frame
        start_time = float(start_frame)/self.fps
        end_time = float(end_frame)/self.fps
        dur = end_time - start_time
        if out_rate is None:
            out_rate = self.fps
        if fname_out is None:
            fname_out = os.path.join(_config.get_clip_folder(), os.path.splitext(self.name)[0] + '_s{:.3f}_e{:.3f}.mp4'.format(start_time, end_time))
        if use_subprocess:
            subprocess.getoutput(f'ffmpeg -ss {start_time} -i "{self.fname}" -r {out_rate} -t {dur} -vcodec h264_nvenc "{fname_out}"')
        else:
            ffmpeg.input(self.fname, ss=start_time).output(fname_out, vcodec='h264_nvenc', t=dur, r=out_rate).run()
        return fname_out


class VideoPlotBrowser(GenericBrowser):
    def __init__(self, vid_name:str, signals:dict, titlefunc=None, figure_handle=None, event_win=None):
        """
        Browse a video and an array of pysampled.Data side by side. 
        Assuming that the time vectors are synchronized across the video and the signals, there will be a black tick at the video frame being viewed.
        Originally created to browse montage videos from optitrack alongside physiological signals from delsys. 
        For example, see projects.fencing.snapshots.browse_trial

        signals is a {signal_name<str> : signal<pysampled.Data>}
        """
        figure_handle = plt.figure(figsize=(20, 12))
        super().__init__(figure_handle)

        self.event_win = event_win
        
        self.vid_name = vid_name
        assert os.path.exists(vid_name)
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, 'rb') as f:
            self.video_data = VideoReader(f)
        self.fps = self.video_data.get_avg_fps()
        
        self.signals = signals
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'

        self.plot_handles = self._setup()
        self.plot_handles['ax']['montage'].set_axis_off()

        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self._len = len(self.video_data)
        self.memoryslots.show(pos='bottom left')

        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        plt.show(block=False)
        self.update()
    
    def __len__(self):
        return self._len

    def _setup(self):
        fig = self.figure
        gs = GridSpec(nrows=len(self.signals), ncols=2, width_ratios=[2, 3])
        ax = {}
        plot_handles = {}
        for signal_count, (signal_name, this_signal) in enumerate(self.signals.items()):
            this_ax = fig.add_subplot(gs[signal_count, 1])
            plot_handles[f'signal{signal_count}'] = this_ax.plot(this_signal.t, this_signal())
            ylim = this_ax.get_ylim()
            plot_handles[f'signal{signal_count}_tick'], = this_ax.plot([0, 0], ylim, 'k')
            this_ax.set_title(signal_name)
            if signal_count < len(self.signals)-1:
                this_ax.get_xaxis().set_ticks([])
            else:
                this_ax.set_xlabel('Time (s)')
            ax[f'signal{signal_count}'] = this_ax

        ax['montage'] = fig.add_subplot(gs[:, 0])
        plot_handles['montage'] = ax['montage'].imshow(self.video_data[0].asnumpy())
        plot_handles['ax'] = ax
        plot_handles['fig'] = fig
        signal_ax = [v for k,v in plot_handles['ax'].items() if 'signal' in k]
        signal_ax[0].get_shared_x_axes().join(*signal_ax)
        plot_handles['signal_ax'] = signal_ax
        return plot_handles
    
    def update(self):
        self.plot_handles['montage'].set_data(self.video_data[self._current_idx].asnumpy())
        self.plot_handles['ax']['montage'].set_title(self.titlefunc(self))
        for signal_count, this_signal in enumerate(self.signals.items()):
            # ylim = self.plot_handles['ax'][f'signal{signal_count}'].get_ylim()
            self.plot_handles[f'signal{signal_count}_tick'].set_xdata([self._current_idx/self.fps]*2)
        if self.event_win is not None:
            curr_t = self._current_idx/self.fps
            self.plot_handles['signal_ax'][0].set_xlim(curr_t+self.event_win[0], curr_t+self.event_win[1])
        super().update()
        plt.draw()

    def onclick(self, event):
        """Right click mouse to seek to that frame."""
        this_frame = round(event.xdata*self.fps)

        # Right click to seek
        if isinstance(this_frame, (int, float)) and (0 <= this_frame < self._len) and (str(event.button) == 'MouseButton.RIGHT'):
            self._current_idx = this_frame
            self.update()
    
    def extract_clip(self, start_frame=None, end_frame=None, sav_dir=None, out_rate=30):
        """Save a video of screengrabs"""
        import shutil
        import subprocess
        if start_frame is None:
            start_frame = self.memoryslots._list['1']
        if end_frame is None:
            end_frame = self.memoryslots._list['2']
        assert end_frame > start_frame
        if sav_dir is None:
            sav_dir = os.path.join(_config.get_clip_folder(), datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(sav_dir):
            os.mkdir(sav_dir)
        print(f"Saving image sequence to {sav_dir}...")
        for frame_count in range(start_frame, end_frame+1):
            self._current_idx = frame_count
            self.update()
            self.figure.savefig(os.path.join(sav_dir, f'{frame_count:08d}.png'))
        print("Creating video from image sequence...")
        cmd = f'cd "{sav_dir}" && ffmpeg -framerate {self.fps} -start_number 0 -i %08d.png -c:v h264_nvenc -b:v 10M -maxrate 12M -bufsize 24M -vf scale="-1:1080" -an "{sav_dir}.mp4"'
        subprocess.getoutput(cmd)

        print("Removing temporary folder...")
        shutil.rmtree(sav_dir)
        
        print("Done")
        # vid_name = os.path.join(Path(sav_dir).parent)
        # f"ffmpeg -framerate {out_rate} -start_number {start_frame} -i "


class ComponentBrowser(GenericBrowser):
    def __init__(self, data, data_transform, labels=None, figure_handle=None, class_names=None, desired_class_names=None, annotation_names=None):
        """
        data is a 2d numpy array with number of signals on dim1, and number of time points on dim2
        data_transform is the transformed data, still a 2d numpy array with number of signals x number of components
            For example, transformed using one of (sklearn.decomposition.PCA, umap.UMAP, sklearn.manifold.TSNE, sklearn.decomposition.FastICA)
        labels are n_signals x 1 array with each entry representing the class of each signal piece. MAKE SURE ALL CLASS LABELS ARE POSITIVE
        class_names is a dictionary, for example {0:'Unclassified', '1:Resonant', '2:NonResonant'}

        This GUI is meant to be used for 
          - 'corrections', where classes are modified / assigned
          - 'annotations', where labels or annotations (separate from a class assignment in the sense that each signal belongs exactly to one class, and a signal my have 0 or more annotations)

        example - 
            import projects.gaitmusic as gm
            mr = gm.MusicRunning01()
            lf = mr(10).ot
            sig_pieces = gm.gait_phase_analysis(lf, muscle_line_name='RSBL_Upper', target_samples=500)
            gui.ComponentBrowser(sig_pieces)

            Single-click on scatter plots to select a gait cycle.
            Press r to refresh 'recent history' plots.
            Double click on the time course plot to select a gait cycle from the time series plot.
        """
        super().__init__(figure_handle)
        self.alpha = {'manual':0.8, 'auto':0.3}
        self.data = data

        n_components = np.shape(data_transform)[1]
        
        if labels is None:
            self.labels = np.zeros(self.n_signals, dtype=int)
        else:
            assert len(labels) == self.n_signals
            self.labels = labels
        assert np.min(self.labels) >= 0 # make sure all class labels are zero or positive!

        class_labels = list(np.unique(self.labels))
        self.class_labels_str = [str(x) for x in class_labels] # for detecting keypresses
        self.n_classes = len(class_labels) 
        if class_names is None:
            self.class_names = {class_label: f'Class_{class_label}' for class_label in class_labels}
        else:
            assert set(class_names.keys()) == set(class_labels)
            self.class_names = class_names
        self.classes = [ClassLabel(label=label, name=self.class_names[label]) for label in self.labels]

        if desired_class_names is None:
            desired_class_names = self.class_names
        self.desired_class_names = desired_class_names

        if annotation_names is None:
            annotation_names = {1:'Representative', 2:'Best', 3:'Noisy', 4:'Worst'}
        self.annotation_names = annotation_names
        self.annotation_idx_str = [str(x) for x in self.annotation_names]

        self.cid.append(self.figure.canvas.mpl_connect('pick_event', self.onpick))
        self.cid.append(self.figure.canvas.mpl_connect('button_press_event', self.select_signal_piece_dblclick))

        n_scatter_plots = int(n_components*(n_components-1)/2)
        self.gs = GridSpec(3, max(n_scatter_plots, 4))

        self._data_index = 0
        self.plot_handles = {}
        self.plot_handles['ax_pca'] = {}
        plot_number = 0
        for xc in range(n_components-1):
            for yc in range(xc+1, n_components):
                this_ax = self.figure.add_subplot(self.gs[1, plot_number])
                this_ax.set_title(str((xc+1, yc+1)))
                self.plot_handles['ax_pca'][plot_number] = this_ax
                self.plot_handles[f'scatter_plot_{xc+1}_{yc+1}'] = this_ax.scatter(data_transform[:, xc], data_transform[:, yc], c=self.colors, alpha=self.alpha['auto'], picker=5)
                self.plot_handles[f'scatter_highlight_{xc+1}_{yc+1}'], = this_ax.plot([], [], 'o', color='darkorange')
                plot_number += 1

        self.plot_handles['signal_plots'] = []
        this_ax = self.figure.add_subplot(self.gs[2, 0])
        self.plot_handles['ax_signal_plots'] = this_ax
        for signal_count in range(self.n_signals):
            self.plot_handles['signal_plots'].append(this_ax.plot(self.data[signal_count, :])[0])

        self.plot_handles['ax_current_signal'] = self.figure.add_subplot(self.gs[2, 1])
        self.plot_handles['current_signal'], = self.plot_handles['ax_current_signal'].plot(list(range(self.n_timepts)), [np.nan]*self.n_timepts)
        self.plot_handles['ax_current_signal'].set_xlim(self.plot_handles['ax_signal_plots'].get_xlim())
        self.plot_handles['ax_current_signal'].set_ylim(self.plot_handles['ax_signal_plots'].get_ylim())

        self.plot_handles['ax_history_signal'] = self.figure.add_subplot(self.gs[2, 2])

        self.plot_handles['ax_signal_full'] = self.figure.add_subplot(self.gs[0, :])
        self.plot_handles['signal_full'] = []
        time_index = np.r_[:self.n_timepts]/self.n_timepts
        for idx, sig in enumerate(self.data):
            this_plot_handle, = self.plot_handles['ax_signal_full'].plot(
                idx + time_index, sig, color=self.colors[idx]
            ) # assumes that there is only one line drawn per signal
            self.plot_handles['signal_full'].append(this_plot_handle)
        self.plot_handles['signal_selected_piece'], = self.plot_handles['ax_signal_full'].plot([], [], color='gray', linewidth=2)
        
        this_ylim = self.plot_handles['ax_signal_full'].get_ylim()
        for x_pos in np.r_[:self.n_signals+1]: # separators between signals
            self.plot_handles['ax_signal_full'].plot([x_pos]*2, this_ylim, 'k', linewidth=0.2)
        self.memoryslots.disable()
        
        self._class_info_text = utils.TextView([], self.figure, pos='bottom left')
        self.update_class_info_text()

        self._mode = 'correction' # ('correction' or 'annotation')
        self._mode_text = utils.TextView([], self.figure, pos='center left')
        self.update_mode_text()
        self.add_key_binding('m', self.toggle_mode)

        self._annotation_text = utils.TextView(['', 'Annotation list:']+[f'{k}:{v}' for k,v in self.annotation_names.items()], self.figure, pos='top left')

        self._message = utils.TextView(['Last action : '], self.figure, pos='bottom right')

        self._desired_class_info_text = utils.TextView([], self.figure, pos='bottom center')
        self.update_desired_class_info_text()

        self.add_key_binding('r', self.clear_axes)
        plt.show(block=False)

    @property
    def n_signals(self):
        return self.data.shape[0]
    
    @property
    def n_timepts(self):
        return self.data.shape[-1]

    @property
    def colors(self):
        return [cl.color for cl in self.classes]

    @property
    def signal(self) -> pysampled.Data:
        """Return the 2d Numpy array as a signal."""
        return pysampled.Data(self.data.flatten(), sr=self.n_timepts)
    
    def select_signal_piece_dblclick(self, event):
        """Double click a signal piece in the timecourse view to highlight that point."""
        if event.inaxes == self.plot_handles['ax_signal_full'] and event.dblclick: # If the click was inside the time course plot
            if 0 <= int(event.xdata) < self.data.shape[0]:
                self._data_index = int(event.xdata)
                self.update()
    
    def onpick(self, event):
        """Single click a projected point."""
        self.pick_event = event
        self._data_index = np.random.choice(event.ind)
        self.update()
        # this_data = event.artist._offsets[event.ind].data

    def update(self):
        super().update()
        for handle_name, handle in self.plot_handles.items():
            if 'scatter_plot_' in handle_name:
                this_data = np.squeeze(handle._offsets[self._data_index].data)
                self.plot_handles[handle_name.replace('_plot_', '_highlight_')].set_data(this_data[0], this_data[1])
        self.plot_handles['ax_history_signal'].plot(self.data[self._data_index, :])
        self.plot_handles['current_signal'].set_ydata(self.data[self._data_index, :])
        self.plot_handles['signal_selected_piece'].set_data(np.arange(self.n_timepts)/self.n_timepts+self._data_index, self.data[self._data_index, :])
        # self.plot_handles['signal_full'][self._data_index].linewidth = 3
        plt.draw()
    
    def update_class_info_text(self, draw=True):
        self._class_info_text.update(['Class list:'] + [f'{k}:{v}' for k,v in self.class_names.items()])
        if draw:
            plt.draw()
    
    def update_desired_class_info_text(self, draw=True):
        self._desired_class_info_text.update(['Desired class list:'] + [f'{k}:{v}' for k,v in self.desired_class_names.items()])
        if draw:
            plt.draw()
    
    def update_mode_text(self, draw=True):
        self._mode_text.update([f'mode: {self._mode}'])
        if draw:
            plt.draw()
    
    def update_message_text(self, text:str, draw=True):
        self._message.update([text])
        if draw:
            plt.draw()

    def toggle_mode(self, event=None): # add key binding to m for switching mode
        self._mode = {'correction':'annotation', 'annotation':'correction'}[self._mode]
        self.update_mode_text()
    
    def update_colors(self, data_idx=None, draw=True):
        if data_idx is None:
            data_idx = list(range(self.n_signals))
        assert isinstance(data_idx, (list, tuple))
        for this_data_idx in data_idx:
            this_color = self.classes[this_data_idx].color
            self.plot_handles['signal_full'][this_data_idx].set_color(this_color)
            for handle_name, handle in self.plot_handles.items():
                if 'scatter_plot_' in handle_name:
                    fc = handle.get_facecolors()
                    fc[this_data_idx, :3] = this_color
                    fc[this_data_idx, -1] = self.alpha['auto'] if self.classes[this_data_idx].is_auto() else self.alpha['manual']
                    handle.set_facecolors(fc)
        if draw:
            plt.draw()
    
    def update_all(self):
        self.update()
        self.update_class_info_text(draw=False)
        self.update_mode_text(draw=False)
        self.update_message_text('Default message', draw=False)
        self.update_colors(draw=False)
        plt.draw()

    def clear_axes(self, event=None):
        self.plot_handles['ax_history_signal'].clear()
        plt.draw()

    def __call__(self, event):
        super().__call__(event)
        if event.name == 'key_press_event' and event.inaxes == self.plot_handles['ax_signal_full'] and (0 <= int(event.xdata) < self.data.shape[0]):
            this_data_idx = int(event.xdata)
            if self._mode == 'correction':
                if (event.key in self.class_labels_str):
                    new_label = int(event.key)
                    original_label = self.classes[this_data_idx].original_label
                    if new_label == original_label:
                        self.classes[this_data_idx].set_auto()
                    else:
                        self.classes[this_data_idx].set_manual()
                    self.classes[this_data_idx].label = new_label
                    self.update_colors([this_data_idx])
            elif self._mode == 'annotation':
                if event.key in self.annotation_idx_str:
                    this_annotation = self.annotation_names[int(event.key)]
                    if this_annotation not in self.classes[this_data_idx].annotations:
                        self.classes[this_data_idx].annotations.append(this_annotation)
                        self.update_message_text(f'Adding annotation {this_annotation} to signal number {this_data_idx}')
    
    def classlabels_to_dict(self):
        fields_to_save = ('label', 'name', 'assignment_type', 'annotations', 'original_label')
        ret = {}
        for class_idx, class_label in enumerate(self.classes):
            ret[class_idx] = {fld:getattr(class_label, fld)for fld in fields_to_save}
        return ret
    
    def set_classlabels(self, classlabels_dict):
        assert set(classlabels_dict.keys()) == set(range(self.n_signals))
        self.classes = [ClassLabel(**this_label) for this_label in classlabels_dict.values()]


class ClassLabel:
    def __init__(self, 
            label:int,                      # class label (0 - unclassified, 1 - non-resonant, 2 - resonant, etc.)
            name:str = None,                # name of the class
            assignment_type:str = 'auto',   # class label was assigned automatically ('auto') or manually ('manual')
            annotations:list = None,        # for adding annotations to a given class instance
            original_label:int = None,
        ):
        assert label >= 0
        self._label = int(label)
        if original_label is None:
            self.original_label = label
        else:
            self.original_label = int(original_label)
        if name is None:
            name = f'Class_{label}'
        self.name = name
        assert assignment_type in ('auto', 'manual')
        self.assignment_type = assignment_type

        self.palette = plt.get_cmap('tab20')([np.r_[0:1.5:0.05]])[0][:, :3]
        # at the moment, colors are assigned automatically
        self._update_colors()

        if annotations is None:
            self.annotations = []
        else:
            assert isinstance(annotations, list)
            self.annotations = annotations

    @property
    def color(self):
        if self.is_auto():
            return self.color_auto
        return self.color_manual

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, val:int):
        self._label = int(val)
        self._update_colors()

    def _update_colors(self):
        if self._label == 0:
            self.color_auto = self.color_manual = (0.0, 0.0, 0.0) # black
        else:
            self.color_auto = self.palette[(self._label-1)*2+1] # lighter
            self.color_manual = self.palette[(self._label-1)*2]
        
    def is_auto(self):
        return (self.assignment_type == 'auto')
    
    def is_manual(self):
        return (self.assignment_type == 'manual')
    
    def set_auto(self):
        """Meant for undo-ing manual assignment"""
        self.assignment_type = 'auto'

    def set_manual(self):
        self.assignment_type = 'manual'
    
    def add_annotation(self, annot:str):
        self.annotations.append(annot)


if __name__ == "__main__":
    vname = r"S:\2201000537 - Operator\data\001_01\ml_models\dlc\opr01-s001_g01-2023-05-04\videos\us_b_009.mp4"
    fname = r"S:\2201000537 - Operator\data\001_01\ml_models\dlc\opr01-s001_g01-2023-05-04\videos\iteration-4\us_b_009DLC_resnet50_opr01May4shuffle1_550000.h5"
    v = VideoAnnotation(fname, vname)
    vname = r"\\192.168.1.5\Studies\2201000537 - Operator\data_opr02\009_01\ml_models\dlc\opr02_s009_t007_u005.mp4"
    v = VideoPointAnnotator(vname, 'test1234')
