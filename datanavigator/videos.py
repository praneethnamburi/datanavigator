"""
Module for browsing videos and extracting clips from videos.

Classes:
    VideoBrowser: Scroll through the frames of a video, extract clips of interest.
    VideoPlotBrowser: Browse a video and an array of `pysampled.Data` side by side.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Callable, Dict, Optional, Union

import pysampled
from decord import VideoReader
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

from . import _config
from .core import GenericBrowser


class VideoBrowser(GenericBrowser):
    """Scroll through the frames of a video, extract clips of interest.

    If figure_handle is an axis handle, the video will be plotted in that axis.

    Future Enhancements:
        - Extend VideoBrowser to play, pause, and extract clips using hotkeys.
        - Show timeline in VideoBrowser.
        - Add clickable navigation.
    """

    def __init__(
        self,
        vid_name: str,
        titlefunc: Optional[Callable] = None,
        figure_or_ax_handle: Optional[Union[plt.Axes, plt.Figure]] = None,
        image_process_func: Callable = lambda im: im,
    ):
        """
        Args:
            vid_name (str): Path to the video file.
            titlefunc (Optional[Callable]): Function to generate the title for the plot.
            figure_or_ax_handle (Optional[Union[plt.Axes, plt.Figure]]): Handle to the figure or axis.
            image_process_func (Callable): Function to process the image.
        """
        assert isinstance(figure_or_ax_handle, (plt.Axes, plt.Figure, type(None)))
        if isinstance(figure_or_ax_handle, plt.Axes):
            figure_handle = figure_or_ax_handle.figure
            ax_handle = figure_or_ax_handle
        else:
            # same behavior for figure_or_ax_handle in (None, isinstance(plt.Figure))
            figure_handle = figure_or_ax_handle
            ax_handle = None
        super().__init__(figure_handle)

        if not os.path.exists(vid_name):
            # try looking in the CLIP FOLDER
            vid_name = os.path.join(
                _config.get_clip_folder(), os.path.split(vid_name)[-1]
            )
        assert os.path.exists(vid_name)
        self.fname = vid_name
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, "rb") as f:
            self.data = VideoReader(f)

        if ax_handle is None:
            self._ax = self.figure.subplots(1, 1)
        else:
            assert isinstance(ax_handle, plt.Axes)
            self._ax = ax_handle
        self._im = self._ax.imshow(self.data[0].asnumpy())
        self._ax.axis("off")

        self.fps = self.data.get_avg_fps()
        if titlefunc is None:
            self.titlefunc = (
                lambda s: f"Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}"
            )

        self.image_process_func = image_process_func

        self.set_default_keybindings()
        self.add_key_binding("e", self.extract_clip)
        self.memoryslots.show(pos="bottom left")

        if self.__class__.__name__ == "VideoBrowser":
            plt.show(block=False)
            self.update()

    def increment_frac(self, n_steps: int = 100) -> None:
        """Browse entire dataset in n_steps.

        Args:
            n_steps (int): Number of steps to increment.
        """
        self._current_idx = min(
            self._current_idx + int(len(self) / n_steps), len(self) - 1
        )
        self.update()

    def decrement_frac(self, n_steps: int = 100) -> None:
        """Browse entire dataset in n_steps.

        Args:
            n_steps (int): Number of steps to decrement.
        """
        self._current_idx = max(self._current_idx - int(len(self) / n_steps), 0)
        self.update()

    def update(self) -> None:
        """Update the video frame."""
        self._im.set_data(
            self.image_process_func(self.data[self._current_idx].asnumpy())
        )
        self._ax.set_title(self.titlefunc(self))
        super().update()
        plt.draw()

    def extract_clip(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        fname_out: Optional[str] = None,
        out_rate: Optional[int] = None,
    ) -> str:
        """Extract a clip from the video.

        Args:
            start_frame (Optional[int]): Starting frame of the clip.
            end_frame (Optional[int]): Ending frame of the clip.
            fname_out (Optional[str]): Output filename.
            out_rate (Optional[int]): Output frame rate.

        Returns:
            str: Path to the extracted clip.
        """
        try:
            import ffmpeg

            use_subprocess = False
        except ModuleNotFoundError:
            import subprocess

            use_subprocess = True

        if start_frame is None:
            start_frame = self.memoryslots._list["1"]
        if end_frame is None:
            end_frame = self.memoryslots._list["2"]
        assert end_frame > start_frame
        start_time = float(start_frame) / self.fps
        end_time = float(end_frame) / self.fps
        dur = end_time - start_time
        if out_rate is None:
            out_rate = self.fps
        if fname_out is None:
            fname_out = os.path.join(
                _config.get_clip_folder(),
                os.path.splitext(self.name)[0]
                + "_s{:.3f}_e{:.3f}.mp4".format(start_time, end_time),
            )
        if use_subprocess:
            subprocess.getoutput(
                f'ffmpeg -ss {start_time} -i "{self.fname}" -r {out_rate} -t {dur} -vcodec h264_nvenc "{fname_out}"'
            )
        else:
            ffmpeg.input(self.fname, ss=start_time).output(
                fname_out, vcodec="h264_nvenc", t=dur, r=out_rate
            ).run()
        return fname_out


class VideoPlotBrowser(GenericBrowser):
    """
    Browse a video and an array of pysampled.Data side by side.

    Args:
        vid_name (str): Path to the video file.
        signals (Dict[str, pysampled.Data]): Dictionary of signals.
        titlefunc (Optional[Callable]): Function to generate the title for the plot.
        figure_handle (Optional[plt.Figure]): Handle to the figure.
        event_win (Optional[tuple]): Event window. Visualize signals around an event. Use this to create "scrolling" plot visualizations, e.g. [-0.5, 1.].
    """

    def __init__(
        self,
        vid_name: str,
        signals: Dict[str, pysampled.Data],
        titlefunc: Optional[Callable] = None,
        figure_handle: Optional[plt.Figure] = None,
        event_win: Optional[tuple] = None,
    ):
        figure_handle = plt.figure(figsize=(20, 12))
        super().__init__(figure_handle)

        self.event_win = event_win

        self.vid_name = vid_name
        assert os.path.exists(vid_name)
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, "rb") as f:
            self.video_data = VideoReader(f)
        self.fps = self.video_data.get_avg_fps()

        self.signals = signals
        if titlefunc is None:
            self.titlefunc = (
                lambda s: f"Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}"
            )

        self.plot_handles = self._setup()
        self.plot_handles["ax"]["montage"].set_axis_off()

        self.set_default_keybindings()
        self.add_key_binding("e", self.extract_clip)
        self._len = len(self.video_data)
        self.memoryslots.show(pos="bottom left")

        self.figure.canvas.mpl_connect("button_press_event", self.onclick)

        plt.show(block=False)
        self.update()

    def __len__(self) -> int:
        """Return the number of frames in the video."""
        return self._len

    def _setup(self) -> dict:
        """Setup the plot handles.

        Returns:
            dict: Dictionary of plot handles.
        """
        fig = self.figure
        gs = GridSpec(nrows=len(self.signals), ncols=2, width_ratios=[2, 3])
        ax = {}
        plot_handles = {}
        for signal_count, (signal_name, this_signal) in enumerate(self.signals.items()):
            this_ax = fig.add_subplot(gs[signal_count, 1])
            plot_handles[f"signal{signal_count}"] = this_ax.plot(
                this_signal.t, this_signal()
            )
            ylim = this_ax.get_ylim()
            (plot_handles[f"signal{signal_count}_tick"],) = this_ax.plot(
                [0, 0], ylim, "k"
            )
            this_ax.set_title(signal_name)
            if signal_count < len(self.signals) - 1:
                this_ax.get_xaxis().set_ticks([])
            else:
                this_ax.set_xlabel("Time (s)")
            ax[f"signal{signal_count}"] = this_ax

        ax["montage"] = fig.add_subplot(gs[:, 0])
        plot_handles["montage"] = ax["montage"].imshow(self.video_data[0].asnumpy())
        plot_handles["ax"] = ax
        plot_handles["fig"] = fig
        signal_ax = [v for k, v in plot_handles["ax"].items() if "signal" in k]
        signal_ax[0].get_shared_x_axes().join(*signal_ax)
        plot_handles["signal_ax"] = signal_ax
        return plot_handles

    def update(self) -> None:
        """Update the video frame and signals."""
        self.plot_handles["montage"].set_data(
            self.video_data[self._current_idx].asnumpy()
        )
        self.plot_handles["ax"]["montage"].set_title(self.titlefunc(self))
        for signal_count, this_signal in enumerate(self.signals.items()):
            self.plot_handles[f"signal{signal_count}_tick"].set_xdata(
                [self._current_idx / self.fps] * 2
            )
        if self.event_win is not None:
            curr_t = self._current_idx / self.fps
            self.plot_handles["signal_ax"][0].set_xlim(
                curr_t + self.event_win[0], curr_t + self.event_win[1]
            )
        super().update()
        plt.draw()

    def onclick(self, event) -> None:
        """Right click mouse to seek to that frame.

        Args:
            event: Matplotlib event.
        """
        this_frame = round(event.xdata * self.fps)

        if (
            isinstance(this_frame, (int, float))
            and (0 <= this_frame < self._len)
            and (str(event.button) == "MouseButton.RIGHT")
        ):
            self._current_idx = this_frame
            self.update()

    def extract_clip(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        sav_dir: Optional[str] = None,
        out_rate: float = None,
    ) -> str:
        """Save a video of screengrabs.

        Args:
            start_frame (Optional[int]): Starting frame of the clip. Defaults to the first memory slot.
            end_frame (Optional[int]): Ending frame of the clip. Defaults to the second memory slot.
            sav_dir (Optional[str]): Directory to save the clip. If not provided, a timestamped directory is created.

        Returns:
            Path to the saved video file.
        """
        import shutil
        import subprocess

        if start_frame is None:
            start_frame = self.memoryslots._list["1"]
        if end_frame is None:
            end_frame = self.memoryslots._list["2"]
        assert end_frame > start_frame

        if sav_dir is None:
            sav_dir = os.path.join(
                _config.get_clip_folder(), datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        if not os.path.exists(sav_dir):
            os.mkdir(sav_dir)

        if out_rate is None:
            out_rate = self.fps

        print(f"Saving image sequence to {sav_dir}...")
        for frame_count in range(start_frame, end_frame + 1):
            self._current_idx = frame_count
            self.update()
            self.figure.savefig(os.path.join(sav_dir, f"{frame_count:08d}.png"))

        print("Creating video from image sequence...")
        cmd = f'cd "{sav_dir}" && ffmpeg -framerate {self.fps} -start_number {start_frame} -i %08d.png -c:v h264_nvenc -b:v 10M -maxrate 12M -bufsize 24M -vf scale="-1:1080" -an "{sav_dir}.mp4"'
        subprocess.getoutput(cmd)

        print("Removing temporary folder...")
        shutil.rmtree(sav_dir)

        print("Done")
        return f"{sav_dir}.mp4"
