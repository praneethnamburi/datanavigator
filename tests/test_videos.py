import os
import pytest
from pathlib import Path
from unittest.mock import patch

import datanavigator
from matplotlib.backend_bases import MouseButton
from matplotlib.backend_bases import MouseEvent

from tests.conftest import simulate_key_press, simulate_mouse_click


@pytest.fixture(scope="function")
def video_fpath(setup_folders):
    return datanavigator.get_example_video()


def test_setup_folders(setup_folders):
    # setup_folders is a fixture that creates temporary directories for clips and cache
    # check if setup_folders is created correctly
    clip_dir, cache_dir = setup_folders
    assert datanavigator.get_clip_folder() == clip_dir
    assert datanavigator.get_cache_folder() == cache_dir


def test_video_browser_init(video_fpath):
    browser = datanavigator.VideoBrowser(video_fpath)
    assert browser.name == "example_video"
    assert browser.fps == browser.data.get_avg_fps() == 29.97
    assert len(browser.data) == 300

    # Check for searching in the clip folder
    clip_dir = datanavigator.get_clip_folder()
    assert os.path.exists(os.path.join(clip_dir, "example_video.mp4"))
    browser2 = datanavigator.VideoBrowser("example_video.mp4")
    assert browser2.name == "example_video"

    with pytest.raises(AssertionError):
        datanavigator.VideoBrowser("non_existent_video.mp4")


def test_video_browser_init2(video_fpath, mock_figure):
    browser = datanavigator.VideoBrowser(video_fpath, figure_or_ax_handle=mock_figure)
    assert browser.name == "example_video"
    assert browser.fps == browser.data.get_avg_fps() == 29.97
    assert len(browser.data) == 300


def test_video_browser_init3(video_fpath, matplotlib_figure):
    fig, ax = matplotlib_figure
    browser = datanavigator.VideoBrowser(video_fpath, figure_or_ax_handle=ax)
    assert browser.name == "example_video"
    assert browser.fps == browser.data.get_avg_fps() == 29.97
    assert len(browser.data) == 300


def test_video_browser_navigation(video_fpath):
    browser = datanavigator.VideoBrowser(video_fpath)
    assert browser._current_idx == 0
    browser(simulate_key_press(browser.figure, "right"))
    assert browser._current_idx == 1
    browser(simulate_key_press(browser.figure, "shift+up"))
    assert browser._current_idx == 299
    browser(simulate_key_press(browser.figure, "shift+down"))
    assert browser._current_idx == 0
    browser(simulate_key_press(browser.figure, "up"))
    browser(simulate_key_press(browser.figure, "up"))
    assert browser._current_idx == 20
    browser(simulate_key_press(browser.figure, "down"))
    assert browser._current_idx == 10
    browser(simulate_key_press(browser.figure, "shift+right"))
    assert browser._current_idx == 13
    browser(simulate_key_press(browser.figure, "shift+left"))
    browser(simulate_key_press(browser.figure, "shift+left"))
    assert browser._current_idx == 7


def test_video_browser_extract_clip(video_fpath):
    browser = datanavigator.VideoBrowser(video_fpath)
    browser(simulate_key_press(browser.figure, "up"))
    browser(simulate_key_press(browser.figure, "up"))
    assert browser._current_idx == 20
    assert browser.memoryslots._list["1"] is None
    browser(simulate_key_press(browser.figure, "1"))
    assert browser.memoryslots._list["1"] == 20
    browser._current_idx = 75
    browser(simulate_key_press(browser.figure, "2"))
    assert browser.memoryslots._list["2"] == 75
    ret_name = browser.extract_clip()
    assert os.path.exists(ret_name)
    fps = browser.fps
    assert Path(ret_name).stem == f"example_video_s{20/fps:.3f}_e{75/fps:.3f}"
    os.remove(ret_name)

    browser._current_idx = 40
    browser(simulate_key_press(browser.figure, "1")) # go to 20
    assert browser.memoryslots._list["1"] == 20
    browser(simulate_key_press(browser.figure, "1")) # clear 20 from memory slot 1
    assert browser.memoryslots._list["1"] is None
    browser._current_idx = 40
    browser(simulate_key_press(browser.figure, "1"))
    assert browser.memoryslots._list["1"] == 40

    browser(simulate_key_press(browser.figure, "2")) # go to 75
    assert browser.memoryslots._list["2"] == 75
    browser(simulate_key_press(browser.figure, "2")) # clear 75 from memory slot 2
    assert browser.memoryslots._list["2"] is None
    browser._current_idx = 92
    browser(simulate_key_press(browser.figure, "2"))
    assert browser.memoryslots._list["2"] == 92

    def mock_import(name, *args):
        if name == "ffmpeg":
            raise ModuleNotFoundError("No module named 'ffmpeg'")
        return original_import(name, *args)

    original_import = __import__
    with patch("builtins.__import__", side_effect=mock_import):
        ret_name = browser.extract_clip()
        assert os.path.exists(ret_name)
        assert Path(ret_name).stem == f"example_video_s{40/fps:.3f}_e{92/fps:.3f}"
        os.remove(ret_name)


def test_video_plot_browser_init(video_fpath, signal_list):
    # generate random signals for 10 s
    signals = dict(zip(["white_noise", "sine_wave", "three_sine_waves"], signal_list))
    browser = datanavigator.VideoPlotBrowser(video_fpath, signals)
    assert browser.fps == browser.video_data.get_avg_fps() == 29.97
    assert len(browser.video_data) == len(browser) == browser._len == 300

    # test _setup
    assert "signal0" in browser.plot_handles
    assert "signal2_tick" in browser.plot_handles
    # montage is the video
    assert "montage" in browser.plot_handles

    # test mouse click
    assert browser._current_idx == 0
    event = simulate_mouse_click(
        (browser.figure, browser.plot_handles["signal_ax"][0]),
        xdata=0.5, 
        ydata=0.5,
        button=3,
        )
    browser.figure.canvas.callbacks.process("button_press_event", event)
    
    assert browser._current_idx == round(browser.fps * 0.5)

    # test event window and clip extraction
    browser = datanavigator.VideoPlotBrowser(video_fpath, signals, event_win=[-0.5, 1.])
    browser._current_idx = 150
    assert browser.memoryslots._list["1"] == None
    browser(simulate_key_press(browser.figure, "1")) # go to 20
    assert browser.memoryslots._list["1"] == 150

    browser._current_idx = 160
    assert browser.memoryslots._list["2"] == None
    browser(simulate_key_press(browser.figure, "2")) # go to 20
    assert browser.memoryslots._list["2"] == 160

    screengrab_clip = browser.extract_clip()
    assert os.path.exists(screengrab_clip)
