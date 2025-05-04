import os
import numpy as np
from pathlib import Path
import pytest

import matplotlib.pyplot as plt
from matplotlib import axes as maxes
from datanavigator import get_example_video
from datanavigator.utils import (
    ticks_from_times,
    _List,
    _parse_fax,
    _parse_pos,
    TextView,
    get_palette,
    is_video,
    Video,
    is_pathname_valid,
    is_path_creatable,
    is_path_exists_or_creatable,
)


@pytest.fixture(scope="session", autouse=True)
def initialize_folder_structure(tmp_path_factory):
    base_temp_directory = tmp_path_factory.getbasetemp()

    # Create a sample text file
    (base_temp_directory / "sample_text_file.txt").touch()

    # Download a sammple video file
    video_path = get_example_video(dest_folder=base_temp_directory)
    assert str(video_path).endswith("example_video.mp4")


@pytest.fixture
def video_fname(tmp_path_factory):
    return os.path.join(tmp_path_factory.getbasetemp(), "example_video.mp4")


def test_ticks_from_times():
    times = [1, 2, 3]
    tick_lim = (0, 1)
    x, y = ticks_from_times(times, tick_lim)
    assert x == [1, 1, np.nan, 2, 2, np.nan, 3, 3, np.nan]
    assert y == [0, 1, np.nan, 0, 1, np.nan, 0, 1, np.nan]


def test_List():
    lst = _List([1, 0, 2, 4, 3])
    assert lst.next(1.1) == 2
    assert lst.next(10) == 4
    assert lst.previous(1.1) == 1
    assert lst.previous(1) == lst.previous(0) == lst.previous(-10) == 0


def test_parse_fax():
    fig, ax = plt.subplots()
    f, a = _parse_fax(fig)
    assert isinstance(f, plt.Figure)
    assert isinstance(a, maxes.Axes)
    plt.close(fig)


def test_parse_pos():
    pos = "top left"
    parsed_pos = _parse_pos(pos)
    assert parsed_pos == (0, 1, "top", "left")


def test_TextView():
    text = ["line1", "line2"]
    tv = TextView(text)
    assert tv.text == text
    assert tv._pos == (0, 0, "bottom", "left")
    plt.close(tv.figure)
    tv = TextView({1: "line1", 2: "line2"}, pos="top right")
    assert tv.text == ["1 - line1", "2 - line2"]


def test_get_palette():
    palette = get_palette(n_colors=3)
    assert len(palette) == 3


def test_is_video(tmp_path_factory, video_fname):
    assert is_video(video_fname) == True
    assert is_video(tmp_path_factory.getbasetemp() / "sample_text_file.txt") == False


def test_Video(video_fname):
    video = Video(video_fname)
    assert video.fname == video_fname
    assert video.name == Path(video_fname).stem
    assert len(video) == 300


def test_is_pathname_valid(tmp_path_factory):
    assert is_pathname_valid(str(tmp_path_factory.getbasetemp() / "abc12")) == True
    assert is_pathname_valid(str(tmp_path_factory.getbasetemp() / "<")) == False


def test_is_path_creatable(tmp_path_factory):
    assert is_path_creatable(str(tmp_path_factory.getbasetemp() / "abc12")) == True


def test_is_path_exists_or_creatable(tmp_path_factory):
    assert is_path_exists_or_creatable(str(tmp_path_factory.getbasetemp())) == True
    assert (
        is_path_exists_or_creatable(str(tmp_path_factory.getbasetemp() / "abc12"))
        == True
    )
    assert (
        is_path_exists_or_creatable(str(tmp_path_factory.getbasetemp() / "<")) == False
    )
