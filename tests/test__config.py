import os
import pytest
from unittest.mock import patch
import datanavigator._config
from datanavigator._config import (
    set_clip_folder,
    get_clip_folder,
    set_cache_folder,
    get_cache_folder,
)


@pytest.fixture
def temp_folder(tmp_path):
    """Fixture to create a temporary folder."""
    return str(tmp_path)


def test_set_clip_folder(temp_folder):
    """Test setting the clip folder."""
    current_clip_folder = get_clip_folder()
    set_clip_folder(temp_folder)
    assert get_clip_folder() == temp_folder
    # Reset to default value
    set_clip_folder(current_clip_folder)


def test_set_clip_folder_invalid():
    """Test setting an invalid clip folder."""
    with pytest.raises(ValueError, match="The provided folder path does not exist"):
        set_clip_folder("nonexistent_folder")


def test_set_cache_folder(temp_folder):
    """Test setting the cache folder."""
    current_cache_folder = get_cache_folder()
    set_cache_folder(temp_folder)
    assert get_cache_folder() == temp_folder
    # Reset to default value
    set_cache_folder(current_cache_folder)


def test_set_cache_folder_invalid():
    """Test setting an invalid cache folder."""
    with pytest.raises(ValueError, match="The provided folder path does not exist"):
        set_cache_folder("nonexistent_folder")


def test_default_clip_and_cache_folder():
    """Test default clip and cache folder values."""
    curr_clip_folder = get_clip_folder()
    curr_cache_folder = get_cache_folder()
    with patch.dict(os.environ, {"CLIP_FOLDER": "C:\\test_clip", "CACHE_FOLDER": "C:\\test_cache"}):
        from importlib import reload
        reload(datanavigator._config)
        assert get_clip_folder() == "C:\\test_clip"
        assert get_cache_folder() == "C:\\test_cache"
    # Reset to default values
    set_clip_folder(curr_clip_folder)
    set_cache_folder(curr_cache_folder)
