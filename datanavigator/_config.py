"""
This module provides functions to set and get the paths for the clip and cache folders.

The clip folder is used to store video clips, for example, when using VideoBrowser. 
The cache folder is used by the :py:mod:`datanavigator.examples` to write json file containing marked events.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)


def _default_data_dir(name: str) -> str:
    """Default storage path for the clip / cache folder, by platform.

    Windows defaults are unchanged from prior releases (``C:\\data\\<name>``).
    On macOS / Linux the default is ``~/datanavigator/<name>``, which
    replaces the previous Windows-only default that could never resolve
    on a non-Windows host. Both can be overridden by the ``CLIP_FOLDER``
    and ``CACHE_FOLDER`` environment variables.
    """
    if sys.platform.startswith("win"):
        return f"C:\\data\\{name}"
    return str(Path.home() / "datanavigator" / name)


# Use environment variables for default paths
CLIP_FOLDER: str = os.getenv("CLIP_FOLDER", _default_data_dir("_clipcollection"))
CACHE_FOLDER: str = os.getenv("CACHE_FOLDER", _default_data_dir("_cache"))


def set_clip_folder(folder: str) -> None:
    """Set the path for storing video clips.

    The default on Windows is ``C:\\data\\_clipcollection``; on
    macOS / Linux it is ``~/datanavigator/_clipcollection``. Either
    can be overridden via the ``CLIP_FOLDER`` environment variable at
    import time, or via this setter at runtime.
    """
    if not os.path.exists(folder):
        raise ValueError(f"The provided folder path does not exist: {folder}")

    global CLIP_FOLDER
    CLIP_FOLDER = folder
    logging.info(f"Clip folder set to: {CLIP_FOLDER}")

    global CACHE_FOLDER
    if not os.path.exists(CACHE_FOLDER):
        logging.info(
            "Setting the cache folder to be the same as the clip folder. To change, use set_cache_folder(<folder_name>)."
        )
        CACHE_FOLDER = folder


def get_clip_folder() -> str:
    """Get the current path of the clip folder."""
    return CLIP_FOLDER


def set_cache_folder(folder: str) -> None:
    """Set the path for the cache folder."""
    if not os.path.exists(folder):
        raise ValueError(f"The provided folder path does not exist: {folder}")

    global CACHE_FOLDER
    CACHE_FOLDER = folder
    logging.info(f"Cache folder set to: {CACHE_FOLDER}")


def get_cache_folder() -> str:
    """Get the current path of the cache folder."""
    return CACHE_FOLDER
