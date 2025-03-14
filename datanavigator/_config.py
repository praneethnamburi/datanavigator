"""
This module provides functions to set and get the paths for the clip and cache folders.

The clip folder is used to store video clips, for example, when using VideoBrowser. 
The cache folder is used by the :py:mod:`datanavigator.examples` to write json file containing marked events.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Use environment variables for default paths
CLIP_FOLDER: str = os.getenv("CLIP_FOLDER", "C:\\data\\_clipcollection")
CACHE_FOLDER: str = os.getenv("CACHE_FOLDER", "C:\\data\\_cache")


def set_clip_folder(folder: str) -> None:
    """Set the path for storing video clips."""
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
