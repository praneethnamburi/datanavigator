# datanavigator

[![src](https://img.shields.io/badge/src-github-blue)](https://github.com/praneethnamburi/datanavigator)
[![PyPI - Version](https://img.shields.io/pypi/v/datanavigator.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/datanavigator/)
[![Documentation Status](https://readthedocs.org/projects/datanavigator/badge/?version=latest)](https://datanavigator.readthedocs.io)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/datanavigator/main/LICENSE)

*Interactive data visualization for signals, videos, and complex data objects.*

`datanavigator` is a matplotlib-based toolkit for interactive data visualization that handles signals, videos, and complex data objects. It provides both simple tools for navigating data with minimal programming and a user-friendly API for building sophisticated data interaction applications. This versatility makes it both powerful and accessible, regardless of a user's programming expertise.

## Installation

```sh
pip install datanavigator
```

If you encounter dependency issues and are using `conda`, set up your environment with `conda` using the `requirements.yml` file in this repository.

```sh
conda env create -n env-datanavigator -f https://github.com/praneethnamburi/datanavigator/raw/master/requirements.yml
conda activate env-datanavigator
pip install datanavigator
```

## Quickstart

### 1. Browse video frames and extract a clip
```python
import datanavigator as dnav

video_browser = dnav.VideoBrowser(dnav.get_example_video())
# Use the arrow keys to browse through frames.
# Press Ctrl+K to bring up a list of available keyboard shortcuts
# To extract a clip, 
#   1. navigate to the start frame and press 1
#   2. navigate to the end frame and press 2
#   3. press e to save the extracted clip
# run dnav.get_clip_folder() to find the saved video clip

# Or, you can extract a clip from the command line
clip_path = video_browser.extract_clip(start_frame=100, end_frame=200)
print(f"Extracted clip saved to: {clip_path}")
```

### 2. Browse time series data and mark events of interest
```python
import datanavigator as dnav
signal_browser = dnav.EventPickerDemo()
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/datanavigator](https://github.com/praneethnamburi/datanavigator)


## Acknowledgments

This tool was developed as part of the ImmersionToolbox initiative at the [MIT.nano Immersion Lab](https://immersion.mit.edu). Thanks to NCSOFT for supporting this initiative.
