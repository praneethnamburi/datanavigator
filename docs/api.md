# API Reference

## Overview
```{eval-rst}
.. automodule:: datanavigator
    :members:
    :special-members: __getitem__
```

## Browsers
```{eval-rst}
.. automodule:: datanavigator.core
    :members:
    :undoc-members:
```

```{eval-rst}
.. automodule:: datanavigator.signals
    :members:
    :undoc-members:
```

```{eval-rst}
.. automodule:: datanavigator.plots
    :members:
    :undoc-members:
```

```{eval-rst}
.. automodule:: datanavigator.videos
    :members:
    :undoc-members:
```

```{eval-rst}
.. automodule:: datanavigator.components
    :members:
    :undoc-members:
```

## Assets

Assets that power the functionality of the `GenericBrowser` class.

```{eval-rst}
.. automodule:: datanavigator.assets
    :members:
    :undoc-members:
    :exclude-members: ERROR_INVALID_NAME
```

```{eval-rst}
.. automodule:: datanavigator.events
    :members:
    :undoc-members:
    :exclude-members: ERROR_INVALID_NAME
```

## Utilities
```{eval-rst}
.. automodule:: datanavigator.utils
    :members:
    :undoc-members:
    :exclude-members: ERROR_INVALID_NAME
```

## Point tracking + optical flow (relocated)

The point-tracking UI, annotation containers
(``VideoAnnotation`` / ``VideoAnnotations``), and Lucas-Kanade helpers
(``lucas_kanade`` / ``lucas_kanade_rstc``) relocated to the
[DUSTrack](https://dustrack.readthedocs.io) package in `1.5.0`
alongside its DeepLabCut workflow. See
[`dustrack.DUSTrack`](https://dustrack.readthedocs.io/en/latest/api.html)
for the new home.