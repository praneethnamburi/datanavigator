"""Canonical modal overlays (``datanavigator.confirm`` / ``prompt_text``).

The Qt overlays themselves need a display; here we cover the *headless* contract
— off a Qt backend (no window) both helpers return the supplied default without
blocking — plus the public exports and default-button resolution.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import datanavigator as dnav  # noqa: E402
from datanavigator import _modals  # noqa: E402


def test_exports():
    assert dnav.confirm is _modals.confirm
    assert dnav.prompt_text is _modals.prompt_text


def test_confirm_offqt_returns_default():
    fig = plt.figure()
    try:
        r = dnav.confirm(
            fig,
            title="T",
            message="m",
            buttons=[("Save", "primary"), ("Cancel", "neutral")],
            default="Cancel",
        )
        assert r == "Cancel"
    finally:
        plt.close(fig)


def test_confirm_offqt_default_is_last_button():
    fig = plt.figure()
    try:
        r = dnav.confirm(
            fig, title="T", message="m", buttons=[("A", "primary"), ("B", "neutral")]
        )
        assert r == "B"  # default falls back to the last button
    finally:
        plt.close(fig)


def test_prompt_text_offqt_returns_default():
    fig = plt.figure()
    try:
        assert dnav.prompt_text(fig, title="Name", default="x") == "x"
        assert dnav.prompt_text(fig, title="Name") == ""
    finally:
        plt.close(fig)


def test_role_qss_and_severity_tables_complete():
    assert set(_modals._ROLE_QSS) == {"primary", "destructive", "neutral"}
    assert set(_modals._SEVERITY_COLOR) == {"info", "warning", "destructive"}
