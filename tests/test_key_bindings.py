"""Tests for the rc2 keybindings cheatsheet + button-face hint surface.

Pins the contract for:
- :class:`datanavigator.core.KeyBinding` dataclass + defaults
- :meth:`GenericBrowser.add_key_binding` with new ``group=`` /
  ``on_button=`` kwargs
- Bidirectional matching between :meth:`Buttons.add` and
  :meth:`add_key_binding` for ``on_button=True``
- :meth:`GenericBrowser.show_key_bindings` stdout fallback on non-Qt
- ``KeyBinding`` storage round-trip through ``__call__`` (the keypress
  dispatch path that consumes ``_keypressdict``)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from datanavigator import GenericBrowser
from datanavigator.core import (
    KeyBinding,
    _format_keybindings_text,
    _group_keybindings,
)


def _make_browser():
    return GenericBrowser(figure_handle=plt.figure())


def test_keybinding_dataclass_defaults():
    """``KeyBinding(callback, description)`` -> group=None, on_button=False."""
    def noop():
        return None

    kb = KeyBinding(callback=noop, description="noop")
    assert kb.callback is noop
    assert kb.description == "noop"
    assert kb.group is None
    assert kb.on_button is False


def test_add_key_binding_defaults_store_keybinding_instance():
    b = _make_browser()

    def my_action():
        pass

    b.add_key_binding("alt+x", my_action, "fire X")
    kb = b._keypressdict["alt+x"]
    assert isinstance(kb, KeyBinding)
    assert kb.callback is my_action
    assert kb.description == "fire X"
    assert kb.group is None
    assert kb.on_button is False


def test_add_key_binding_records_group():
    b = _make_browser()
    b.add_key_binding("alt+y", lambda: None, "fire Y", group="Annotation")
    assert b._keypressdict["alt+y"].group == "Annotation"


def test_group_keybindings_pushes_none_group_to_end():
    """``_group_keybindings`` keeps insertion order but places None last."""
    kpd = {
        "a": KeyBinding(lambda: None, "alpha", group="First"),
        "b": KeyBinding(lambda: None, "beta"),  # group=None -> "Other"
        "c": KeyBinding(lambda: None, "gamma", group="Second"),
        "d": KeyBinding(lambda: None, "delta", group="First"),
    }
    sections = _group_keybindings(kpd)
    names = [name for name, _ in sections]
    assert names == ["First", "Second", "Other"]
    first_rows = dict(sections[0][1])
    assert "a" in first_rows and "d" in first_rows
    other_rows = dict(sections[-1][1])
    assert other_rows == {"b": "beta"}


def test_group_keybindings_respects_section_order():
    """``section_order`` re-orders named buckets; unlisted ones follow,
    ``None`` still goes last."""
    kpd = {
        # Insertion order: First, Other (None), Second, Third
        "a": KeyBinding(lambda: None, "alpha", group="First"),
        "b": KeyBinding(lambda: None, "beta"),  # None -> "Other"
        "c": KeyBinding(lambda: None, "gamma", group="Second"),
        "d": KeyBinding(lambda: None, "delta", group="Third"),
    }
    # Pin Second to lead, then Third; First is not in section_order so it
    # follows them in insertion order; Other stays last.
    sections = _group_keybindings(kpd, section_order=("Second", "Third"))
    names = [name for name, _ in sections]
    assert names == ["Second", "Third", "First", "Other"]


def test_group_keybindings_section_order_with_missing_entries():
    """Group names in ``section_order`` that don't appear in keypressdict
    are silently skipped."""
    kpd = {
        "a": KeyBinding(lambda: None, "alpha", group="First"),
    }
    sections = _group_keybindings(kpd, section_order=("MissingGroup", "First"))
    names = [name for name, _ in sections]
    assert names == ["First"]


def test_button_hint_binding_first_then_button():
    """Declare the binding, then add the button: hint attaches."""
    b = _make_browser()

    def my_action(event=None):
        pass

    b.add_key_binding(
        "ctrl+r", my_action, "Replace from overlay",
        group="Annotation", on_button=True,
    )
    btn = b.buttons.add(text="Replace from overlay", action_func=my_action)
    assert btn.name == "Replace from overlay  (ctrl+r)"
    assert btn.label.get_text() == "Replace from overlay  (ctrl+r)"


def test_button_hint_button_first_then_binding():
    """Add the button, then declare the binding: hint attaches."""
    b = _make_browser()

    def my_action(event=None):
        pass

    btn = b.buttons.add(text="Replace from overlay", action_func=my_action)
    assert btn.name == "Replace from overlay"  # no hint yet
    b.add_key_binding(
        "ctrl+r", my_action, "Replace from overlay",
        group="Annotation", on_button=True,
    )
    assert btn.name == "Replace from overlay  (ctrl+r)"
    assert btn.label.get_text() == "Replace from overlay  (ctrl+r)"


def test_button_hint_idempotent_on_repeat_call():
    """Applying the same hint twice doesn't append the suffix twice."""
    b = _make_browser()

    def my_action(event=None):
        pass

    btn = b.buttons.add(text="Save", action_func=my_action)
    b.add_key_binding("s", my_action, "Save", on_button=True)
    b.add_key_binding("s", my_action, "Save", on_button=True)  # re-register
    assert btn.name == "Save  (s)"
    assert btn.name.count("(s)") == 1


def test_button_hint_skipped_when_on_button_false():
    """Default ``on_button=False`` leaves the button label alone."""
    b = _make_browser()

    def my_action(event=None):
        pass

    btn = b.buttons.add(text="Save", action_func=my_action)
    b.add_key_binding("s", my_action, "Save")  # on_button defaults to False
    assert btn.name == "Save"


def test_show_key_bindings_stdout_on_non_qt(capsys):
    """Agg has no QMainWindow -> fall back to stdout dump."""
    b = _make_browser()
    b.set_default_keybindings()
    b.show_key_bindings()
    out = capsys.readouterr().out
    assert "[Navigation]" in out
    assert "[View]" in out
    # Sample bindings should appear under their declared groups.
    assert "left" in out
    assert "ctrl+k" in out


def test_format_keybindings_text_section_headers():
    """Spot-check the stdout formatter layout."""
    kpd = {
        "s": KeyBinding(lambda: None, "Save", group="File"),
        "x": KeyBinding(lambda: None, "Other thing"),  # falls into Other
    }
    text = _format_keybindings_text(kpd)
    assert "[File]" in text
    assert "[Other]" in text
    # File section comes before Other (None group goes last).
    assert text.index("[File]") < text.index("[Other]")


def test_keypress_dispatch_still_fires_after_dataclass_migration():
    """Regression guard: ``__call__`` reads ``.callback``, not ``[0]``."""
    from unittest.mock import MagicMock
    from tests.conftest import simulate_key_press  # type: ignore[import-not-found]

    b = _make_browser()
    cb = MagicMock()
    b.add_key_binding("alt+z", cb, "fire Z")
    event = simulate_key_press(b.figure, key="alt+z")
    b(event)
    cb.assert_called_once()
