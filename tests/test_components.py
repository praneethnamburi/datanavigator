import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from datanavigator.components import ComponentBrowser, ClassLabel
from .conftest import simulate_key_press_at_xy

import matplotlib.pyplot as plt

# Assuming conftest.py is in the parent directory 'tests'
# and the current file is tests/test_components.py
# and the module to test is datanavigator/components.py


@pytest.fixture
def component_data():
    """Fixture for sample ComponentBrowser data."""
    np.random.seed(0)
    n_signals = 10
    n_timepts = 50
    n_components = 3
    data = np.random.randn(n_signals, n_timepts)
    data_transform = np.random.randn(n_signals, n_components)
    labels = np.random.randint(0, 3, n_signals)  # Labels 0, 1, 2
    class_names = {0: "Class_0", 1: "Class_1", 2: "Class_2"}
    return data, data_transform, labels, class_names


def test_component_browser_init_defaults(component_data, mock_figure):
    """Test initialization with minimal arguments."""
    data, data_transform, _, _ = component_data
    with patch("matplotlib.pyplot.show"):  # Prevent plot window from showing
        browser = ComponentBrowser(
            data=data, data_transform=data_transform, figure_handle=mock_figure
        )

    assert browser.data is data
    assert browser.n_signals == data.shape[0]
    assert browser.n_timepts == data.shape[1]
    assert np.array_equal(browser.labels, np.zeros(data.shape[0], dtype=int))
    assert browser.class_names == {0: "Class_0"}
    assert browser.desired_class_names == {0: "Class_0"}
    assert browser.annotation_names == {
        1: "Representative",
        2: "Best",
        3: "Noisy",
        4: "Worst",
    }
    assert browser._mode == "correction"
    assert (
        len(browser.plot_handles["ax_pca"]) == 3
    )  # n_components * (n_components - 1) / 2 = 3 * 2 / 2 = 3
    assert len(browser.plot_handles["signal_plots"]) == browser.n_signals
    assert len(browser.plot_handles["signal_full"]) == browser.n_signals
    assert browser._class_info_text is not None
    assert browser._mode_text is not None
    assert browser._annotation_text is not None
    assert browser._message is not None
    assert browser._desired_class_info_text is not None


def test_component_browser_init_with_labels(component_data, mock_figure):
    """Test initialization with provided labels."""
    data, data_transform, labels, class_names = component_data
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            figure_handle=mock_figure,
        )

    assert np.array_equal(browser.labels, labels)
    assert browser.class_names == class_names
    assert browser.n_classes == len(np.unique(labels))
    assert len(browser.classes) == browser.n_signals
    assert all(isinstance(c, ClassLabel) for c in browser.classes)


def test_component_browser_init_with_names(component_data, mock_figure):
    """Test initialization with provided class_names, desired_class_names, and annotation_names."""
    data, data_transform, labels, _ = component_data
    custom_class_names = {0: "Zero", 1: "One", 2: "Two"}
    custom_desired_names = {0: "Desired_0", 1: "Desired_1", 2: "Desired_2"}
    custom_annotation_names = {1: "Annot_A", 2: "Annot_B"}

    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            class_names=custom_class_names,
            desired_class_names=custom_desired_names,
            annotation_names=custom_annotation_names,
            figure_handle=mock_figure,
        )

    assert browser.class_names == custom_class_names
    assert browser.desired_class_names == custom_desired_names
    assert browser.annotation_names == custom_annotation_names
    assert browser._annotation_text.text[-len(custom_annotation_names) :] == [
        f"{k}:{v}" for k, v in custom_annotation_names.items()
    ]


def test_component_browser_properties(component_data, mock_figure):
    """Test properties of the ComponentBrowser."""
    data, data_transform, labels, _ = component_data
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            figure_handle=mock_figure,
        )

    assert browser.n_signals == data.shape[0]
    assert browser.n_timepts == data.shape[1]
    assert len(browser.colors) == browser.n_signals
    # assert all(isinstance(c, tuple) and len(c) == 3 for c in browser.colors) # Check if colors are RGB tuples
    # Cannot directly test pysampled.Data equality easily without implementing __eq__
    assert browser.signal._sig.shape == (data.size,)
    assert browser.signal.sr == browser.n_timepts


@patch("matplotlib.pyplot.draw")
def test_component_browser_event_pick(mock_draw, component_data, mock_figure):
    """Test the onpick event handler."""
    data, data_transform, labels, _ = component_data
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            figure_handle=mock_figure,
        )

    # Simulate a pick event
    mock_event = MagicMock()
    mock_event.ind = [3, 5]  # Indices picked
    browser.update = MagicMock()  # Mock update to avoid complex plot checks

    browser.onpick(mock_event)

    assert browser.pick_event is mock_event
    assert browser._data_index in mock_event.ind  # Should pick one of the indices
    browser.update.assert_called_once()


@patch("matplotlib.pyplot.draw")
def test_component_browser_event_dblclick(mock_draw, component_data, mock_figure):
    """Test the select_signal_piece_dblclick event handler."""
    data, data_transform, labels, _ = component_data
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            figure_handle=mock_figure,
        )

    # Simulate a double-click event in the correct axes
    mock_event = MagicMock()
    mock_event.inaxes = browser.plot_handles["ax_signal_full"]
    mock_event.dblclick = True
    mock_event.xdata = 4.5  # Corresponds to signal index 4

    browser.update = MagicMock()  # Mock update

    browser.select_signal_piece_dblclick(mock_event)

    assert browser._data_index == 4
    browser.update.assert_called_once()

    # Simulate a single click (should do nothing)
    browser.update.reset_mock()
    mock_event.dblclick = False
    browser.select_signal_piece_dblclick(mock_event)
    browser.update.assert_not_called()

    # Simulate double click outside valid range
    mock_event.dblclick = True
    mock_event.xdata = browser.n_signals + 1
    browser.select_signal_piece_dblclick(mock_event)
    browser.update.assert_not_called()  # _data_index should not change


@patch("matplotlib.pyplot.draw")
def test_component_browser_event_keypress(mock_draw, component_data, mock_figure):
    """Test key press events for mode toggle, clear, correction, and annotation."""
    data, data_transform, labels, class_names = component_data
    annotation_names = {1: "Annot_A", 2: "Annot_B"}
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            class_names=class_names,
            annotation_names=annotation_names,
            figure_handle=mock_figure,
        )

    # --- Test Mode Toggle ---
    browser.update_mode_text = MagicMock()
    event_m = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]), key="m", xdata=1.5
    )
    browser(event_m)
    assert browser._mode == "annotation"
    browser.update_mode_text.assert_called_once()
    browser(event_m)  # Toggle back
    assert browser._mode == "correction"
    assert browser.update_mode_text.call_count == 2

    # --- Test Clear Axes ---
    browser.plot_handles["ax_history_signal"].clear = MagicMock()
    event_r = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]), key="r", xdata=1.5
    )
    browser(event_r)
    browser.plot_handles["ax_history_signal"].clear.assert_called_once()
    mock_draw.assert_called()  # draw is called by clear_axes
    mock_draw.reset_mock()

    # --- Test Correction Mode ---
    browser._mode = "correction"
    target_idx = 2
    original_label = browser.classes[target_idx].label
    original_assignment = browser.classes[target_idx].assignment_type
    # Find a valid key different from the original label
    new_label_key = str((original_label + 1) % browser.n_classes)
    new_label = int(new_label_key)

    browser.update_colors = MagicMock()
    event_correct = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]),
        key=new_label_key,
        xdata=target_idx + 0.5,
    )
    browser(event_correct)

    assert browser.classes[target_idx].label == new_label
    assert browser.classes[target_idx].is_manual()
    browser.update_colors.assert_called_once_with([target_idx])

    # Test setting back to original label (should become auto)
    browser.update_colors.reset_mock()
    event_correct_back = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]),
        key=str(original_label),
        xdata=target_idx + 0.5,
    )
    browser(event_correct_back)
    assert browser.classes[target_idx].label == original_label
    # Check if it reverted to original assignment type or became auto if original was manual
    if browser.classes[target_idx].original_label == original_label:
        assert browser.classes[
            target_idx
        ].is_auto()  # Becomes auto if key matches original_label
    browser.update_colors.assert_called_once_with([target_idx])

    # --- Test Annotation Mode ---
    browser._mode = "annotation"
    target_idx_annot = 4
    assert not browser.classes[
        target_idx_annot
    ].annotations  # Should be empty initially
    annot_key = "1"  # Corresponds to "Annot_A"
    annot_name = annotation_names[int(annot_key)]

    browser.update_message_text = MagicMock()
    event_annot = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]),
        key=annot_key,
        xdata=target_idx_annot + 0.5,
    )
    browser(event_annot)

    assert annot_name in browser.classes[target_idx_annot].annotations
    browser.update_message_text.assert_called_once()
    assert (
        f"Adding annotation {annot_name}" in browser.update_message_text.call_args[0][0]
    )

    # Test adding same annotation again (should not duplicate)
    browser.update_message_text.reset_mock()
    browser(event_annot)
    assert (
        browser.classes[target_idx_annot].annotations.count(annot_name) == 1
    )  # Still only one
    browser.update_message_text.assert_not_called()  # No message if annotation exists

    # Test invalid key in annotation mode
    event_invalid_annot = simulate_key_press_at_xy(
        (browser.figure, browser.plot_handles["ax_signal_full"]),
        key="x",
        xdata=target_idx_annot + 0.5,
    )
    browser(event_invalid_annot)
    assert (
        browser.classes[target_idx_annot].annotations.count(annot_name) == 1
    )  # No change
    browser.update_message_text.assert_not_called()  # No message


@patch("matplotlib.pyplot.draw")
def test_component_browser_classlabels_dict(mock_draw, component_data, mock_figure):
    """Test classlabels_to_dict and set_classlabels."""
    data, data_transform, labels, class_names = component_data
    with patch("matplotlib.pyplot.show"):
        browser = ComponentBrowser(
            data=data,
            data_transform=data_transform,
            labels=labels,
            class_names=class_names,
            figure_handle=mock_figure,
        )

    # Get initial dict
    initial_dict = browser.classlabels_to_dict()
    assert isinstance(initial_dict, dict)
    assert len(initial_dict) == browser.n_signals
    assert 0 in initial_dict
    assert "label" in initial_dict[0]
    assert "name" in initial_dict[0]
    assert "assignment_type" in initial_dict[0]
    assert "annotations" in initial_dict[0]
    assert "original_label" in initial_dict[0]
    assert initial_dict[0]["label"] == browser.classes[0].label
    assert initial_dict[0]["original_label"] == browser.classes[0].original_label

    # Modify the dict
    modified_dict = initial_dict.copy()
    modified_dict[0]["label"] = (modified_dict[0]["label"] + 1) % browser.n_classes
    modified_dict[0]["assignment_type"] = "manual"
    modified_dict[0]["annotations"] = ["TestAnnot"]
    modified_dict[1]["label"] = (modified_dict[1]["label"] + 1) % browser.n_classes

    # Set the modified dict
    browser.set_classlabels(modified_dict)

    # Verify changes
    assert browser.classes[0].label == modified_dict[0]["label"]
    assert browser.classes[0].assignment_type == "manual"
    assert browser.classes[0].annotations == ["TestAnnot"]
    assert browser.classes[1].label == modified_dict[1]["label"]
    # Check that original_label was preserved if present in dict
    assert browser.classes[0].original_label == initial_dict[0]["original_label"]

    # Check that a new dict reflects the changes
    new_dict = browser.classlabels_to_dict()
    assert new_dict[0]["label"] == modified_dict[0]["label"]
    assert new_dict[0]["assignment_type"] == "manual"
    assert new_dict[0]["annotations"] == ["TestAnnot"]
    assert new_dict[1]["label"] == modified_dict[1]["label"]
