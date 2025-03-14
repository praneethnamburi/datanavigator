"""
This module is for visualizing and interacting with time series data that
can be analyzed using dimesionality reduction techniques such as PCA. It
was originally developed to visualize and classify periodic motion data
during running.
"""

from __future__ import annotations

import numpy as np
import pysampled
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional, List, Dict, Union, Any

from . import utils
from .core import GenericBrowser


class ComponentBrowser(GenericBrowser):
    def __init__(
        self,
        data: np.ndarray,
        data_transform: np.ndarray,
        labels: Optional[np.ndarray] = None,
        figure_handle: Optional[plt.Figure] = None,
        class_names: Optional[Dict[int, str]] = None,
        desired_class_names: Optional[Dict[int, str]] = None,
        annotation_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the ComponentBrowser.

        Args:
            data (np.ndarray): 2D array with number of signals on dim1, and number of time points on dim2.
            data_transform (np.ndarray): Transformed data, still a 2D array with number of signals x number of components. 
                For example, transformed using one of (sklearn.decomposition.PCA, umap.UMAP, sklearn.manifold.TSNE, sklearn.decomposition.FastICA)
            labels (Optional[np.ndarray]): n_signals x 1 array with each entry representing the class of each signal piece.
            figure_handle (Optional[plt.Figure]): Matplotlib figure handle.
            class_names (Optional[Dict[int, str]]): Dictionary of class names.
            desired_class_names (Optional[Dict[int, str]]): Dictionary of desired class names.
            annotation_names (Optional[Dict[int, str]]): Dictionary of annotation names.

        This GUI is meant to be used and extended for
          - 'corrections', where classes are modified / assigned
          - 'annotations', where labels or annotations (separate from a class assignment in the sense that each signal belongs exactly to one class, and a signal my have 0 or more annotations)
        """
        super().__init__(figure_handle)
        self.alpha = {"manual": 0.8, "auto": 0.3}
        self.data = data

        n_components = np.shape(data_transform)[1]

        if labels is None:
            self.labels = np.zeros(self.n_signals, dtype=int)
        else:
            assert len(labels) == self.n_signals
            self.labels = labels
        # make sure all class labels are zero or positive!
        assert np.min(self.labels) >= 0 

        class_labels = list(np.unique(self.labels))
        
        # for detecting keypresses
        self.class_labels_str = [str(x) for x in class_labels]
        self.n_classes = len(class_labels)
        if class_names is None:
            self.class_names = {class_label: f"Class_{class_label}" for class_label in class_labels}
        else:
            assert set(class_names.keys()) == set(class_labels)
            self.class_names = class_names
        self.classes = [ClassLabel(label=label, name=self.class_names[label]) for label in self.labels]

        if desired_class_names is None:
            desired_class_names = self.class_names
        self.desired_class_names = desired_class_names

        if annotation_names is None:
            annotation_names = {1: "Representative", 2: "Best", 3: "Noisy", 4: "Worst"}
        self.annotation_names = annotation_names
        self.annotation_idx_str = [str(x) for x in self.annotation_names]

        self.cid.append(self.figure.canvas.mpl_connect("pick_event", self.onpick))
        self.cid.append(self.figure.canvas.mpl_connect("button_press_event", self.select_signal_piece_dblclick))

        n_scatter_plots = int(n_components * (n_components - 1) / 2)
        self.gs = GridSpec(3, max(n_scatter_plots, 4))

        self._data_index = 0
        self.plot_handles = {}
        self.plot_handles["ax_pca"] = {}
        plot_number = 0
        for xc in range(n_components - 1):
            for yc in range(xc + 1, n_components):
                this_ax = self.figure.add_subplot(self.gs[1, plot_number])
                this_ax.set_title(str((xc + 1, yc + 1)))
                self.plot_handles["ax_pca"][plot_number] = this_ax
                self.plot_handles[f"scatter_plot_{xc+1}_{yc+1}"] = this_ax.scatter(
                    data_transform[:, xc],
                    data_transform[:, yc],
                    c=self.colors,
                    alpha=self.alpha["auto"],
                    picker=5,
                )
                (self.plot_handles[f"scatter_highlight_{xc+1}_{yc+1}"],) = this_ax.plot(
                    [], [], "o", color="darkorange"
                )
                plot_number += 1

        self.plot_handles["signal_plots"] = []
        this_ax = self.figure.add_subplot(self.gs[2, 0])
        self.plot_handles["ax_signal_plots"] = this_ax
        for signal_count in range(self.n_signals):
            self.plot_handles["signal_plots"].append(this_ax.plot(self.data[signal_count, :])[0])

        self.plot_handles["ax_current_signal"] = self.figure.add_subplot(self.gs[2, 1])
        (self.plot_handles["current_signal"],) = self.plot_handles["ax_current_signal"].plot(
            list(range(self.n_timepts)), [np.nan] * self.n_timepts
        )
        self.plot_handles["ax_current_signal"].set_xlim(self.plot_handles["ax_signal_plots"].get_xlim())
        self.plot_handles["ax_current_signal"].set_ylim(self.plot_handles["ax_signal_plots"].get_ylim())

        self.plot_handles["ax_history_signal"] = self.figure.add_subplot(self.gs[2, 2])

        self.plot_handles["ax_signal_full"] = self.figure.add_subplot(self.gs[0, :])
        self.plot_handles["signal_full"] = []
        time_index = np.r_[: self.n_timepts] / self.n_timepts
        for idx, sig in enumerate(self.data):
            (this_plot_handle,) = self.plot_handles["ax_signal_full"].plot(
                idx + time_index, sig, color=self.colors[idx]
            )  # assumes that there is only one line drawn per signal
            self.plot_handles["signal_full"].append(this_plot_handle)
        self.plot_handles["signal_selected_piece"], = self.plot_handles["ax_signal_full"].plot([], [], color="gray", linewidth=2)

        this_ylim = self.plot_handles["ax_signal_full"].get_ylim()
        for x_pos in np.r_[: self.n_signals + 1]:  # separators between signals
            self.plot_handles["ax_signal_full"].plot([x_pos] * 2, this_ylim, "k", linewidth=0.2)
        self.memoryslots.disable()

        self._class_info_text = utils.TextView([], self.figure, pos="bottom left")
        self.update_class_info_text()

        self._mode = "correction"  # ('correction' or 'annotation')
        self._mode_text = utils.TextView([], self.figure, pos="center left")
        self.update_mode_text()
        self.add_key_binding("m", self.toggle_mode)

        self._annotation_text = utils.TextView(
            ["", "Annotation list:"] + [f"{k}:{v}" for k, v in self.annotation_names.items()],
            self.figure,
            pos="top left",
        )

        self._message = utils.TextView(["Last action : "], self.figure, pos="bottom right")

        self._desired_class_info_text = utils.TextView([], self.figure, pos="bottom center")
        self.update_desired_class_info_text()

        self.add_key_binding("r", self.clear_axes)
        plt.show(block=False)

    @property
    def n_signals(self) -> int:
        """Return the number of signals."""
        return self.data.shape[0]

    @property
    def n_timepts(self) -> int:
        """Return the number of time points."""
        return self.data.shape[-1]

    @property
    def colors(self) -> List[tuple]:
        """Return the colors for each class."""
        return [cl.color for cl in self.classes]

    @property
    def signal(self) -> pysampled.Data:
        """Return the 2D Numpy array as a signal."""
        return pysampled.Data(self.data.flatten(), sr=self.n_timepts)

    def select_signal_piece_dblclick(self, event: Any) -> None:
        """Double click a signal piece in the timecourse view to highlight that point.

        Args:
            event (Any): The mouse event.
        """
        if event.inaxes == self.plot_handles["ax_signal_full"] and event.dblclick:
            if 0 <= int(event.xdata) < self.data.shape[0]:
                self._data_index = int(event.xdata)
                self.update()

    def onpick(self, event: Any) -> None:
        """Single click a projected point.

        Args:
            event (Any): The pick event.
        """
        self.pick_event = event
        self._data_index = np.random.choice(event.ind)
        self.update()

    def update(self) -> None:
        """Update the plot with the current data index."""
        super().update()
        for handle_name, handle in self.plot_handles.items():
            if "scatter_plot_" in handle_name:
                this_data = np.squeeze(handle._offsets[self._data_index].data)
                self.plot_handles[handle_name.replace("_plot_", "_highlight_")].set_data(this_data[0], this_data[1])
        self.plot_handles["ax_history_signal"].plot(self.data[self._data_index, :])
        self.plot_handles["current_signal"].set_ydata(self.data[self._data_index, :])
        self.plot_handles["signal_selected_piece"].set_data(
            np.arange(self.n_timepts) / self.n_timepts + self._data_index,
            self.data[self._data_index, :],
        )
        # self.plot_handles['signal_full'][self._data_index].linewidth = 3
        plt.draw()

    def update_class_info_text(self, draw: bool = True) -> None:
        """Update the class info text.

        Args:
            draw (bool): Whether to draw the plot after updating.
        """
        self._class_info_text.update(
            ["Class list:"] + [f"{k}:{v}" for k, v in self.class_names.items()]
        )
        if draw:
            plt.draw()

    def update_desired_class_info_text(self, draw: bool = True) -> None:
        """Update the desired class info text.

        Args:
            draw (bool): Whether to draw the plot after updating.
        """
        self._desired_class_info_text.update(
            ["Desired class list:"] + [f"{k}:{v}" for k, v in self.desired_class_names.items()]
        )
        if draw:
            plt.draw()

    def update_mode_text(self, draw: bool = True) -> None:
        """Update the mode text.

        Args:
            draw (bool): Whether to draw the plot after updating.
        """
        self._mode_text.update([f"mode: {self._mode}"])
        if draw:
            plt.draw()

    def update_message_text(self, text: str, draw: bool = True) -> None:
        """Update the message text.

        Args:
            text (str): The message text.
            draw (bool): Whether to draw the plot after updating.
        """
        self._message.update([text])
        if draw:
            plt.draw()

    def toggle_mode(self, event: Optional[Any] = None) -> None:
        """Toggle between correction and annotation modes.

        Args:
            event (Optional[Any]): The key event.
        """
        self._mode = {"correction": "annotation", "annotation": "correction"}[self._mode]
        self.update_mode_text()

    def update_colors(self, data_idx: Optional[List[int]] = None, draw: bool = True) -> None:
        """Update the colors of the plot.

        Args:
            data_idx (Optional[List[int]]): List of data indices to update.
            draw (bool): Whether to draw the plot after updating.
        """
        if data_idx is None:
            data_idx = list(range(self.n_signals))
        assert isinstance(data_idx, (list, tuple))
        for this_data_idx in data_idx:
            this_color = self.classes[this_data_idx].color
            self.plot_handles["signal_full"][this_data_idx].set_color(this_color)
            for handle_name, handle in self.plot_handles.items():
                if "scatter_plot_" in handle_name:
                    fc = handle.get_facecolors()
                    fc[this_data_idx, :3] = this_color
                    fc[this_data_idx, -1] = (
                        self.alpha["auto"]
                        if self.classes[this_data_idx].is_auto()
                        else self.alpha["manual"]
                    )
                    handle.set_facecolors(fc)
        if draw:
            plt.draw()

    def update_all(self) -> None:
        """Update all components of the plot."""
        self.update()
        self.update_class_info_text(draw=False)
        self.update_mode_text(draw=False)
        self.update_message_text("Default message", draw=False)
        self.update_colors(draw=False)
        plt.draw()

    def clear_axes(self, event: Optional[Any] = None) -> None:
        """Clear the axes.

        Args:
            event (Optional[Any]): The key event.
        """
        self.plot_handles["ax_history_signal"].clear()
        plt.draw()

    def __call__(self, event: Any) -> None:
        """Handle events.

        Args:
            event (Any): The event.
        """
        super().__call__(event)
        if (
            event.name == "key_press_event"
            and event.inaxes == self.plot_handles["ax_signal_full"]
            and (0 <= int(event.xdata) < self.data.shape[0])
        ):
            this_data_idx = int(event.xdata)
            if self._mode == "correction":
                if event.key in self.class_labels_str:
                    new_label = int(event.key)
                    original_label = self.classes[this_data_idx].original_label
                    if new_label == original_label:
                        self.classes[this_data_idx].set_auto()
                    else:
                        self.classes[this_data_idx].set_manual()
                    self.classes[this_data_idx].label = new_label
                    self.update_colors([this_data_idx])
            elif self._mode == "annotation":
                if event.key in self.annotation_idx_str:
                    this_annotation = self.annotation_names[int(event.key)]
                    if this_annotation not in self.classes[this_data_idx].annotations:
                        self.classes[this_data_idx].annotations.append(this_annotation)
                        self.update_message_text(
                            f"Adding annotation {this_annotation} to signal number {this_data_idx}"
                        )

    def classlabels_to_dict(self) -> Dict[int, Dict[str, Union[int, str, List[str]]]]:
        """Convert class labels to a dictionary.

        Returns:
            Dict[int, Dict[str, Union[int, str, List[str]]]]: Dictionary of class labels.
        """
        fields_to_save = ("label", "name", "assignment_type", "annotations", "original_label")
        ret = {}
        for class_idx, class_label in enumerate(self.classes):
            ret[class_idx] = {fld: getattr(class_label, fld) for fld in fields_to_save}
        return ret

    def set_classlabels(self, classlabels_dict: Dict[int, Dict[str, Union[int, str, List[str]]]]) -> None:
        """Set class labels from a dictionary.

        Args:
            classlabels_dict (Dict[int, Dict[str, Union[int, str, List[str]]]]): Dictionary of class labels.
        """
        assert set(classlabels_dict.keys()) == set(range(self.n_signals))
        self.classes = [ClassLabel(**this_label) for this_label in classlabels_dict.values()]


class ClassLabel:
    def __init__(
        self,
        label: int,
        name: Optional[str] = None,
        assignment_type: str = "auto",
        annotations: Optional[List[str]] = None,
        original_label: Optional[int] = None,
    ):
        """
        Initialize a ClassLabel.

        Args:
            label (int): Class label (0 - unclassified, 1 - non-resonant, 2 - resonant, etc.).
            name (Optional[str]): Name of the class.
            assignment_type (str): Class label was assigned automatically ('auto') or manually ('manual').
            annotations (Optional[List[str]]): List of annotations for the class instance.
            original_label (Optional[int]): Original class label.
        """
        assert label >= 0
        self._label = int(label)
        if original_label is None:
            self.original_label = label
        else:
            self.original_label = int(original_label)
        if name is None:
            name = f"Class_{label}"
        self.name = name
        assert assignment_type in ("auto", "manual")
        self.assignment_type = assignment_type

        self.palette = plt.get_cmap("tab20")([np.r_[0:1.5:0.05]])[0][:, :3]
        self._update_colors()

        if annotations is None:
            self.annotations = []
        else:
            assert isinstance(annotations, list)
            self.annotations = annotations

    @property
    def color(self) -> tuple:
        """Return the color of the class."""
        if self.is_auto():
            return self.color_auto
        return self.color_manual

    @property
    def label(self) -> int:
        """Return the class label."""
        return self._label

    @label.setter
    def label(self, val: int) -> None:
        """Set the class label.

        Args:
            val (int): The new class label.
        """
        self._label = int(val)
        self._update_colors()

    def _update_colors(self) -> None:
        """Update the colors based on the class label."""
        if self._label == 0:
            self.color_auto = self.color_manual = (0.0, 0.0, 0.0)  # black
        else:
            self.color_auto = self.palette[(self._label - 1) * 2 + 1]  # lighter
            self.color_manual = self.palette[(self._label - 1) * 2]

    def is_auto(self) -> bool:
        """Return whether the class label was assigned automatically.

        Returns:
            bool: True if the class label was assigned automatically, False otherwise.
        """
        return self.assignment_type == "auto"

    def is_manual(self) -> bool:
        """Return whether the class label was assigned manually.

        Returns:
            bool: True if the class label was assigned manually, False otherwise.
        """
        return self.assignment_type == "manual"

    def set_auto(self) -> None:
        """Set the class label assignment type to automatic."""
        self.assignment_type = "auto"

    def set_manual(self) -> None:
        """Set the class label assignment type to manual."""
        self.assignment_type = "manual"

    def add_annotation(self, annot: str) -> None:
        """Add an annotation to the class instance.

        Args:
            annot (str): The annotation to add.
        """
        self.annotations.append(annot)
