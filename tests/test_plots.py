import pytest
import numpy as np
import matplotlib.pyplot as plt

from datanavigator.plots import PlotBrowser
from tests.conftest import simulate_key_press, press_browser_button

@pytest.fixture(scope="module")
def figure_with_two_subplots():
    # Set up: create the figure and axis
    fig, ax_all = plt.subplots(2, 1, figsize=(5, 10))
    yield fig, ax_all
    # Teardown: close the figure
    plt.close(fig)

def plotter(figure_with_two_subplots):
    fig, ax_all = figure_with_two_subplots
    def setup(data):
        plot_handles = {}
        plot_handles["ax1"] = ax_all[0]
        plot_handles["ax2"] = ax_all[1]
        plot_handles["ax"] = ax_all
        plot_handles["figure"] = fig
        plot_handles["p1"], = ax_all[0].plot([], [], label="p1")
        plot_handles["p2"], = ax_all[1].plot([], [], label="p2")
        return plot_handles
    
    def update(data, plot_handles):
        plot_handles["p1"].set_data(data.t, data())
        plot_handles["p2"].set_data(data.t, data.lowpass(2)())

    return setup, update

def plotter_no_setup(figure_with_two_subplots):
    fig, ax_all = figure_with_two_subplots
    def update(data, figure_handle):
        ax_all[0].plot(data.t, data())
        ax_all[1].plot(data.t, data.lowpass(2)())

    return update

def test_plot_browser(signal_list, figure_with_two_subplots):
    pb = PlotBrowser(signal_list, plotter(figure_with_two_subplots))
    assert "p1" in pb.plot_handles
    event = simulate_key_press(pb.figure, "right")
    assert pb._current_idx == 0
    pb(event)
    assert pb._current_idx == 1
    assert np.allclose(pb.get_current_data()(), signal_list[1]())

    assert pb.buttons["Auto limits"].state is False
    press_browser_button(pb.buttons["Auto limits"])
    assert pb.buttons["Auto limits"].state is True
    pb.update()

def test_plot_browser_no_setup(signal_list, figure_with_two_subplots):
    pb = PlotBrowser(signal_list, plotter_no_setup(figure_with_two_subplots))
    assert pb.plot_handles is None
    assert pb.setup_func is None
    event = simulate_key_press(pb.figure, "right")
    assert pb._current_idx == 0
    pb(event)
    assert pb._current_idx == 1
    assert np.allclose(pb.get_current_data()(), signal_list[1]())
