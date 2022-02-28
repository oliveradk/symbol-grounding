from typing import Callable, List, Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt

from stable_baselines3.common.results_plotter import window_func, load_results, ts2xy

X_TIMESTEPS = "timesteps"
X_EPISODES = "episodes"
X_WALLTIME = "walltime_hrs"
POSSIBLE_X_AXES = [X_TIMESTEPS, X_EPISODES, X_WALLTIME]
EPISODES_WINDOW = 100



def plot_results_modified(
    dirs: List[str], num_timesteps: Optional[int], x_axis: str, task_name: str, labels: list, figsize: Tuple[int, int] = (8, 2),
        show_points=False) -> None:
    """
    Plot the results using csv files from ``Monitor`` wrapper.

    :param dirs: the save location of the results to plot
    :param num_timesteps: only plot the points below this value
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: the title of the task to plot
    :param labels: list of labels used for the plot legend
    :param figsize: Size of the figure (width, height)
    :param show_points: Whether to show points (in addition to mean curves)
    """

    data_frames = []
    for folder in dirs:
        data_frame = load_results(folder)
        if num_timesteps is not None:
            data_frame = data_frame[data_frame.l.cumsum() <= num_timesteps]
        data_frames.append(data_frame)
    xy_list = [ts2xy(data_frame, x_axis) for data_frame in data_frames]
    plot_curves(xy_list, x_axis, task_name, labels, figsize, show_points)


def plot_curves(
    xy_list: List[Tuple[np.ndarray, np.ndarray]], x_axis: str, title: str, labels: list, figsize: Tuple[int, int] = (8, 2),
        show_points=False) -> None:
    """
    plot the curves

    :param xy_list: the x and y coordinates to plot
    :param x_axis: the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: the title of the plot
    :param labels: list of labels used for the plot legend
    :param figsize: Size of the figure (width, height)
    :param show_points: Whether to show points (in addition to mean curves)
    """

    plt.figure(title, figsize=figsize)
    max_x = max(xy[0][-1] for xy in xy_list)
    min_x = 0
    for (i, (x, y)) in enumerate(xy_list):
        if show_points:
            plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, label=labels[i])
    plt.xlim(min_x, max_x)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.legend(loc='lower right', fontsize='xx-small')
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
