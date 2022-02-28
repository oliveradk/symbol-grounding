import os
import argparse
import matplotlib.pyplot as plt
from config import LOG_DIR

from stable_baselines3.common.results_plotter import X_TIMESTEPS

from utils.plot_utils import plot_results_modified


def plot_results(dirs=["ppo_arrows", "ppo_arrows_transfer", "ppo_arrows_translated"], time_steps=1e5,
                 title="Arrows Transfer Results", show_points=False):
    parent = os.path.join(os.path.dirname(__file__), LOG_DIR)
    root_dirs = [os.path.join(parent, d) for d in dirs]
    plot_results_modified(root_dirs, time_steps, X_TIMESTEPS, title, labels=dirs, show_points=show_points)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=list)
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--title')

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    plot_results(**kwargs)

