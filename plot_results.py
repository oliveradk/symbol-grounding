import os
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from config import LOG_DIR


def plot_results(dirs=["ppo_arrows", "ppo_arrows_transfer", "ppo_arrows_translated"], time_steps=1e5,
                 title="Arrows Transfer Results"):
    parent = os.path.join(os.path.dirname(__file__), LOG_DIR)
    dirs = [os.path.join(parent, d) for d in dirs]
    results_plotter.plot_results(dirs, time_steps, results_plotter.X_TIMESTEPS, title)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirs', type=list)
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--title')

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    plot_results(**kwargs)

