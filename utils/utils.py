import os
from shutil import rmtree
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3.common.monitor import Monitor


def wrap_env(env, log_dir):
    env = FullyObsWrapper(env)
    env = ImgObsWrapper(env)
    env = Monitor(env, log_dir)
    return env

def make_dir(dir_path):
    if os.path.isdir(dir_path):
        rmtree(dir_path)
    os.mkdir(dir_path)
