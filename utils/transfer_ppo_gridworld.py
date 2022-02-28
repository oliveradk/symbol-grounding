import os
import argparse

import gym
from stable_baselines3 import PPO

from utils.callbacks import SaveOnBestTrainingRewardCallback
from config import LOG_DIR, MODEL_DIR
from utils.utils import make_dir, wrap_env


def transfer_ppo_gridworld(env_name, old_model_name, new_model_name, time_steps=1e4, log_freq=1000, verbose=1):
    # make log dir
    log_dir = os.path.join(LOG_DIR, new_model_name)
    make_dir(os.path.join(LOG_DIR, new_model_name))

    # wrap env
    env = wrap_env(gym.make(env_name), log_dir)

    # load model
    model_path = os.path.join(MODEL_DIR, old_model_name)
    model = PPO.load(model_path)
    model.set_env(env)

    # train model
    callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, log_dir=log_dir)
    model.learn(total_timesteps=time_steps, callback=callback)

    # save model
    save_path = os.path.join(MODEL_DIR, new_model_name)
    model.save(save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    parser.add_argument('old_model_name')
    parser.add_argument('new_model_name')
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--verbose', type=bool)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    transfer_ppo_gridworld(**kwargs)
