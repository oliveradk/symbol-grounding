import os
import argparse

import gym
from stable_baselines3 import PPO

from utils.feature_extractors import GridWorldCNN
from utils.callbacks import SaveOnBestTrainingRewardCallback
from config import LOG_DIR, MODEL_DIR
from utils.utils import wrap_env, make_dir


def train_ppo_gridworld(env_name, model_name, time_steps=1e4, log_freq=1000, verbose=1):
    #make log dir
    log_dir = os.path.join(LOG_DIR, model_name)
    make_dir(os.path.join(LOG_DIR, model_name))

    #wrap env
    env = wrap_env(gym.make(env_name), log_dir)

    # load model
    policy_kwargs = {"features_extractor_class": GridWorldCNN}
    model = PPO("CnnPolicy", env, verbose=verbose, policy_kwargs=policy_kwargs)

    # train model
    callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, log_dir=log_dir)
    model.learn(total_timesteps=time_steps, callback=callback)

    # save model
    save_path = os.path.join(MODEL_DIR, model_name)
    model.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name')
    parser.add_argument('model_name')
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--verbose', type=bool)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    train_ppo_gridworld(**kwargs)


