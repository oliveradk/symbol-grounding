import os
import argparse

import gym
from stable_baselines3 import PPO

from utils.callbacks import SaveOnBestTrainingRewardCallback
from config import LOG_DIR, MODEL_DIR
from utils.utils import wrap_env, make_dir

from arrow_to_lava.translatorwrapper import TranslateArrows


def translate_arrows_to_lava(base_model_name="ppo_lava", new_model_name="ppo_arrows_translated", time_steps=1e5,
                             log_freq=1000, verbose=1, update_interval=1000):
    # make log dir
    log_dir = os.path.join(LOG_DIR, new_model_name)
    make_dir(os.path.join(LOG_DIR, new_model_name))

    # wrap env
    env = wrap_env(gym.make("MiniGrid-ArrowsRandomS5-v0"), log_dir)
    env = TranslateArrows(env, update_interval=update_interval, max_steps=time_steps)
    env.reset()

    #load model
    model_path = os.path.join(MODEL_DIR, base_model_name)
    model = PPO.load(model_path)
    model.set_env(env)
    model.learning_rate = 0 #stop training
    model._setup_lr_schedule()

    # "train" model
    callback = SaveOnBestTrainingRewardCallback(check_freq=log_freq, log_dir=log_dir)
    model.learn(total_timesteps=time_steps, callback=callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--verbose', type=bool)
    parser.add_argument('--update_interval', type=int)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    translate_arrows_to_lava(**kwargs)

