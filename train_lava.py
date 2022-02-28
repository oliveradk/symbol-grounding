import argparse
from utils.train_ppo_gridworld import train_ppo_gridworld


def train_ppo_lava(model_name="ppo_lava", time_steps=3e5, log_freq=1000, verbose=1):
    train_ppo_gridworld(env_name="MiniGrid-LavaRandomS5-v0", model_name=model_name,
                        time_steps=time_steps, log_freq=log_freq, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name')
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--verbose', type=bool)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    train_ppo_lava(**kwargs)
