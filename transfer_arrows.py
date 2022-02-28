import argparse
from utils.transfer_ppo_gridworld import transfer_ppo_gridworld


def transfer_arrows(old_model_name="ppo_lava", new_model_name="ppo_arrows_transfer", time_steps=1e5, log_freq=1000, verbose=1):
    return transfer_ppo_gridworld("MiniGrid-ArrowsRandomS5-v0", old_model_name=old_model_name,
                                  new_model_name=new_model_name, time_steps=time_steps, log_freq=log_freq,
                                  verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_model_name')
    parser.add_argument('--new_model_name')
    parser.add_argument('--time_steps', type=int)
    parser.add_argument('--log_freq', type=int)
    parser.add_argument('--verbose', type=bool)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    transfer_arrows(**kwargs)

#NOTE: average reward ~.45 after 1e5 iters




