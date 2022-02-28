import argparse

from train_lava import train_ppo_lava
from train_arrows import train_ppo_arrows
from transfer_arrows import transfer_arrows
from translate_arrow_lava import translate_arrows_to_lava
from arrow_to_lava.train_translator import train_translator
from plot_results import plot_results


def main(pretrain_timesteps=int(3e5), transfer_timesteps=int(1e5), save_interval=int(1e2), dataset_size=int(1e4), epochs=int(1e1)):
    # train lava
    train_ppo_lava(time_steps=pretrain_timesteps)

    # train arrows
    train_ppo_arrows(time_steps=transfer_timesteps)

    # train transfer arrows
    transfer_arrows(time_steps=transfer_timesteps)

    # train translator
    train_translator(dataset_size=dataset_size, epochs=epochs + 1, save_interval=save_interval)

    # run translator
    translate_arrows_to_lava(time_steps=transfer_timesteps, update_interval=save_interval)

    # plot results
    plot_results(time_steps=transfer_timesteps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_timesteps', type=int)
    parser.add_argument('--transfer_timesteps', type=int)
    parser.add_argument('--save_interval', type=int)
    parser.add_argument('--dataset_size', type=int)
    parser.add_argument('--epochs', type=int)

    args_dict = vars(parser.parse_args())
    kwargs = {key: value for key, value in args_dict.items() if value is not None}
    main(**kwargs)





