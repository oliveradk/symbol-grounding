import numpy as np
import torch
from torch.utils.data import Dataset

from gym_minigrid.envs.lavarandom import LavaRandomEnv
from gym_minigrid.envs.lavarandom import RandomAgentLavaRandomEnv
from gym_minigrid.wrappers import FullyObsWrapper, ImgObsWrapper
from .translatorwrapper import ImgObsHideAgentWrapper


class ArrowLavaDataset(Dataset):
    """Arrows to Lava Locations dataset."""
    def __init__(self, length=100, size=5, hide_agent=True, randomize_agent=True, seed=True):
        """
        Args:
            length (int): Number of data points
            size (int): size of grid which data is generated from
            hide_agent (bool): whether to hide agent in data
            randomize_agent (bool): whether to randomize the agent location data
            seed (bool): whehter to fix the seed for each index
        """
        self.hide_agent = hide_agent
        self.randomize_agent = randomize_agent
        obs_wrapper = ImgObsHideAgentWrapper if hide_agent else ImgObsWrapper
        env = RandomAgentLavaRandomEnv if randomize_agent else LavaRandomEnv
        self.env = obs_wrapper(FullyObsWrapper(env(num_obstacles=1, show_arrows=True, size=size)))
        self.length = length
        self.size = size
        self.grid = np.array(range((size-2)**2)).reshape(size-2, size-2)
        self.seed = seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.seed:
            self.env.seed(idx)
        obs = self.env.reset()
        arrow_env = torch.Tensor(obs)
        lava_loc_idx = self.loc_to_index(self.env.lava_locs[0])
        return (arrow_env, lava_loc_idx)

    def loc_to_index(self, loc):
        loc = np.array(loc)
        loc = loc - 1
        return self.grid[loc[0]][loc[1]]