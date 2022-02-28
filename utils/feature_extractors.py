# from itertools import zip_longest
# from typing import Dict, List, Tuple, Type, Union

import gym
import torch as th
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
# from stable_baselines3.common.type_aliases import TensorDict
# from stable_baselines3.common.utils import get_device

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class GridWorldCNN(BaseFeaturesExtractor):
    """
    CNN from rl-starter-files
    https://github.com/lcswillems/rl-starter-files/blob/e604b36915a13e25ac8a8a912f9a9a15e2d4a170/model.py#L18
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64):
        """
        :param observation_space:
        :param features_dim: Number of features to be extracted
            This corresponds to the number of units for the last layer
        """
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), (
            "You should use GridWorldCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            #import ipdb; ipdb.set_trace()
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None, ...]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        observations.squeeze()
        return self.linear(self.cnn(observations))
