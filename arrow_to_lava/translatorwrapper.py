import os.path
from os.path import dirname
import numpy as np
import torch
import gym

from gym_minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX
from gym_minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from gym_minigrid.envs.placedobject import PlacedObject

from .arrowtranslator import ArrowTranlsator
from .utils import index_to_loc


class TranslateArrows(gym.core.ObservationWrapper):
    '''
    Wrapper for Arrows environment that translates observations to the equivalent Lava environment,
    updating the parameters of the translator to simulate the training process
    '''

    def __init__(self, env, update_interval=1000, max_steps=150000, hide_agent=True, model_type=ArrowTranlsator,
                 model_dir='saved_params', model_name='translator_net'):
        '''
        :param env: Arrow environment to translate from
        :param update_interval: how often to update the weights of the translator
        :param max_steps: maximum steps to update the translator weights at
        :param hide_agent: whether agent should be hidden from translator
        :param model_type: type of translator model
        :param model_dir: location of pretrained translator params
        :param model_name: name of pretrained translator params (actual params should be followed by step_num,
        e.g. translator_net_1000)
        '''
        super().__init__(env)
        self.env = env
        self.size = self.env.width
        self.update_interval = update_interval
        self.max_steps = max_steps
        self.model_type = model_type
        self.model_dir = os.path.join(dirname(__file__), model_dir)
        self.model_name = model_name
        self.steps = 0
        self.model = self.model_type()
        self.hide_agent = hide_agent
        obs_wrapper = ImgObsHideAgentWrapper if hide_agent else ImgObsWrapper
        self.preimage_env = obs_wrapper(FullyObsWrapper(PlacedObject(size=env.width)))
        self.image_env = ImgObsWrapper(FullyObsWrapper(PlacedObject(size=env.width)))

    def reset(self, **kwargs):
        self.preimage_env.reset()
        self.image_env.reset()
        return self.observation(self.env.reset(**kwargs), step=False)

    def observation(self, observation, step=True):
        if self.steps < self.max_steps and self.steps % self.update_interval == 0:
            path = os.path.join(self.model_dir, f"{self.model_name}_{self.steps}.pth")
            self.model.load_state_dict(torch.load(path))

        obs = self._translate(observation)
        if step:
            self.steps += 1
        return obs

    def _translate(self, observation):
        self.preimage_env.reset(agent_pos=self.env.agent_pos, agent_dir=self.env.agent_dir)
        self.image_env.reset(agent_pos=self.env.agent_pos, agent_dir=self.env.agent_dir)
        arrow_locs = self.env.arrow_locs
        for arrow_loc in arrow_locs:
            arrow = self.env.grid.get(arrow_loc[0], arrow_loc[1])
            self.preimage_env.place_arrow(color=arrow.color, orientation=arrow.orientation, loc=arrow_loc)
            preimage = self._no_op(self.preimage_env)
            lava_idx = int(self.model.predict(torch.Tensor(preimage[None, ...])))
            lava_loc = index_to_loc(lava_idx, size=self.size)
            self.image_env.place_lava(lava_loc)
            self.preimage_env.reset()
        obs = self._no_op(self.image_env)
        return obs

    def render(self, mode='arrows'):
        if mode == 'lava':
            return self.image_env.render()
        else:
            return self.env.render()

    def _no_op(self, env):
        obs, _, _, _ = env.step(MiniGridEnv.Actions.pickup)
        return obs


class ImgObsHideAgentWrapper(ImgObsWrapper):
    '''
    Used instead of ImgObsWrapper, provides same functionality but removes agent from observation
    '''

    def observation(self, observation):
        observation = super().observation(observation)
        agent_pos = self.env.agent_pos
        observation[agent_pos[0]][agent_pos[1]] = np.array([OBJECT_TO_IDX['empty'], 0, 0])
        return observation
