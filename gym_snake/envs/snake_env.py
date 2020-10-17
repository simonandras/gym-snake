
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

from gym_snake.utilities.utils import asd1


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, shape):
        self.shape = shape
        self.map = None
        self.snake_body = None

    def step(self, action):
        pass

    def reset(self):
        self.map = np.zeros(self.shape)
        self.snake_body = np.zeros((4, 2))

    def render(self, mode='human'):
        asd1()

    def close(self):
        pass
