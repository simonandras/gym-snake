
import gym
from gym import error, spaces, utils
from gym.utils import seeding

from utilities import test_asd


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        pass

    def step(self, action):
        test_asd()

    def reset(self):
        pass

    def render(self, mode='human'):
        print("asd")

    def close(self):
        pass

