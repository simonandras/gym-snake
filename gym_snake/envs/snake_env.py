
import numpy as np
import gym
from gym_snake.objects.snake import Snake


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, shape):
        self.shape = shape
        self.map = None
        self.snake = None

    def step(self, action):
        pass

    def reset(self) -> None:
        self.snake = Snake(map_shape=self.shape, initial_length=4)

        self.map = np.zeros(self.shape)
        for i, part in enumerate(self.snake.snake_body):
            # head part
            if i == 0:
                self.map[part[0], part[1]] = 2
            # body parts
            else:
                self.map[part[0], part[1]] = 1

    def render(self, mode='human'):
        print(self.map)

    def close(self):
        pass


env = SnakeEnv(shape=(5, 5))
env.reset()
env.render()
