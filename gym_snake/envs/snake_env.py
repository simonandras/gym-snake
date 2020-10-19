
import numpy as np
import gym
from gym_snake.envs.objects import Snake, Food
from gym_snake.utilities.utils import array_in_collection


class SnakeEnv(gym.Env):
    """
    Actions:
        0 : turn left
        1 : go ahead
        2 : turn right
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, shape: tuple, initial_snake_length: int = 4):
        self.shape = shape
        assert shape[0] >= 5 and shape[1] >= 5, "The map size should be at least 5x5"
        self.initial_snake_length = initial_snake_length
        assert initial_snake_length >= 2, "The initial snake length should be at least 2"

        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(2, shape[0], shape[1]), dtype=np.float32)
        self.reward_range = (-1., 1.)

        self.map = None            # 2d np.array
        self.previous_map = None
        self.snake = None          # Snake object
        self.food = None           # Food object
        self.done = True           # status of the episode

    def step(self, action: int) -> tuple:
        if self.done:
            raise EnvironmentError("Cant make step when the episode is done")

        new_head = self.snake.get_new_head(action)

        if self.snake.valid_part(new_head):
            tail = self.snake.snake_body.pop()
            self.snake.snake_body.appendleft(new_head)
            if np.array_equal(self.food.location, new_head):
                self.snake.snake_body.append(tail)  # restore tail
                self.snake.length += 1  # increase length
                self.food.create()
                reward = 1.
            else:
                reward = 0.
            self.snake.update_direction()
            self.update_map(start=False)
        # out of bound or new_head intersects with the other body parts
        else:
            reward = -1.
            self.end_episode()

        return np.array([self.map, self.previous_map]), reward, self.done, {}

    def end_episode(self) -> None:
        self.map = np.zeros(self.shape, dtype=np.float32)
        self.previous_map = np.zeros(self.shape, dtype=np.float32)
        self.snake = None
        self.food = None
        self.done = True

    def reset(self) -> np.ndarray:
        # reset the episode done parameter
        self.done = False

        # creating random snake
        self.snake = Snake(map_shape=self.shape, initial_length=self.initial_snake_length)

        # creating random food
        self.food = Food(map_shape=self.shape, snake=self.snake)

        # adding snake and food to the map
        self.update_map(start=True)

        # returning initial observation
        return np.array([self.map, self.previous_map])

    def update_map(self, start: bool) -> None:
        """
        Updates the observations

        start: if True, then the previous map is set differently
        """

        # update the previous map
        if start:
            self.previous_map = np.zeros(self.shape, dtype=np.float32)
        else:
            self.previous_map = np.copy(self.map)

        # clear the map
        self.map = np.zeros(self.shape, dtype=np.float32)

        # show the snake on the map
        for part in self.snake.snake_body:
            self.map[part[0], part[1]] = 1.

        # show the food on the map
        self.map[self.food.location[0], self.food.location[1]] = 1.

    def render(self, mode='human') -> None:
        if not self.done:
            print(self.map)
        else:
            print("The episode has ended")

    def close(self):
        pass
