
import numpy as np
import gym
from gym_snake.envs.objects import Snake, Apple
from gym_snake.utilities.utils import array_in_collection, increase_resolution


class SnakeEnv(gym.Env):
    """
    An environment which implements the snake game. The snake can be controlled with the actions on the map and
    the goal is to get the apple. If the snake eats the apple it becomes larger by one. If the snake hits the wall
    or itself the game is over.

    The environment is based on the gym.Env object and can be used with the gym.make function.

    Observation:
        An enlarged 2 dimensional numpy array of the map

    Rewards:
        - Eating the apple: +1
        - The snake hits the wall or itself: -1
        - Otherwise: -0.01

    Actions:
        0 : turn left
        1 : go ahead
        2 : turn right
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, shape: tuple, initial_snake_length: int = 4, enlargement: int = 1):

        assert shape[0] == shape[1], "The map should be square shaped"
        assert initial_snake_length + 1 <= shape[0] * shape[1], "Snake is too long for this map"
        assert 2 <= initial_snake_length, "The initial snake length should be at least 2"
        assert isinstance(enlargement, int), "enlargement should be int"
        assert enlargement > 0, "enlargement should be positive"

        self.shape = shape  # shape of the map
        self.initial_snake_length = initial_snake_length
        self.enlargement = enlargement  # the enlarged map is the observation

        self.observation_shape = (self.enlargement * self.shape[0], self.enlargement * self.shape[1])
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=self.observation_shape, dtype=np.float32)
        self.reward_range = (-1., 1.)

        self.map = None            # 2d np.array
        self.snake = None          # Snake object
        self.apple = None          # Apple object
        self.done = True           # status of the episode (bool)

    def step(self, action: int) -> tuple:
        if self.done:
            raise EnvironmentError("Cant make step when the episode is done")

        # Get new snake head based on the action
        new_head = self.snake.get_new_head(action)

        # Valid step
        if self.snake.valid_part(new_head):
            tail = self.snake.snake_body.pop()
            self.snake.snake_body.appendleft(new_head)

            # The snake ate the apple
            if np.array_equal(self.apple.location, new_head):
                self.snake.snake_body.append(tail)  # restore tail
                self.snake.length += 1  # increase length
                reward = 1.

                # The game is won
                if self.snake.length == self.shape[0] * self.shape[1]:
                    self.end_episode()

                # The game continues
                else:
                    self.apple.create()
                    self.snake.update_direction()
                    self.update_map()

            # The snake did not eat the apple
            else:
                reward = -0.01
                self.snake.update_direction()
                self.update_map()

        # out of bound or new_head intersects with the other body parts
        else:
            reward = -1.
            self.end_episode()

        return increase_resolution(self.map, self.enlargement), reward, self.done, {}

    def end_episode(self) -> None:
        self.map = np.zeros(self.shape, dtype=np.float32)
        self.snake = None
        self.apple = None
        self.done = True

    def reset(self, spec_reset: bool = False, spec_snake_length: int = 4) -> np.ndarray:
        """
        Resets the environment and return the initial observation
        """

        # Reset the episode done parameter
        self.done = False

        # Create random snake
        if spec_reset:  # other initial snake length can be used
            self.snake = Snake(map_shape=self.shape, initial_length=spec_snake_length)
        else:
            self.snake = Snake(map_shape=self.shape, initial_length=self.initial_snake_length)

        # Add random apple
        self.apple = Apple(map_shape=self.shape, snake=self.snake)

        # Add snake and apple to the map
        self.update_map()

        # Return the initial observation
        return increase_resolution(self.map, self.enlargement)

    def update_map(self) -> None:
        """
        Updates the observations
        """

        # Clear the map
        self.map = np.zeros(self.shape, dtype=np.float32)

        # Show the snake
        for i, part in enumerate(self.snake.snake_body):
            # Show the head of the snake on the map
            if i == 0:
                self.map[part[0], part[1]] = 0.5

            # Show the other parts of the snake on the map
            else:
                self.map[part[0], part[1]] = 0.25

        # Show the apple on the map
        self.map[self.apple.location[0], self.apple.location[1]] = 1.

    def render(self, mode='human') -> None:
        if not self.done:
            print(increase_resolution(self.map, self.enlargement))
        else:
            print("The episode has ended")

    def close(self):
        pass
