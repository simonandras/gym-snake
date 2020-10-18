
import numpy as np
import gym
from gym_snake.objects.snake import Snake
from gym_snake.utilities.utils import array_in_collection


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    reward_range = (-1., 1.)

    action_space = gym.spaces.Discrete(3)

    map = None            # 2d np.array
    previous_map = None
    snake = None          # Snake object
    food_location = None  # np.array([a, b])
    done = True           # status of the episode

    def __init__(self, shape: tuple, initial_snake_length: int = 4):
        self.shape = shape
        self.initial_snake_length = initial_snake_length
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2, shape[0], shape[1]), dtype=np.float32)

    def step(self, action: int) -> tuple:
        """
        actions:
            0 : turn left
            1 : go ahead
            2 : turn right
        """

        if self.done:
            raise EnvironmentError("Cant make step when the episode is done")

        new_head = self.get_new_head(action)
        reward = None

        # remove the last part of the tail before check new_part validation
        tail = self.snake.snake_body.pop()

        if self.snake.valid_part(new_head):
            self.snake.snake_body.appendleft(new_head)
            if np.array_equal(self.food_location, new_head):
                self.snake.snake_body.append(tail)  # restore tail
                self.snake.length += 1  # increase length
                self.create_food()
                reward = 1.
            else:
                reward = 0.
            self.snake.update_direction()
            self.update_map(start=False)
            observation = np.array([self.map, self.previous_map])
        # out of bound or new_head intersects with the other body parts
        else:
            reward = -1.
            self.end_episode()
            observation = None

        return observation, reward, self.done, {}

    def get_new_head(self, action: int) -> np.ndarray:
        """
        actions:
            0 : turn left
            1 : go ahead
            2 : turn right
        """

        head = self.snake.snake_body[0]

        if self.snake.direction == 'left':
            if action == 0:
                return np.array([head[0] + 1, head[1]])
            elif action == 1:
                return np.array([head[0], head[1] - 1])
            elif action == 2:
                return np.array([head[0] - 1, head[1]])
        elif self.snake.direction == 'up':
            if action == 0:
                return np.array([head[0], head[1] - 1])
            elif action == 1:
                return np.array([head[0] - 1, head[1]])
            elif action == 2:
                return np.array([head[0], head[1] + 1])
        elif self.snake.direction == 'right':
            if action == 0:
                return np.array([head[0] - 1, head[1]])
            elif action == 1:
                return np.array([head[0], head[1] + 1])
            elif action == 2:
                return np.array([head[0] + 1, head[1]])
        elif self.snake.direction == 'down':
            if action == 0:
                return np.array([head[0], head[1] + 1])
            elif action == 1:
                return np.array([head[0] + 1, head[1]])
            elif action == 2:
                return np.array([head[0], head[1] - 1])

    def end_episode(self) -> None:
        self.map = None
        self.previous_map = None
        self.snake = None
        self.food_location = None
        self.done = True

    def reset(self) -> np.ndarray:
        # reset the episode done parameter
        self.done = False

        # creating random snake
        self.snake = Snake(map_shape=self.shape, initial_length=self.initial_snake_length)

        # creating random food
        self.create_food()

        # adding snake and food to the map
        self.update_map(start=True)

        # returning initial observation
        return np.array([self.map, self.previous_map])

    def create_food(self) -> None:
        while True:
            new_food_location = np.array([np.random.randint(self.shape[0]),
                                          np.random.randint(self.shape[1])])
            if not array_in_collection(self.snake.snake_body, new_food_location):
                self.food_location = new_food_location
                break

    def update_map(self, start: bool) -> None:
        """
        Updates the observations (map, previous_map)

        start: if True, then the previous map is set differently
        """

        # save the map to the previous map
        if start:
            self.previous_map = np.zeros(self.shape, dtype=np.float32)
        else:
            self.previous_map = np.copy(self.map)

        # clear the map
        self.map = np.zeros(self.shape, dtype=np.float32)

        # show the snake on the map
        for part in self.snake.snake_body:
            self.map[part[0], part[1]] = 0.5

        # show the food on the map
        self.map[self.food_location[0], self.food_location[1]] = 1.0

    def render(self, mode='human'):
        if not self.done:
            print(self.map)
        else:
            print("The episode has ended")

    def close(self):
        pass


env = SnakeEnv(shape=(5, 5))
env.reset()

reward = "start"

for i in range(10):
    print(i)
    env.render()
    print("...")
    print(env.previous_map)
    if not env.done:
        a = env.action_space.sample()
        print(f"action: {a}")
        print(f"reward: {reward}")
        print("----------------")
        observation, reward, done, info = env.step(a)
        print(observation)
    else:
        print(f"reward: {reward}")
        print("----------------")
        break
