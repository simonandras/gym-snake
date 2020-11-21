
import numpy as np
from collections import deque
from gym_snake.envs.snake_functions import beside_each_other, around, where_is_compared
from gym_snake.utilities.utils import xor, array_in_collection


class Snake:

    def __init__(self, map_shape: tuple, initial_length: int):

        assert map_shape[0] >= 5 and map_shape[1] >= 5, "The map size should be at least 5x5"

        self.map_shape = map_shape
        self.initial_length = initial_length

        self.length = None
        self.snake_body = None
        self.direction = None

        self.reset()

    def reset(self) -> None:
        """
        Creates a new random snake with the initial_length
        """

        # reset the length
        self.length = self.initial_length

        # clear the snake
        self.snake_body = deque()

        # first add the head inside of the border
        self.snake_body.append(np.array([np.random.randint(2, self.map_shape[0] - 2),
                                         np.random.randint(2, self.map_shape[1] - 2)]))

        # adds some valid part connected to the tail
        for _ in range(self.initial_length - 1):
            self.add_random_part_to_tail()

        # reset the direction
        self.update_direction()

    def update_direction(self) -> None:
        self.direction = where_is_compared(self.snake_body[1], self.snake_body[0])

    def add_random_part_to_tail(self) -> None:
        """
        Appends a new part to the snake's tail randomly
        This method is used when creating a new initial snake
        """

        available_positions = []
        parts_around = around(self.snake_body[-1])

        for i, part in enumerate(parts_around):
            if self.valid_part(part, to_tail=True):
                available_positions.append(i)

        if available_positions:
            position = available_positions[np.random.randint(len(available_positions))]
            self.snake_body.append(parts_around[position])

    def valid_part(self, part: np.ndarray, to_tail=False) -> bool:
        """
        Checks whether part can be appended or not
        Part can be appended if:
            - it is not out of bound
            - it isn't intersects with the other body parts
            - it connects to the tail or to_tail is False

        part: np.array([a, b)]
        """

        # out of bound
        if part[0] < 0 or part[0] >= self.map_shape[0] or part[1] < 0 or part[1] >= self.map_shape[1]:
            return False
        # part intersect with the body
        elif array_in_collection(self.snake_body, part):
            return False
        # part not connect to the tail
        elif to_tail and not beside_each_other(self.snake_body[-1], part):
            return False
        else:
            return True

    def get_new_head(self, action: int) -> np.ndarray:
        """
        Actions:
            0 : turn left
            1 : go ahead
            2 : turn right
        """

        head = self.snake_body[0]

        if self.direction == 'left':
            if action == 0:
                return np.array([head[0] + 1, head[1]])
            elif action == 1:
                return np.array([head[0], head[1] - 1])
            elif action == 2:
                return np.array([head[0] - 1, head[1]])
        elif self.direction == 'up':
            if action == 0:
                return np.array([head[0], head[1] - 1])
            elif action == 1:
                return np.array([head[0] - 1, head[1]])
            elif action == 2:
                return np.array([head[0], head[1] + 1])
        elif self.direction == 'right':
            if action == 0:
                return np.array([head[0] - 1, head[1]])
            elif action == 1:
                return np.array([head[0], head[1] + 1])
            elif action == 2:
                return np.array([head[0] + 1, head[1]])
        elif self.direction == 'down':
            if action == 0:
                return np.array([head[0], head[1] + 1])
            elif action == 1:
                return np.array([head[0] + 1, head[1]])
            elif action == 2:
                return np.array([head[0], head[1] - 1])


class Apple:

    def __init__(self, map_shape: tuple, snake: Snake):
        self.map_shape = map_shape
        self.snake = snake
        self.location = None

        self.create()

    def create(self):
        while True:
            new_location = np.array([np.random.randint(self.map_shape[0]),
                                     np.random.randint(self.map_shape[1])])
            if not array_in_collection(self.snake.snake_body, new_location):
                self.location = new_location
                break
