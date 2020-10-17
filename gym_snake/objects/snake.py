
import numpy as np
from collections import deque
from gym_snake.utilities.utils import array_in_collection, beside_each_other, around


class Snake:

    def __init__(self, map_shape, initial_length):
        self.map_shape = map_shape
        self.initial_length = initial_length

        self.length = self.initial_length
        self.snake_body = deque()

        self.reset()

    def reset(self) -> None:
        self.snake_body = deque()

        # first add the head
        self.snake_body.append(np.array([np.random.randint(self.map_shape[0]),
                                         np.random.randint(self.map_shape[1])]))

        # adds some valid part connected to the tail
        for _ in range(self.initial_length):
            self.random_add_part()

    def random_add_part(self) -> None:
        """
        Appends a new part to the snake's tail randomly
        """

        available_positions = []
        parts_around = around(self.snake_body[-1])

        for i, part in enumerate(parts_around):
            if self.valid_part(part):
                available_positions.append(i)

        if not available_positions:
            raise ValueError("Can't append new part to the snake's tail because all positions are invalid.")

        position = available_positions[np.random.randint(len(available_positions))]

        self.snake_body.append(parts_around[position])

    def valid_part(self, part) -> bool:
        """
        Checks whether part can be appended or not
        Part can be appended if:
            - it is not out of bound
            - it isn't intersects with the other body parts
            - it connects to the tail

        part: np.array([a, b)]
        """

        # out of bound
        if part[0] < 0 or part[0] >= self.map_shape[0] or part[1] < 0 or part[1] >= self.map_shape[1]:
            return False
        # part intersect with the body
        elif array_in_collection(self.snake_body, part):
            return False
        # part not connect to the tail
        elif not beside_each_other(self.snake_body[-1], part):
            return False
        else:
            return True
