
import numpy as np
from collections import deque
from gym_snake.utilities.utils import xor, array_in_collection, beside_each_other, around, where_is_compared


class Snake:
    length = None
    snake_body = None
    direction = None

    def __init__(self, map_shape: tuple, initial_length: int):
        self.map_shape = map_shape
        self.initial_length = initial_length

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
        self.snake_body.append(np.array([np.random.randint(1, self.map_shape[0] - 1),
                                         np.random.randint(1, self.map_shape[1] - 1)]))

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

        if not available_positions:
            raise ValueError("Can't append new part to the snake's tail because all positions are invalid.")

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
