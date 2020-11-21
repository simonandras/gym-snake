
import numpy as np


class ShortTermMemory:
    """
    Stores the last observations in a 3 dimensional channel first np.ndarray
    """

    def __init__(self, capacity: int, observation_shape: tuple):

        self.capacity = capacity
        self.observation_shape = observation_shape
        self.memory_shape = (self.capacity, *self.observation_shape)  # channel first

        # initialize the memory with zeros
        self.state = np.zeros(self.memory_shape)

    def update(self, observation: np.ndarray) -> None:
        """
        observation: get from the env
        """

        new_state = np.zeros(self.memory_shape)
        new_state[:-1] = self.state[1:]
        new_state[-1] = observation

        self.state = new_state

    def reset(self) -> None:
        self.state = np.zeros(self.memory_shape)
