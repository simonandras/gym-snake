
import numpy as np


class ShortTermMemory:
    """
    Stores the last observations in a (len(observation_shape) + 1) dimensional np.ndarray
    """

    def __init__(self, capacity: int, observation_shape: tuple):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.memory_shape = (self.capacity, *self.observation_shape)

        # initialize the memory with zeros
        self.observations = np.zeros(self.memory_shape)

    def update(self, observation: np.ndarray) -> None:

        new_observations = np.zeros(self.memory_shape)
        new_observations[:-1] = self.observations[1:]
        new_observations[-1] = observation

        self.observations = new_observations

    def get(self):
        """
        Returns the memory content with channel last ordering
        """

        return np.moveaxis(self.observations, -1, 0)
