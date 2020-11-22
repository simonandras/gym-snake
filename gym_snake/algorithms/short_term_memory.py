
import numpy as np


class ShortTermMemory:

    def __init__(self, capacity: int, latent_vector_length: int):

        self.capacity = capacity
        self.latent_vector_length = latent_vector_length  # length of the encoded observation
        self.memory_length = self.capacity * self.latent_vector_length

        # initialize the memory with zeros
        self.state = np.zeros(self.memory_length)

    def update(self, encoded_observation: np.ndarray) -> None:
        """
        encoded_observation: 1 dim, the env observation encoded
        """

        new_state = np.zeros(self.memory_length)
        new_state[:(-1)*self.latent_vector_length] = self.state[self.latent_vector_length:]
        new_state[(-1)*self.latent_vector_length:] = encoded_observation

        self.state = new_state

    def reset(self) -> None:
        self.state = np.zeros(self.memory_length)
