
import random


class Memory:

    def __init__(self, capacity: int):
        self.capacity = capacity

        self.samples = []

    def add(self, sample: tuple) -> None:
        """
        One sample is stored as (state, action, reward, new_state)
        """

        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, number_of_samples: int) -> list:
        """
        Sampling without replacement
        """
        
        return random.sample(self.samples, min(number_of_samples, self.capacity))
