
import random
from gym_snake.algorithms.brain import Brain


class Memory:
    """
    The previous experiences are stored and used for training
    """

    def __init__(self, capacity: int, brain: Brain):
        self.capacity = capacity
        self.brain = brain

        self.experiences = []
        self.priorities = []

    def add(self, experience: tuple) -> None:
        """
        One experience is stored as (state, action, reward, new_state)
        """

        self.experiences.append(experience)

        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

    def sample(self, number_of_samples: int) -> list:
        """
        Sampling without replacement
        """
        
        return random.sample(self.experiences, min(number_of_samples, len(self.experiences)))
