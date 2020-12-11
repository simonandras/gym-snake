
import random
import numpy as np
from gym_snake.algorithms.brain import Brain


class Memory:
    """
    The previous experiences are stored and used for training
    """

    def __init__(self, capacity: int, brain: Brain, gamma: float):
        self.capacity = capacity
        self.brain = brain
        self.gamma = gamma

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

    def get_priority(self, experience):
        state, action, reward, new_state = experience

        # Q(state, action) prediction using the primary model
        q = self.brain.predict_one(np.array([state]), target=False)[action]

        # Best action choice given the new_state using the primary model
        a = np.argmax(self.brain.predict_one(np.array([new_state]), target=False))

        # Q(new_state, a) prediction using the target model
        q_ = self.brain.predict_one(np.array([new_state]), target=True)[a]

        # Target value
        t = reward + self.gamma * q_

        error = np.abs(q - t)

        return error + 0.1

    def update_priorities(self):
        self.priorities = []

        for experience in self.experiences:
            self.priorities.append(self.get_priorty(experience))

        sum_priorities = sum(self.priorities)

        for i in range(len(self.priorities)):
            self.priorities[i] /= sum_priorities


