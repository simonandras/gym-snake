
import random
import numpy as np
from gym_snake.algorithms.brain import Brain


class Memory:
    """
    The previous experiences an their priorities are stored and used for training
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
        self.priorities.append(self.get_priority(experience))

        if len(self.experiences) != len(self.priorities):
            raise ValueError

        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)
            self.priorities.pop(0)

    def sample(self, number_of_samples: int, using_priorities=False) -> list:
        """
        Sampling without replacement
        """

        if using_priorities:
            temp_priorities = self.priorities.copy()
            n = len(temp_priorities)
            if n != len(self.experiences):
                raise ValueError

            sum_priorities = sum(self.priorities)
            for i in range(n):
                temp_priorities[i] /= sum_priorities

            if n < number_of_samples:
                replace = True
            else:
                replace = False
            indexes = np.random.choice(n, number_of_samples, p=temp_priorities, replace=replace)
            return [self.experiences[i] for i in indexes]
        else:
            return random.sample(self.experiences, min(number_of_samples, len(self.experiences)))

    def get_priority(self, experience) -> float:
        state, action, reward, new_state = experience

        # Q(state, action) prediction using the primary model
        q = self.brain.predict_one(np.array([state]), target=False)[action]

        # Best action choice given the new_state using the primary model
        a = np.argmax(self.brain.predict_one(np.array([new_state]), target=False))

        # Q(new_state, a) prediction using the target model
        q_ = self.brain.predict_one(np.array([new_state]), target=True)[a]

        # Target value
        if not np.any(new_state):  # new_state is terminal state
            t = reward
        else:
            t = reward + self.gamma * q_

        error = np.abs(q - t)

        return error + 0.1

    def update_priorities(self, show_progress=True) -> None:
        if show_progress:
            print("Updating priorities")
            print(f"Number of experiences: {len(self.experiences)}")

        self.priorities = []

        for i, experience in enumerate(self.experiences):
            if show_progress and i % 100 == 0:
                print(i)

            self.priorities.append(self.get_priority(experience))
