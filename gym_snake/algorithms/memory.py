
import random
import numpy as np
from gym_snake.algorithms.brain import Brain


class Memory:
    """
    The previous experiences and their priorities are stored and used for training.
    If using_priority is True, then prioritized experience replay is used. This functionality is in an experimental
    state for now. It runs too slow and should not be used.
    """

    def __init__(self, capacity: int, using_priority: bool, brain: Brain, gamma: float):

        # Main memory parameter
        self.capacity = capacity

        # Used in case of prioritized experience replay
        self.using_priority = using_priority
        self.brain = brain
        self.gamma = gamma

        # List of the experiences
        self.experiences = []

        # List of the priorities
        if using_priority:
            self.priorities = []

    def add(self, experience: tuple) -> None:
        """
        Stores one experience.
        One experience has the form (state, action, reward, new_state).
        """

        # Append the experience
        self.experiences.append(experience)

        if self.using_priority:
            # Append the priority
            self.priorities.append(self.get_priority(experience))

            # Check the lengths of the lists
            if len(self.experiences) != len(self.priorities):
                raise ValueError

        # If the memory list is full, then the oldest memory is deleted
        if len(self.experiences) > self.capacity:
            self.experiences.pop(0)

            if self.using_priority:
                self.priorities.pop(0)

    def sample(self, number_of_samples: int) -> list:
        """
        Get samples from the memory.
        In case of prioritized experience replay the choosing distribution is based on the priorities.
        In the other case the sampling is done uniformly without replacement.
        """

        if self.using_priority:
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
        """
        Can be used only in case of prioritized experience replay.
        Computes the priority value of an experience.
        """

        if not self.using_priority:
            raise ValueError

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
        """
        Can be used only in case of prioritized experience replay.
        Updates the priorities list.
        """

        if not self.using_priority:
            raise ValueError

        if show_progress:
            print("Updating priorities")
            print(f"Number of experiences: {len(self.experiences)}")

        # Initialize the priorities with an empty list
        self.priorities = []

        for i, experience in enumerate(self.experiences):
            if show_progress and i % 100 == 0:
                print(i)

            self.priorities.append(self.get_priority(experience))
