
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.algorithms.short_term_memory import ShortTermMemory
from gym_snake.algorithms.memory import Memory
from gym_snake.algorithms.brain import Brain


class Agent:

    def __init__(self, env: SnakeEnv, long_term_memory_capacity: int = 10_000, short_term_memory_capacity: int = 2,
                 exploration_fraction: float = 0.1, gamma: float = 0.95,
                 batch_size: int = 64, number_of_epochs: int = 1,
                 alpha: float = 0.01, momentum: float = 0.0, nesterov: bool = False):

        # SnakeEnv
        self.env = env

        # Memory parameters
        self.long_term_memory_capacity = long_term_memory_capacity
        self.short_term_memory_capacity = short_term_memory_capacity

        # Reinforcement learning parameters
        self.exploration_fraction = exploration_fraction  # the probability of exploration
        self.gamma = gamma  # discount parameter

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # SGD parameters
        self.alpha = alpha
        self.momentum = momentum
        self.nesterov = nesterov

        # Create memory and Keras CNN model
        self.short_term_memory = ShortTermMemory(capacity=self.short_term_memory_capacity,
                                                 observation_shape=self.env.shape)
        self.long_term_memory = Memory(capacity=self.long_term_memory_capacity)
        self.brain = Brain(input_shape=self.short_term_memory.memory_output_shape,
                           number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size, number_of_epochs=self.number_of_epochs,
                           alpha=self.alpha, momentum=self.momentum, nesterov=self.nesterov)

    def act(self, greedy: bool = True) -> int:
        """
        The state is read from the short term memory
        """

        # random action
        if greedy and np.random.rand() < self.exploration_fraction:
            return self.env.action_space.sample()
        # Best action
        else:
            return np.argmax(self.brain.predict_one(self.short_term_memory.observations))

    def observe(self, observation: np.ndarray) -> np.ndarray:
        """
        observations: k dim
        return: (k + 1) dim
        """

        self.short_term_memory.update(observation)

        return self.short_term_memory.get()

    def memorize(self, sample: tuple) -> None:
        """
        One sample is stored as (state, action, reward, new_state)
        state has the short term memory output shape
        """

        self.long_term_memory.add(sample)

    def replay(self, replay_size: int, verbose: int = 0) -> None:

        # list of samples in format: (state, action, reward, new_state)
        # state and new_state has the short term memory output shape
        experiences = self.long_term_memory.sample(number_of_samples=replay_size)
        number_of_experiences = len(experiences)

        states = np.array([i[0] for i in experiences])

        new_states = np.array([i[3] for i in experiences])

        predictions_of_states = self.brain.predict(states)
        predictions_of_new_states = self.brain.predict(new_states)

        X = np.zeros((number_of_experiences, *self.short_term_memory.memory_output_shape))
        y = np.zeros((number_of_experiences, self.env.action_space.n))

        for i in range(number_of_experiences):
            state, action, reward, new_state = experiences[i]

            target = predictions_of_states[i]

            # new_state is terminal state
            # The observation is the zero array in case of termination
            # The last channel of the state is the last most recent observation
            # The states has channel last ordering
            if not np.any(new_state[..., -1]):
                target[action] = reward

            # new_state is non-terminal
            else:
                target[action] = reward + self.gamma * np.argmax(predictions_of_new_states[i])

            X[i] = state
            y[i] = target

        self.brain.train(X, y, verbose=verbose)

    def learn(self, number_of_episodes: int, replay_size: int, verbose: int = 0):

        for episode in range(number_of_episodes):
            print(episode)

            observation = self.env.reset()
            experience = self.observe(observation)

            while True:
                total_reward = 0.0

                action = self.act(greedy=True)

                observation, reward, done, info = self.env.step(action)
                new_experience = self.observe(observation)

                self.memorize((experience, action, reward, new_experience))

                self.replay(replay_size=replay_size, verbose=verbose)

                experience = new_experience
                total_reward += reward

                if done:
                    print(f"Total reward: {total_reward}")
                    break
