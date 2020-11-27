
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.algorithms.short_term_memory import ShortTermMemory
from gym_snake.algorithms.memory import Memory
from gym_snake.algorithms.brain import Brain


class Agent:
    """
    Observation: 2d array; get from env
    State: 3d array; get from short term memory
    Experience: (state, action, reward, new_state); get form long term memory
    """

    def __init__(self, env: SnakeEnv, long_term_memory_capacity: int = 1_000_000, short_term_memory_capacity: int = 2,
                 exploration_fraction: float = 0.15, gamma: float = 0.99,
                 batch_size: int = 32, number_of_epochs: int = 1,
                 lr: float = 0.00025, rho: float = 0.95, epsilon: float = 0.01):

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

        # RMSprop parameters
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Create memory and Keras CNN model
        self.short_term_memory = ShortTermMemory(capacity=self.short_term_memory_capacity,
                                                 observation_shape=self.env.observation_shape)
        self.long_term_memory = Memory(capacity=self.long_term_memory_capacity)
        self.brain = Brain(input_shape=self.short_term_memory.memory_shape,
                           number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size, number_of_epochs=self.number_of_epochs,
                           lr=self.lr, rho=self.rho, epsilon=self.epsilon)

        # Storing learning history
        self.length_history = []
        self.reward_history = []

    def act(self, greedy: bool = True) -> int:
        """
        The state is read from the short term memory
        """

        # random action
        if greedy and np.random.rand() < self.exploration_fraction:
            return self.env.action_space.sample()
        # Best action
        else:
            return np.argmax(self.brain.predict_one(self.short_term_memory.state))

    def observe(self, observation: np.ndarray) -> np.ndarray:
        """
        observation: k dim, get from the env
        return: (k + 1) dim, get from the short term memory
        """

        self.short_term_memory.update(observation)

        return self.short_term_memory.state

    def memorize(self, experience: tuple) -> None:
        """
        One experience is stored as (state, action, reward, new_state)
        state has the short term memory output shape
        """

        self.long_term_memory.add(experience)

    def replay(self, replay_size: int, verbose: int = 0) -> None:

        # One experience is stored as (state, action, reward, new_state)
        # state and new_state has the short term memory output shape
        experiences = self.long_term_memory.sample(number_of_samples=replay_size)
        number_of_experiences = len(experiences)

        states = np.array([i[0] for i in experiences])

        new_states = np.array([i[3] for i in experiences])

        predictions_of_states = self.brain.predict(states)
        predictions_of_new_states = self.brain.predict(new_states)

        X = np.zeros((number_of_experiences, *self.short_term_memory.memory_shape))
        y = np.zeros((number_of_experiences, self.env.action_space.n))

        for i in range(number_of_experiences):
            state, action, reward, new_state = experiences[i]

            target = predictions_of_states[i]

            # new_state is terminal state
            # The observation is the zero array in case of termination
            # The last channel of the state is the last most recent observation
            # The state has channel first ordering
            if not np.any(new_state[-1]):
                target[action] = reward

            # new_state is non-terminal
            else:
                target[action] = reward + self.gamma * np.max(predictions_of_new_states[i])

            X[i] = state
            y[i] = target

        self.brain.train(X, y, verbose=verbose)

    def learn(self, number_of_episodes: int, replay_size: int, verbose: int = 0):

        for episode in range(number_of_episodes):
            print(episode)

            episode_length = 1
            total_reward = 0.

            observation = self.env.reset()
            state = self.observe(observation)

            while True:
                action = self.act(greedy=True)

                observation, reward, done, info = self.env.step(action)
                new_state = self.observe(observation)

                # store experience
                self.memorize((state, action, reward, new_state))

                self.replay(replay_size=replay_size, verbose=verbose)

                state = new_state

                episode_length += 1
                total_reward += reward

                if done:
                    self.short_term_memory.reset()

                    self.length_history.append(episode_length)
                    self.reward_history.append(total_reward)
                    print(f"Episode length: {episode_length}")
                    print(f"Total reward: {total_reward}")
                    break
