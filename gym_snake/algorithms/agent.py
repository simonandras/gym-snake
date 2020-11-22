
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

    def __init__(self, env: SnakeEnv, long_term_memory_capacity: int = 100_000, short_term_memory_capacity: int = 2,
                 latent_vector_length: int = 64,
                 min_exp_ratio: float = 0.01, max_exp_ratio: float = 1.0, decay: float = 0.001, gamma: float = 0.95,
                 batch_size: int = 32, number_of_epochs: int = 1,
                 lr: float = 0.0001, rho: float = 0.95, epsilon: float = 0.01):

        # SnakeEnv
        self.env = env

        # Memory parameters
        self.long_term_memory_capacity = long_term_memory_capacity
        self.short_term_memory_capacity = short_term_memory_capacity

        # Encoding parameter
        self.latent_vector_length = latent_vector_length

        # Reinforcement learning parameters
        self.min_exp_ratio = min_exp_ratio
        self.max_exp_ratio = max_exp_ratio
        self.exploration_ratio = self.max_exp_ratio  # the probability of exploration
        self.decay = decay
        self.gamma = gamma  # discount parameter

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop parameters
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Create memory and Keras models
        self.short_term_memory = ShortTermMemory(capacity=self.short_term_memory_capacity,
                                                 latent_vector_length=self.latent_vector_length)
        self.long_term_memory = Memory(capacity=self.long_term_memory_capacity)
        self.brain = Brain(input_length=self.short_term_memory.memory_length,
                           number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size, number_of_epochs=self.number_of_epochs,
                           lr=self.lr, rho=self.rho, epsilon=self.epsilon)

        # Count the number of steps
        self.steps = 0

        # Storing learning history
        self.length_history = []
        self.reward_history = []

    def act(self, greedy: bool = True) -> int:
        """
        The state is read from the short term memory
        """

        # random action
        if greedy and np.random.rand() < self.exploration_ratio:
            return self.env.action_space.sample()

        # Best action
        else:
            return np.argmax(self.brain.predict_one(self.short_term_memory.state))

    def observe(self, encoded_observation: np.ndarray) -> np.ndarray:
        """
        encoded_observation: 1 dim, the env observation encoded
        return: 1 dim, get from the short term memory
        """

        self.short_term_memory.update(encoded_observation)
        self.steps += 1
        self.update_exploration_ratio()

        return self.short_term_memory.state

    def update_exploration_ratio(self):
        self.exploration_ratio = self.min_exp_ratio + \
                                 (self.max_exp_ratio - self.min_exp_ratio) * math.exp((-1)*self.decay * self.steps)

    def memorize(self, experience: tuple) -> None:
        """
        One experience is stored as (state, action, reward, new_state)
        state has short term memory memory_length
        """

        self.long_term_memory.add(experience)

    def replay(self, replay_size: int, verbose: int = 0) -> None:

        # One experience is stored as (state, action, reward, new_state)
        # state has short term memory memory_length
        # new_state is None in case of terminal state
        experiences = self.long_term_memory.sample(number_of_samples=replay_size)
        number_of_experiences = len(experiences)

        no_state = np.zeros(self.short_term_memory.memory_length)

        states = np.array([i[0] for i in experiences], dtype=np.float32)
        new_states = np.array([no_state if i[3] is None else i[3] for i in experiences], dtype=np.float32)

        predictions_of_states = self.brain.predict(states)
        predictions_of_new_states = self.brain.predict(new_states)

        X = np.zeros((number_of_experiences, self.short_term_memory.memory_length))
        y = np.zeros((number_of_experiences, self.env.action_space.n))

        for i in range(number_of_experiences):
            state, action, reward, new_state = experiences[i]

            target = predictions_of_states[i]

            # new_state is terminal state
            # The observation is the zero array in case of termination
            # The last channel of the state is the last most recent observation
            # The state has channel first ordering
            if new_state is None:
                target[action] = reward

            # new_state is non-terminal
            else:
                target[action] = reward + self.gamma * np.argmax(predictions_of_new_states[i])

            X[i] = state
            y[i] = target

        self.brain.train(X, y, verbose=verbose)

    def learn(self, number_of_episodes: int, replay_size: int, verbose: int = 0) -> None:

        for episode in range(number_of_episodes):
            print(episode)

            episode_length = 1
            total_reward = 0.

            observation = self.brain.encode_one(self.env.reset())
            state = self.observe(observation)

            while True:
                action = self.act(greedy=True)

                raw_observation, reward, done, info = self.env.step(action)
                observation = self.brain.encode_one(raw_observation)
                new_state = self.observe(observation)

                if done:
                    new_state = None

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
