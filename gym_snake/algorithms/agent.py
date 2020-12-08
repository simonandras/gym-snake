
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

    def __init__(self, env: SnakeEnv, long_term_memory_capacity: int = 1_000_000,
                 min_exp_ratio: float = 0., max_exp_ratio: float = 1., decay: float = 0.001, gamma: float = 0.99,
                 batch_size: int = 32, number_of_epochs: int = 1,
                 lr: float = 0.00025, rho: float = 0.95, epsilon: float = 0.01):

        # SnakeEnv
        self.env = env

        # Memory parameters
        self.long_term_memory_capacity = long_term_memory_capacity

        # Reinforcement learning parameters
        self.gamma = gamma  # discount parameter
        self.min_exp_ratio = min_exp_ratio
        self.max_exp_ratio = max_exp_ratio
        self.decay = decay
        self.exploration_ratio = self.max_exp_ratio  # the probability of exploration

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop parameters
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Create memory and Keras CNN model
        self.long_term_memory = Memory(capacity=self.long_term_memory_capacity)
        self.brain = Brain(input_shape=(1, *self.env.observation_shape),
                           number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size,
                           number_of_epochs=self.number_of_epochs,
                           lr=self.lr,
                           rho=self.rho,
                           epsilon=self.epsilon)

        # Count the number of steps
        self.steps = 0

        # Storing learning history
        self.length_history = []
        self.reward_history = []

    def act(self, state: np.ndarray, greedy: bool = True) -> int:
        """
        The state is a 2d np array
        """

        # random action
        if greedy and np.random.rand() < self.exploration_ratio:
            return self.env.action_space.sample()

        # Best action
        else:
            return np.argmax(self.brain.predict_one(np.array([state])))

    def memorize(self, experience: tuple) -> None:
        """
        One experience is stored as (state, action, reward, new_state)
        """

        self.long_term_memory.add(experience)

    def update_exploration_ratio(self):
        self.exploration_ratio = self.min_exp_ratio + \
                                 (self.max_exp_ratio - self.min_exp_ratio) * np.exp((-1)*self.decay * self.steps)

    def replay(self, replay_size: int, verbose: int = 0) -> None:

        # One experience is stored as (state, action, reward, new_state)
        experiences = self.long_term_memory.sample(number_of_samples=replay_size)
        number_of_experiences = len(experiences)

        states = np.array([[i[0]] for i in experiences])

        new_states = np.array([[i[3]] for i in experiences])

        predictions_of_states = self.brain.predict(states)
        predictions_of_new_states = self.brain.predict(new_states)

        X = np.zeros((number_of_experiences, *self.brain.input_shape))
        y = np.zeros((number_of_experiences, self.env.action_space.n))

        for i in range(number_of_experiences):
            state, action, reward, new_state = experiences[i]

            target = predictions_of_states[i]

            # new_state is terminal state
            # The observation is the zero array in case of termination
            if not np.any(new_state[0]):
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

            state = self.env.reset()

            while True:
                self.steps += 1

                self.update_exploration_ratio()

                action = self.act(state=state, greedy=True)

                new_state, reward, done, info = self.env.step(action)

                # store experience
                self.memorize((state, action, reward, new_state))

                self.replay(replay_size=replay_size, verbose=verbose)

                state = new_state

                episode_length += 1
                total_reward += reward

                if done:
                    self.length_history.append(episode_length)
                    self.reward_history.append(total_reward)
                    print(f"Exploration ratio at the end of the episode: {self.exploration_ratio}")
                    print(f"Episode length: {episode_length}")
                    print(f"Total reward: {total_reward}")
                    break
