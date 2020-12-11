
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.algorithms.memory import Memory
from gym_snake.algorithms.brain import Brain
from gym_snake.utilities.utils import rotate_times, mirror


class Agent:

    def __init__(self, env: SnakeEnv, memory_capacity: int = 1_000_000,
                 min_exp_ratio: float = 0., max_exp_ratio: float = 1., decay: float = 0.001, gamma: float = 0.99,
                 batch_size: int = 32, number_of_epochs: int = 1, lr: float = 0.00025):

        # SnakeEnv
        self.env = env

        # Memory parameters
        self.memory_capacity = memory_capacity

        # Reinforcement learning parameters
        self.gamma = gamma  # discount parameter
        self.min_exp_ratio = min_exp_ratio
        self.max_exp_ratio = max_exp_ratio
        self.decay = decay
        self.exploration_ratio = self.max_exp_ratio  # the probability of exploration

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop learning rate
        self.lr = lr

        # Create memory and Keras CNN model
        self.brain = Brain(input_shape=(1, *self.env.observation_shape),
                           number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size,
                           number_of_epochs=self.number_of_epochs,
                           lr=self.lr)
        self.memory = Memory(capacity=self.memory_capacity, brain=self.brain)

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
            return np.argmax(self.brain.predict_one(np.array([state]), target=False))

    def memorize(self, experience: tuple) -> None:
        """
        Store the experience and its transformations
        One experience is stored as (state, action, reward, new_state)
        """

        self.memory.add(experience)

        state, action, reward, new_state = experience

        # In case of rotation, the actions do not changes
        self.memory.add((rotate_times(state, 1), action, reward, rotate_times(new_state, 1)))
        self.memory.add((rotate_times(state, 2), action, reward, rotate_times(new_state, 2)))
        self.memory.add((rotate_times(state, 3), action, reward, rotate_times(new_state, 3)))

        # In case of mirroring the right and left actions are interchanged
        self.memory.add((mirror(state, axis=0), action, 2 - reward, mirror(new_state, axis=0)))
        self.memory.add((mirror(state, axis=1), action, 2 - reward, mirror(new_state, axis=1)))

    def update_exploration_ratio(self):
        self.exploration_ratio = self.min_exp_ratio + \
                                 (self.max_exp_ratio - self.min_exp_ratio) * np.exp((-1)*self.decay * self.steps)

    def replay(self, replay_size: int, verbose: int = 0) -> None:

        # One experience is stored as (state, action, reward, new_state)
        experiences = self.memory.sample(number_of_samples=replay_size)
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

    def learn(self, number_of_episodes: int, replay_size: int, synchronization_episode_number: int = 100,
              verbose: int = 0):
        for episode in range(number_of_episodes):
            if episode % synchronization_episode_number == 0:
                self.brain.synchronization()
                print("Target model weights synchronized")

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

    def play(self, number_of_episodes: int, episode_max_length: int = 100, random=False, memorize=False,
             show_results=True):

        for episode in range(number_of_episodes):
            if show_results:
                print(episode)

            episode_length = 1
            total_reward = 0.

            state = self.env.reset()

            while True:
                if random:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state=state, greedy=False)

                new_state, reward, done, info = self.env.step(action)

                # store experience
                if memorize:
                    self.memorize((state, action, reward, new_state))

                state = new_state

                episode_length += 1

                total_reward += reward

                if done or episode_max_length < episode_length:
                    if memorize:
                        self.length_history.append(episode_length)
                        self.reward_history.append(total_reward)
                    if show_results:
                        print(f"Episode length: {episode_length}")
                        print(f"Total reward: {total_reward}")
                    break
