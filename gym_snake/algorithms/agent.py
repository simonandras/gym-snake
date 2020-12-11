
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv
from gym_snake.algorithms.memory import Memory
from gym_snake.algorithms.brain import Brain
from gym_snake.utilities.utils import rotate_times, mirror


class Agent:
    """
    A DDQN (Double Deep Q Network) based agent for the SnakeEnv environment.
    The agent has 3 important objects:
        - Environment: SnakeEnv
        - Brain: double convolutional neural network
        - Memory: for replaying experiences
    """

    def __init__(self, env: SnakeEnv, memory_capacity: int = 1_000_000, using_priority: bool = False,
                 min_exp_ratio: float = 0., max_exp_ratio: float = 1., decay: float = 0.001, gamma: float = 0.99,
                 batch_size: int = 32, number_of_epochs: int = 1, lr: float = 0.00025):

        # SnakeEnv
        self.env = env

        # Memory parameters
        self.memory_capacity = memory_capacity
        self.using_priority = using_priority  # if True, then prioritized experience replay is used

        # Reinforcement learning parameters
        self.gamma = gamma  # discount parameter
        self.min_exp_ratio = min_exp_ratio
        self.max_exp_ratio = max_exp_ratio
        self.decay = decay  # used in update_exploration_ratio function
        self.exploration_ratio = self.max_exp_ratio  # the probability of exploration

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop learning rate
        self.lr = lr

        # Create Keras CNN models and memory
        self.brain = Brain(input_shape=(1, *self.env.observation_shape), number_of_actions=self.env.action_space.n,
                           batch_size=self.batch_size, number_of_epochs=self.number_of_epochs, lr=self.lr)
        self.memory = Memory(capacity=self.memory_capacity, using_priority=self.using_priority, brain=self.brain,
                             gamma=self.gamma)

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

        # The actual experience
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
        """
        The exploration probability decreases with every step
        """

        self.exploration_ratio = self.min_exp_ratio + \
                                 (self.max_exp_ratio - self.min_exp_ratio) * np.exp((-1) * self.decay * self.steps)

    def replay(self, replay_size: int, verbose: int = 0) -> None:
        """
        Experiences are sampled from the memory and used for training the model
        """

        # Load the experiences
        # One experience is stored as (state, action, reward, new_state)
        experiences = self.memory.sample(number_of_samples=replay_size)  # list
        number_of_experiences = len(experiences)

        # Get the states and new states from the experiences
        states = np.array([[i[0]] for i in experiences])
        new_states = np.array([[i[3]] for i in experiences])

        # Using the target network get the action-value function predictions for every action
        # Has (number_of_experiences * number_of_actions) shape
        predictions_of_states = self.brain.predict(states, target=True)
        predictions_of_new_states = self.brain.predict(new_states, target=True)

        # Initialize the training data
        X = np.zeros((number_of_experiences, *self.brain.input_shape))
        print(X)
        print(X.shape)
        y = np.zeros((number_of_experiences, self.env.action_space.n))

        for i in range(number_of_experiences):
            state, action, reward, new_state = experiences[i]

            target = predictions_of_states[i]

            # new_state is terminal state
            # The observation is the zero array in case of termination
            if not np.any(new_state):
                target[action] = reward

            # new_state is non-terminal
            else:
                target[action] = reward + self.gamma * np.max(predictions_of_new_states[i])

            X[i] = state
            y[i] = target

        print(X)
        print(X.shape)

        self.brain.train(X, y, verbose=verbose)

    def learn(self, number_of_episodes: int, replay_size: int, synchronization_episode_number: int = 100,
              priority_update_episode_number: int = 100, saving_episode_number: int = 100, verbose: int = 0):
        """
        This function performs the main training loop
        """

        for episode in range(number_of_episodes):

            # Save the models periodically
            if episode % saving_episode_number == 0:
                self.brain.save()
                print("Model weights saved")

            # Update target model weights periodically
            if episode % synchronization_episode_number == 0:
                self.brain.synchronization()
                print("Target model weights synchronized")

            # Update priorities periodically
            if self.using_priority and episode != 0 and episode % priority_update_episode_number == 0:
                self.memory.update_priorities(show_progress=True)

            print(episode)

            episode_length = 1
            total_reward = 0.

            # Start new episode and get initial observation
            state = self.env.reset()

            while True:
                self.steps += 1
                episode_length += 1

                # The exploration probability is decreased with every step
                self.update_exploration_ratio()

                # Take a greedy action using the primary model
                action = self.act(state=state, greedy=True)

                # Make a step and observe the new state and reward
                new_state, reward, done, info = self.env.step(action)

                # Store experience
                self.memorize((state, action, reward, new_state))

                # Learn from previous experiences
                self.replay(replay_size=replay_size, verbose=verbose)

                # Update the current state
                state = new_state

                total_reward += reward

                if done:  # The episode has ended
                    # Storing learning history
                    self.length_history.append(episode_length)
                    self.reward_history.append(total_reward)

                    # Print episode information
                    print(f"Exploration ratio at the end of the episode: {self.exploration_ratio}")
                    print(f"Episode length: {episode_length}")
                    print(f"Total reward: {total_reward}")
                    break

    def play(self, number_of_episodes: int, episode_max_length: int = 100, random=False, memorize=False,
             show_results=True):
        """
        The agent plays the game without learning. It can be used to initialization using random
        agent and to check performance.
        """

        for episode in range(number_of_episodes):
            if show_results:
                print(episode)

            episode_length = 1
            total_reward = 0.

            # Start new episode and get initial observation
            state = self.env.reset()

            while True:
                episode_length += 1

                if random:
                    action = self.env.action_space.sample()
                else:
                    action = self.act(state=state, greedy=False)

                # Make a step and observe the new state and reward
                new_state, reward, done, info = self.env.step(action)

                # store experience
                if memorize:
                    self.memorize((state, action, reward, new_state))

                # Update the current state
                state = new_state

                total_reward += reward

                if done or episode_max_length < episode_length:  # Avoid infinite episodes
                    if memorize:
                        self.length_history.append(episode_length)
                        self.reward_history.append(total_reward)
                    if show_results:
                        print(f"Episode length: {episode_length}")
                        print(f"Total reward: {total_reward}")
                    break
