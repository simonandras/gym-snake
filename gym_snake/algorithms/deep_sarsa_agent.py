
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, \
    Dropout, BatchNormalization, Concatenate
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from gym_snake.envs.snake_env import SnakeEnv


class DeepSarsaAgent:

    def __init__(self, env: SnakeEnv, epsilon: float = 0.1, alpha: float = 0.01,
                 momentum: float = 0.0,
                 nesterov=False):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.momentum = momentum
        self.nesterov = nesterov

        self.one_hot_actions = np.array([[1., 0., 0.],
                                         [0., 1., 0.],
                                         [0., 0., 1.]])

        self.model = self.reset_model()

    def reset_model(self) -> Model:
        map_rows, map_cols = self.env.shape

        # State input
        t_input_1 = Input(shape=(map_rows, map_cols, 1))

        t1 = Conv2D(16, kernel_size=2, activation='relu', padding='same')(t_input_1)
        t1 = Dropout(0.3)(t1)

        t1 = Flatten()(t1)
        t1 = Dropout(0.3)(t1)
        t1 = Dense(10, activation='relu')(t1)

        # Action input
        t_input_2 = Input(shape=(3,))

        # Merge state features and actions
        t = Concatenate()([t1, t_input_2])

        t_output = Dense(1, activation='sigmoid')(t)

        model = Model([t_input_1, t_input_2], t_output)

        model.compile(optimizer=SGD(learning_rate=self.alpha,
                                    momentum=self.momentum,
                                    nesterov=self.nesterov),
                      loss='mean_squared_error')

        return model

    def take_action(self, obs: np.ndarray, epsilon_greedy: bool = True) -> int:
        preds = self.model.predict([np.array([obs, obs, obs]),
                                    self.one_hot_actions])

        if epsilon_greedy and np.random.rand() < self.epsilon:
            random_action = np.random.randint(3)
            return random_action, preds[random_action]
        else:
            best_action = np.argmax(preds)
            return best_action, preds[best_action]

    def learn(self, episodes: int):
        for episode_number in range(episodes):
            print(f"Episode: {episode_number}")

            # using only the actual map
            obs = self.env.reset()[0, ...]
            a, q = self.take_action(obs=obs, epsilon_greedy=True)

            i = 0
            sum_reward = 0.0

            while True:
                i += 1
                new_obs_, reward, done, info = self.env.step(action=a)
                sum_reward += reward
                new_obs = new_obs_[0, ...]
                if done:
                    self.model.fit([np.array([obs]), np.array([self.one_hot_actions[a]])],
                                   np.array([reward]),
                                   epochs=1, verbose=False)
                    break
                else:
                    new_a, new_q = self.take_action(obs=new_obs, epsilon_greedy=True)
                    self.model.fit([np.array([obs]), np.array([self.one_hot_actions[a]])],
                                   np.array([reward + new_q]),
                                   epochs=1, verbose=False)
                obs = new_obs
                a = new_a

            print(i, sum_reward)
