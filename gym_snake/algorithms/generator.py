
import numpy as np
import keras
from gym_snake.envs.snake_env import SnakeEnv


class SnakeDataGenerator(keras.utils.Sequence):
    """
    Generates random snake map data for Keras encoder
    """

    def __init__(self, bach_per_epoch=10, batch_size=16, shape=(32, 32), enlargement: int = 1):

        self.batch_per_epoch = bach_per_epoch
        self.batch_size = batch_size
        self.shape = shape
        self.enlargement = enlargement

        self.enlarged_shape = (self.enlargement * self.shape[0], self.enlargement * self.shape[1])
        self.max_snake_length = shape[0] * shape[1]
        self.env = SnakeEnv(shape=self.shape, enlargement=self.enlargement)

    def __len__(self):
        """
        Returns the number of batches per epoch
        """

        return self.batch_per_epoch

    def __getitem__(self, idx):
        """
        Generate one batch of data
        """

        X = np.zeros((self.batch_size, 1, *self.enlarged_shape))

        for i in range(self.batch_size):
            length = np.random.randint(low=4, high=self.max_snake_length)
            observation = self.env.reset(spec_reset=True, spec_snake_length=length)
            X[i] = np.array([observation])

        return X, X
