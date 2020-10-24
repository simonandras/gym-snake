
import numpy as np
#import keras
from gym_snake.envs.snake_env import SnakeEnv


# class SnakeDataGenerator(keras.utils.Sequence):
class SnakeDataGenerator:
    """
    Generates random snake map data for Keras
    """

    def __init__(self, bach_per_epoch=10, batch_size=32, shape=(10, 10)):
        self.batch_per_epoch = bach_per_epoch
        self.batch_size = batch_size
        self.shape = shape

        self.max_snake_length = shape[0] * shape[1]
        self.env = SnakeEnv(shape=self.shape)

    def __len__(self):
        """
        Returns the number of batches per epoch
        """

        return self.batch_per_epoch

    def __getitem__(self):
        """
        Generate one batch of data
        """

        X = np.empty((self.batch_size, *self.shape))

        for i in range(self.batch_per_epoch):
            length = np.random.randint(low=4, high=self.max_snake_length)
            observation = self.env.reset(spec_reset=True, spec_snake_length=length)
            X[i, ...] = observation[0, ...]

        return X, X
