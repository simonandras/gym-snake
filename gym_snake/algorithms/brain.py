
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model


class Brain:

    def __init__(self, input_length: int, number_of_actions: int,
                 batch_size: int, number_of_epochs: int,
                 lr: float, rho: float, epsilon: float):

        # NN shape parameters
        self.input_length = input_length  # The short term memory memory_length
        self.number_of_actions = number_of_actions

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop parameters
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Used for encoding the observations
        self.encoder_model = None

        # Used for predicting action values
        self.model = self.create_model()

    def create_model(self) -> Model:
        inp = Input((self.input_length,))
        x = Dense(32, activation='relu')(inp)
        output = Dense(self.number_of_actions)(x)

        model = Model(inp, output)

        model.compile(optimizer=RMSprop(lr=self.lr,
                                        rho=self.rho,
                                        epsilon=self.epsilon),
                      loss="mean_squared_error")

        return model

    def train(self, X, y, verbose: int = 0):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.number_of_epochs, verbose=verbose)

    def predict(self, states):
        return self.model.predict(states)

    def predict_one(self, state):
        return self.predict(np.array([state]))[0]

    def encode_one(self, observation):
        """
        observation: comes from the env; shape: (m, n)
        predict needs shape (1, 1, m, n)
        the last of the predictions are the encoded array of one array
        finally sample the one element of the array
        """
        return self.encoder_model.predict(np.array([np.array([observation])]))[-1][0]
