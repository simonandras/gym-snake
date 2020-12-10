
import numpy as np
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model


class Brain:

    def __init__(self, input_shape: tuple, number_of_actions: int,
                 batch_size: int, number_of_epochs: int,
                 lr: float, rho: float, epsilon: float):

        assert len(input_shape) == 3, "input_shape should be 3 dimensional tuple"

        # CNN shape parameters
        self.input_shape = input_shape  # 3d, channel first
        self.number_of_actions = number_of_actions

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop parameters
        self.lr = lr
        self.rho = rho
        self.epsilon = epsilon

        # Creating Keras models with the same starting weights
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.synchronization()

    def create_model(self) -> Model:

        input_x = Input(shape=self.input_shape)

        x = Conv2D(16, kernel_size=8, strides=(4, 4),
                   activation='relu', padding='same',
                   data_format='channels_first')(input_x)
        x = Conv2D(32, kernel_size=4, strides=(2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first')(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        output_x = Dense(3)(x)

        model = Model(input_x, output_x)
        model.compile(optimizer=RMSprop(lr=self.lr,
                                        rho=self.rho,
                                        epsilon=self.epsilon),
                      loss="mean_squared_error")

        return model

    def train(self, X, y, verbose: int = 0):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.number_of_epochs, verbose=verbose)

    def predict(self, states):
        """
        Using the target model for prediction
        """

        return self.target_model.predict(states)

    def predict_one(self, state):
        return self.predict(np.array([state]))[0]

    def synchronization(self):
        self.target_model.set_weights(self.model.get_weights())
