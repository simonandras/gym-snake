
import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model


class Brain:
    """
    Contains 2 CNN models for the DDQN agent. Using Keras framework.
    RMSprop is used as optimizer.
    Huber loss is used instead of the mean squared error in the training.
    """

    def __init__(self, input_shape: tuple, number_of_actions: int,
                 batch_size: int, number_of_epochs: int, lr: float,):

        assert len(input_shape) == 3, "input_shape should be 3 dimensional tuple"

        # CNN shape parameters
        self.input_shape = input_shape  # 3d, channel first
        self.number_of_actions = number_of_actions

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # RMSprop learning rate
        self.lr = lr

        # Creating Keras models with the same starting weights
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.synchronization()

    def create_model(self) -> Model:
        input_x = Input(shape=self.input_shape)

        x = Conv2D(32, kernel_size=8, strides=(4, 4),
                   activation='relu', padding='same',
                   data_format='channels_first')(input_x)
        x = Conv2D(64, kernel_size=4, strides=(2, 2),
                   activation='relu', padding='same',
                   data_format='channels_first')(x)
        x = Conv2D(64, kernel_size=3,
                   activation='relu', padding='same',
                   data_format='channels_first')(x)
        x = Flatten()(x)
        x = Dense(512, activation='relu')(x)
        output_x = Dense(self.number_of_actions, activation='linear')(x)

        model = Model(input_x, output_x)
        model.compile(optimizer=RMSprop(lr=self.lr),
                      loss=huber_loss)

        return model

    def train(self, X, y, verbose: int = 0):
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.number_of_epochs, verbose=verbose)

    def predict(self, states, target=True):
        if target:
            return self.target_model.predict(states)
        else:
            return self.model.predict(states)

    def predict_one(self, state, target=True):
        return self.predict(np.array([state]), target)[0]

    def synchronization(self):
        """
        Loads the weights of the primary model to the target model
        """

        self.target_model.set_weights(self.model.get_weights())

    def save(self):
        self.model.save_weights('model_weights.h5')
        self.target_model.save_weights('target_model_weights.h5')


def huber_loss(y_true, y_pred):
    """
    Huber loss is used instead of the mean squared error in the training
    """

    err = y_true - y_pred

    cond = K.abs(err) < 1.
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    loss = tf.where(cond, L2, L1)

    return K.mean(loss)
