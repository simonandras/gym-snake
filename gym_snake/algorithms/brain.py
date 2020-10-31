
from tensorflow.keras.models import Model
from tensorflow.keras import Input


class Brain:

    def __init__(self, input_shape: tuple, number_of_actions: int,
                 batch_size: int, number_of_epochs: int,
                 alpha: float, momentum: float, nesterov: bool):

        assert len(input_shape) == 3, "input_shape should be 3 dimensional tuple"

        # CNN shape parameters
        self.input_shape = input_shape  # 3d
        self.number_of_actions = number_of_actions

        # Training parameters
        self.batch_size = batch_size
        self.number_of_epochs = number_of_epochs

        # SGD parameters
        self.alpha = alpha  # learning rate
        self.momentum = momentum
        self.nesterov = nesterov

        # Create Keras model
        self.model = self.create_model()

    def create_model(self) -> Model:

        # channel last ordering in Keras
        input_x = Input(shape=self.input_shape)

        x = Conv2D(8, kernel_size=2, activation='relu', padding='same')(input_x)
        x = BatchNormalization()(x)

        x = Conv2D(16, kernel_size=2, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        output_x = Dense(3, activation='sigmoid')(x)

        model = Model(input_x, output_x)
        model.compile(optimizer=SGD(learning_rate=self.alpha,
                                    momentum=self.momentum,
                                    nesterov=self.nesterov),
                      loss='mean_squared_error')

        return model

    def train(self, X, y, verbose: int = 0):
        self.model.fit(X, y, batch_size=self.batch_size, nb_epoch=self.number_of_epochs, verbose=verbose)

    def predict(self, states):
        return self.model.predict(states)

    def predict_one(self, state):
        return self.predict(np.array([state]))[0]
