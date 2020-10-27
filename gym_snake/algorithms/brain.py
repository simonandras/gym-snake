
from keras.models import Model


class Brain:

    def __init__(self, state_shape: tuple, number_of_actions: int,
                 alpha: float = 0.01, momentum: float = 0.0, nesterov=False):

        assert len(state_shape) == 3, "state_shape should be 3 dimensional tuple"

        # CNN shape parameters
        self.state_shape = state_shape  # 3d
        self.number_of_actions = number_of_actions

        # SGD parameters
        self.alpha = alpha  # learning rate
        self.momentum = momentum
        self.nesterov = nesterov

        # Create Keras model
        self.model = self.create_model()

    def create_model(self) -> Model:

        # channel last ordering in Keras
        input_x = Input(shape=self.state_shape)

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

    def train(self, X, y, epoch=1, batch_size=64, verbose=0):
        self.model.fit(X, y, batch_size=batch_size, nb_epoch=epoch, verbose=verbose)

    def predict(self, states):
        return self.model.predict(states)

    def predict_one(self, state):
        return self.predict(np.array([state]))[0]
