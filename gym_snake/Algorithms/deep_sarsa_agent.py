
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from gym_snake.envs.snake_env import SnakeEnv


class DeepSarsaAgent:

    def __init__(self, env: SnakeEnv):
        self.env = env
        self.model = self.reset_model()

    def reset_model(self) -> Model:
        map_rows, map_cols = self.env.shape

        t_input = Input(shape=(map_rows, map_cols, 1))
        t = Conv2D(32, kernel_size=2, activation='relu', padding='same')(t_input)
        t = Dropout(0.5)(t)

        t = Conv2D(16, kernel_size=2, activation='relu', input_shape=(img_rows, img_cols, 1), padding='same')(t)
        t = BatchNormalization()(t)

        t = Flatten()(t)
        t = Dropout(0.5)(t)
        t = Dense(10, activation='relu')(t)

        t_output = Dense(1, activation='sigmoid')(t)

        model = Model(t_input, t_output)

        model.compile(optimizer='adam', loss='categorical_crossentropy')

        return model
