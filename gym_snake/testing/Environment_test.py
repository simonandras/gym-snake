
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv


# Testing the behavior of the environment

# Initialize a new environment using double enlargement
env = SnakeEnv(shape=(2, 2), initial_snake_length=2, enlargement=2)

# Get initial observation
observation = env.reset()

# Show the observation (2d numpy array)
print(observation, '\n')
print("----------------")

for i in range(10):
    print(i)

    # Make random action
    action = env.action_space.sample()

    # Get observation adn reward
    observation, reward, done, info = env.step(action)

    # Show information
    print(f"action: {action}")
    print(f"reward: {reward}")

    # Show the observation (2d numpy array)
    print(observation, observation.shape)

    if env.done:  # the state is terminal
        print("END")
        break
    print("----------------")
