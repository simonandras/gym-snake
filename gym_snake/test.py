
import numpy as np
from gym_snake.envs.snake_env import SnakeEnv


env = SnakeEnv(shape=(5, 5))
observation = env.reset()
print(observation)
print("----------------")

for i in range(10):
    print(i)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(f"action: {action}")
    print(f"reward: {reward}")
    print(observation)
    if env.done:
        print("END")
        break
    print("----------------")
