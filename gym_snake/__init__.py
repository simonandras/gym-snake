
from gym.envs.registration import register

register(
    id='snake-v0',
    entry_point='snake_foo.envs:SnakeEnv',
)

register(
    id='snake-extrahard-v0',
    entry_point='gym_snake.envs:SnakeExtraHardEnv',
)

