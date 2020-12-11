# gym-snake

This project contains a snake-game environment based on OpenAI AI gym and a DDQN agent, which can learn to play the game using only image observation of the game.

## Folders

In the gym_snake forlder is 4 folders: <br>
- algorithms: contains the DDQN agent
- envs: contains the snake-game environment
- testing: contains the test files
- utilities: contains utility functions

### Testing and reproducing the results

The training and testing of the agent on the environment is in the DDQN_agent_test.ipynb file. It can be used without modification on Google Colab. <br>
The behavior of the snake-game environment is tested in the Environment_test.py. It can be used after downloading and installing the package, like in the DDQN_agent_test.ipynb file.
