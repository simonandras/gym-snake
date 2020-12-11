# gym-snake

This project contains a snake-game environment based on OpenAI AI gym and a DDQN agent, which can learn to play the game using only image observation of the game.

## Folders
The gym_snake forlder contains 4 folders: <br>
- algorithms: contains the DDQN agent
- envs: contains the snake-game environment
- testing: contains the test files
- utilities: contains utility functions

## Testing and reproducing the results

The training and testing of the agent on the environment is in the DDQN_agent_test.ipynb file. It can be used without modification on Google Colab. <br>
The behavior of the snake-game environment is tested in the environment_test.py. It can be used after downloading and installing the package.

## Install

### Using Google Colab

If using Google Colab, then just run te first code cell of the DDQN_agent_test.ipynb file: <br> <br>
!git clone --single-branch --branch CNN https://github.com/simonandras/gym-snake <br>
%cd gym-snake/ <br>
!pip install -e . <br>

### Using locally

If using locally, then for first download the package with the command: <br> <br>
git clone --single-branch --branch CNN https://github.com/simonandras/gym-snake <br> <br>
than go into the downloaded folder (gym-snake) and use pip to install: <br> <br>
pip install -e . <br> <br>
The packages in the setup.py file should be installed to use this package.
