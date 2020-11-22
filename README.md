# gym-snake

### Second Milestone

The DQN angent is implemented in the algorithms folder. It uses memory for replaying the experiences. There is a training process in the rf_agent_test.ipynb, but unfortunately it perfoms poorly for now. I think the problem is with the model: i have to use an autoencoder (possibly a vae) to create state representation and the RL model should only use this representation for learing the policy. If this works, then i will implement the double learning version of the current model. Some other improvements can also be used. The documentation is not ready for now, and i have to do some experiment to solve the problem properly.
