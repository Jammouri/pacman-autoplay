# Deep Convolution Q-Network (DQN) for Ms. Pac-Man

 This repository contains an implementation of a Deep convolution Q-Network (DQN) to play the Ms. Pac-Man game using the Gymnasium environment. The agent utilizes convolutional neural networks to process image inputs and learn optimal actions through reinforcement learning.

# Project Structure

DQN_MsPacman.ipynb: Jupyter notebook containing the entire implementation, including the neural network, DQN agent, and training process.

# Requirements
Python 3.8+

Jupyter Notebook

PyTorch

Gymnasium

NumPy

Pillow

torchvision

# Neural Network Architecture

## The neural network consists of four convolutional layers followed by three fully connected layers:

Conv Layer 1: 32 filters, 8x8 kernel, stride 4

Conv Layer 2: 64 filters, 4x4 kernel, stride 2

Conv Layer 3: 64 filters, 3x3 kernel, stride 1

Conv Layer 4: 128 filters, 3x3 kernel, stride 1

FC Layer 1: 512 units

FC Layer 2: 256 units

Output Layer: Number of actions

Batch normalization and ReLU activation are applied after each convolutional layer.



# Learning Process

## Initialization:

The environment is initialized using the Gymnasium Ms. Pac-Man environment with simplified actions.

The DQN agent is initialized with the neural network, target network, optimizer, and replay memory.

## Preprocessing:

Frames from the environment are preprocessed by resizing to 128x128 and normalizing pixel values using the preprocess_frame function.

## Experience Replay:

The agent stores experiences as tuples (state, action, reward, next_state, done) in a replay memory.
When the replay memory size exceeds the minibatch size, a random sample of experiences is used for learning.

## Training Loop:

### For each episode:
1-The environment is reset to obtain the initial state.

2-The agent interacts with the environment for a maximum number of timesteps per episode.

3-At each timestep, the agent selects an action using an epsilon-greedy policy:

   3.1-With probability epsilon, a random action is selected (exploration).

   3.2-With probability (1 - epsilon), the action with the highest Q-value is selected (exploitation).

4-The agent performs the action, observes the next state and reward, and stores the experience in the replay memory.

5-The agent samples a minibatch of experiences from the replay memory and performs a learning step:

6-Computes the target Q-values using the target network.

7-Computes the expected Q-values using the local network.

8-Updates the local network parameters by minimizing the loss between the target and expected Q-values.

9-The agent's epsilon value decays after each episode.

## Model Saving:

The agent's performance is tracked, and the model weights are saved to checkpoint.pth once the agent achieves an average score of 500 over 100 episodes.

# Results

The agent's performance is tracked and printed during training. The model weights are saved,once the agent achieves an average score of 500 over 100 episodes.

# References

1-Deep Q-Learning

2-Gymnasium Ms. Pac-Man Environment

# Acknowledgments

Special thanks to the developers of PyTorch and Gymnasium for providing excellent tools for machine learning and reinforcement learning research.
