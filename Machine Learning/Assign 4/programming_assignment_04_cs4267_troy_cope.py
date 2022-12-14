# -*- coding: utf-8 -*-
"""Programming Assignment-04-CS4267-Troy Cope.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GOkREOPorEbCBKZoLWm6k4_sBfi5HEfa

## **We train the neural network as follows:**

 

During the exploration phase, we perform a random action that has the highest value in the output layer.

Then, we store the action, the next state, the reward, and the flag stating whether the game was complete in memory.

In a given state, if the game is not complete, the Q-value of taking an action in a given state will be calculated; that is, reward + discount factor x maximum possible Q-value of all actions in the next state.

The Q-values of the current state-action combinations remain unchanged except for the action that is taken in step 2.

Perform steps 1 to 4 multiple times and store the experiences.

Fit a model that takes the state as input and the action values as the expected outputs (from memory and replay experience) and minimize the MSE loss.

Repeat the preceding steps over multiple episodes while decreasing the exploration rate.

With the preceding strategy in place, let's code up deep Q-learning so that we can perform CartPole balancing:

 

**Import the relevant packages:**
"""

import gym

import numpy as np

import cv2

from collections import deque

import torch

import torch.nn as nn

import torch.nn.functional as F

import random

from collections import namedtuple, deque

import torch.optim as optim

!pip install pygame

device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""Define the environment:"""

env = gym.make('CartPole-v1')

"""Define the network architecture:"""

class DQNetwork(nn.Module):

  def __init__ (self, state_size, action_size): 

    super(DQNetwork, self).__init__() 

    self.fc1 = nn.Linear(state_size, 24) 

    self.fc2 = nn.Linear(24, 24)

    self.fc3 = nn.Linear(24, action_size) 

  def forward(self, state):

    x = F.relu(self.fc1(state)) 

    x = F.relu(self.fc2(x))

    x = self.fc3(x)

    return x

"""Note that the architecture is fairly simple since it only contains 24 units in the two hidden layers. The output layer contains as many units as there are possible actions.

 

Define the Agent class, as follows:
 

Define the __init__ method with the various parameters, network, and experience defined:

Define the step function, which fetches data from memory and fits it to the model by calling the learn function:
"""

class Agent():

  def __init__ (self, state_size, action_size):

    self.state_size = state_size

    self.action_size = action_size

    self.seed = random.seed(0)

    

    ## hyperparameters

    self.buffer_size = 2000

    self.batch_size = 64

    self.gamma = 0.99

    self.lr = 0.0025

    self.update_every = 4

    # Q-Network

    self.local = DQNetwork(state_size, action_size).to(device)

    self.optimizer=optim.Adam(self.local.parameters(), lr=self.lr)

    

    # Replay memory

    self.memory = deque(maxlen=self.buffer_size) 

    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    

    self.t_step = 0

  def step(self, state, action, reward, next_state, done): 

    # Save experience in replay memory 

    self.memory.append(self.experience(state, action, reward, next_state, done)) 

    # Learn every update_every time steps.

    self.t_step = (self.t_step + 1) % self.update_every 

    if self.t_step == 0:

    # If enough samples are available in memory, 

    # get random subset and learn

      if len(self.memory) > self.batch_size: 
        experiences = self.sample_experiences() 

        self.learn(experiences, self.gamma)

  def act(self, state, eps=0.):

    # Epsilon-greedy action selection 

    if random.random() > eps:

      state = torch.from_numpy(state).float().unsqueeze(0).to(device)

      self.local.eval()

      with torch.no_grad():

        action_values = self.local(state) 

      self.local.train()

      return np.argmax(action_values.cpu().data.numpy()) 

    else:

      return random.choice(np.arange(self.action_size))



  def learn(self, experiences, gamma): 

    states,actions,rewards,next_states,dones= experiences 

    # Get expected Q values from local model

    Q_expected = self.local(states).gather(1, actions)

    

    # Get max predicted Q values (for next states) 

    # from local model

    Q_targets_next = self.local(next_states).detach().max(1)[0].unsqueeze(1)           

    # Compute Q targets for current states

    Q_targets = rewards+(gamma*Q_targets_next*(1-dones)) 

    # Compute loss

    loss = F.mse_loss(Q_expected, Q_targets)

    # Minimize the loss 

    self.optimizer.zero_grad()

    loss.backward()

    self.optimizer.step()

  def sample_experiences(self):

    experiences = random.sample(self.memory, k=self.batch_size)            

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)

    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)

    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)

    next_states=torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)

    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)     

    return (states, actions, rewards, next_states, dones)

"""Define the agent object:"""

#agent = Agent()
agent = Agent(env.observation_space.shape[0], env.action_space.n)

"""
Perform deep Q-learning, as follows:
Initialize your lists:"""

scores = [] # list containing scores from each episode

scores_window = deque(maxlen=100) # last 100 scores 

n_episodes=5000

max_t=5000 

eps_start=1.0 

eps_end=0.001 

eps_decay=0.9995

eps = eps_start

"""Reset the environment in each episode and fetch the state's shape.
Furthermore, reshape it so that we can pass it to a network:

Loop through max_t time steps, identify the action to be performed, and perform (step) it. Next, reshape it so that the reshaped state is passed to the neural network:

Fit the model by specifying agent.step on top of the current state and resetting the state to the next state so that it can be useful in the next iteration:

Store, print periodically, and stop training if the mean of the scores in the previous 10 steps is greater than 450:
"""

for i_episode in range(1, n_episodes+1):

  state = env.reset()

  state_size = env.observation_space.shape[0] 

  state = np.reshape(state, [1, state_size])

  score = 0

  for i in range(max_t):

    action = agent.act(state, eps)

    next_state, reward, done, _ = env.step(action)

    next_state = np.reshape(next_state, [1, state_size])

    reward = reward if not done or score == 499 else -10 

    agent.step(state, action, reward, next_state, done) 

    state = next_state

    score += reward 

    if done:
      break

  scores_window.append(score) # save most recent score

  scores.append(score) # save most recent score

  eps = max(eps_end, eps_decay*eps) # decrease epsilon

  print('\rEpisode {}\tReward {} \tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode,score, np.mean(scores_window), eps), end="")

  if i_episode % 100 == 0:

    print('\rEpisode {}\tAverage Score: {:.2f} \tEpsilon: {}'.format(i_episode, np.mean(scores_window), eps))

  if i_episode>10 and np.mean(scores[-10:])>450:
    break

"""Plot the variation in scores over increasing episodes:"""

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt

# %matplotlib inline

plt.plot(scores)

plt.title('Scores over increasing episodes')