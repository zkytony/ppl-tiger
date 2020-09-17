# The POMDP version of Tiger

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt
import numpy as np

states = ["tiger-left", "tiger-right"]
observations = ["growl-left", "growl-right"]
actions = ["open-left", "open-right", "listen"]

# All of the models have to return a tensor.

def observation_model(next_state, action, noise=0.15, t=0):
    """
    Args:
        next_state (str)  next state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the observation probability
    """
    if action == "listen":
        if next_state == "tiger-left":
            probs = [1.0-noise, noise]
        else:
            probs = [noise, 1.0-noise]
    else:
        probs = [0.5, 0.5]
    observation_dist = dist.Categorical(torch.tensor(probs))
    return pyro.sample('o_%d' % t, observation_dist)

def transition_model(state, action, t=1):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the transition probability
    """    
    probs = torch.zeros(len(states))
    probs[states.index(state)] = 1.0
    return pyro.sample('s_%d' % t, dist.Categorical(probs))

def reward_model(state, action, t=0):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the reward function
    """
    reward = 0
    if action == "open-left":
        if state== "tiger-right":
            reward = 10
        else:
            reward = -100
    elif action == "open-right":
        if state== "tiger-left":
            reward = 10
        else:
            reward = -100
    elif action == "listen":
        reward = -1
    return pyro.sample('r_%d' % t, dist.Delta(reward))

# World Model
# Pr (b, o, a, r)
# Observable: b0, rewards
# I think the example on Agent Book (https://agentmodels.org/chapters/3c-pomdp.html)
# is wrong because from the perspective of the POMDP agent it
# cannot simulate forward observations based on the real state!
# ... I realize they are trying to simulate both the world and
# the agent together instead of just doing POMDP planning

