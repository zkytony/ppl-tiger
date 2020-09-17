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

def observation_model(next_state, action, noise=0.15):
    """
    Args:
        next_state (str)  next state
        action (str)  action
    Returns:
        The observation distribution to sample from
    """
    if action == "listen":
        if next_state == "tiger-left":
            obs_probs = [1.0-noise, noise]
        else:
            obs_probs = [noise, 1.0-noise]
    else:
        obs_probs = [0.5, 0.5]
    return dist.Categorical(torch.tensor(obs_probs))

def transition_model(state, action):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        The transition distribution to sample from
    """    
    trans_probs = torch.zeros(len(states))
    trans_probs[states.index(state)] = 1.0
    return dist.Categorical(trans_probs)

def reward_model(state, action):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the reward function
    """
    reward = 0
    if action == "open-left":
        if state == "tiger-right":
            reward = 10
        else:
            reward = -100
    elif action == "open-right":
        if state == "tiger-left":
            reward = 10
        else:
            reward = -100
    elif action == "listen":
        reward = -1
    return dist.Delta(tensor(reward))

def uniform_policy_model():
    return dist.Categorical(tensor([1.,1.,1.]))

def expected_reward_model(state, policy_model, t=0,
                          discount=1.0, discount_factor=0.95, thresh=1e-4):
    if discount < thresh:
        return 0.0
    action = actions[pyro.sample("a_%d" % t, policy_model())]
    reward = pyro.sample("r_%d" % t, reward_model(state, action))
    next_state = states[pyro.sample("s_%d" % (t+1),
                                    transition_model(state, action))]
    observation = observations[pyro.sample("o_%d" % t,
                                           observation_model(next_state, action))]
    return reward + discount*expected_reward_model(next_state,
                                                   policy_model,
                                                   t=t+1,
                                                   discount=discount*discount_factor,
                                                   discount_factor=discount_factor,
                                                   thresh=thresh)

print(expected_reward_model("tiger-left", uniform_policy_model))















# # World Model
# # Pr (b, o, a, r)
# # Observable: b0, rewards
# # I think the example on Agent Book (https://agentmodels.org/chapters/3c-pomdp.html)
# # is wrong because from the perspective of the POMDP agent it
# # cannot simulate forward observations based on the real state!
# # ... I realize they are trying to simulate both the world and
# # the agent together instead of just doing POMDP planning
# def step_model(belief, t=0):
#     """
#     belief is a distribution.

#     The world model essentially samples an action to take
#     from a categorical distribution, then computes the
#     expected reward R(b,a) where b' is the updated belief.
#     """
#     action = actions[
#         pyro.sample("a_%d" % t,
#                     dist.Categorical(tensor([1.,1.,1.])))]
