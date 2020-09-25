# POMDP Planning with PPL
# on tiger domain.

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
import sys
import matplotlib.pyplot as plt
import numpy as np

states = ["tiger-left", "tiger-right", "terminal"]
observations = ["growl-left", "growl-right"]
actions = ["open-right", "listen", "open-left"]


def observation_dist(next_state, action, noise=0.15):
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
        elif next_state == "tiger-right":
            obs_probs = [noise, 1.0-noise]
        else:  # terminal
            obs_probs = [0.5, 0.5]
    else:
        obs_probs = [0.5, 0.5]
    return dist.Categorical(torch.tensor(obs_probs))

def transition_dist(state, action):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        The transition distribution to sample from
    """
    if action == "open-left" or action == "open-right":
        next_state = "terminal"
    else:
        next_state = state
    trans_probs = torch.zeros(len(states))
    trans_probs[states.index(next_state)] = 1.0
    return dist.Categorical(trans_probs)

def reward_dist(state, action, next_state):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the reward function
    """
    reward = 1e-9
    if next_state == "terminal":
        if action == "open-left":
            if state == "tiger-right":
                reward = 10.0
            elif state == "tiger-left":
                reward = -100.0
        elif action == "open-right":
            if state == "tiger-left":
                reward = 10.0
            elif state == "tiger-right":
                reward = -100.0
        elif action == "listen":
            reward = -1.0
    else:
        assert action == "listen", "state (%s) --%s--> next_state (%s) is Problematic."\
            % (state, action, next_state)
        reward -= 1.0
    return dist.Delta(tensor(reward))


def policy_model(state, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """Returns Pr(a|b)"""
    # Weight the actions based on the value, and return the most
    # likely action
    if t >= max_depth:
        return pyro.sample("a%d" % t, dist.Categorical(tensor([1., 1., 1.])))
    action_weights = []
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = value_model(state, action, t,
                                discount=discount,
                                discount_factor=discount_factor,
                                max_depth=max_depth)
            action_weights.append(value)  # action prior is uniform
    # Make the weights positive, then subtract from max
    action_weights = -1*tensor(action_weights)
    max_weight = torch.max(action_weights)
    action_weights = max_weight - action_weights
    return actions[pyro.sample("a%d" % t, dist.Categorical(action_weights))]

def value_model(state, action, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """Returns Pr(Value | b,a)"""
    if t >= max_depth:
        return tensor(1e-9)

    # Somehow compute the value
    next_state = states[pyro.sample("next_s%d" % t, transition_dist(state, action))]
    observation = observations[pyro.sample("o%d" % t, observation_dist(next_state, action))]
    reward = pyro.sample("r%d" % t, reward_dist(state, action, next_state))
    
    if next_state == "terminal":
        return pyro.sample("v%d" % t, dist.Delta(reward))
    else:
        # compute future value
        discount = discount*discount_factor
        next_action = policy_model(next_state, t+1,
                                   discount=discount,
                                   discount_factor=discount_factor,
                                   max_depth=max_depth)
        return reward + discount*value_model(next_state, next_action, t+1,
                                             discount=discount,
                                             discount_factor=discount_factor,
                                             max_depth=max_depth)

def policy_model_guide(state, t, discount=1.0, discount_factor=0.95, max_depth=10):
    # You must reproduce the same structure in model...
    weights = pyro.param("action_weights", tensor([0.1, 0.1, 0.1]),
                         constraint=dist.constraints.simplex)
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = value_model(state, action, t,
                                discount=discount, discount_factor=discount_factor,
                                max_depth=max_depth)
    action = pyro.sample("a%d" % t, dist.Categorical(weights))


def Infer(svi, *args, num_steps=100, print_losses=True, **kwargs):
    losses = []
    for t in range(num_steps):
        losses.append(svi.step(*args, **kwargs))
        if print_losses:
            print("Loss [%d] = %.3f" % (t, losses[-1]))

        
def main():
    state = "tiger-left"
    max_depth = 3
    discount_factor = 0.95
    svi = pyro.infer.SVI(policy_model,
                         policy_model_guide,
                         pyro.optim.Adam({"lr": 0.01}),
                         loss=pyro.infer.Trace_ELBO())
    Infer(svi, state, 0, discount=1.0, discount_factor=discount_factor, max_depth=max_depth,
          num_steps=100, print_losses=True)
    weights = pyro.param("action_weights")
    print("Action to take: %s" % actions[torch.argmax(weights).item()])
    print("Action weights: %s" % str(pyro.param("action_weights")))
    for action in actions:
        value = value_model(state, action, 0,
                            discount=1.0,
                            discount_factor=discount_factor,
                            max_depth=max_depth)
        print(action, value)
    # print(actions[policy_model(state, 0, discount=1.0, discount_factor=0.95, max_depth=5)])

if __name__ == "__main__":
    main()
