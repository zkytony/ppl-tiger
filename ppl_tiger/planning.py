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
actions = ["open-left", "open-right", "listen"]


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

def reward_dist(state, action):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the reward function
    """
    reward = 0
    if state != "terminal":
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


# The model
#   Pr (at | bt)
GLOBAL_COUNT = 0
def name(text):
    global GLOBAL_COUNT
    GLOBAL_COUNT += 1
    return "%s-%d" % (text, GLOBAL_COUNT)


def policy_model(state, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """Returns Pr(a|b)"""
    # Weight the actions based on the value, and return the most
    # likely action
    if t >= max_depth:
        return pyro.sample(name("a%d" % t), dist.Categorical(tensor([1., 1., 1.])))
    print("policy model")    
    action_weights = []
    
    # state = pyro.sample("s", belief)
    for i, action in enumerate(actions):
        print("--- model %d, %s ---" % (t, action))
        with scope(prefix=action):
            value = value_model(state, action, t,
                                discount=discount,
                                discount_factor=discount_factor,
                                max_depth=max_depth)
            action_weights.append(value)  # action prior is uniform
    # Make the weights positive, then subtract from max
    action_weights = torch.abs(tensor(action_weights))
    max_weight = torch.max(action_weights)
    action_weights -= max_weight
    return pyro.sample(name("a%d" % t), dist.Categorical(action_weights))

def value_model(state, action, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """Returns Pr(Value | b,a)"""
    if t >= max_depth:
        return tensor(0)
    print("Value model %d" % t)

    # Somehow compute the value
    next_state = states[pyro.sample(name("next_s%d" % t), transition_dist(state, action))]
    observation = observations[pyro.sample(name("o%d" % t), observation_dist(next_state, action))]
    reward = pyro.sample(name("r%d" % t), reward_dist(state, action))
    
    if next_state == "terminal":
        return pyro.sample(name("v%d" % t), dist.Delta(reward))
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
                                             max_depth = max_depth)

def policy_model_guide(state, t, discount=1.0, discount_factor=0.95, max_depth=10):
    # print("---guide---%d" % t)
    weights = pyro.param("action_weights", tensor([0.1, 0.1, 0.1]),
                         constraint=dist.constraints.simplex)
    action = actions[pyro.sample(name("a%d" % t), dist.Categorical(weights))]
    # print("--- guide ---")
    # value = value_model(state, action, t,
    #                     discount=discount, discount_factor=discount_factor)


def Infer(svi, *args, num_steps=100, print_losses=True, **kwargs):
    losses = []
    for t in range(num_steps):
        print("Step %d" % t)
        losses.append(svi.step(*args, **kwargs))
        if print_losses:
            print("Loss [%d] = %.3f" % (t, losses[-1]))

        
def main():
    # print(observations[pyro.sample("o", observation_model("tiger-left", "listen", noise=0.15))])
    # print(states[pyro.sample("s1", transition_model("tiger-left", "open-left"))])
    # print(states[pyro.sample("s2", transition_model("tiger-left", "listen"))])

    state = "tiger-left"
    svi = pyro.infer.SVI(policy_model,
                         policy_model_guide,
                         pyro.optim.Adam({"lr": 0.1}),
                         loss=pyro.infer.Trace_ELBO())
    Infer(svi, state, 0, discount=1.0, discount_factor=0.95,
          num_steps=100, print_losses=True)
    weights = pyro.param("action_weights")
    print("Action to take: %s" % actions[torch.argmax(weights).item()])
    print("Action weights: %s" % str(pyro.param("action_weights")))
    # print(actions[policy_model(state, 0, discount=1.0, discount_factor=0.95, max_depth=5)])

if __name__ == "__main__":
    main()
