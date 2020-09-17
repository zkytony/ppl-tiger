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


def Infer(svi, *args, num_steps=100, print_losses=True, **kwargs):
    losses = []
    for t in range(num_steps):
        losses.append(svi.step(*args, **kwargs))
        if print_losses:
            print("Loss [%d] = %.3f" % (t, losses[-1]))


def belief_update(belief, action, observation, num_steps=100):
    def belief_update_model(belief, action, observation):
        state = states[pyro.sample("state", belief)]
        next_state = states[pyro.sample("next_state", transition_model(state, action))]
        with pyro.condition(data={"obs": observation}):
            predicted_observation = pyro.sample("obs", observation_model(next_state, action))

    def belief_guide(belief, action, observation):
        belief_weights = pyro.param("belief_weights", torch.ones(len(states)),
                                    constraint=dist.constraints.simplex)
        state = states[pyro.sample("state", dist.Categorical(belief_weights))]
        next_state = states[pyro.sample("next_state", transition_model(state, action))]

    svi = pyro.infer.SVI(belief_update_model,
                         belief_guide,
                         pyro.optim.Adam({"lr": 0.001}),
                         loss=pyro.infer.Trace_ELBO())
    if type(observation) == str:
        # !! Having to call observations.index is really annoying. I wish
        # Pyro can directly work with strings / or tensors support strings. !!
        # ... To work around this, either your models only work with integers
        # or you call index every time everywhere.
        observation = tensor(observations.index(observation))
    Infer(svi, belief, action, observation, num_steps=num_steps)
    return dist.Categorical(pyro.param("belief_weights"))

prior_belief = dist.Categorical(tensor([1., 1.]))
action = "listen"
observation = "growl-left"
new_belief = belief_update(prior_belief, action, observation, num_steps=100)
for name in pyro.get_param_store():
    print("{}: {}".format(name, pyro.param(name)))
