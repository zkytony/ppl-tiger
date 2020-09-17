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


def belief_update(belief, action, observation, num_steps=100, print_losses=False):
    def belief_update_model(belief, action, observation):
        state = states[pyro.sample("bu_state", belief)]
        next_state = states[pyro.sample("bu_next_state", transition_model(state, action))]
        with pyro.condition(data={"bu_obs": observation}):
            predicted_observation = pyro.sample("bu_obs", observation_model(next_state, action))

    def belief_guide(belief, action, observation):
        belief_weights = pyro.param("bu_belief_weights", torch.ones(len(states)),
                                    constraint=dist.constraints.simplex)
        state = states[pyro.sample("bu_state", dist.Categorical(belief_weights))]
        next_state = states[pyro.sample("bu_next_state", transition_model(state, action))]

    svi = pyro.infer.SVI(belief_update_model,
                         belief_guide,
                         pyro.optim.Adam({"lr": 0.1}),  # hyper parameter matters
                         loss=pyro.infer.Trace_ELBO(retain_graph=True))
    if type(observation) == str:
        # !! Having to call observations.index is really annoying. I wish
        # Pyro can directly work with strings / or tensors support strings. !!
        # ... To work around this, either your models only work with integers
        # or you call index every time everywhere.
        observation = tensor(observations.index(observation))
    Infer(svi, belief, action, observation, num_steps=num_steps,
          print_losses=print_losses)
    return dist.Categorical(pyro.param("bu_belief_weights"))


def belief_value_model(t, belief, policy_model, discount=1.0, discount_factor=0.95, **kwargs):
    if discount < 1e-4:
        return dist.Normal(0, 0.01)
    state = states[pyro.sample("state_%d" % t, belief)]
    action = actions[pyro.sample("action_%d" % t, policy_model())]
    next_state = states[pyro.sample("next_state_%d" % t, transition_model(state, action))]
    observation = observations[pyro.sample("observation_%d" % t, observation_model(next_state, action))]
    reward = pyro.sample("reward_%d" % t, reward_model(state, action))

    next_belief = belief_update(belief, action, observation, **kwargs)
    cum_reward = reward + discount*pyro.sample("next_reward_%d" % t,
                                               belief_value_model(t+1, next_belief, policy_model,
                                                                  discount=discount*discount_factor,
                                                                  discount_factor=discount_factor,
                                                                  **kwargs))
    return dist.Normal(cum_reward, 0.01)

# def belief_value_guide(t, belief, policy_model, discount=1.0, discount_factor=0.95, **kwargs):
    
                                               


####### TESTS #######
def _test_belief_update():
    prior_belief = dist.Categorical(tensor([1., 1.]))
    action = "listen"
    observation = "growl-right"
    new_belief = belief_update(prior_belief, action, observation, num_steps=10)
    for name in pyro.get_param_store():
        print("{}: {}".format(name, pyro.param(name)))

    res = belief_value_model(0, new_belief, uniform_policy_model, discount=1.0, discount_factor=0.95)
    print(res)


if __name__ == "__main__":
    _test_belief_update()

