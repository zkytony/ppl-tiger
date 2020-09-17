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


def belief_update(belief, action, observation, num_steps=100, print_losses=False,
                  suffix=""):
    def belief_update_model(belief, action, observation):
        state = states[pyro.sample("bu_state-%s" % suffix, belief)]
        next_state = states[pyro.sample("bu_next_state-%s" % suffix, transition_model(state, action))]
        with pyro.condition(data={"bu_obs-%s" % suffix: observation}):
            predicted_observation = pyro.sample("bu_obs-%s" % suffix, observation_model(next_state, action))

    def belief_guide(belief, action, observation):
        belief_weights = pyro.param("bu_belief_weights-%s" % suffix, torch.ones(len(states)),
                                    constraint=dist.constraints.simplex)
        state = states[pyro.sample("bu_state-%s" % suffix, dist.Categorical(belief_weights))]
        next_state = states[pyro.sample("bu_next_state-%s" % suffix, transition_model(state, action))]

    svi = pyro.infer.SVI(belief_update_model,
                         belief_guide,
                         pyro.optim.Adam({"lr": 0.1}),  # hyper parameter matters
                         loss=pyro.infer.Trace_ELBO())
    if type(observation) == str:
        # !! Having to call observations.index is really annoying. I wish
        # Pyro can directly work with strings / or tensors support strings. !!
        # ... To work around this, either your models only work with integers
        # or you call index every time everywhere.
        observation = tensor(observations.index(observation))
    if print_losses:
        print("Inferring belief update...")
    Infer(svi, belief, action, observation, num_steps=num_steps,
          print_losses=print_losses)
    return dist.Categorical(pyro.param("bu_belief_weights-%s" % suffix))

def expected_reward(belief, action, t, discount=1.0, discount_factor=0.95,
                    **kwargs):
    """This should return a tensor where each element is
    the expected reward when taking action at that index."""
    if discount < 0.1:
        return 0.0
    state = states[pyro.sample("state_%d" % t, belief)]
    reward = pyro.sample("reward_%d" % t, reward_model(state, action))

    next_state = states[pyro.sample("next_state_%d" % t, transition_model(state, action))]
    observation = observations[pyro.sample("observation_%d" % t, observation_model(next_state, action))]
    next_belief = belief_update(belief, action, observation, suffix=str(t), **kwargs)
    next_action = pyro.sample("next_action_%d" % t, plan_action(t, next_belief))
    cum_reward = reward + discount*pyro.sample("next_reward_%d" % t,
                                               expected_reward(next_belief, next_action, t,
                                                               discount=discount*discount_factor,
                                                               discount_factor=discount_factor))
    return cum_reward

def plan_action(t, belief, num_steps=20, print_losses=True):
    """Infer action given belief"""
    def policy_model(t, belief):
        probs = tensor([1., 1., 1.])
        for i, action in enumerate(actions):
            value = expected_reward(belief, action, t,
                                    num_steps=num_steps,
                                    print_losses=print_losses)
            probs[i] = value
        return dist.Categorical(probs)

    def policy_model_guide(t, belief):
        action_weights = pyro.param("action_weights_%d" % t, tensor([0.1, 0.1, 0.1]),
                                    constraint=dist.constraints.simplex)
        action = pyro.sample("action_%d" % t, dist.Categorical(action_weights))

    svi = pyro.infer.SVI(policy_model,
                         policy_model_guide,
                         pyro.optim.Adam({"lr":0.1}),
                         loss=pyro.infer.Trace_ELBO())
    if print_losses:
        print("Inferring next action (%d)..." % t)
    Infer(svi, t, belief, num_steps=num_steps, print_losses=print_losses)
    return dist.Categorical(pyro.param("action_weights_%d" % t))


####### TESTS #######
def _test_belief_update():
    prior_belief = dist.Categorical(tensor([1., 1.]))
    action = "listen"
    observation = "growl-right"
    new_belief = belief_update(prior_belief, action, observation, num_steps=10)
    for name in pyro.get_param_store():
        print("{}: {}".format(name, pyro.param(name)))

def _test_plan_action():
    prior_belief = dist.Categorical(tensor([1., 1.]))
    action_dist = plan_action(0, prior_belief, num_steps=20, print_losses=True)
    for name in pyro.get_param_store():
        print("{}: {}".format(name, pyro.param(name)))


if __name__ == "__main__":
    _test_belief_update()
    _test_plan_action()

