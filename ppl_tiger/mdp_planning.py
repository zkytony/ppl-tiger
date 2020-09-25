# POMDP Planning with PPL
# on tiger domain.

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
import numpy as np

from domain import *
from utils import remap, Infer


# My thought process:
#
# I am eventually obtaining a distribution Pr(a | s)
# and I know that this distribution should be weighted
# by Value(s,a).
#
# Then, what I do is I write a model for Value(s,a),
# use that as the action weights, and sample the action
# from this distribution as the policy_model.
#
# Then, if I set those action weights as the parameters,
# SVI will figure out what those parameters should be
# in order to be aligned with the Value(s,a) distribution
# encoded by the T/O/R models of this domain.
#
# Hence the policy_model, policy_model_guide, and value_model.

def policy_model(state, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """Returns Pr(a|s)"""
    # Weight the actions based on the value, and return the most
    # likely action
    if t >= max_depth:
        return pyro.sample("a%d" % t, dist.Categorical(tensor([1., 1., 1.])))
    action_weights = torch.zeros(len(actions))
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = value_model(state, action, t,
                                discount=discount,
                                discount_factor=discount_factor,
                                max_depth=max_depth)
            action_weights[i] = value  # action prior is uniform
    # Make the weights positive, then subtract from max
    min_weight = torch.min(action_weights)
    max_weight = torch.max(action_weights)
    action_weights = tensor([remap(action_weights[i], min_weight, max_weight, 0., 1.)
                             for i in range(len(action_weights))])
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
    # prior weights is uniform
    weights = pyro.param("action_weights", tensor([0.1, 0.1, 0.1]),
                         constraint=dist.constraints.simplex)
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = value_model(state, action, t,
                                discount=discount, discount_factor=discount_factor,
                                max_depth=max_depth)
    action = pyro.sample("a%d" % t, dist.Categorical(weights))
        
def main():
    state = "tiger-right"
    max_depth = 3
    discount_factor = 0.95
    svi = pyro.infer.SVI(policy_model,
                         policy_model_guide,
                         pyro.optim.Adam({"lr": 0.05}),
                         loss=pyro.infer.Trace_ELBO())
    Infer(svi, state, 0, discount=1.0, discount_factor=discount_factor, max_depth=max_depth,
          num_steps=300, print_losses=True)
    weights = pyro.param("action_weights")
    print("Action to take: %s" % actions[torch.argmax(weights).item()])
    print("Action weights: %s" % str(pyro.param("action_weights")))

if __name__ == "__main__":
    main()
