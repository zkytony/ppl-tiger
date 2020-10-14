# An attempt to do pomdp planning without
# nested belief update - but it does not work;
# Still equivalent to QMDP planning (observation
# information is not used).

from domain import *
from mdp_planning import\
    policy_model, value_model
from belief_update import\
    belief_update
from qmdp_planning import plan as qmdp_plan

import torch
torch.set_printoptions(sci_mode=False)
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
import sys
import matplotlib.pyplot as plt
import numpy as np
from utils import Infer, remap

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# These counters will show that taking LISTEN and STAY leads
# to a different distribution in the allocation of states among
# histories; LISTEN will allocate more states to the branch which
# contains the observation that matches the state, while STAY will
# not be able to do so. LISTEN_POINTS is expected to be greater than
# STAY_POINTS.
LISTEN_POINTS = 0
STAY_POINTS = 0

# History-based policy and value models
def history_policy_model(state, history, t, discount=1.0, discount_factor=0.95, max_depth=10):
    """history is a string"""
    with scope(prefix=history):
        if t > max_depth:
            return pyro.sample("a%d" % t, dist.Categorical(torch.ones(len(actions))))
        action_weights = torch.zeros(len(actions))
        for i, action in enumerate(actions):
            with scope(prefix="%s%d" % (action, t)):
                value = history_value_model(state, action, history, t,
                                            discount=discount,
                                            discount_factor=discount_factor,
                                            max_depth=max_depth)
                action_weights[i] = torch.exp(value)
    # Make the weights positive, then subtract from max
    min_weight = torch.min(action_weights)
    max_weight = torch.max(action_weights)
    action_weights = tensor([remap(action_weights[i], min_weight, max_weight, 0., 1.)
                             for i in range(len(action_weights))])
    return actions[pyro.sample("a%d" % t, dist.Categorical(action_weights))]

def history_value_model(state, action, history, t, discount=1.0, discount_factor=0.95, max_depth=10):
    global LISTEN_POINTS, STAY_POINTS
    if t > max_depth:
        return tensor(1e-9)

    next_state = states[pyro.sample("next_s%d" % t, transition_dist(state, action))]
    reward = pyro.sample("r%d" % t, reward_dist(state, action, next_state))
    observation = observations[pyro.sample("o%d" % t, observation_dist(next_state, action))]

    if next_state == "terminal":
        return pyro.sample("v%d" % t, dist.Delta(reward))
    else:
        # compute future value
        discount = discount*discount_factor
        next_history = history + "_%s:%s" % (action, observation)

        if len(next_history) > 0:
            h1 = next_history.split("_")[1]
            if (h1.endswith("left") and state.endswith("left"))\
               or (h1.endswith("right") and state.endswith("right")):
                if h1.startswith("listen"):
                    LISTEN_POINTS += 1
                elif h1.startswith("stay"):
                    STAY_POINTS += 1

        next_action = history_policy_model(next_state, next_history, t+1,
                                           discount=discount, discount_factor=discount_factor,
                                           max_depth=max_depth)
        rew = reward + discount * history_value_model(next_state, next_action, next_history, t+1,
                                                       discount=discount,
                                                       discount_factor=discount_factor,
                                                       max_depth=max_depth)
        return rew

def history_policy_model_guide(state, history, t, discount=1.0,
                               discount_factor=0.95, max_depth=10):
    with scope(prefix=history):
        weights = pyro.param("action_weights", torch.ones(len(actions)),
                             constraint=dist.constraints.simplex)
        for i, action in enumerate(actions):
            with scope(prefix="%s%d" % (action, t)):
                value = history_value_model(state, action, history, t,
                                            discount=discount,
                                            discount_factor=discount_factor,
                                            max_depth=max_depth)
    return actions[pyro.sample("a%d" % t, dist.Categorical(weights))]

# belief policy model based on history models
def belief_policy_model(belief, t, discount=1.0, discount_factor=0.95, max_depth=10):
    state = states[pyro.sample("s%d" % t, belief)]
    history = ""  # we can start from empty history
    return history_policy_model(state, history, t, discount=discount,
                                discount_factor=discount_factor,
                                max_depth=max_depth)

def belief_policy_model_guide(belief, t, discount=1.0, discount_factor=0.95, max_depth=10):
    # prior weights is uniform
    weights = pyro.param("action_weights", torch.ones(len(actions)),
                         constraint=dist.constraints.simplex)
    state = states[pyro.sample("s%d" % t, belief)]
    history = ""  # we can start from empty history
    # This is just to generate the other variables
    with poutine.block(hide=["a%d" % t]):
        # Need to hide 'at' generated by the policy_model;
        # we don't care about it; because that's what we
        # are inferring -- we are calling pyro.sample("a%d" % t ..) next.
        history_policy_model(state, history, t, discount=discount,
                             discount_factor=discount_factor,
                             max_depth=max_depth)
    # We eventually generate actions based on the weights
    action = pyro.sample("a%d" % t, dist.Categorical(weights))


def plan(belief, max_depth=3, discount_factor=0.95, lr=0.1, nsteps=100, print_losses=True):
    """nsteps (int) number of iterations to reduce the loss"""
    pyro.clear_param_store()
    svi = pyro.infer.SVI(belief_policy_model,
                         belief_policy_model_guide,
                         pyro.optim.Adam({"lr": lr}),
                         loss=pyro.infer.Trace_ELBO(retain_graph=True))
    Infer(svi, belief, 0,
          discount=1.0, discount_factor=discount_factor, max_depth=max_depth,
          num_steps=nsteps, print_losses=print_losses)
    return pyro.param("action_weights")


### TESTS

def test_history_models():
    state = "tiger-left"
    history = ""
    max_depth = 3
    discount_factor = 0.95

    # Testing history model
    svi = pyro.infer.SVI(history_policy_model,
                         history_policy_model_guide,
                         pyro.optim.Adam({"lr": 0.001}),
                         loss=pyro.infer.Trace_ELBO())
    Infer(svi, state, history, 0,
          discount=1.0, discount_factor=discount_factor, max_depth=max_depth,
          num_steps=300, print_losses=True)
    weights = pyro.param("action_weights")
    print("Action to take: %s" % actions[torch.argmax(weights).item()])
    print("Action weights: %s" % str(pyro.param("action_weights")))



def test_pomdp_planning():
    belief = dist.Categorical(tensor([1., 1., 0.]))
    state = "tiger-left"
    max_depth = 2
    discount_factor = 0.95

    weights = plan(belief, max_depth=max_depth,
                   discount_factor=discount_factor, lr=0.1,
                   nsteps=100, print_losses=True)
    action = actions[torch.argmax(weights).item()]
    action_weights = pyro.param("action_weights").detach().numpy()
    print("=== History based ===")
    print("Action to take: %s" % action)
    print("Action weights: %s" % str(action_weights))

    print("listen:", LISTEN_POINTS)
    print("stay:", STAY_POINTS)


    weights = qmdp_plan(belief, max_depth=max_depth,
                        discount_factor=discount_factor, lr=0.1,
                        nsteps=200, print_losses=False)
    action = actions[torch.argmax(weights).item()]
    action_weights = pyro.param("action_weights").detach().numpy()
    print("\n=== QMDP ===")
    print("Action to take: %s" % action)
    print("Action weights: %s" % str(action_weights))

if __name__ == "__main__":
    # test_history_models()
    test_pomdp_planning()
