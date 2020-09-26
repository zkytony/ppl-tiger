# The POMDP version of Tiger

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt
import numpy as np
from domain import *
from utils import Infer

# For this particular example we only care about tiger-left, tiger-right states.

def belief_update(belief, action, observation, num_steps=100, print_losses=False,
                  suffix=""):
    def belief_update_model(belief, action, observation):
        state = states[pyro.sample("bu_state-%s" % suffix, belief)]
        next_state = states[pyro.sample("bu_next_state-%s" % suffix, transition_dist(state, action))]
        with pyro.condition(data={"bu_obs-%s" % suffix: observation}):
            predicted_observation = pyro.sample("bu_obs-%s" % suffix, observation_dist(next_state, action))

    def belief_guide(belief, action, observation):
        belief_weights = pyro.param("bu_belief_weights-%s" % suffix, torch.ones(len(states)),
                                    constraint=dist.constraints.simplex)
        state = states[pyro.sample("bu_state-%s" % suffix, dist.Categorical(belief_weights))]
        next_state = states[pyro.sample("bu_next_state-%s" % suffix, transition_dist(state, action))]

    pyro.clear_param_store()        
    svi = pyro.infer.SVI(belief_update_model,
                         belief_guide,
                         pyro.optim.Adam({"lr": 0.01}),  # hyper parameter matters
                         loss=pyro.infer.Trace_ELBO(retain_graph=True))
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


####### TESTS #######
# states = states_without_terminal
def _test_belief_update():
    prior_belief = dist.Categorical(tensor([1., 1., 0.]))
    action = "listen"
    observation = "growl-right"
    new_belief = belief_update(prior_belief, action, observation, num_steps=1000)
    for name in pyro.get_param_store():
        print("{}: {}".format(name, pyro.param(name)))
        
if __name__ == "__main__":
    _test_belief_update()

