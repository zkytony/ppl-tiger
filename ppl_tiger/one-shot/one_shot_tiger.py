# One-shot tiger
# - The world is an MDP.
# - The agent takes just one action
# The world is deterministic

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt
import numpy as np

states = ["you-win", "you-lose"]
actions = ["open-left", "open-right", "listen"]
        
def transition_model(state, action):
    if state == "tiger-left":
        if action == "open-right":
            return tensor(states.index("you-win"))
        else:
            return tensor(states.index("you-lose"))
    else:
        if action == "open-left":
            return tensor(states.index("you-win"))
        else:
            return tensor(states.index("you-lose"))

def world_model(state):
    a = pyro.sample("a", dist.Categorical(tensor([1.,1.,1.])))
    action = actions[a]
    loc = transition_model(state, action)
    if loc.item() == 0:
        next_state = pyro.sample("next_state", dist.Categorical(tensor([1.0, 0.0])))
    else:
        next_state = pyro.sample("next_state", dist.Categorical(tensor([0.0, 1.0])))
    return next_state
    
def guide(state):
    weights = pyro.param("action_weights", tensor([0.1, 0.1, 0.1]),
                         constraint=dist.constraints.simplex)
    action_dist = dist.Categorical(weights)
    action = actions[pyro.sample("a", action_dist)]

def train(svi, init_state, num_steps=2500):
    elbo = pyro.infer.Trace_ELBO()
    losses = []
    num_steps = 2500
    for t in range(num_steps):
        sys.stdout.write("%d/%d\r" % (t+1, num_steps))
        losses.append(svi.step(init_state))
        if t % 100 == 0:
            print("Loss [%d] = %.3f" % (t, losses[-1]))
    sys.stdout.write("\n")
    return losses

### Inference
if __name__ == "__main__":
    pyro.clear_param_store()
    with pyro.condition(data={"next_state": tensor(states.index("you-win"))}):
        svi = pyro.infer.SVI(world_model, guide,
                             pyro.optim.Adam({"lr": 0.001}),
                             loss=pyro.infer.Trace_ELBO())
        losses = train(svi, "tiger-left")
        for name in pyro.get_param_store():
            print("{}: {}".format(name, pyro.param(name)))
        weights = pyro.param("action_weights")
        print("Action to take: %s" % actions[torch.argmax(weights).item()])


