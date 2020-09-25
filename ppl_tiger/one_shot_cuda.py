# One-shot tiger
# - The world is an MDP.
# - The agent takes just one action
# The world is deterministic


# According to https://forum.pyro.ai/t/using-gpu-slows-down-the-training-process/65/3
# Also: https://github.com/pyro-ppl/numpyro/issues/204
# this is expected behavior. there is no guarantee that code run on gpus is
# faster than code run on the cpu. in this case you have very small tensors and
# so the all the overhead associated with moving data/instructions to the gpu
# far outweight any gains made from fast floating point arithmetic.

import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt
import numpy as np

states = ["you-win", "you-lose"]
actions = ["open-left", "open-right", "listen"]

device = "cuda:0"
        
def transition_model(state, action):
    if state == "tiger-left":
        if action == "open-right":
            return tensor(states.index("you-win")).to(device)
        else:
            return tensor(states.index("you-lose")).to(device)
    else:
        if action == "open-left":
            return tensor(states.index("you-win")).to(device)
        else:
            return tensor(states.index("you-lose")).to(device)

def world_model(state):
    """
    This describes p(s,a)
    """
    action = actions[pyro.sample("a", dist.Categorical(tensor([1.,1.,1.]).to(device)))]
    loc = transition_model(state, action)
    # Give the next state given by the transition model 100% probability
    # and zero everywhere else.
    next_state_weights = torch.zeros(len(states)).to(device)
    next_state_weights[loc] = 1.0
    next_state = pyro.sample("next_state", dist.Categorical(next_state_weights))
    return next_state
    
def guide(state):
    """
    The variational family for q(a)
    """
    # Previously I separated the three elements of these weights,
    # and there was no constraint for them. The SVI algorithm was
    # never able to infer the right parameters. But now with constraints,
    # it works!
    weights = pyro.param("action_weights", tensor([0.1, 0.1, 0.1]).to(device),
                         constraint=dist.constraints.simplex)
    action_dist = dist.Categorical(weights)
    action = actions[pyro.sample("a", action_dist)]

    
def train(svi, init_state, num_steps=2500):
    """Performs variational inference"""
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
    with pyro.condition(data={"next_state": tensor(states.index("you-win")).to(device)}):
        svi = pyro.infer.SVI(world_model, guide,
                             pyro.optim.Adam({"lr": 0.01}),
                             loss=pyro.infer.Trace_ELBO())
        losses = train(svi, "tiger-right")
        for name in pyro.get_param_store():
            print("{}: {}".format(name, pyro.param(name)))
        weights = pyro.param("action_weights")
        print("Action to take: %s" % actions[torch.argmax(weights).item()])


