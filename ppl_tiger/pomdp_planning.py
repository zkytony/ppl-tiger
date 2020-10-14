from belief_update import\
    belief_update
from domain import *

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
from utils import remap, Infer
import time

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def belief_policy_model(belief, t, discount=1.0, discount_factor=0.95, max_depth=10,
                        bu_nsteps=10, bu_lr=0.1):
    if t > max_depth:
        return pyro.sample("a%d" % t, dist.Categorical(tensor([1., 1., 1.])))
    action_weights = torch.zeros(len(actions))
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = belief_value_model(belief, action, t,
                                       discount=discount,
                                       discount_factor=discount_factor,
                                       max_depth=max_depth,
                                       bu_nsteps=bu_nsteps,
                                       bu_lr=bu_lr)
            action_weights[i] = torch.exp(value)  # action weight is softmax of value
    # Make the weights positive, then subtract from max
    min_weight = torch.min(action_weights)
    max_weight = torch.max(action_weights)
    action_weights = tensor([remap(action_weights[i], min_weight, max_weight, 0., 1.)
                             for i in range(len(action_weights))])
    return actions[pyro.sample("a%d" % t, dist.Categorical(action_weights))]


def belief_value_model(belief, action, t, discount=1.0, discount_factor=0.95, max_depth=10,
                       bu_nsteps=10, bu_lr=0.1):
    """Returns Pr(Value | b,a)"""
    if t > max_depth:
        return tensor(1e-9)

    # Somehow compute the value
    state = states[pyro.sample("s%d" % t, belief)]
    next_state = states[pyro.sample("next_s%d" % t, transition_dist(state, action))]
    reward = pyro.sample("r%d" % t, reward_dist(state, action, next_state))

    if next_state == "terminal":
        return pyro.sample("v%d" % t, dist.Delta(reward))
    else:
        # compute future value
        discount = discount*discount_factor
        observation = observations[pyro.sample("o%d" % t,
                                               observation_dist(next_state, action))]
        with poutine.block(hide_fn=lambda site: site["name"].startswith("bu")):
            next_belief = belief_update(belief, action, observation,
                                        num_steps=bu_nsteps, lr=bu_lr, suffix=str(t))
        # action_weights = pyro.param("action_weights", action_weights)
        next_action = belief_policy_model(next_belief, t+1,
                                          discount=discount,
                                          discount_factor=discount_factor,
                                          max_depth=max_depth)
        return reward + discount*belief_value_model(next_belief, next_action, t+1,
                                                    discount=discount,
                                                    discount_factor=discount_factor,
                                                    max_depth=max_depth,
                                                    bu_nsteps=bu_nsteps,
                                                    bu_lr=bu_lr)


def belief_policy_model_guide(belief, t, discount=1.0, discount_factor=0.95, max_depth=10,
                              bu_nsteps=10, bu_lr=0.1):
    weights = pyro.param("action_weights", torch.ones(len(actions)),
                         constraint=dist.constraints.simplex)
    for i, action in enumerate(actions):
        with scope(prefix="%s%d" % (action,t)):
            value = belief_value_model(belief, action, t,
                                       discount=discount, discount_factor=discount_factor,
                                       max_depth=max_depth,
                                       bu_nsteps=bu_nsteps,
                                       bu_lr=bu_lr)
    action = pyro.sample("a%d" % t, dist.Categorical(weights))


def plan(belief, max_depth=3, discount_factor=0.95,
         lr=0.1, nsteps=100, print_losses=True,
         bu_nsteps=10, bu_lr=0.01):
    """nsteps (int) number of iterations to reduce the loss"""
    pyro.clear_param_store()
    try:
        svi = pyro.infer.SVI(belief_policy_model,
                             belief_policy_model_guide,
                             pyro.optim.Adam({"lr": lr}),
                             loss=pyro.infer.Trace_ELBO(retain_graph=True))
        Infer(svi, belief, 0,
              discount=1.0, discount_factor=discount_factor, max_depth=max_depth,
              num_steps=nsteps, print_losses=print_losses,
              bu_nsteps=bu_nsteps, bu_lr=bu_lr)
    finally:
        print("------------")
        for name in pyro.get_param_store():
            print("{}: {}".format(name, pyro.param(name)))
        print("------------")
    return pyro.param("action_weights")


def main(state="tiger-left", sim_steps=10):
    # Simulate agent and planning and belief updates
    max_depth = 2
    discount_factor = 0.95
    bu_nsteps = 10   # number of steps for belief update (increasing this blows up planning time)
    bu_lr = 0.5   # learning rate when performing belief update
    nsteps = 100
    lr = 0.1

    # prior belief
    belief = dist.Categorical(tensor([1., 1., 0.]))

    plan_times = []
    rewards = []

    for i in range(sim_steps):
        print("\n--- Step %d ---" % i)
        print("State: %s" % state)
        print("Belief: %s" % belief.probs)
        start_time = time.time()
        weights = plan(belief, max_depth=max_depth,
                       discount_factor=discount_factor,
                       nsteps=nsteps, lr=lr, print_losses=False,
                       bu_nsteps=bu_nsteps, bu_lr=bu_lr)
        plan_times.append(time.time() - start_time)
        print("Time taken %.4fs" % plan_times[-1])
        action = actions[torch.argmax(weights).item()]
        action_weights = pyro.param("action_weights").detach().numpy()
        print("Action to take: %s" % action)
        print("Action weights: %s" % str(action_weights))

        # state transition, observation, reward, belief update
        next_state = states[pyro.sample("s'", transition_dist(state, action))]
        # The environment gives an observation without noise
        observation = observations[pyro.sample("o", observation_dist(next_state, action, noise=0.0))]
        reward = pyro.sample("r", reward_dist(state, action, next_state))
        rewards.append(reward)
        print("Next State: %s" % next_state)
        print("Observation: %s" % observation)
        print("Reward: %s" % reward.item())

        # Plot action weights
        state_weights = belief.probs.detach().numpy()  # i.e. belief
        df_a = pd.DataFrame({"actions": actions,
                             "action_weights": action_weights})
        df_b = pd.DataFrame({"states": states,
                             "state_weights": state_weights})

        sns.barplot(data=df_a,
                    x="actions",
                    y="action_weights")
        plt.title("t = %d | state = %s" % (i, state))
        plt.savefig("figs/pomdp-tiger_%d_%s_action.png" % (i, state))
        plt.clf()

        # Plot belief weights
        sns.barplot(data=df_b,
                    x="states",
                    y="state_weights",
                    palette="rocket")
        plt.title("t = %d | state = %s" % (i, state))
        plt.savefig("figs/pomdp-tiger_%d_%s_belief.png" % (i, state))
        plt.clf()

        print("Updating belief...")
        belief = belief_update(belief, action, observation, num_steps=1000,
                               print_losses=False)

        if next_state == "terminal":
            print("Done.")
            break

        # update state
        state = next_state

    d=1.0
    disc_reward = 0
    for r in rewards:
        disc_reward += d*r
        d *= discount_factor
    print("Average time: %.4fs" % np.mean(plan_times))
    print("Discounted cumulative reward: %.4f" % disc_reward)


if __name__ == "__main__":
    main()
