# Models for the Tiger problem

import torch
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt

states = ["tiger-left", "tiger-right"]
observations = ["growl-left", "growl-right"]
actions = ["open-left", "open-right", "listen"]

# Generative model for observation
def observation_model(next_state, action, noise=0.15, t=0):
    if action == "listen":
        if next_state == "tiger-left":
            probs = [1.0-noise, noise]
        else:
            probs = [noise, 1.0-noise]
    else:
        probs = [0.5, 0.5]
    o = pyro.sample('o_%d' % t, dist.Categorical(torch.tensor(probs)))
    return observations[o]

# Generative model for state; deterministic
def transition_model(state, action, noise=0.15):
    return state

# Generative model for reward; deterministic
def reward_model(state, action):
    reward = 0
    if action == "open-left":
        if state== "tiger-right":
            reward += 10
        else:
            reward -= 100
    elif action == "open-right":
        if state== "tiger-left":
            reward += 10
        else:
            reward -= 100
    elif action == "listen":
        reward -= 1
    return reward

print(observation_model("tiger-left", "listen"))
print(transition_model("tiger-left", "listen"))
print(reward_model("tiger-left", "listen"))

# # We want to sample a1, ..., aT from the following distribution
# # Pr ( a1, ... aT | b0, Criterion_T=c)
# def forward_model(b0, T=5):
#     # Sample a number of actions uniformly at random
#     # Compute updated beliefs
#     # Compute the criterion at the time step T
#     for t in range(T):
#         index = pyro.sample("a_%d" % t, dist.Categorical(torch.tensor([1.0, 1.0, 1.0])))
#         at = actions[index]



# Write a model for Pr (a1,r1, ..., aT,rT | s);  Condition on state because I don't
# know how to write when conditioned on belief.
# Then infer Pr(a1,...,aT | s, r1, ..., rT)
def forward_model(s0, T=5,
                  action_prior=[1.0, 1.0, 1.0]):
    history = []
    st = s0
    for t in range(T):
        a_params = []
        for i in range(len(actions)):
            p = pyro.param("pr_a%d=%d" % (t, i), torch.tensor(action_prior[i]))
            a_params.append(p)
        index = pyro.sample("a_%d" % t, dist.Categorical(torch.tensor(a_params)))
        at = actions[index]
        st_plus1 = transition_model(st, at)
        ot = observation_model(st, at, t=t)
        rt = pyro.sample("r_%d" % t, dist.Normal(loc=reward_model(st, at), scale=1e-12))

        history.append((st, at, ot, rt))        
        st = st_plus1
    return history

rewards = [-1,-1,-1,-1,10]
observation = {}
for t in range(len(rewards)):
    observation["r_%d" % t] = rewards[t]
conditioned_forward_model = pyro.condition(forward_model, data=observation)

# I don't understand yet how this works and why inference involves a loss function.
pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_forward_model,
                     guide=forward_model,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.1}),
                     loss=pyro.infer.Trace_ELBO())

losses = [] #, pr_left, pr_right, pr_listen  = [], [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step("tiger-left"))
    sys.stdout.write("%d/%d\r" % (t+1, num_steps))
sys.stdout.write("\n")

plt.plot(losses)
plt.title("ELBO")  # Evidence lower bound
plt.xlabel("step")
plt.ylabel("loss");
for t in range(len(rewards)):
    for i in range(len(actions)):
        param_name = "pr_a%d=%d" % (t, i)
        print("Pr(a_%d = %s) = %.3f" % (t, actions[i], pyro.param(param_name).item()))
plt.show()
# The parameters a and b turn out to be very close to the analytical solution.

# This tells me that if you want to use this, your distribution needs to
# be parameterized - instead of an arbitrary histogram.
