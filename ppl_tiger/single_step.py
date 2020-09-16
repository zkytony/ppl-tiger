import torch
import torch.tensor as tensor
import pyro
import pyro.distributions as dist
import sys
import matplotlib.pyplot as plt

actions = ["open-left", "open-right", "listen"]

# Let's model a single step Pr (a, r | s0)
#
# Pr (a, r | s0) = Pr (r | a, s0) Pr(a | s0)

# Generative model for reward; deterministic
def reward_model(state, action):
    reward = 0
    if action == "open-left":
        if state == "tiger-right":
            reward += 10
        else:
            reward -= 100
    elif action == "open-right":
        if state == "tiger-left":
            reward += 10
        else:
            reward -= 100
    elif action == "listen":
        reward -= 1
    return reward

def model(s0):
    prob_open_left = tensor(1.0/3.0)
    prob_open_right = tensor(1.0/3.0)
    prob_listen = tensor(1.0/3.0)
    a = pyro.sample("a", dist.Categorical(
        tensor([prob_open_left, prob_open_right, prob_listen])))
    r = pyro.deterministic("r", tensor(reward_model(s0, actions[a])))
    return a, r

def guide(s0):
    prob_open_left = pyro.param("prob_open_left", tensor(1.0/3.0))
    prob_open_right = pyro.param("prob_open_right", tensor(1.0/3.0))
    prob_listen = pyro.param("prob_listen", tensor(1.0/3.0))
    pyro.sample("a", dist.Categorical(
        tensor([prob_open_left, prob_open_right, prob_listen])))


observation = {"r": tensor(-1)}
conditioned_forward_model = pyro.condition(model, data=observation)
print(conditioned_forward_model("tiger-left"))

# I don't understand yet how this works and why inference involves a loss function.
pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_forward_model,
                     guide=guide,
                     optim=pyro.optim.Adam({"lr": 0.0001}),
                     loss=pyro.infer.Trace_ELBO())

losses = [] #, pr_left, pr_right, pr_listen  = [], [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step("tiger-left"))
    sys.stdout.write("%d/%d\r" % (t+1, num_steps))
    if t % 100 == 0:
        print("Loss [%d] = %.3f" % (t, losses[-1]))
sys.stdout.write("\n")

plt.plot(losses)
plt.title("ELBO")  # Evidence lower bound
plt.xlabel("step")
plt.ylabel("loss");
print("prob_open_left = %.3f" % pyro.param("prob_open_left").item())
print("prob_open_right = %.3f" % pyro.param("prob_open_right").item())
print("prob_listen = %.3f" % pyro.param("prob_listen").item())
plt.show()
# The parameters a and b turn out to be very close to the analytical solution.

# This tells me that if you want to use this, your distribution needs to
# be parameterized - instead of an arbitrary histogram.

