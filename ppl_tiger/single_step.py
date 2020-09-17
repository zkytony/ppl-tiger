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
    alpha = torch.tensor(5.0)
    beta = torch.tensor(10.0)
    a_probs = pyro.sample("a_probs", dist.Beta(alpha, beta).expand([3]).independent(1))
    a_probs = a_probs / torch.sum(a_probs)  # normalize

    a = pyro.sample("a", dist.Categorical(a_probs))
    r = pyro.deterministic("r", tensor(reward_model(s0, actions[a])))
    return a, r

def guide(s0):
    alpha = pyro.param('alphas', torch.tensor(6.).expand([3]), constraint=dist.constraints.positive)
    beta = pyro.param('betas', torch.tensor(7.).expand([3]), constraint=dist.constraints.positive)
    a_probs = pyro.sample("a_probs", dist.Beta(alpha, beta).independent(1))

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
num_steps = 20000
for t in range(num_steps):
    loss = svi.step("tiger-left")
    sys.stdout.write("%d/%d\r" % (t+1, num_steps))
    if t % 100 == 0:
        print("Loss [%d] = %.3f" % (t, loss))
        losses.append(loss)
sys.stdout.write("\n")

plt.plot(losses)
plt.title("ELBO")  # Evidence lower bound
plt.xlabel("step")
plt.ylabel("loss");
alphas = pyro.param("alphas")
betas = pyro.param("betas")
print("alphas =", alphas)
print("betas =", betas)
for i in range(10):
    a_probs = pyro.sample("a_probs", dist.Beta(alphas, betas).independent(1))
    a_probs = a_probs / torch.sum(a_probs)  # normalize
    a = pyro.sample("a", dist.Categorical(a_probs))
    print(i, a_probs, actions[a])
plt.show()
# The parameters a and b turn out to be very close to the analytical solution.

# This tells me that if you want to use this, your distribution needs to
# be parameterized - instead of an arbitrary histogram.

