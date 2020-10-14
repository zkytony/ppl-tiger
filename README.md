# ppl-tiger

Using [Pyro](http://pyro.ai/) probabilistic programming package to implement
the Tiger POMDP domain, and solving it with planning as inference. This tutorial about [Modeling Agents with Probabilistic Programs](https://agentmodels.org/) using WebPPL is very helpful.

One shot, MDP, QMDP planning works. Belief update works. But POMDP planning does not work properly, most likely due to belief update SVI nested inside planning SVI does not accurately performs the belief update - planning also becomes very slow when you try to increase the number of optimization steps for the belief update SVI. Also contains a WebPPL implementation of Tiger [here]( ppl-tiger/tiger.webppl), which has no compilation error when running inside an input box on the [agent models site](https://agentmodels.org/), but does not finish (too slow?)...(a better algorithm to solve POMDPs using PPL is needed, one which does not require nested inference for belief update).

**More on POMDP planning does not work properly** I added a `stay` action to the tiger domain which does not change the state. A POMDP planner should be able to differentiate `stay` from `listen` and still chooses to take `listen`. POMCP can do this (check out [pomdp_py](https://github.com/h2r/pomdp-py)). QMDP cannot. The POMDP planner that is implemented here cannot.
