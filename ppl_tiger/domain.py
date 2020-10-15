import pyro.distributions as dist
import torch.tensor as tensor
import torch


states = ["tiger-left", "tiger-right"]#, "terminal"]
states_without_terminal = ["tiger-left", "tiger-right"]
observations = ["growl-left", "growl-right"]
actions = ["open-left", "open-right", "listen"]#, "stay"]

# state. Otherwise, the game resets.
TERMINATES = "terminal" in states


def observation_dist(next_state, action, noise=0.15):
    """
    Args:
        next_state (str)  next state
        action (str)  action
    Returns:
        The observation distribution to sample from
    """
    if action == "listen":
        if next_state == "tiger-left":
            obs_probs = [1.0-noise, noise]
        elif next_state == "tiger-right":
            obs_probs = [noise, 1.0-noise]
        else:  # terminal
            obs_probs = [0.5, 0.5]
    else:
        obs_probs = [0.5, 0.5]
    return dist.Categorical(tensor(obs_probs))

def transition_dist(state, action):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        The transition distribution to sample from
    """
    if action == "open-left" or action == "open-right":
        if not TERMINATES:
            return dist.Categorical(tensor([.5, .5]))
        else:
            next_state = "terminal"
    else:
        next_state = state

    trans_probs = torch.zeros(len(states))
    trans_probs[states.index(next_state)] = 1.0

    return dist.Categorical(trans_probs)

def reward_dist(state, action, next_state):
    """
    Args:
        state (str)  state
        action (str)  action
    Returns:
        A tensor (with a single integer) sampled from the reward function
    """
    if not TERMINATES:
        if action == "open-left":
            if state == "tiger-right":
                reward = 10.
            else:
                reward = -100.
        elif action == "open-right":
            if state == "tiger-left":
                reward = 10.
            else:
                reward = -100.
        else:  # listen
            reward = -1.
    else:
        reward = 1e-9
        if next_state == "terminal":
            if action == "open-left":
                if state == "tiger-right":
                    reward = 10.0
                elif state == "tiger-left":
                    reward = -100.0
            elif action == "open-right":
                if state == "tiger-left":
                    reward = 10.0
                elif state == "tiger-right":
                    reward = -100.0
            else:
                reward = -1.0
        else:
            reward = -1.0
    return dist.Delta(tensor(reward))
