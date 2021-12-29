from os import environ
import os
import time
from typing import Dict, List
from IPython import embed

import gym
from gym.core import Env
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax


class FrozenLakeAgent:
    """
    Class for experimenting with TD and MCMC methods on the frozen lake gym environment.
    """

    env: Env
    history: Dict[str, List[float]]

    def __init__(self, stepwiseDiscount: float) -> None:
        """
        Create an instance of an agent.

        params:
            stepwiseDiscount: float
                The discount for return calculations at the step-level.
            episodicDiscount: float
                The discount for quality updates at the episode-level.
        """

        self.stepwiseDiscount = stepwiseDiscount
        self.qualityFn = np.zeros((16, 4))
        self.resetEnv()
        self.returns = np.zeros((16, 4))
        self.hitCounts = np.zeros((16, 4))
        self.history = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

    def resetEnv(self) -> None:
        """Spawn new environment (refresh board)"""
        self.env = gym.make("FrozenLake-v1")

    def renderEnv(self) -> None:
        """Print the current state and previous action on screen, if applicable."""

        self.env.render()

    def doEpisode(
        self, numSteps: int, render: bool = False, fps: int = 10
    ) -> tuple[int, bool]:
        """
        Interact with the environment and save the history of these interactions.
        """

        state = self.env.reset()
        self.history["states"] = [state]
        # Prefix with 0 for consistent indexing
        self.history["actions"] = [0.0]
        self.history["rewards"] = [0.0]

        if render:
            time.sleep(1 / fps)
            os.system("clear")
            self.renderEnv()

        t = 0
        reward = 0
        for t in range(numSteps):
            action = self.getGreedyAction(state)
            newState, reward, done = self.act(action)

            # Changing the reward to punish falling into a hole
            if done and reward == 0:
                reward = -1

            # Do something with this information..?
            self.history["states"].append(newState)
            self.history["actions"].append(action)
            self.history["rewards"].append(reward)

            state = newState
            if done:
                break

        return t, reward == 1

    def updateQualityFn(self) -> None:
        """
        Update quality function with returns from previous episodes.  This will clear
        the existing episode history.
        """

        ret = 0
        n = len(self.history["states"])

        # Skip the 0th time step because no action was taken
        for i in reversed(range(1, n)):
            state = self.history["states"][i]
            action = self.history["actions"][i]
            reward = self.history["rewards"][i]
            ret = reward + self.stepwiseDiscount * ret

            # Update running average observed return, but only for first visit at
            # state
            if state not in self.history["states"][:i]:
                self.hitCounts[state, action] += 1
                m = self.hitCounts[state, action]
                self.returns[state, action] = ((m - 1) / m) * self.returns[
                    state, action
                ] + (1 / m) * ret

        self.qualityFn = self.returns
        self.history = {
            "states": [],
            "actions": [],
            "rewards": [],
        }

    def act(self, action: int) -> tuple[int, float, bool]:
        """Take action and return relevant information."""

        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def getGreedyAction(self, state: int) -> int:
        """Follows best action accoridng to quality function."""

        return self.getEpsilonGreedyAction(state, eps=0.0)

    def getEpsilonGreedyAction(self, state: int, eps: float) -> int:
        """
        Follows best action according to quality function with 1-eps probability.
        Otherwise picks random action.
        """

        e = np.random.rand()
        if e < eps:
            return np.random.randint(0, 4)
        else:
            policy = softmax(self.qualityFn[state])
            return np.random.choice(4, p=policy)


if __name__ == "__main__":
    numEpisodes = int(environ.get("EPISODES", 100000))
    numSteps = int(environ.get("STEPS", 100))
    # This doesn't affect the model, just vizualization
    render = bool(environ.get("RENDER", False))
    fps = int(environ.get("FPS", 10))

    wins = np.zeros(numEpisodes)
    agent = FrozenLakeAgent(stepwiseDiscount=0.9)
    for i in range(numEpisodes):
        _, win = agent.doEpisode(numSteps, render=render, fps=fps)
        wins[i] = win
        agent.updateQualityFn()

    plt.figure()
    plt.plot(wins.cumsum(), label="total wins")
    plt.legend()
    plt.xlabel("episode")
    plt.show()
