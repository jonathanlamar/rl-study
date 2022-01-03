from os import environ
import os
import time
from typing import List

import gym
from gym.core import Env
import matplotlib.pyplot as plt
import numpy as np


class FrozenLakeAgent:
    """
    Class for experimenting with TD and MCMC methods on the frozen lake gym environment.
    """

    env: Env
    returnAverages: np.ndarray
    hitCounts: np.ndarray
    stateHistory: List[int]
    actionHistory: List[int]
    rewardHistory: List[float]

    def __init__(self, stepwiseDiscount: float) -> None:
        """
        Create an instance of an agent.

        params:
            stepwiseDiscount: float
                The discount for return calculations at the step-level.
        """

        self.stepwiseDiscount = stepwiseDiscount
        self.qualityFn = np.zeros((16, 4))
        self.env = gym.make("FrozenLake-v1")
        self.returnAverages = np.zeros((16, 4))
        self.hitCounts = np.zeros((16, 4))
        self.resetHistory()

    def resetHistory(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []

    def renderEnv(self) -> None:
        """Print the current state and previous action on screen, if applicable."""
        self.env.render()

    def doEpisode(
        self, numSteps: int, epsilon: float, render: bool = False, fps: int = 10
    ) -> tuple[int, bool]:
        """
        Interact with the environment and save the history of these interactions.
        """

        state = self.env.reset()

        if render:
            time.sleep(1 / fps)
            os.system("clear")
            self.renderEnv()

        t = 0
        reward = 0
        for t in range(numSteps):
            action = self.getEpsilonGreedyAction(state, eps=epsilon)
            newState, reward, done = self.act(action)

            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
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
        for i in reversed(range(len(self.states))):
            state = self.states[i]
            action = self.actions[i]
            reward = self.rewards[i]
            ret = self.stepwiseDiscount * ret + reward

            # Update running average observed return, but only for first visit at
            # (state, action)
            if (state, action) not in zip(self.states[:i], self.actions[:i]):
                self.hitCounts[state, action] += 1
                m = self.hitCounts[state, action]
                self.returnAverages[state, action] = (
                    (m - 1) / m
                ) * self.returnAverages[state, action] + (1 / m) * ret

        self.qualityFn = self.returnAverages
        self.resetHistory()

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
            # equivalent to argmax(actionValues), but breaks ties randomly
            actionValues = self.qualityFn[state]
            return np.random.choice(np.flatnonzero(actionValues == actionValues.max()))


if __name__ == "__main__":
    numEpisodes = int(environ.get("EPISODES", 100000))
    numSteps = int(environ.get("STEPS", 100))
    # This doesn't affect the model, just vizualization
    render = bool(environ.get("RENDER", False))
    fps = int(environ.get("FPS", 10))

    wins = np.zeros(numEpisodes)
    agent = FrozenLakeAgent(stepwiseDiscount=0.9)
    for i in range(numEpisodes):
        eps = 10e-4
        _, win = agent.doEpisode(numSteps, epsilon=eps, render=render, fps=fps)
        wins[i] = win
        agent.updateQualityFn()

    plt.figure()
    plt.plot(wins.cumsum(), label="total wins")
    plt.legend()
    plt.xlabel("episode")
    plt.grid()
    plt.show()
