from os import environ
import os
import time
from typing import Dict, List

from scipy.special import softmax
import gym
from gym.core import Env
import numpy as np


class FrozenLakeAgent:
    """
    Class for experimenting with TD and MCMC methods on the frozen lake gym environment.
    """

    env: Env
    history: List[Dict[str, List[float]]]

    def __init__(self, stepwiseDiscount: float, episodicDiscount: float) -> None:
        """
        Create an instance of an agent.

        params:
            stepwiseDiscount: float
                The discount for return calculations at the step-level.
            episodicDiscount: float
                The discount for policy updates at the episode-level.
        """

        self.stepwiseDiscount = stepwiseDiscount
        self.episodicDiscount = episodicDiscount
        self.policy = softmax(np.random.rand(16, 4), axis=1)
        self.resetEnv()
        self.history = []

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
        states = [state]
        # Prefix with 0 for consistent indexing
        actions = [0.0]
        rewards = [0.0]

        if render:
            time.sleep(1 / fps)
            os.system("clear")
            self.renderEnv()

        t = 0
        reward = 0
        for t in range(numSteps):
            action = self.getGreedyAction(state)
            newState, reward, done = self.act(action)

            # Do something with this information..?
            states.append(newState)
            actions.append(action)
            rewards.append(reward)

            state = newState
            if done:
                break

        self.history.append(
            {
                "states": states,
                "actions": actions,
                "rewards": rewards,
            }
        )

        return t, reward == 1

    def updatePolicy(self) -> None:
        """
        Update policy function with returns from previous episodes.  This will clear
        the existing episode history.
        """

        returns = np.zeros((16, 4))
        hitCounts = np.zeros((16, 4))

        for ep in self.history:
            ret = 0
            n = len(ep["states"])

            # Skip the 0th time step because no action was taken
            for i in reversed(range(1, n)):
                state = ep["states"][i]
                action = ep["actions"][i]
                reward = ep["rewards"][i]
                ret = reward + self.stepwiseDiscount * ret

                # Update running average observed return
                hitCounts[state, action] += 1
                m = hitCounts[state, action]
                returns[state, action] = ((m - 1) / m) * returns[state, action] + (
                    1 / m
                ) * ret

        self.policy = softmax(
            self.episodicDiscount * self.policy + (1 - self.episodicDiscount) * returns,
            axis=1,
        )
        self.history = []

    def act(self, action: int) -> tuple[int, float, bool]:
        """Take action and return relevant information."""

        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def getGreedyAction(self, state: int) -> int:
        """Follows best action accoridng to policy function."""

        return self.getEpsilonGreedyAction(state, eps=0.0)

    def getEpsilonGreedyAction(self, state: int, eps: float) -> int:
        """
        Follows best action according to policy function with 1-eps probability.
        Otherwise picks random action.
        """

        e = np.random.rand()
        if e < eps:
            return np.random.randint(0, 4)
        else:
            return np.random.choice(4, p=self.policy[state])


if __name__ == "__main__":
    numEpochs = int(environ.get("EPOCHS", 1000))
    numEpisodes = int(environ.get("EPISODES", 1000))
    numSteps = int(environ.get("STEPS", 100))
    # This doesn't affect the model, just vizualization
    render = bool(environ.get("RENDER", False))
    fps = int(environ.get("FPS", 10))

    agent = FrozenLakeAgent(stepwiseDiscount=0.5, episodicDiscount=0.0)
    avgStepCounts = np.zeros(numEpochs)
    winRates = np.zeros(numEpochs)

    for i in range(numEpochs):
        agent.resetEnv()
        stepCounts = np.zeros(numEpisodes)
        wins = np.zeros(numEpisodes)

        for j in range(numEpisodes):
            t, win = agent.doEpisode(numSteps, render=render, fps=fps)
            stepCounts[j] = t
            wins[j] = int(win)

        avgStepCount = stepCounts.mean()
        winRate = wins.mean()
        avgStepCounts[i] = avgStepCount
        winRates[i] = winRate

        os.system("clear")
        print(
            f"Epoch {i:04d}: win rate: {winRate:1.4f}, ",
            f"avg step count: {avgStepCount:2.2f}",
        )

        agent.updatePolicy()
