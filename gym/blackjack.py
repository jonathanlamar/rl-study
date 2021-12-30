from os import environ
import os
import time
from typing import Dict, List, Tuple

import gym
from gym.core import Env
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

BlackjackState = Tuple[int, int, bool]


class BlackjackAgent:
    """
    Class for experimenting with TD and MCMC methods on the frozen lake gym environment.
    """

    stepwiseDiscount: float
    env: Env
    qualityFn: Dict[BlackjackState, Dict[bool, float]]
    returnAverages: Dict[BlackjackState, Dict[bool, float]]
    hitCounts: Dict[BlackjackState, Dict[bool, float]]
    stateHistory: List[BlackjackState]
    actionHistory: List[bool]
    rewardHistory: List[float]

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
        self.env = gym.make("Blackjack-v1")
        self.qualityFn = {
            (s, d, bool(a)): {bool(h): 0.0 for h in range(2)}
            for s in range(32)
            for d in range(11)
            for a in range(2)
        }
        self.returnAverages = {
            (s, d, bool(a)): {bool(h): 0.0 for h in range(2)}
            for s in range(32)
            for d in range(11)
            for a in range(2)
        }
        self.hitCounts = {
            (s, d, bool(a)): {bool(h): 0 for h in range(2)}
            for s in range(32)
            for d in range(11)
            for a in range(2)
        }
        self.resetHistory()

    def resetHistory(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []

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
            # if done and reward == 0:
            #     reward = -1

            # Do something with this information..?
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
            ret = reward + self.stepwiseDiscount * ret

            # Update running average observed return, but only for first visit at state
            if state not in self.states[:i]:
                self.hitCounts[state][action] += 1
                m = self.hitCounts[state][action]
                self.returnAverages[state][action] = (
                    (m - 1) / m
                ) * self.returnAverages[state][action] + (1 / m) * ret

        self.qualityFn = self.returnAverages
        self.resetHistory()

    def act(self, action: int) -> BlackjackState:
        """Take action and return relevant information."""
        state, reward, done, _ = self.env.step(action)
        return state, reward, done

    def getGreedyAction(self, state: BlackjackState) -> int:
        """Follows best action accoridng to quality function."""
        return self.getEpsilonGreedyAction(state, eps=0.0)

    def getEpsilonGreedyAction(self, state: BlackjackState, eps: float) -> bool:
        """
        Follows best action according to quality function with 1-eps probability.
        Otherwise picks random action.
        """
        e = np.random.rand()
        if e < eps:
            return bool(np.random.choice(2))
        else:
            policy = softmax(list(self.qualityFn[state].values()))
            return bool(np.random.choice(2, p=policy))


if __name__ == "__main__":
    numEpisodes = int(environ.get("EPISODES", 100000))
    numSteps = int(environ.get("STEPS", 100))
    # This doesn't affect the model, just vizualization
    render = bool(environ.get("RENDER", False))
    fps = int(environ.get("FPS", 10))

    wins = np.zeros(numEpisodes)
    agent = BlackjackAgent(stepwiseDiscount=0.9)
    for i in range(numEpisodes):
        _, win = agent.doEpisode(numSteps, render=render, fps=fps)
        wins[i] = win
        agent.updateQualityFn()

    plt.figure()
    plt.plot(wins.cumsum(), label="total wins")
    plt.legend()
    plt.xlabel("episode")
    plt.show()
