from typing import Tuple, Optional, Any
import gymnasium as gym
from utils.replay_buffer import Experience, ReplayBuffer

import numpy as np
import torch
from torch import nn


class OffPolicyAgent:
    def __init__(
        self,
        env: gym.Env,
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()

    def reset(self, *, seed=None, options=None) -> None:
        self.state, info = self.env.reset(seed=seed, options=options)

    def get_action(self, net: nn.Module, epsilon: float, device: str = "cpu") -> int:
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(self.state).unsqueeze(0).to(device)
            q_values = net(state)
            return q_values.argmax(dim=1).item()

    @torch.no_grad()
    def play_step(
        self, net: nn.Module, epsilon: float, device: str = "cpu"
    ) -> Tuple[float, bool]:
        action = self.get_action(net, epsilon, device)
        state, reward, term, trunc, _ = self.env.step(action)
        done = term or trunc
        exp = Experience(self.state, action, reward, done, state)
        self.replay_buffer.append(exp)
        self.state = state
        if done:
            self.reset()
        return reward, done
