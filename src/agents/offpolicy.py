from typing import Tuple
import gymnasium as gym
from utils.replay_buffer import Experience, ReplayBuffer

import numpy as np
import torch
from torch import nn


class OffPolicyAgent:
    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state, info = self.env.reset()

    def reset(self) -> None:
        self.state, info = self.env.reset()

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
        s2, r, term, trunc, _ = self.env.step(action)
        d = term or trunc
        exp = Experience(self.state, action, r, d, s2)
        self.replay_buffer.append(exp)
        self.state = s2
        if d:
            self.reset()
        return r, d
