from typing import Any, Sequence, Tuple, Optional, Union
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from lightning.pytorch.cli import OptimizerCallable

import os
import os.path as osp

import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as L

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from module.offpolicy import OffPolicyAgent
from models.mlp import MLP
from utils.replay_buffer import ReplayBuffer, Experience
from utils.rl_dataset import RLDataset, MapRLDataset


class DQNModule(L.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        batch_size: int = 16,
        env: Optional[gym.Env] = None,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 200,
        warm_start_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["env"])
        self.optimizer = optimizer

        self.env = env
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = MLP(obs_size, n_actions)
        self.target_net = MLP(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = OffPolicyAgent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def configure_optimizers(self) -> Any:
        opt = self.optimizer(self.net.parameters())
        return opt

    def populate(self, steps: int = 1000) -> None:
        for _ in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def dqn_mse_loss(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        s, a, r, d, s2 = batch
        q = self.net(s).gather(1, a[..., None]).squeeze(-1)

        with torch.no_grad():
            q_next = self.target_net(s2).max(1)[0]
            q_next[d] = 0.0
            q_next = q_next.detach()

        expected_q = q_next * self.hparams.gamma + r

        return nn.MSELoss()(q, expected_q)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start + self.global_step * (end - start) / frames

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch):
        device = self.get_device(batch)
        epsilon = self.get_epsilon(
            self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame
        )
        self.log("epsilon", epsilon)

        reward, done = self.agent.play_step(self.net, epsilon, device=device)
        self.episode_reward += reward
        self.log("episode_reward", self.episode_reward)

        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward += self.episode_reward
            self.episode_reward = 0

        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({"reward": reward, "loss": loss})

        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return loss

    def validation_step(self, *args: Any, **kwargs: Any):
        return

    def on_validation_epoch_end(self):
        device = next(self.net.parameters()).device
        self.save_dir = self.logger.log_dir or self.logger.save_dir
        save_dir = osp.join(self.save_dir, "videos", f"video_{self.current_epoch}")

        record_env = RecordVideo(self.env, video_folder=save_dir, disable_logger=True)
        agent = OffPolicyAgent(record_env, self.buffer)
        agent.reset()

        for _ in range(1000):
            reward, done = agent.play_step(self.net, epsilon=0.0, device=device)
            if done:
                agent.reset()

        record_env.close()

    def __dataloader(self):
        # dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataset = MapRLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return dataloader

    def train_dataloader(self):
        return self.__dataloader()

    def val_dataloader(self):
        return self.__dataloader()
