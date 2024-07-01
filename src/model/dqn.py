from typing import Any, Sequence, Tuple, Optional, Union
from lightning.pytorch.callbacks.callback import Callback
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, STEP_OUTPUT
from torch import Tensor
from lightning.pytorch.cli import OptimizerCallable

import os
import os.path as osp
import glob

import torch
from torch import nn
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from model.offpolicy import OffPolicyAgent
from policy.mlp import MLP
from utils.replay_buffer import ReplayBuffer, Experience
from utils.rl_dataset import IterableRLDataset, RLDataset


class DQNModel(L.LightningModule):
    def __init__(
        self,
        optimizer: OptimizerCallable = torch.optim.Adam,
        batch_size: int = 16,
        env: Optional[str] = None,
        env_kwargs: Optional[dict[str, Any]] = None,
        reset_options: Optional[dict[str, Any]] = None,
        gamma: float = 0.99,
        sync_rate: int = 10,
        replay_size: int = 1000,
        warm_start_size: int = 1000,
        eps_last_frame: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        episode_length: int = 1000,
        warm_start_steps: int = 1000,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_rate = sync_rate
        self.replay_size = replay_size
        self.warm_start_size = warm_start_size
        self.eps_last_frame = eps_last_frame
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.episode_length = episode_length
        self.warm_start_steps = warm_start_steps

        self.env: gym.Env = gym.make(env, **env_kwargs)
        self.reset_options = reset_options
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = MLP(obs_size, n_actions)
        self.target_net = MLP(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.replay_size)
        self.agent = OffPolicyAgent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.populate(self.warm_start_steps)

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

        expected_q = q_next * self.gamma + r

        return nn.MSELoss()(q, expected_q)

    def get_epsilon(self, start: int, end: int, frames: int) -> float:
        if self.global_step > frames:
            return end
        return start + self.global_step * (end - start) / frames

    def get_device(self, batch):
        return batch[0].device.index if self.on_gpu else "cpu"

    def training_step(self, batch: Tuple[Tensor, Tensor], nb_batch):
        device = self.get_device(batch)
        epsilon = self.get_epsilon(self.eps_start, self.eps_end, self.eps_last_frame)
        self.log("epsilon", epsilon)

        reward, done = self.agent.play_step(self.net, epsilon, device=device)
        self.episode_reward += reward
        self.log("episode_reward", self.episode_reward)

        loss = self.dqn_mse_loss(batch)

        if done:
            self.total_reward += self.episode_reward
            self.episode_reward = 0
            self.episode_count += 1

        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict({"reward": reward, "loss": loss, "episode": self.episode_count})

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
        agent = OffPolicyAgent(record_env, ReplayBuffer(1))
        agent.reset()

        for _ in range(self.episode_length):
            reward, done = agent.play_step(self.net, epsilon=0.0, device=device)
            if done:
                agent.reset()

        record_env.close()

        if isinstance(self.logger, WandbLogger):
            video_file_list = glob.glob(osp.join(save_dir, "*.mp4"))
            video_file = (
                video_file_list[-2] if len(video_file_list) > 1 else video_file_list[0]
            )
            self.logger.experiment.log(
                {"videos": wandb.Video(video_file, fps=4, format="gif")}
            )

    def __dataloader(self):
        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )
        return dataloader

    def train_dataloader(self):
        return self.__dataloader()

    def val_dataloader(self):
        return self.__dataloader()
