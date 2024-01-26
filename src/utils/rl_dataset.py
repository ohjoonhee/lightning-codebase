import torch
from typing import Iterator, Tuple
from .replay_buffer import ReplayBuffer

from torch.utils.data import IterableDataset
from torch.utils.data import Dataset


class IterableRLDataset(IterableDataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        state, action, reward, done, next_state = self.buffer.sample(self.sample_size)
        for i in range(len(done)):
            yield state[i], action[i], reward[i], done[i], next_state[i]


class RLDataset(Dataset):
    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        super().__init__()

        self.buffer = buffer
        self.sample_size = sample_size

    def __getitem__(self, idx: int) -> Tuple:
        s, a, r, d, s2 = self.buffer.buffer[idx]
        return [
            torch.tensor(s),
            torch.tensor(a),
            torch.tensor(r, dtype=torch.float32),
            torch.tensor(d, dtype=torch.bool),
            torch.tensor(s2),
        ]

    def __len__(self) -> int:
        return len(self.buffer.buffer)
